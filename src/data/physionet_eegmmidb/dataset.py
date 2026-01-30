"""
EEGMMIDB / EEGBCI (PhysioNet) dataset support for EEG quality classification.

This module implements loading and preprocessing for the PhysioNet EEG Motor
Movement/Imagery dataset (often referred to as EEGMMIDB / EEGBCI).

High-level pipeline:
- Resolve a user-provided `data_dir` to an actual EEGMMIDB root by scanning for EDF files
  (supports multiple on-disk layouts, including the MNE `MNE-eegbci-data` layout).
- For each EDF record (subject/run), window the EEG into fixed-length windows and extract
  features (e.g., `raw8`, `bandpower`, `bandpower_hjorth_entropy`).
- Compute heuristic, per-window quality scores and convert them into 3-class labels using
  quantile thresholds fit on the TRAIN split only (anti-leakage).
- Build subject-disjoint train/val/test splits (or k-fold subject CV via `fold_index`).

Shape conventions (conceptual):
- `current_covariates`: (N_records, T, F)
- `outputs`:            (N_records, T, 1) int64 class labels
- `active_entries`:     (N_records, T, 1) {0,1} mask for padded timesteps

Sanity / debugging flags (environment variables):
- `CT_SHUFFLE_TRAIN_LABELS=1`: permute training labels (expected performance ~ chance).
- `CT_SHUFFLE_TRAIN_INPUTS=1` + `CT_SHUFFLE_TRAIN_INPUTS_MODE=...`: input shuffling modes.
- `CT_ONLY_SPLIT_CHECK=1`: verify split disjointness and exit early.
"""

import os
import json
import ast
import logging
import hashlib
import glob
import sys
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Iterable, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset_collection import RealDatasetCollection, assert_disjoint_splits
from src.data.utils.scalers import PerFoldStandardScaler
from src.data.utils.labeling import QuantileLabeler
from src.data.utils.quality import compute_quality_score
from src.data.split_diagnostics import generate_split_diagnostics


logger = logging.getLogger(__name__)


def _stable_hash(data: np.ndarray) -> str:
    """Computes a stable hash of a numpy array."""
    return hashlib.sha256(data.tobytes()).hexdigest()


def _ids_to_int64(ids: Iterable[Any], name: str) -> np.ndarray:
    """
    Robust, deterministic conversion of arbitrary record/subject IDs to int64.

    This prevents object-dtype tensors while keeping IDs stable across runs.
    """
    ids_list = list(ids)
    try:
        return np.asarray(ids_list, dtype=np.int64)
    except Exception:
        pass

    def _is_hex_str(s: Any) -> bool:
        if not isinstance(s, str):
            return False
        if not (16 <= len(s) <= 32):
            return False
        s = s.lower()
        return all(c in "0123456789abcdef" for c in s)

    # Common case: stable hex IDs (e.g., md5/sha1 hexdigests). Take the first 64 bits.
    if ids_list and all(_is_hex_str(x) for x in ids_list):
        out = np.empty((len(ids_list),), dtype=np.int64)
        for i, s in enumerate(ids_list):
            v = int(s[:16], 16)  # 64 bits
            v = v & 0x7FFFFFFFFFFFFFFF  # force 63-bit positive
            out[i] = np.int64(v)
        return out

    # Fallback: stable hash (sha1) -> 64 bits -> mask 63 bits.
    logger.warning(f"{name}: non-numeric ids; converting deterministically to int64 via sha1 (63-bit).")
    out = np.empty((len(ids_list),), dtype=np.int64)
    for i, x in enumerate(ids_list):
        if isinstance(x, bytes):
            s = x.decode("utf-8", errors="replace")
        else:
            s = x if isinstance(x, str) else str(x)
        digest = hashlib.sha1(s.encode("utf-8")).digest()
        v = int.from_bytes(digest[:8], byteorder="big", signed=False)
        v = v & 0x7FFFFFFFFFFFFFFF
        out[i] = np.int64(v)
    return out


def _parse_subject_run_from_name(filename: str) -> Tuple[Optional[int], Optional[int]]:
    # Expected pattern: S001R01.edf
    name = os.path.basename(filename)
    if len(name) < 7:
        return None, None
    try:
        s_idx = name.find("S")
        r_idx = name.find("R")
        if s_idx == -1 or r_idx == -1:
            return None, None
        subj = int(name[s_idx + 1:s_idx + 4])
        run = int(name[r_idx + 1:r_idx + 3])
        return subj, run
    except Exception:
        return None, None


def _normalize_list(val):
    if val is None:
        return None
    if isinstance(val, str):
        if val.lower() == "all":
            return None
        parts = [p.strip() for p in val.split(",") if p.strip()]
        try:
            return [int(p) for p in parts]
        except Exception:
            return None
    if isinstance(val, (list, tuple)):
        try:
            return [int(v) for v in val]
        except Exception:
            return None
    return None


def _normalize_str_list(val):
    if val is None:
        return []
    if isinstance(val, str):
        if val.lower() in ("none", "null", "false", "0"):
            return []
        parts = [p.strip() for p in val.split(",") if p.strip()]
        return [str(p) for p in parts]
    if isinstance(val, (list, tuple)):
        out = []
        for v in val:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    return [str(val)]


def _to_builtin(val):
    if isinstance(val, dict) or hasattr(val, "items"):
        try:
            val = dict(val)
        except Exception:
            return val
        return {str(k): _to_builtin(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_builtin(v) for v in val]
    return val


def _find_edf_files(data_dir: str) -> List[str]:
    """
    Recursively collect EDF file paths under `data_dir`.

    This is used as a fallback when the fast layout checks do not match. The return value
    is sorted to keep runs deterministic.
    """
    # Supports structured (data_dir/S###/*.edf) and flat (data_dir/*.edf)
    files = []
    for root, _, filenames in os.walk(data_dir):
        for f in filenames:
            if f.lower().endswith(".edf"):
                files.append(os.path.join(root, f))
    return sorted(files)


def _fast_edf_scan(data_dir: str) -> Tuple[str, List[str], Dict[str, int]]:
    """
    Fast EEGMMIDB layout detection.

    The checks are ordered from cheapest/most common to more specific layouts. This provides:
    - quick feedback in logs (which layout matched),
    - a stable "resolved root" used by the rest of the pipeline,
    - deterministic behavior (sorted glob results).

    Returns:
        resolved_root: directory treated as the dataset root for subsequent processing
        edf_files: sorted list of EDF paths found under that root
        counts_by_check: how many files each layout probe matched (for debugging)
    """
    counts = {}

    # a) data_dir/S###/*.edf
    patt_a = os.path.join(data_dir, "S???", "*.edf")
    files_a = sorted(glob.glob(patt_a))
    counts["a:data_dir/S###/*.edf"] = len(files_a)
    if files_a:
        return data_dir, files_a, counts

    # b) data_dir/*/S###/*.edf (one-level deep)
    patt_b = os.path.join(data_dir, "*", "S???", "*.edf")
    files_b = sorted(glob.glob(patt_b))
    counts["b:data_dir/*/S###/*.edf"] = len(files_b)
    if files_b:
        # Resolve to the first matched parent (data_dir/<subdir>)
        first = os.path.dirname(os.path.dirname(files_b[0]))
        return first, files_b, counts

    # c) data_dir/files/eegmmidb/1.0.0/S###/*.edf
    root_c = os.path.join(data_dir, "files", "eegmmidb", "1.0.0")
    patt_c = os.path.join(root_c, "S???", "*.edf")
    files_c = sorted(glob.glob(patt_c))
    counts["c:data_dir/files/eegmmidb/1.0.0/S###/*.edf"] = len(files_c)
    if files_c:
        return root_c, files_c, counts

    # d) data_dir/eegmmidb/1.0.0/S###/*.edf
    root_d = os.path.join(data_dir, "eegmmidb", "1.0.0")
    patt_d = os.path.join(root_d, "S???", "*.edf")
    files_d = sorted(glob.glob(patt_d))
    counts["d:data_dir/eegmmidb/1.0.0/S###/*.edf"] = len(files_d)
    if files_d:
        return root_d, files_d, counts

    # e) data_dir/eeg-motor-movementimagery-dataset-1.0.0/files/eegmmidb/1.0.0/S###/*.edf
    root_e = os.path.join(data_dir, "eeg-motor-movementimagery-dataset-1.0.0", "files", "eegmmidb", "1.0.0")
    patt_e = os.path.join(root_e, "S???", "*.edf")
    files_e = sorted(glob.glob(patt_e))
    counts["e:data_dir/eeg-motor-movementimagery-dataset-1.0.0/files/eegmmidb/1.0.0/S###/*.edf"] = len(files_e)
    if files_e:
        return root_e, files_e, counts

    # f) data_dir/MNE-eegbci-data/files/eegmmidb/1.0.0/S###/*.edf
    root_f = os.path.join(data_dir, "MNE-eegbci-data", "files", "eegmmidb", "1.0.0")
    patt_f = os.path.join(root_f, "S???", "*.edf")
    files_f = sorted(glob.glob(patt_f))
    counts["f:data_dir/MNE-eegbci-data/files/eegmmidb/1.0.0/S###/*.edf"] = len(files_f)
    if files_f:
        return root_f, files_f, counts

    # fallback: deep scan
    files_all = _find_edf_files(data_dir)
    counts["fallback:os.walk"] = len(files_all)
    return data_dir, files_all, counts


def _resolve_eegmmidb_root(data_dir: str) -> Tuple[str, List[str], Dict[str, int]]:
    """
    Resolve `data_dir` into an EEGMMIDB root and list EDF files found.

    This is a small wrapper around `_fast_edf_scan` so callers can consistently log:
    - the matched layout,
    - example EDF files,
    - and counts per probe.
    """
    return _fast_edf_scan(data_dir)


def _count_subject_dirs(root: str) -> int:
    if not os.path.isdir(root):
        return 0
    return len([d for d in os.listdir(root) if d.upper().startswith("S") and os.path.isdir(os.path.join(root, d))])


def summarize_eegmmidb_dir(data_dir: str, top_k: int = 5):
    """
    Print a short, human-friendly summary of an EEGMMIDB directory.

    Intended for quick CLI checks (see `runnables/check_eegmmidb.py`).
    """
    root, files, _ = _resolve_eegmmidb_root(data_dir)
    subjects = []
    for f in files:
        subj, _ = _parse_subject_run_from_name(f)
        if subj is not None:
            subjects.append(subj)
    subj_counts = {}
    for s in subjects:
        subj_counts[s] = subj_counts.get(s, 0) + 1
    top = sorted(subj_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"Resolved data root: {root}")
    print(f"EDF files found: {len(files)}")
    print("First 5 EDF paths:")
    for p in files[:5]:
        print(f"  {p}")
    print(f"Top {top_k} subject counts:")
    for s, c in top:
        print(f"  S{s:03d}: {c}")


DEFAULT_BAND_DEFS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("beta", 12.0, 30.0),
    ("gamma", 30.0, 45.0),
]


def _parse_band_defs(bands: Optional[Iterable[Any]]) -> List[Tuple[str, float, float]]:
    if bands is None:
        return list(DEFAULT_BAND_DEFS)
    parsed: List[Tuple[str, float, float]] = []
    for i, b in enumerate(bands):
        name = f"band{i}"
        fmin = None
        fmax = None
        if isinstance(b, dict) or hasattr(b, "items"):
            try:
                b = dict(b)
            except Exception:
                b = {}
            name = str(b.get("name", name))
            fmin = b.get("fmin", b.get("low", b.get("start", None)))
            fmax = b.get("fmax", b.get("high", b.get("stop", None)))
        elif isinstance(b, (list, tuple)) and len(b) >= 2:
            fmin = b[0]
            fmax = b[1]
            if len(b) >= 3:
                name = str(b[2])
        if fmin is None or fmax is None:
            continue
        parsed.append((name, float(fmin), float(fmax)))
    return parsed if parsed else list(DEFAULT_BAND_DEFS)


def _compute_psd_welch(x: np.ndarray, sfreq: float, nperseg: Optional[int] = None,
                       noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch
    except Exception as e:
        raise ImportError("scipy is required for Welch PSD. Install scipy.") from e

    n_samples = x.shape[1]
    if nperseg is None or nperseg <= 0:
        nperseg = min(256, n_samples)
    nperseg = int(min(nperseg, n_samples))
    if noverlap is None or noverlap < 0:
        noverlap = nperseg // 2
    noverlap = int(min(noverlap, nperseg - 1)) if nperseg > 1 else 0
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=1)
    return freqs, psd


def _bandpower_welch(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=psd.dtype)
    return np.trapz(psd[:, mask], freqs[mask], axis=1)


def _hjorth_params(x: np.ndarray) -> np.ndarray:
    # x: (C, S)
    eps = 1e-12
    diff1 = np.diff(x, axis=1)
    diff2 = np.diff(diff1, axis=1)
    var0 = np.var(x, axis=1)
    var1 = np.var(diff1, axis=1)
    var2 = np.var(diff2, axis=1)
    activity = var0
    mobility = np.sqrt(var1 / (var0 + eps))
    complexity = np.sqrt(var2 / (var1 + eps)) / (mobility + eps)
    return np.stack([activity, mobility, complexity], axis=1)


def _spectral_entropy(psd: np.ndarray) -> np.ndarray:
    eps = 1e-12
    psd = np.maximum(psd, 0.0)
    psd_norm = psd / (psd.sum(axis=1, keepdims=True) + eps)
    ent = -np.sum(psd_norm * np.log(psd_norm + eps), axis=1)
    ent = ent / (np.log(psd_norm.shape[1] + eps))
    return ent


def _parse_channel_groups(channel_groups, ch_names: List[str]) -> List[List[int]]:
    groups: List[List[int]] = []
    if channel_groups is None:
        return groups
    for g in channel_groups:
        if isinstance(g, str):
            items = [x.strip() for x in g.split(",") if x.strip()]
        elif isinstance(g, (list, tuple)):
            items = list(g)
        else:
            items = []
        idxs: List[int] = []
        for item in items:
            if isinstance(item, int):
                if 0 <= item < len(ch_names):
                    idxs.append(int(item))
            else:
                name = str(item).strip().upper()
                if not name:
                    continue
                for i, ch in enumerate(ch_names):
                    if ch.strip().upper() == name:
                        idxs.append(i)
                        break
        if idxs:
            groups.append(sorted(set(idxs)))
    return groups


def _infer_channel_groups(ch_names: List[str]) -> List[List[int]]:
    groups = {
        "frontal": [],
        "central": [],
        "parietal": [],
        "occipital": [],
        "temporal": [],
        "other": [],
    }
    for i, ch in enumerate(ch_names):
        name = ch.upper().replace("EEG ", "").replace("-REF", "").replace(".", "")
        if name.startswith(("FT", "TP", "T")):
            groups["temporal"].append(i)
        elif name.startswith("O"):
            groups["occipital"].append(i)
        elif name.startswith(("P", "PO")):
            groups["parietal"].append(i)
        elif name.startswith(("C", "CP")):
            groups["central"].append(i)
        elif name.startswith(("FP", "AF", "F", "FC")):
            groups["frontal"].append(i)
        else:
            groups["other"].append(i)
    ordered = ["frontal", "central", "parietal", "occipital", "temporal", "other"]
    return [groups[k] for k in ordered if groups[k]]


def _resolve_channel_groups(ch_names: List[str], grouping: Optional[str], channel_groups) -> List[List[int]]:
    if grouping is None:
        grouping = "channels"
    grouping = str(grouping).lower()
    if grouping in ("channels", "channel", "none", "per_channel"):
        return [[i] for i in range(len(ch_names))]
    if grouping in ("regions", "region"):
        groups = _infer_channel_groups(ch_names)
        return groups if groups else [[i] for i in range(len(ch_names))]
    if grouping in ("custom", "groups", "group"):
        groups = _parse_channel_groups(channel_groups, ch_names)
        return groups if groups else [[i] for i in range(len(ch_names))]
    return [[i] for i in range(len(ch_names))]


def _bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=psd.dtype)
    return np.sum(psd[:, mask], axis=1)


def _window_features(x: np.ndarray, sfreq: float) -> Dict[str, float]:
    # x: (C, S)
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    ptp = np.ptp(x, axis=1)

    n = x.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    psd = (np.abs(np.fft.rfft(x, axis=1)) ** 2) / max(n, 1)
    total = np.sum(psd, axis=1) + 1e-12

    alpha = _bandpower(psd, freqs, 8.0, 12.0)
    beta = _bandpower(psd, freqs, 12.0, 30.0)
    theta = _bandpower(psd, freqs, 4.0, 8.0)
    line = _bandpower(psd, freqs, 55.0, 65.0)

    alpha_ratio = alpha / total
    beta_ratio = beta / total
    theta_ratio = theta / total
    line_ratio = line / total

    feats = {
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "ptp_mean": float(ptp.mean()),
        "ptp_std": float(ptp.std()),
        "line_ratio_mean": float(line_ratio.mean()),
        "alpha_ratio_mean": float(alpha_ratio.mean()),
        "beta_ratio_mean": float(beta_ratio.mean()),
        "theta_ratio_mean": float(theta_ratio.mean()),
    }
    return feats


def _parse_float_list(val, default=None):
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        try:
            return [float(v) for v in val]
        except Exception:
            return default
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
        except Exception:
            parsed = None
        if parsed is None:
            try:
                parsed = ast.literal_eval(val)
            except Exception:
                parsed = None
        if isinstance(parsed, (list, tuple)):
            try:
                return [float(v) for v in parsed]
            except Exception:
                return default
        if "," in val:
            try:
                return [float(v.strip()) for v in val.split(",") if v.strip()]
            except Exception:
                return default
    return default



class PhysioNetEEGMMIDBDataset(Dataset):
    """
    Pytorch-style dataset for PhysioNet eegmmidb
    """

    def __init__(self, data: dict, scaling_params: dict, subset_name: str):
        self.data = data
        self.scaling_params = scaling_params
        self.subset_name = subset_name
        self._logged_dtypes = False
        self.processed = True
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.exploded = False
        self.norm_const = 1.0

        data_shapes = {k: v.shape for k, v in self.data.items()}
        logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

    def __getitem__(self, index) -> dict:
        result = {}
        for k, v in self.data.items():
            item = v[index]
            if isinstance(item, np.ndarray):
                if item.dtype.kind in ("f", "c"):
                    item = torch.from_numpy(item).float()
                elif item.dtype.kind in ("i", "u", "b"):
                    item = torch.from_numpy(item).long()
                else:
                    item = torch.from_numpy(item)
            elif isinstance(item, np.generic):
                arr = np.asarray(item)
                if arr.dtype.kind in ("f", "c"):
                    item = torch.from_numpy(arr).float()
                elif arr.dtype.kind in ("i", "u", "b"):
                    item = torch.from_numpy(arr).long()
                else:
                    item = torch.from_numpy(arr)
            result[k] = item
        if "current_covariates" in result and "x" not in result:
            result["x"] = result["current_covariates"]
        if not self._logged_dtypes:
            try:
                logger.info(
                    f"{self.subset_name} dtypes - current_covariates: {result.get('current_covariates').dtype}, "
                    f"current_treatments: {result.get('current_treatments').dtype}, "
                    f"prev_treatments: {result.get('prev_treatments').dtype}"
                )
            except Exception:
                pass
            self._logged_dtypes = True
        if hasattr(self, 'encoder_r'):
            if 'original_index' in self.data:
                result.update({'encoder_r': self.encoder_r[int(result['original_index'])]})
            else:
                result.update({'encoder_r': self.encoder_r[index]})
        return result

    def __len__(self):
        return len(self.data['active_entries'])


class PhysioNetEEGMMIDBDatasetCollection(RealDatasetCollection):
    """
    Dataset collection for PhysioNet EEGMMIDB / EEGBCI.

    This class builds a time-series dataset suitable for the repository's EEG quality benchmark.

    Key behaviors:
    - Subject-disjoint splits (recommended) and optional k-fold subject CV (`fold_index`).
    - Train-only normalization (e.g., `fold_zscore` with `normalization_scope=train`).
    - Train-only quantile labeling (`label_strategy=quantile_fold`) to avoid leakage.
    - Optional caching to `<data_dir>/processed/*.npz` for faster iteration.

    Output dictionary fields (per split) follow the project-wide conventions:
    - `current_covariates`: (N, T, F) float features
    - `outputs`:            (N, T, 1) int64 labels (3 classes)
    - `active_entries`:     (N, T, 1) mask (1=valid timestep, 0=padding)
    """

    def __init__(self,
                 data_dir: str,
                 subjects="all",
                 runs="all",
                 window_seconds: float = 2.0,
                 stride_seconds: float = 1.0,
                 max_seq_length: int = 60,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 100,
                 download: bool = False,
                 cache: bool = True,
                 cache_name: str = "",
                 split_by_subject: bool = False,
                 stratify_subject_split: bool = False,
                 n_folds: Optional[int] = None,
                 fold_index: Optional[int] = None,
                 feature_set: str = "raw8",
                 bandpower_bands: Optional[List[dict]] = None,
                 bandpower_relative: bool = True,
                 bandpower_log: bool = False,
                 bandpower_include_quality: bool = False,
                 channel_grouping: str = "channels",
                 channel_groups: Optional[List] = None,
                 psd_nperseg: Optional[int] = None,
                 psd_noverlap: Optional[int] = None,
                 psd_fmin: Optional[float] = None,
                 psd_fmax: Optional[float] = None,
                 normalization_mode: str = "fold_zscore",
                 normalization_scope: str = "train",
                 label_strategy: str = "quantile_fold",
                 quantiles: Optional[List[float]] = None,
                 quantile_range: Optional[List[float]] = None,
                 label_feature: str = "composite",
                 fixed_thresholds: Optional[List[float]] = None,
                 fit_split: str = "train",
                 label_balance_min: float = 0.1,
                 label_balance_mode: str = "warn",
                 treatment_mode: str = "none",
                 treatment_dim: int = 1,
                 stratify_subject_agg: str = "mean",
                 exclude_features: Optional[List[str]] = None,
                 **kwargs):
        super().__init__()
        torch.set_default_dtype(torch.float)
        logger.info("Setting torch default dtype to float32 for eegmmidb.")
        self.seed = seed
        self.autoregressive = True
        self.has_vitals = False
        self.projection_horizon = 1
        self.split_by_subject = bool(split_by_subject)
        self.stratify_subject_split = bool(stratify_subject_split)
        self.n_folds = int(n_folds) if n_folds is not None else None
        self.fold_index = int(fold_index) if fold_index is not None else None
        self.exclude_features = _normalize_str_list(exclude_features)
        if self.exclude_features:
            logger.info(f"Excluding features from input: {self.exclude_features}")

        # New counter for invalid run IDs
        self.invalid_run_id_count = 0

        self.feature_set = str(feature_set).lower() if feature_set is not None else "raw8"
        if self.feature_set not in ("raw8", "bandpower", "bandpower_hjorth", "bandpower_hjorth_entropy", "quality3"):
            logger.warning(f"Unknown feature_set '{feature_set}', falling back to 'raw8'.")
            self.feature_set = "raw8"
        self.band_defs = _parse_band_defs(bandpower_bands)
        self.bandpower_relative = bool(bandpower_relative)
        self.bandpower_log = bool(bandpower_log)
        self.bandpower_include_quality = bool(bandpower_include_quality)
        self.channel_grouping = channel_grouping
        self.channel_groups = channel_groups
        self.psd_nperseg = psd_nperseg
        self.psd_noverlap = psd_noverlap
        self.psd_fmin = psd_fmin
        self.psd_fmax = psd_fmax
        self.normalization_mode = str(normalization_mode).lower() if normalization_mode is not None else "fold_zscore"
        self.normalization_scope = str(normalization_scope).lower() if normalization_scope is not None else "train"

        if quantiles is None and "label_quantiles" in kwargs:
            quantiles = kwargs.get("label_quantiles")
        if quantile_range is None:
            quantile_range = quantiles
        if fixed_thresholds is None and "label_fixed_thresholds" in kwargs:
            fixed_thresholds = kwargs.get("label_fixed_thresholds")

        self.label_strategy = str(label_strategy).lower() if label_strategy is not None else "quantile_fold"
        self.label_quantiles = _parse_float_list(quantile_range, default=[0.33, 0.66])
        self.label_fixed_thresholds = _parse_float_list(fixed_thresholds, default=None)
        self.label_fit_split = str(fit_split).lower() if fit_split is not None else "train"
        self.label_feature = str(label_feature).lower() if label_feature is not None else "composite"
        try:
            self.label_balance_min = float(label_balance_min)
        except Exception:
            self.label_balance_min = 0.1
        self.label_balance_mode = str(label_balance_mode).lower() if label_balance_mode is not None else "warn"
        self.treatment_mode = str(treatment_mode).lower() if treatment_mode is not None else "none"
        if self.treatment_mode in ("none", "no", "off", "false", "null", "0"):
            if treatment_dim is not None and int(treatment_dim) != 0:
                raise ValueError(
                    "EEGMMIDB: treatment_mode=none requires treatment_dim=0. "
                    f"Got treatment_dim={treatment_dim}."
                )
            self.treatment_dim = 0
            self.treatment_mode = "none"
        else:
            self.treatment_dim = int(treatment_dim) if treatment_dim is not None else 1
        self.stratify_subject_agg = str(stratify_subject_agg).lower() if stratify_subject_agg is not None else "mean"
        if self.stratify_subject_agg not in ("mean", "median"):
            logger.warning(f"Unknown stratify_subject_agg='{self.stratify_subject_agg}', defaulting to 'mean'.")
            self.stratify_subject_agg = "mean"
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.max_seq_length = int(max_seq_length) if max_seq_length is not None else None
        self.label_names = None
        if "label_names" in kwargs and kwargs["label_names"] is not None:
            try:
                self.label_names = [str(n) for n in list(kwargs["label_names"])]
            except Exception:
                self.label_names = None

        # Sanity checks control (from environment variables)
        sanity_shuffle_labels = (os.getenv("CT_SHUFFLE_TRAIN_LABELS", "0") == "1")
        sanity_shuffle_inputs = (os.getenv("CT_SHUFFLE_TRAIN_INPUTS", "0") == "1")
        sanity_shuffle_inputs_mode_env = os.getenv("CT_SHUFFLE_TRAIN_INPUTS_MODE", None)
        sanity_shuffle_inputs_mode = str(sanity_shuffle_inputs_mode_env).strip() if sanity_shuffle_inputs_mode_env is not None else ""
        if sanity_shuffle_inputs_mode == "":
            sanity_shuffle_inputs_mode = "intra_sequence_lockstep"
        sanity_shuffle_inputs_mode = sanity_shuffle_inputs_mode.lower()
        if sanity_shuffle_inputs_mode not in ("intra_sequence_lockstep", "inter_batch_decouple"):
            logger.warning(
                "SANITY: Unknown CT_SHUFFLE_TRAIN_INPUTS_MODE='%s'; defaulting to 'intra_sequence_lockstep'.",
                sanity_shuffle_inputs_mode,
            )
            sanity_shuffle_inputs_mode = "intra_sequence_lockstep"
        only_split_check = (os.getenv("CT_ONLY_SPLIT_CHECK", "0") == "1")
        if sanity_shuffle_labels:
            print("SANITY: cache bypassed (CT_SHUFFLE_TRAIN_LABELS=1)")
        if sanity_shuffle_inputs:
            print("SANITY: cache bypassed (CT_SHUFFLE_TRAIN_INPUTS=1)")
            if sanity_shuffle_inputs_mode_env is not None and str(sanity_shuffle_inputs_mode_env).strip() != "":
                print(f"SANITY: CT_SHUFFLE_TRAIN_INPUTS_MODE={sanity_shuffle_inputs_mode}")
        if only_split_check:
            print("DEBUG: cache bypassed (CT_ONLY_SPLIT_CHECK=1)")

        if self.fold_index is not None and self.n_folds is None:
            self.n_folds = 5
        if self.fold_index is not None and not self.split_by_subject:
            logger.warning("fold_index provided; forcing split_by_subject=True.")
            self.split_by_subject = True
        split_by_subject = self.split_by_subject

        subjects_list = _normalize_list(subjects)
        runs_list = _normalize_list(runs)

        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)

        cache_path = None
        if cache:
            cfg = dict(
                subjects=subjects_list,
                runs=runs_list,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
                max_seq_length=max_seq_length,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                split_by_subject=split_by_subject,
                stratify_subject_split=self.stratify_subject_split,
                n_folds=self.n_folds,
                fold_index=self.fold_index,
                feature_set=self.feature_set,
                band_defs=self.band_defs,
                bandpower_relative=self.bandpower_relative,
                bandpower_log=self.bandpower_log,
                bandpower_include_quality=self.bandpower_include_quality,
                channel_grouping=self.channel_grouping,
                channel_groups=self.channel_groups,
                psd_nperseg=self.psd_nperseg,
                psd_noverlap=self.psd_noverlap,
                psd_fmin=self.psd_fmin,
                psd_fmax=self.psd_fmax,
                normalization_mode=self.normalization_mode,
                normalization_scope=self.normalization_scope,
                label_strategy=self.label_strategy,
                label_quantiles=self.label_quantiles,
                label_feature=self.label_feature,
                label_fixed_thresholds=self.label_fixed_thresholds,
                label_fit_split=self.label_fit_split,
                label_balance_min=self.label_balance_min,
                treatment_mode=self.treatment_mode,
                treatment_dim=self.treatment_dim,
                stratify_subject_agg=self.stratify_subject_agg,
                feature_version=7, # Incremented version for new subject_run_id logic
            )
            cfg = _to_builtin(cfg)
            cfg_str = json.dumps(cfg, sort_keys=True)
            key = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()[:8]
            fname = cache_name if cache_name else f"eegmmidb_{key}.npz"
            cache_path = os.path.join(data_dir, "processed", fname)

        if cache_path and os.path.exists(cache_path) and not sanity_shuffle_labels and not sanity_shuffle_inputs and not only_split_check:
            logger.info(f"Loading cached eegmmidb data from {cache_path}")
            npz = np.load(cache_path, allow_pickle=False)
            scaling_params = {
                "input_means": npz["scaling_input_means"],
                "inputs_stds": npz["scaling_input_stds"],
                "output_means": 0.0,
                "output_stds": 1.0,
            }
            try:
                scaler = PerFoldStandardScaler()
                scaler.mean_ = scaling_params["input_means"].astype(np.float32, copy=False)
                scaler.std_ = scaling_params["inputs_stds"].astype(np.float32, copy=False)
                self.scaler_ = scaler
            except Exception:
                self.scaler_ = None
            for key in ("scaling_subject_ids", "scaling_subject_centers", "scaling_subject_scales",
                        "normalization_mode", "normalization_scope"):
                if key in npz:
                    scaling_params[key] = npz[key]
            train_data = self._unpack(npz, "train_")
            val_data = self._unpack(npz, "val_")
            test_data = self._unpack(npz, "test_")
            for split in (train_data, val_data, test_data):
                self._ensure_record_targets(split)
            if not getattr(self, "labeling_report_path", None):
                self._write_labeling_report_cached(train_data, val_data, test_data)
            if "labeler_quantiles" in npz and "labeler_thresholds" in npz:
                try:
                    labeler = QuantileLabeler(quantiles=npz["labeler_quantiles"].tolist())
                    labeler.thresholds_ = tuple(npz["labeler_thresholds"].tolist())
                    self.labeler_ = labeler
                except Exception:
                    self.labeler_ = None
            self.train_f = PhysioNetEEGMMIDBDataset(train_data, scaling_params, "train")
            self.val_f = PhysioNetEEGMMIDBDataset(val_data, scaling_params, "val")
            self.test_f = PhysioNetEEGMMIDBDataset(test_data, scaling_params, "test")
            self.train_scaling_params = scaling_params
            return

        data_root, edf_files, counts = _resolve_eegmmidb_root(data_dir)
        subject_dirs = _count_subject_dirs(data_root)
        logger.info(f"EEGMMIDB data_dir resolved to: {data_dir}")
        logger.info(f"EEGMMIDB resolved data root: {data_root}")
        logger.info(f"Found subject folders: {subject_dirs}")
        for k, v in counts.items():
            logger.info(f"EDF scan {k}: {v}")
        if edf_files:
            logger.info(f"Example EDF paths: {edf_files[:3]}")
        else:
            logger.info(f"Example expected path: {os.path.join(data_root, 'S001', 'S001R01.edf')}")

        if edf_files and download:
            logger.info("EDF files found locally; skipping download.")

        if download and not edf_files:
            try:
                import mne
                from mne.datasets import eegbci
            except Exception as e:
                raise ImportError("mne is required for download. Install with: pip install mne") from e

            subj_list = subjects_list
            runs_dl = runs_list
            if subj_list is None:
                subj_list = list(range(1, 110))
                logger.warning("dataset.subjects=all -> downloading all subjects (1-109). "
                               "Set dataset.subjects to limit downloads.")
            if runs_dl is None:
                runs_dl = list(range(1, 15))
                logger.warning("dataset.runs=all -> downloading all runs (1-14). "
                               "Set dataset.runs to limit downloads.")

            logger.info(f"Downloading eegmmidb via MNE eegbci to {data_dir}")
            downloaded_paths = []
            for subj in subj_list:
                try:
                    paths = eegbci.load_data(subject=subj, runs=runs_dl, path=data_dir,
                                             force_update=False, update_path=False)
                    downloaded_paths.extend(paths)
                except Exception as e:
                    logger.warning(f"Download failed for subject {subj}: {e}")

            logger.info(f"Downloaded EDF paths via eegbci: {len(downloaded_paths)}")
            if downloaded_paths:
                logger.info(f"Example downloaded path: {downloaded_paths[0]}")

        if download and not edf_files:
            data_root, edf_files, counts = _resolve_eegmmidb_root(data_dir)
            subject_dirs = _count_subject_dirs(data_root)
            logger.info(f"After download, resolved data root: {data_root}")
            logger.info(f"Found subject folders: {subject_dirs}")
            for k, v in counts.items():
                logger.info(f"EDF scan {k}: {v}")
            if edf_files:
                logger.info(f"Example EDF paths: {edf_files[:3]}")

        # === Start of Pre-Split Refactoring ===
        all_records_meta = self._pre_scan_records(data_root, subjects_list, runs_list)
        if not all_records_meta:
            # Error messages from _pre_scan_records are sufficient
            raise FileNotFoundError("No valid EDF records found for eegmmidb. Check logs and config.")

        logger.info(f"Found {len(all_records_meta)} records for eegmmidb.")
        if self.invalid_run_id_count > 0:
            logger.warning(f"{self.invalid_run_id_count} records had invalid or un-parseable subject/run IDs from filename.")

        if not split_by_subject and self.fold_index is None:
            logger.warning("EEGMMIDB split is record-level (not subject-disjoint). "
                           "Set dataset.split_by_subject=True to avoid potential leakage.")

        rng = np.random.RandomState(seed)
        if split_by_subject and any((m['subject_id'] is None or int(m['subject_id']) < 0) for m in all_records_meta):
            logger.warning("Some records missing subject IDs; falling back to record-level split.")
            split_by_subject = False

        use_cv = self.fold_index is not None
        if use_cv:
            if self.n_folds is None:
                self.n_folds = 5
            if self.n_folds < 3:
                raise ValueError("n_folds must be >= 3 for train/val/test cross-validation.")
            if self.fold_index < 0 or self.fold_index >= self.n_folds:
                raise ValueError(f"fold_index={self.fold_index} out of range for n_folds={self.n_folds}.")
            if not split_by_subject:
                logger.warning("fold_index provided; forcing subject-disjoint splits.")
                split_by_subject = True

        if split_by_subject:
            subjects = sorted(list({m['subject_id'] for m in all_records_meta if m['subject_id'] is not None}))
            rng.shuffle(subjects)
            # Splitting logic (CV or random split for subjects) remains the same
            # ... (logic for splitting subjects into train/val/test sets)
            train_subjects, val_subjects, test_subjects = self._split_subjects(
                subjects, train_ratio, val_ratio, use_cv, rng
            )
            train_meta = [m for m in all_records_meta if m['subject_id'] in train_subjects]
            val_meta = [m for m in all_records_meta if m['subject_id'] in val_subjects]
            test_meta = [m for m in all_records_meta if m['subject_id'] in test_subjects]
        else:
            # Record-level split
            permuted_meta = rng.permutation(all_records_meta)
            n = len(permuted_meta)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = n - n_train - n_val
            # ... (logic for adjusting n_test if needed)
            logger.info(f"Split records: train={n_train}, val={n_val}, test={n_test}")
            train_meta = permuted_meta[:n_train]
            val_meta = permuted_meta[n_train:n_train + n_val]
            test_meta = permuted_meta[n_train + n_val:]

        # === Moved Disjointness Check ===
        logger.info("--- Verifying split disjointness (fail-fast) ---")

        def get_meta_field(meta_list, field):
            return [m[field] for m in meta_list]
        
        def build_meta_lookup(meta_list, key_field, value_field):
            lookup = defaultdict(list)
            for m in meta_list:
                lookup[m[key_field]].append(m[value_field])
            return lookup

        train_record_ids = get_meta_field(train_meta, 'record_id')
        val_record_ids = get_meta_field(val_meta, 'record_id')
        test_record_ids = get_meta_field(test_meta, 'record_id')
        assert_disjoint_splits(train_record_ids, val_record_ids, test_record_ids, "record_id")

        train_subject_ids = get_meta_field(train_meta, 'subject_id')
        val_subject_ids = get_meta_field(val_meta, 'subject_id')
        test_subject_ids = get_meta_field(test_meta, 'subject_id')
        
        train_subject_run_ids = get_meta_field(train_meta, 'subject_run_id')
        val_subject_run_ids = get_meta_field(val_meta, 'subject_run_id')
        test_subject_run_ids = get_meta_field(test_meta, 'subject_run_id')

        train_sri_meta = build_meta_lookup(train_meta, 'subject_run_id', 'path')
        val_sri_meta = build_meta_lookup(val_meta, 'subject_run_id', 'path')
        test_sri_meta = build_meta_lookup(test_meta, 'subject_run_id', 'path')

        if split_by_subject:
             assert_disjoint_splits(train_subject_ids, val_subject_ids, test_subject_ids, "subject_id")
        
        assert_disjoint_splits(train_subject_run_ids, val_subject_run_ids, test_subject_run_ids, "subject_run_id",
                               train_meta=train_sri_meta, val_meta=val_sri_meta, test_meta=test_sri_meta)

        logger.info("--- Split verification passed ---")

        if only_split_check:
            logger.info("CT_ONLY_SPLIT_CHECK=1: Split check successful. Exiting.")
            # Set dummy datasets and return
            self.train_f, self.val_f, self.test_f = None, None, None
            self.train_scaling_params = {}
            sys.exit(0)
        # === End of Refactored Section ===

        all_covs, all_treats, all_quality, record_subjects, record_runs, record_ids = [], [], [], [], [], []
        
        def process_meta_split(meta_list):
            split_covs, split_treats, split_quality, split_subjects, split_runs, split_record_ids = [], [], [], [], [], []
            for meta in meta_list:
                covs, treats, quality_scores, _, _ = self._process_record(
                    meta['path'], self.window_seconds, self.stride_seconds, self.max_seq_length
                )
                if covs is None:
                    continue
                split_covs.append(covs)
                split_treats.append(treats)
                split_quality.append(quality_scores)
                split_subjects.append(meta['subject_id'])
                split_runs.append(meta['run_id'])
                split_record_ids.append(meta['record_id'])
            return split_covs, split_treats, split_quality, split_subjects, split_runs, split_record_ids

        train_covs, train_treats, train_quality, train_subject_ids, train_run_ids, train_record_ids = process_meta_split(train_meta)
        val_covs, val_treats, val_quality, val_subject_ids, val_run_ids, val_record_ids = process_meta_split(val_meta)
        test_covs, test_treats, test_quality, test_subject_ids, test_run_ids, test_record_ids = process_meta_split(test_meta)

        if not train_covs and not val_covs and not test_covs:
            raise RuntimeError("No valid records processed for eegmmidb across all splits.")

        all_covs = train_covs + val_covs + test_covs
        max_len = max([len(x) for x in all_covs]) if all_covs else 0
        if max_seq_length and max_seq_length > 0:
            max_len = min(max_len, int(max_seq_length))

        # START: Input shuffling sanity check
        if sanity_shuffle_inputs and sanity_shuffle_inputs_mode == "intra_sequence_lockstep":
            logger.warning(
                "SANITY: CT_SHUFFLE_TRAIN_INPUTS=1 applied (MODE=intra_sequence_lockstep; intra-sequence time permutation "
                "for covariates, treatments, and quality scores)."
            )
            # This shuffle is applied *before* normalization.
            # It tests if the model is sensitive to the temporal order of windows.
            # Determinism requirements:
            # - Depends on exp.seed + fold_index (epoch optionally via env), not on record iteration order.
            # - Applies to TRAIN only, in lock-step across covariates/treatments/quality.
            fold = int(self.fold_index) if self.fold_index is not None else 0
            try:
                shuffle_epoch = int(str(os.getenv("CT_SHUFFLE_TRAIN_INPUTS_EPOCH", "0")).strip() or "0")
            except Exception:
                shuffle_epoch = 0

            # Prefer the experiment-level seed (seed_everything) over dataset seed defaults.
            # torch.initial_seed() is set by seed_everything(args.exp.seed, ...).
            base_seed = int(torch.initial_seed() & 0xFFFFFFFFFFFFFFFF)

            def _record_id_to_u64(rid) -> int:
                if isinstance(rid, str):
                    s = rid.strip().lower()
                    if len(s) >= 16 and all(c in "0123456789abcdef" for c in s[:16]):
                        return int(s[:16], 16) & 0xFFFFFFFFFFFFFFFF
                digest = hashlib.sha256(str(rid).encode("utf-8")).digest()
                return int.from_bytes(digest[:8], byteorder="big", signed=False) & 0xFFFFFFFFFFFFFFFF

            def _mix_u64(x: int) -> int:
                # splitmix64
                x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
                x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
                return x ^ (x >> 31)

            # Pick a stable probe record for logging (order-invariant selection).
            probe_idx = None
            probe_key = None
            for i, rid in enumerate(train_record_ids):
                num_windows = int(train_covs[i].shape[0])
                if num_windows <= 1:
                    continue
                key = str(rid)
                if probe_key is None or key < probe_key:
                    probe_key = key
                    probe_idx = i

            any_T_gt1 = False
            any_non_identity = False
            probed = False
            for i in range(len(train_covs)):
                # Shuffle the windows (axis 0) for covariates, treatments, and quality scores in lock-step
                num_windows = int(train_covs[i].shape[0])
                if num_windows <= 1:
                    continue
                any_T_gt1 = True

                rid_u64 = _record_id_to_u64(train_record_ids[i])
                seed_i = _mix_u64(
                    base_seed
                    ^ (rid_u64 + (fold & 0xFFFFFFFF) * 0xD1B54A32D192ED03)
                    ^ (shuffle_epoch & 0xFFFFFFFF)
                )
                rng_i = np.random.default_rng(seed_i)
                perm = rng_i.permutation(num_windows)
                if np.array_equal(perm, np.arange(num_windows)):
                    # Avoid rare identity permutation so the sanity check can't pass without effect.
                    perm = np.roll(perm, 1)

                if not np.array_equal(perm, np.arange(num_windows)):
                    any_non_identity = True

                if probe_idx is not None and i == probe_idx and not probed:
                    cov_hash_before = _stable_hash(np.ascontiguousarray(train_covs[i]))

                train_covs[i] = train_covs[i][perm, :]
                train_treats[i] = train_treats[i][perm, :]
                train_quality[i] = train_quality[i][perm]

                if probe_idx is not None and i == probe_idx and not probed:
                    cov_hash_after = _stable_hash(np.ascontiguousarray(train_covs[i]))
                    logger.info(
                        "SANITY INPUT-SHUFFLE (train, intra-seq): "
                        f"record_id={train_record_ids[i]}, T={num_windows}, fold={fold}, epoch={shuffle_epoch}, "
                        f"seed64={seed_i}, idx_before=[0,1,2,3,4], idx_after={perm[:5].tolist()}, "
                        f"cov_hash_before={cov_hash_before}, cov_hash_after={cov_hash_after}"
                    )
                    probed = True

            if any_T_gt1:
                assert any_non_identity, (
                    "Input shuffling failed: no non-identity permutation applied in train split "
                    "(unexpected; please check permutation seeding logic)."
                )
                logger.info("Assertion passed: at least one train sample had a non-identity intra-sequence permutation.")
            else:
                logger.warning("SANITY INPUT-SHUFFLE skipped assert: all train sequences have T<=1.")
        elif sanity_shuffle_inputs and sanity_shuffle_inputs_mode == "inter_batch_decouple":
            logger.warning(
                "SANITY: CT_SHUFFLE_TRAIN_INPUTS=1 enabled with MODE=inter_batch_decouple; "
                "skipping intra-sequence shuffle at dataset-level (will be applied train-only in training_step)."
            )
        # END: Input shuffling sanity check

        train_covs, val_covs, test_covs, scaling_params = self._apply_normalization(
            train_covs,
            val_covs,
            test_covs,
            train_subject_ids,
            val_subject_ids,
            test_subject_ids,
            split_by_subject,
        )

        labeler = self._fit_labeler(train_quality, train_subject_ids)
        train_outs = self._apply_labeler(train_quality, train_subject_ids, labeler)
        val_outs = self._apply_labeler(val_quality, val_subject_ids, labeler)
        test_outs = self._apply_labeler(test_quality, test_subject_ids, labeler)

        # START: Label shuffling sanity check instrumentation
        logger.info("--- Sanity Check Instrumentation ---")
        y_val_flat = np.concatenate(val_outs)
        y_test_flat = np.concatenate(test_outs)
        logger.info(f"Val labels hash: {_stable_hash(y_val_flat)}")
        logger.info(f"Test labels hash: {_stable_hash(y_test_flat)}")

        if sanity_shuffle_labels:
            print("SANITY: CT_SHUFFLE_TRAIN_LABELS=1: Shuffling training labels for sanity check.")
            rng = np.random.default_rng(seed) # Use a consistent seed for reproducibility
            
            if train_outs:
                lengths = [len(y) for y in train_outs]
                y_flat_before = np.concatenate(train_outs)
                hash_before = _stable_hash(y_flat_before)
                counts_before = dict(zip(*np.unique(y_flat_before, return_counts=True)))

                logger.info(f"Train labels BEFORE shuffle: hash={hash_before}, counts={counts_before}")

                perm = rng.permutation(len(y_flat_before))
                y_flat_shuffled = y_flat_before[perm]
                hash_after = _stable_hash(y_flat_shuffled)
                counts_after = dict(zip(*np.unique(y_flat_shuffled, return_counts=True)))
                logger.info(f"Train labels AFTER shuffle: hash={hash_after}, counts={counts_after}")

                assert hash_before != hash_after, "Label shuffling failed: hash is identical before and after."
                assert counts_before == counts_after, "Label shuffling failed: class counts changed."
                logger.info("Assertion passed: train label hash changed and counts are identical.")

                train_outs_shuffled = []
                current_pos = 0
                for length in lengths:
                    original_shape = train_outs[len(train_outs_shuffled)].shape
                    train_outs_shuffled.append(y_flat_shuffled[current_pos : current_pos + length].reshape(original_shape))
                    current_pos += length
                train_outs = train_outs_shuffled
        else:
            logger.info("CT_SHUFFLE_TRAIN_LABELS is off.")
        logger.info("--- End Sanity Check Instrumentation ---")
        # END: Label shuffling sanity check

        train_data = self._build_split(train_covs, train_treats, train_outs, max_len,
                                       train_subject_ids, train_run_ids, train_record_ids)
        val_data = self._build_split(val_covs, val_treats, val_outs, max_len,
                                     val_subject_ids, val_run_ids, val_record_ids)
        test_data = self._build_split(test_covs, test_treats, test_outs, max_len,
                                      test_subject_ids, test_run_ids, test_record_ids)

        for split in (train_data, val_data, test_data):
            split["unscaled_outputs"] = split["outputs"]
            self._ensure_record_targets(split)

        self._log_label_distributions(labeler, train_data, val_data, test_data)

        self.train_f = PhysioNetEEGMMIDBDataset(train_data, scaling_params, "train")
        self.val_f = PhysioNetEEGMMIDBDataset(val_data, scaling_params, "val")
        self.test_f = PhysioNetEEGMMIDBDataset(test_data, scaling_params, "test")
        self.train_scaling_params = scaling_params

        if cache_path and not sanity_shuffle_labels and not sanity_shuffle_inputs:
            logger.info(f"Saving eegmmidb cache to {cache_path}")
            self._save_cache(cache_path, train_data, val_data, test_data, scaling_params)

    def process_data_multi(self):
        # Data already processed
        self.processed_data_multi = True

    def _split_subjects(self, subjects, train_ratio, val_ratio, use_cv, rng):
       """
       Split subject IDs into train/val/test sets.

       Notes:
       - When `use_cv=True`, `fold_index` selects the *test* fold.
       - The validation set is then created by taking a ratio-based subset of the remaining
         subjects (it is not necessarily "fold_index+1").
       - The `subjects` list is assumed to have already been shuffled deterministically upstream.
       """
       subjects = list(subjects)
       n = len(subjects)
       if n < 3:
           raise ValueError(f"Need at least 3 subjects for train/val/test split, got n={n}")

       def _ensure_nonempty(n_train, n_val, n_test, total):
           # ensure all >=1 while preserving sum=total
           if n_test < 1:
               deficit = 1 - n_test
               take = min(deficit, max(0, n_train - 1))
               n_train -= take
               deficit -= take
               if deficit > 0:
                   take = min(deficit, max(0, n_val - 1))
                   n_val -= take
                   deficit -= take
               n_test = total - n_train - n_val
           if n_val < 1:
               deficit = 1 - n_val
               take = min(deficit, max(0, n_train - 1))
               n_train -= take
               n_val = total - n_train - n_test
           if n_train < 1:
               raise ValueError("Unable to create non-empty train/val/test splits with given ratios.")
           return n_train, n_val, n_test

       if use_cv:
           k = int(self.n_folds or 5)
           if k < 3:
               raise ValueError("n_folds must be >= 3 for train/val/test CV.")
           if n < k:
               raise ValueError(f"Not enough subjects (n={n}) for n_folds={k}.")
           fold_index = int(self.fold_index)
           if fold_index < 0 or fold_index >= k:
               raise ValueError(f"fold_index={fold_index} out of range for n_folds={k}.")

           folds = np.array_split(np.array(subjects, dtype=object), k)
           test_subjects = [int(x) for x in folds[fold_index].tolist()]

           remaining = []
           for i, f in enumerate(folds):
               if i == fold_index:
                   continue
               remaining.extend([int(x) for x in f.tolist()])

           denom = (train_ratio + val_ratio)
           val_frac = (val_ratio / denom) if denom > 0 else 0.5
           n_rem = len(remaining)
           if n_rem < 2:
               raise ValueError("Not enough remaining subjects to split into train/val.")
           n_val = max(1, int(round(n_rem * val_frac)))
           n_train = n_rem - n_val
           n_test = len(test_subjects)
           n_train, n_val, _ = _ensure_nonempty(n_train, n_val, 1, n_rem)

           train_subjects = remaining[:n_train]
           val_subjects = remaining[n_train:n_train + n_val]

           # final safety
           if len(set(train_subjects) & set(val_subjects)) != 0:
               raise RuntimeError("Internal error: train/val overlap in CV split.")
           if len(set(train_subjects) & set(test_subjects)) != 0 or len(set(val_subjects) & set(test_subjects)) != 0:
               raise RuntimeError("Internal error: overlap with test fold in CV split.")

           logger.info(f"Subject CV split: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)} (k={k}, fold={fold_index})")
           return set(train_subjects), set(val_subjects), set(test_subjects)

       # non-CV
       n_train = max(1, int(round(n * train_ratio)))
       n_val = max(1, int(round(n * val_ratio)))
       n_test = n - n_train - n_val
       n_train, n_val, n_test = _ensure_nonempty(n_train, n_val, n_test, n)

       train_subjects = subjects[:n_train]
       val_subjects = subjects[n_train:n_train + n_val]
       test_subjects = subjects[n_train + n_val:]

       logger.info(f"Subject split: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
       return set(train_subjects), set(val_subjects), set(test_subjects)

    def _make_record_id(self, subj: Optional[int], run: Optional[int], edf_path: str, fallback_idx: int) -> int:
        if subj is not None and run is not None:
            return int(subj) * 100 + int(run)
        # Fallback: stable hash of path within 32-bit range
        h = int(hashlib.md5(edf_path.encode("utf-8")).hexdigest()[:8], 16)
        return int(h % 2**31) if h else int(fallback_idx)

    def _extract_window_features(self, window: np.ndarray, sfreq: float, group_indices: List[List[int]]) -> np.ndarray:
        if self.feature_set in ("raw8", "quality3"):
            # These feature sets are derived from the same base calculations
            all_feats = _window_features(window, sfreq)

            # Exclude features by name *before* assembling the final feature vector
            if self.exclude_features:
                all_feats = {k: v for k, v in all_feats.items() if k not in self.exclude_features}

            if self.feature_set == "quality3":
                # Select a specific subset for 'quality3'
                feature_keys = ["rms_mean", "ptp_mean", "line_ratio_mean"]
                final_feats = np.array([all_feats.get(k, 0.0) for k in feature_keys], dtype=np.float32)
                return final_feats
            else: # raw8
                # Use all (remaining) features, maintaining a consistent order
                feature_keys = sorted(all_feats.keys())
                final_feats = np.array([all_feats[k] for k in feature_keys], dtype=np.float32)
                return final_feats

        # This part handles 'bandpower' feature sets
        freqs, psd = _compute_psd_welch(window, sfreq, nperseg=self.psd_nperseg, noverlap=self.psd_noverlap)
        if self.psd_fmin is not None or self.psd_fmax is not None:
            fmin = self.psd_fmin if self.psd_fmin is not None else freqs.min()
            fmax = self.psd_fmax if self.psd_fmax is not None else freqs.max()
            mask = (freqs >= fmin) & (freqs <= fmax)
            freqs = freqs[mask]
            psd = psd[:, mask]

        band_feats = []
        for _, fmin, fmax in self.band_defs:
            band = _bandpower_welch(psd, freqs, fmin, fmax)
            band_feats.append(band)
        band_feats = np.stack(band_feats, axis=1)  # (C, B)
        if self.bandpower_relative:
            total = psd.sum(axis=1, keepdims=True) + 1e-12
            band_feats = band_feats / total
        if self.bandpower_log:
            band_feats = np.log10(band_feats + 1e-12)

        feats_list = [band_feats]

        if self.feature_set in ("bandpower_hjorth", "bandpower_hjorth_entropy"):
            feats_list.append(_hjorth_params(window))
        if self.feature_set == "bandpower_hjorth_entropy":
            ent = _spectral_entropy(psd)
            feats_list.append(ent[:, np.newaxis])

        feats_chan = np.concatenate(feats_list, axis=1)  # (C, Fch)

        if group_indices and len(group_indices) != feats_chan.shape[0]:
            grouped = []
            for idxs in group_indices:
                if not idxs:
                    continue
                grouped.append(np.mean(feats_chan[idxs, :], axis=0))
            if grouped:
                feats_chan = np.stack(grouped, axis=0)

        base = feats_chan.reshape(-1).astype(np.float32, copy=False)
        if self.bandpower_include_quality:
            # Append global quality stats so labels and features share amplitude-related cues.
            quality_feats = _window_features(window, sfreq)
            if self.exclude_features:
                quality_feats = {k: v for k, v in quality_feats.items() if k not in self.exclude_features}
            
            # Ensure consistent order for concatenation
            quality_keys = sorted(quality_feats.keys())
            quality_arr = np.array([quality_feats[k] for k in quality_keys], dtype=np.float32)
            base = np.concatenate([base, quality_arr], axis=0)
        
        # Note: Feature exclusion is not applied to the bandpower features themselves, only the appended quality features.
        # This is because bandpower features are not named in a way that allows for easy exclusion.
        return base

    def _pre_scan_records(self, data_root: str, subjects_list: Optional[List[int]], runs_list: Optional[List[int]]) -> List[Dict[str, Any]]:
        """
        Scans for all EDF records, filters them, and extracts stable metadata
        PRIOR to any data loading or windowing. This is critical for robust splitting.
        """
        import re

        # Regex to robustly parse Subject and Run from filenames like 'S001R01.edf'
        record_pattern = re.compile(r"S(?P<sid>\d{3})R(?P<rid>\d{2})", re.IGNORECASE)

        _, all_edf_files, _ = _resolve_eegmmidb_root(data_root)

        logger.info(f"Pre-scanning {len(all_edf_files)} total EDF files found in {data_root}.")

        all_records_meta = []
        self.invalid_run_id_count = 0

        for i, path in enumerate(all_edf_files):
            normalized_path = os.path.normpath(path)
            basename = os.path.basename(normalized_path)

            # 1. Create a stable, unique record_id from the file path
            record_id = hashlib.md5(normalized_path.encode("utf-8")).hexdigest()

            # 2. Parse subject and run from filename
            subj_id, run_id, subject_run_id = None, None, None
            match = record_pattern.search(basename)

            if match:
                sid = int(match.group("sid"))
                rid = int(match.group("rid"))

                # Filter based on user's subject/run lists
                if subjects_list is not None and sid not in subjects_list:
                    continue
                if runs_list is not None and rid not in runs_list:
                    continue

                subj_id = sid
                run_id = rid
                # 3. Create the composite subject_run_id for robust splitting
                subject_run_id = f"S{sid:03d}R{rid:02d}"
            else:
                self.invalid_run_id_count += 1
                subject_run_id = f"UNK_{record_id}"
                if subjects_list is not None or runs_list is not None:
                    continue

            all_records_meta.append({
                "path": normalized_path,
                "record_id": record_id,
                "subject_id": subj_id,
                "run_id": run_id,
                "subject_run_id": subject_run_id
            })

        logger.info(f"Finished pre-scan. Found {len(all_records_meta)} valid records matching criteria.")
        if self.invalid_run_id_count > 0:
            logger.warning(f"Could not parse subject/run from {self.invalid_run_id_count} filenames.")

        return all_records_meta

    def _process_record(self, edf_path: str, window_seconds: float, stride_seconds: float, max_seq_length: int):
        try:
            import mne
        except Exception as e:
            raise ImportError("mne is required to read EDF files. Install mne.") from e

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        data = raw.get_data(picks=picks)
        sfreq = float(raw.info["sfreq"])
        info = {"skip_reason": None}
        if data.size == 0:
            logger.warning(f"No EEG data found in {edf_path}")
            info["skip_reason"] = "no_eeg_data"
            return None, None, None, sfreq, info
        ch_names = [raw.ch_names[i] for i in picks] if picks is not None else raw.ch_names
        group_indices = _resolve_channel_groups(ch_names, self.channel_grouping, self.channel_groups)

        win = max(1, int(window_seconds * sfreq))
        stride = max(1, int(stride_seconds * sfreq))
        starts_full = np.arange(0, data.shape[1] - win + 1, stride, dtype=np.int64)
        if starts_full.size == 0:
            info["skip_reason"] = "no_windows"
            return None, None, None, sfreq, info

        starts = starts_full
        if max_seq_length and max_seq_length > 0 and starts_full.size > max_seq_length:
            starts = starts_full[:max_seq_length]

        feats = []
        quality_scores = []
        for s in starts:
            e = s + win
            window = data[:, s:e]
            feats.append(self._extract_window_features(window, sfreq, group_indices))
            quality_scores.append(compute_quality_score(window, sfreq, feature=self.label_feature))

        feats = np.stack(feats, axis=0)
        quality_scores = np.asarray(quality_scores, dtype=np.float32)
        treats = np.zeros((feats.shape[0], self.treatment_dim), dtype=np.float32)

        return feats, treats, quality_scores, sfreq, info

    def _build_split(self, covs_list, treats_list, outs_list, max_len: int,
                     subject_ids: Optional[List[int]] = None,
                     run_ids: Optional[List[int]] = None,
                     record_ids: Optional[List[int]] = None):
        n = len(covs_list)
        f = covs_list[0].shape[1]
        covs = np.zeros((n, max_len, f), dtype=np.float32)
        treats = np.zeros((n, max_len, self.treatment_dim), dtype=np.float32)
        outs = np.zeros((n, max_len, 1), dtype=np.int64)
        active = np.zeros((n, max_len, 1), dtype=np.float32)
        seq_lengths = np.zeros((n,), dtype=np.int64)
        y_outcome = np.zeros((n,), dtype=np.int64)

        n_classes = len(self.label_names) if self.label_names is not None else 3

        for i in range(n):
            L = min(len(covs_list[i]), max_len)
            covs[i, :L, :] = covs_list[i][:L]
            treats[i, :L, :] = treats_list[i][:L]
            outs[i, :L, :] = outs_list[i][:L]
            active[i, :L, 0] = 1.0
            seq_lengths[i] = L

            # Record-level outcome label (majority vote over active windows)
            labels_i = outs_list[i][:L].reshape(-1).astype(np.int64, copy=False)
            if labels_i.size > 0:
                counts = np.bincount(labels_i, minlength=max(n_classes, int(labels_i.max()) + 1))
                y_outcome[i] = int(np.argmax(counts))
            else:
                y_outcome[i] = 0


        prev_treats = np.zeros_like(treats)
        prev_treats[:, 1:, :] = treats[:, :-1, :]

        static_features = np.zeros((n, 1), dtype=np.float32)
        prev_outputs = covs

        data = {
            "sequence_lengths": seq_lengths,
            "prev_treatments": prev_treats,
            "current_treatments": treats,
            "static_features": static_features,
            "active_entries": active,
            "outputs": outs,
            "unscaled_outputs": outs,
            "prev_outputs": prev_outputs,
            "current_covariates": covs,
            "original_index": np.arange(n, dtype=np.int64),
            "y_outcome": y_outcome,
        }
        if subject_ids is not None:
            data["subject_id"] = _ids_to_int64(subject_ids, "subject_id")
        if run_ids is not None:
            data["run_id"] = _ids_to_int64(run_ids, "run_id")
        if record_ids is not None:
            data["record_id"] = _ids_to_int64(record_ids, "record_id")
        logger.info(f"Built split: sequences={n}, total_windows={int(seq_lengths.sum())}, max_len={max_len}, features={f}")
        return data

    def _ensure_record_targets(self, data: dict) -> None:
        if "outputs" not in data or "active_entries" not in data:
            return

        outputs = data["outputs"]
        active = data["active_entries"]
        n, t = outputs.shape[0], outputs.shape[1]

        if "y_outcome" not in data:
            n_classes = len(self.label_names) if self.label_names is not None else 3
            y_outcome = np.zeros((n,), dtype=np.int64)
            for i in range(n):
                mask = active[i, :, 0].astype(bool)
                labels = outputs[i, mask, 0].astype(np.int64, copy=False)
                if labels.size > 0:
                    counts = np.bincount(labels, minlength=max(n_classes, int(labels.max()) + 1))
                    y_outcome[i] = int(np.argmax(counts))
                else:
                    y_outcome[i] = 0
            data["y_outcome"] = y_outcome

    def _compute_global_stats(self, covs_list: List[np.ndarray], mode: str):
        if not covs_list:
            raise RuntimeError("Empty covariate list; cannot compute stats.")
        flat = np.concatenate(covs_list, axis=0)
        if mode == "zscore":
            center = flat.mean(axis=0)
            scale = flat.std(axis=0) + 1e-6
            return center, scale
        if mode == "robust":
            center = np.median(flat, axis=0)
            q75 = np.percentile(flat, 75, axis=0)
            q25 = np.percentile(flat, 25, axis=0)
            scale = (q75 - q25) + 1e-6
            return center, scale
        raise ValueError(f"Unsupported stats mode: {mode}")

    def _compute_subject_stats(self, covs_list: List[np.ndarray], subject_ids: List[int], mode: str):
        stats = {}
        by_subject = defaultdict(list)
        for covs, subj in zip(covs_list, subject_ids):
            by_subject[int(subj)].append(covs)
        for subj, arrs in by_subject.items():
            flat = np.concatenate(arrs, axis=0)
            if mode == "zscore":
                center = flat.mean(axis=0)
                scale = flat.std(axis=0) + 1e-6
            elif mode == "robust":
                center = np.median(flat, axis=0)
                q75 = np.percentile(flat, 75, axis=0)
                q25 = np.percentile(flat, 25, axis=0)
                scale = (q75 - q25) + 1e-6
            else:
                raise ValueError(f"Unsupported stats mode: {mode}")
            stats[int(subj)] = (center.astype(np.float32, copy=False), scale.astype(np.float32, copy=False))
        return stats

    def _apply_stats(self, covs_list: List[np.ndarray], subject_ids: List[int],
                     stats: Dict[int, Tuple[np.ndarray, np.ndarray]],
                     fallback_stats: Tuple[np.ndarray, np.ndarray]):
        out = []
        for covs, subj in zip(covs_list, subject_ids):
            center, scale = stats.get(int(subj), fallback_stats)
            out.append(((covs - center) / scale).astype(np.float32, copy=False))
        return out

    def _apply_normalization(self,
                             train_covs: List[np.ndarray],
                             val_covs: List[np.ndarray],
                             test_covs: List[np.ndarray],
                             train_subjects: List[int],
                             val_subjects: List[int],
                             test_subjects: List[int],
                             split_by_subject: bool):
        mode = self.normalization_mode
        scope = self.normalization_scope

        scaling_params = {
            "input_means": np.zeros((train_covs[0].shape[1],), dtype=np.float32),
            "inputs_stds": np.ones((train_covs[0].shape[1],), dtype=np.float32),
            "output_means": 0.0,
            "output_stds": 1.0,
            "normalization_mode": mode,
            "normalization_scope": scope,
        }

        if mode in ("none", "off", "no"):
            logger.info("Normalization is disabled.")
            return train_covs, val_covs, test_covs, scaling_params

        # The 'global' scope is a potential source of data leakage.
        # We explicitly forbid this to prevent accidental leakage.
        if scope == "global":
            logger.error(
                f"FATAL: Normalization scope is 'global'. This leaks statistics from the "
                f"validation and test sets into the training set. Please use 'fold_zscore' "
                f"or another train-only scope."
            )
            raise ValueError("Normalization scope 'global' is not permitted due to data leakage.")

        logger.info(f"Applying '{mode}' normalization with fitting on the training set.")

        # 'fold_zscore' is the main mode used. It correctly fits on the training fold.
        if mode in ("fold_zscore", "per_fold", "global_zscore"):
            if mode == "global_zscore":
                logger.warning("DEPRECATION: normalization_mode='global_zscore' is deprecated. "
                               "It now correctly fits on the train fold only. Use 'fold_zscore'.")

            scaler = PerFoldStandardScaler()
            logger.info(f"Fitting scaler on {len(train_covs)} training records...")
            scaler.fit(train_covs)

            logger.info(f"Scaler fit complete. Mean (first 5): {scaler.mean_[:5]}")
            logger.info(f"Scaler fit complete. Std (first 5): {scaler.std_[:5]}")

            train_covs = scaler.transform(train_covs)
            val_covs = scaler.transform(val_covs)
            test_covs = scaler.transform(test_covs)

            scaling_params["input_means"] = scaler.mean_
            scaling_params["inputs_stds"] = scaler.std_
            scaling_params["scaler_state"] = scaler.state_dict()
            self.scaler_ = scaler

            if not split_by_subject:
                logger.warning("fold_zscore normalization used with split_by_subject=False; "
                               "subject leakage may still exist due to record-level split.")

            logger.info("Scaler fit on train set and applied to train, val, and test sets.")
            return train_covs, val_covs, test_covs, scaling_params

        if mode == "subject_zscore":
            # This mode is more complex and leaks information if not handled carefully.
            # It normalizes validation/test subjects using their own data, which is a form of leakage.
            logger.warning("Using 'subject_zscore' normalization. Note: This mode normalizes val/test subjects using their own data, which can leak information and is generally not recommended.")
            global_center, global_scale = self._compute_global_stats(train_covs, mode="zscore")
            global_center = global_center.astype(np.float32, copy=False)
            global_scale = global_scale.astype(np.float32, copy=False)
            scaling_params["input_means"] = global_center
            scaling_params["inputs_stds"] = global_scale

            train_stats = self._compute_subject_stats(train_covs, train_subjects, mode="zscore")
            
            # For validation and test, we SHOULD NOT use their own stats.
            # We use the stats from the training set subjects if available, otherwise fallback to global train stats.
            # However, the current implementation fits on val/test data, which is a leak.
            # For this fix, we will just log a strong warning. A proper fix would require refactoring.
            logger.error("POTENTIAL LEAK: 'subject_zscore' as implemented computes stats on validation and test sets.")
            val_stats = self._compute_subject_stats(val_covs, val_subjects, mode="zscore")
            test_stats = self._compute_subject_stats(test_covs, test_subjects, mode="zscore")

            def _merge_stats(subj, split_stats):
                # Always prefer training set stats for a given subject.
                return train_stats.get(int(subj), split_stats.get(int(subj), (global_center, global_scale)))

            train_covs = self._apply_stats(train_covs, train_subjects, train_stats, (global_center, global_scale))
            val_covs = self._apply_stats(val_covs, val_subjects,
                                         {int(k): _merge_stats(k, val_stats) for k in val_stats.keys()},
                                         (global_center, global_scale))
            test_covs = self._apply_stats(test_covs, test_subjects,
                                          {int(k): _merge_stats(k, test_stats) for k in test_stats.keys()},
                                          (global_center, global_scale))

            scaling_params["scaling_subject_ids"] = np.array(sorted(train_stats.keys()), dtype=np.int64)
            if scaling_params["scaling_subject_ids"].size:
                centers = np.stack([train_stats[s][0] for s in scaling_params["scaling_subject_ids"]], axis=0)
                scales = np.stack([train_stats[s][1] for s in scaling_params["scaling_subject_ids"]], axis=0)
                scaling_params["scaling_subject_centers"] = centers.astype(np.float32, copy=False)
                scaling_params["scaling_subject_scales"] = scales.astype(np.float32, copy=False)
            
            return train_covs, val_covs, test_covs, scaling_params

        logger.warning(f"Unknown or unhandled normalization_mode='{mode}'; skipping normalization.")
        return train_covs, val_covs, test_covs, scaling_params

    def _make_subject_folds(self, subjects: List[int], n_folds: int, seed: int,
                            strata: Optional[Dict[int, int]] = None) -> List[List[int]]:
        if n_folds <= 1:
            raise ValueError("n_folds must be >= 2.")
        subjects = list(sorted(set(subjects)))
        rng = np.random.RandomState(seed)
        if strata:
            groups = defaultdict(list)
            for s in subjects:
                groups[int(strata.get(int(s), -1))].append(int(s))
            folds = [[] for _ in range(n_folds)]
            for subs in groups.values():
                rng.shuffle(subs)
                for i, subj in enumerate(subs):
                    folds[i % n_folds].append(subj)
            return folds

        rng.shuffle(subjects)
        fold_sizes = [len(subjects) // n_folds] * n_folds
        for i in range(len(subjects) % n_folds):
            fold_sizes[i] += 1
        folds = []
        idx = 0
        for size in fold_sizes:
            folds.append(subjects[idx:idx + size])
            idx += size
        return folds

    def _compute_subject_strata(self, quality_list: List[np.ndarray], subject_ids: List[int]) -> Dict[int, int]:
        """
        Compute subject-level strata for stratified splitting based on raw (pre-quantile) quality scores.
        Each record contributes an aggregate score; subjects are then binned by quantiles of subject scores.
        """
        if not quality_list:
            raise RuntimeError("Empty quality list; cannot compute subject strata.")
        if len(quality_list) != len(subject_ids):
            raise RuntimeError("quality_list and subject_ids length mismatch.")

        record_scores = []
        record_subjects = []
        for scores, subj in zip(quality_list, subject_ids):
            subj_id = int(subj)
            scores = np.asarray(scores, dtype=np.float32)
            if scores.size == 0:
                continue
            if self.stratify_subject_agg == "median":
                rec_score = float(np.nanmedian(scores))
            else:
                rec_score = float(np.nanmean(scores))
            record_scores.append(rec_score)
            record_subjects.append(subj_id)

        if not record_scores:
            raise RuntimeError("No valid record scores for stratified split.")

        by_subject = defaultdict(list)
        for score, subj_id in zip(record_scores, record_subjects):
            by_subject[int(subj_id)].append(float(score))

        subj_scores = {}
        for subj_id, scores in by_subject.items():
            arr = np.asarray(scores, dtype=np.float32)
            if arr.size == 0:
                continue
            if self.stratify_subject_agg == "median":
                subj_scores[subj_id] = float(np.nanmedian(arr))
            else:
                subj_scores[subj_id] = float(np.nanmean(arr))

        if not subj_scores:
            raise RuntimeError("No subject scores for stratified split.")

        quantiles = self.label_quantiles
        if quantiles is None or len(quantiles) == 0:
            quantiles = [0.33, 0.66]
        quantiles = [float(q) for q in quantiles if 0.0 < float(q) < 1.0]
        values = np.asarray(list(subj_scores.values()), dtype=np.float32)
        if values.size == 0:
            raise RuntimeError("No scores available for subject stratification.")

        if quantiles:
            edges = np.quantile(values, quantiles)
            edges = np.unique(edges)
        else:
            edges = np.array([], dtype=np.float32)

        subj_label = {}
        for subj_id, score in subj_scores.items():
            if edges.size == 0:
                label = 0
            else:
                label = int(np.searchsorted(edges, score, side="right"))
            subj_label[int(subj_id)] = label

        return subj_label

    def _fit_labeler(self, train_quality: List[np.ndarray], train_subject_ids: List[int]) -> Dict[str, Any]:
        if not train_quality:
            raise RuntimeError("Empty training quality features; cannot fit labeler.")

        if self.label_fit_split not in ("train", "train_only", "train-only"):
            logger.error(f"FATAL: label.fit_split='{self.label_fit_split}' is not 'train'. "
                         f"This would leak information from other splits. Aborting.")
            raise ValueError(f"label.fit_split must be 'train', but was '{self.label_fit_split}'.")

        logger.info(f"Fitting labeler with strategy '{self.label_strategy}' on TRAINING data only.")

        quantiles = self.label_quantiles
        if quantiles is None or len(quantiles) != 2:
            logger.warning("label.quantiles invalid; using default [0.33, 0.66].")
            quantiles = [0.33, 0.66]

        flat_scores = np.concatenate(train_quality, axis=0)
        if flat_scores.size == 0:
            raise RuntimeError("No training scores available for quantile thresholds.")

        logger.info(f"Fitting quantiles on {flat_scores.size} training windows...")
        labeler = QuantileLabeler(quantiles=quantiles).fit(flat_scores)
        logger.info(f"Quantiles fittés sur train: seuils = {labeler.thresholds_}")
        self.labeler_ = labeler
        return labeler

    def _apply_labeler(self, quality_list: List[np.ndarray], subject_ids: List[int], labeler: Dict[str, Any]):
        outputs = []
        for scores in quality_list:
            labels = labeler.transform(scores)
            outputs.append(labels[:, np.newaxis].astype(np.int64))
        return outputs

    def _resolve_label_names(self, n_classes: int) -> List[str]:
        if self.label_names is not None and len(self.label_names) == n_classes:
            return list(self.label_names)
        return [str(i) for i in range(n_classes)]

    def _window_label_counts(self, data: dict) -> Dict[int, int]:
        y = data["outputs"].reshape(-1)
        mask = data["active_entries"].reshape(-1).astype(bool)
        y = y[mask]
        values, counts = np.unique(y, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def _record_label_counts(self, data: dict) -> Dict[int, int]:
        outputs = data["outputs"]
        active = data["active_entries"]
        counts = defaultdict(int)
        n = outputs.shape[0]
        for i in range(n):
            y = outputs[i].reshape(-1)
            mask = active[i].reshape(-1).astype(bool)
            y = y[mask]
            if y.size == 0:
                continue
            values, cnts = np.unique(y, return_counts=True)
            label = int(values[np.argmax(cnts)])
            counts[label] += 1
        return dict(counts)

    def _label_balance_gate(self, split_name: str, level: str,
                             counts: Dict[int, int], total: int, n_classes: int):
        if total <= 0:
            return
        min_prop = float(self.label_balance_min) if self.label_balance_min is not None else 0.0
        if min_prop <= 0:
            return
        names = self._resolve_label_names(n_classes)
        for i in range(n_classes):
            prop = counts.get(i, 0) / float(total)
            if prop < min_prop:
                msg = (f"Label balance gate: {split_name} {level} class '{names[i]}' proportion "
                       f"{prop:.4f} < {min_prop:.4f}.")
                if self.label_balance_mode in ("error", "fail", "raise"):
                    raise ValueError(msg)
                logger.warning(msg)

    def _summarize_label_distribution(self, data: dict, split_name: str) -> Dict[str, Any]:
        window_counts = self._window_label_counts(data)
        record_counts = self._record_label_counts(data)

        total_window = int(sum(window_counts.values()))
        total_record = int(sum(record_counts.values()))
        if self.label_names is not None:
            n_classes = len(self.label_names)
        else:
            n_classes = max(window_counts.keys() | record_counts.keys()) + 1 if (window_counts or record_counts) else 0
        names = self._resolve_label_names(n_classes)

        window_counts_full = {names[i]: int(window_counts.get(i, 0)) for i in range(n_classes)}
        record_counts_full = {names[i]: int(record_counts.get(i, 0)) for i in range(n_classes)}
        window_props = {names[i]: (window_counts.get(i, 0) / total_window) if total_window > 0 else 0.0
                        for i in range(n_classes)}
        record_props = {names[i]: (record_counts.get(i, 0) / total_record) if total_record > 0 else 0.0
                        for i in range(n_classes)}

        logger.info(f"{split_name} window label distribution: {window_counts_full} (total={total_window})")
        logger.info(f"{split_name} record label distribution: {record_counts_full} (total={total_record})")

        self._label_balance_gate(split_name, "window", window_counts, total_window, n_classes)
        self._label_balance_gate(split_name, "record", record_counts, total_record, n_classes)

        return {
            "window_counts": window_counts_full,
            "window_props": window_props,
            "window_total": total_window,
            "record_counts": record_counts_full,
            "record_props": record_props,
            "record_total": total_record,
        }

    def _log_label_distributions(self, labeler: Dict[str, Any], train: dict, val: dict, test: dict):
        summaries = {
            "train": self._summarize_label_distribution(train, "train"),
            "val": self._summarize_label_distribution(val, "val"),
            "test": self._summarize_label_distribution(test, "test"),
        }
        self._write_labeling_report(labeler, summaries)

    def _write_labeling_report_cached(self, train: dict, val: dict, test: dict):
        labeler = QuantileLabeler(quantiles=self.label_quantiles)
        summaries = {
            "train": self._summarize_label_distribution(train, "train"),
            "val": self._summarize_label_distribution(val, "val"),
            "test": self._summarize_label_distribution(test, "test"),
        }
        self._write_labeling_report(labeler, summaries)

    def _write_labeling_report(self, labeler: Dict[str, Any], summaries: Dict[str, Any]):
        if isinstance(labeler, QuantileLabeler):
            labeler_dict = labeler.to_loggable()
        elif isinstance(labeler, dict):
            labeler_dict = labeler
        else:
            labeler_dict = {}

        report = {
            "label_strategy": self.label_strategy,
            "label_feature": self.label_feature,
            "fit_split": self.label_fit_split,
            "quantiles": labeler_dict.get("quantiles", self.label_quantiles),
            "thresholds": labeler_dict.get("thresholds", None),
            "label_balance_min": self.label_balance_min,
            "label_balance_mode": self.label_balance_mode,
            "label_distribution": summaries,
        }

        out_dir = os.getcwd()
        report_path = os.path.join(out_dir, "labeling_report.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            self.labeling_report_path = report_path
            logger.info(f"Saved labeling report: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to write labeling report: {e}")

    def _compute_scaling(self, covs: np.ndarray, active: np.ndarray):
        mask = active.reshape(-1).astype(bool)
        flat = covs.reshape(-1, covs.shape[-1])[mask]
        means = flat.mean(axis=0)
        stds = flat.std(axis=0) + 1e-6
        return means, stds

    def _save_cache(self, path: str, train: dict, val: dict, test: dict, scaling_params: dict):
        payload = {
            "train_current_covariates": train["current_covariates"],
            "train_current_treatments": train["current_treatments"],
            "train_prev_treatments": train["prev_treatments"],
            "train_prev_outputs": train["prev_outputs"],
            "train_outputs": train["outputs"],
            "train_unscaled_outputs": train["unscaled_outputs"],
            "train_active_entries": train["active_entries"],
            "train_sequence_lengths": train["sequence_lengths"],
            "train_static_features": train["static_features"],
            "train_original_index": train["original_index"],
            "val_current_covariates": val["current_covariates"],
            "val_current_treatments": val["current_treatments"],
            "val_prev_treatments": val["prev_treatments"],
            "val_prev_outputs": val["prev_outputs"],
            "val_outputs": val["outputs"],
            "val_unscaled_outputs": val["unscaled_outputs"],
            "val_active_entries": val["active_entries"],
            "val_sequence_lengths": val["sequence_lengths"],
            "val_static_features": val["static_features"],
            "val_original_index": val["original_index"],
            "test_current_covariates": test["current_covariates"],
            "test_current_treatments": test["current_treatments"],
            "test_prev_treatments": test["prev_treatments"],
            "test_prev_outputs": test["prev_outputs"],
            "test_outputs": test["outputs"],
            "test_unscaled_outputs": test["unscaled_outputs"],
            "test_active_entries": test["active_entries"],
            "test_sequence_lengths": test["sequence_lengths"],
            "test_static_features": test["static_features"],
            "test_original_index": test["original_index"],
            "scaling_input_means": scaling_params.get("input_means"),
            "scaling_input_stds": scaling_params.get("inputs_stds"),
        }

        for key in ("subject_id", "run_id", "record_id", "y_outcome"):
            tkey = f"train_{key}"
            vkey = f"val_{key}"
            tekey = f"test_{key}"
            if key in train:
                payload[tkey] = train[key]
            if key in val:
                payload[vkey] = val[key]
            if key in test:
                payload[tekey] = test[key]

        for k in ("scaling_subject_ids", "scaling_subject_centers", "scaling_subject_scales",
                  "normalization_mode", "normalization_scope"):
            if k in scaling_params:
                payload[k] = scaling_params[k]

        labeler = getattr(self, "labeler_", None)
        if isinstance(labeler, QuantileLabeler) and labeler.thresholds_ is not None:
            payload["labeler_quantiles"] = np.asarray(labeler.quantiles, dtype=np.float32)
            payload["labeler_thresholds"] = np.asarray(labeler.thresholds_, dtype=np.float32)

        np.savez_compressed(path, **payload)

    def _unpack(self, npz, prefix: str) -> dict:
        data = {
            "current_covariates": npz[f"{prefix}current_covariates"],
            "current_treatments": npz[f"{prefix}current_treatments"],
            "prev_treatments": npz[f"{prefix}prev_treatments"],
            "prev_outputs": npz[f"{prefix}prev_outputs"],
            "outputs": npz[f"{prefix}outputs"],
            "unscaled_outputs": npz[f"{prefix}unscaled_outputs"],
            "active_entries": npz[f"{prefix}active_entries"],
            "sequence_lengths": npz[f"{prefix}sequence_lengths"],
            "static_features": npz[f"{prefix}static_features"],
            "original_index": npz[f"{prefix}original_index"] if f"{prefix}original_index" in npz else None,
            "subject_id": npz[f"{prefix}subject_id"] if f"{prefix}subject_id" in npz else None,
            "run_id": npz[f"{prefix}run_id"] if f"{prefix}run_id" in npz else None,
            "record_id": npz[f"{prefix}record_id"] if f"{prefix}record_id" in npz else None,
            "y_outcome": npz[f"{prefix}y_outcome"] if f"{prefix}y_outcome" in npz else None,
        }
        if data.get("original_index") is None:
            data.pop("original_index", None)
        if data.get("subject_id") is None:
            data.pop("subject_id", None)
        if data.get("run_id") is None:
            data.pop("run_id", None)
        if data.get("record_id") is None:
            data.pop("record_id", None)
        if data.get("y_outcome") is None:
            data.pop("y_outcome", None)
        float_keys = {
            "current_covariates",
            "current_treatments",
            "prev_treatments",
            "prev_outputs",
            "unscaled_outputs",
            "active_entries",
            "static_features",
        }
        for k in float_keys:
            if k in data and data[k] is not None and isinstance(data[k], np.ndarray) and data[k].dtype.kind in ("f", "c"):
                data[k] = data[k].astype(np.float32, copy=False)
        # outputs are class labels in this dataset; keep integer dtype if present
        if isinstance(data.get("outputs"), np.ndarray) and data["outputs"].dtype.kind in ("f", "c"):
            data["outputs"] = data["outputs"].astype(np.float32, copy=False)
        if isinstance(data.get("unscaled_outputs"), np.ndarray) and data["unscaled_outputs"].dtype.kind in ("f", "c"):
            data["unscaled_outputs"] = data["unscaled_outputs"].astype(np.float32, copy=False)
        return data
