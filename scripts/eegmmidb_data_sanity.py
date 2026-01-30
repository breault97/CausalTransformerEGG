"""
Standalone EEGMMIDB sanity checks (no Hydra).

This script is intended for quick local debugging of:
- dataset discovery/layout resolution,
- feature dimensions and basic non-degeneracy checks,
- and label distributions for a small subset.

It is not a training script.
"""

import argparse
import json

import numpy as np

from src.data.physionet_eegmmidb.dataset import PhysioNetEEGMMIDBDatasetCollection, _resolve_eegmmidb_root


def _parse_list(val):
    """Parse CLI list arguments like '1,2,3' or 'all' into Python lists."""
    if val is None:
        return None
    if isinstance(val, str):
        if val.lower() == "all":
            return "all"
        parts = [p.strip() for p in val.split(",") if p.strip()]
        try:
            return [int(p) for p in parts]
        except Exception:
            return "all"
    return val


def main():
    """CLI entrypoint for quick, local EEGMMIDB data sanity checks."""
    parser = argparse.ArgumentParser(description="EEGMMIDB data sanity checks (labels + features).")
    parser.add_argument("--data-dir", required=True, help="Path to eegmmidb root.")
    parser.add_argument("--subjects", default="all", help="Subjects list or 'all' (e.g., '1,2,3').")
    parser.add_argument("--runs", default="all", help="Runs list or 'all' (e.g., '1,2').")
    parser.add_argument("--max-seq-length", type=int, default=60, help="Override max_seq_length.")
    parser.add_argument("--feature-set", default="bandpower", help="raw8 / bandpower / quality3 / bandpower_hjorth / bandpower_hjorth_entropy")
    parser.add_argument("--bandpower-relative", type=int, default=1, help="1 to use relative bandpower, 0 for absolute.")
    parser.add_argument("--bandpower-log", type=int, default=0, help="1 to log bandpower features.")
    parser.add_argument("--bandpower-include-quality", type=int, default=1, help="1 to append RMS/PTP/line_ratio features.")
    parser.add_argument("--normalization-mode", default="global_zscore", help="none / global_zscore / subject_zscore / robust")
    parser.add_argument("--split-by-subject", type=int, default=1, help="1 to split by subject, 0 for record split.")
    parser.add_argument("--stratify-subject-split", type=int, default=1, help="1 to stratify subject split, 0 for random.")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset cache.")
    args = parser.parse_args()

    print("Standalone sanity script (no Hydra).")
    print("If you see 'UnsupportedInterpolationType hydra', update to the latest script version.")

    ds_cfg = {
        "data_dir": args.data_dir,
        "subjects": _parse_list(args.subjects),
        "runs": _parse_list(args.runs),
        "window_seconds": 2.0,
        "stride_seconds": 1.0,
        "max_seq_length": int(args.max_seq_length),
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "seed": 100,
        "download": False,
        "cache": not args.no_cache,
        "cache_name": "",
        "split_by_subject": bool(args.split_by_subject),
        "stratify_subject_split": bool(args.stratify_subject_split),
        "feature_set": args.feature_set,
        "bandpower_relative": bool(args.bandpower_relative),
        "bandpower_log": bool(args.bandpower_log),
        "bandpower_include_quality": bool(args.bandpower_include_quality),
        "channel_grouping": "channels",
        "channel_groups": None,
        "psd_nperseg": 256,
        "psd_noverlap": 128,
        "psd_fmin": 0.5,
        "psd_fmax": 45.0,
        "normalization_mode": args.normalization_mode,
        "normalization_scope": "global",
        "label_names": ["mauvais", "moyen", "excellent"],
        "label_strategy": "quantile_global",
        "quantiles": [0.33, 0.66],
        "fixed_thresholds": None,
        "fit_split": "train",
        "label_balance_min": 0.1,
        "label_balance_mode": "warn",
        "val_batch_size": 256,
        "treatment_mode": "multilabel",
        "num_workers": 0,
    }

    print("Loading dataset with config (standalone, no Hydra):")
    print(json.dumps({k: ds_cfg.get(k) for k in sorted(ds_cfg.keys())}, indent=2))

    # Basic EDF stats before processing
    try:
        root, files, counts = _resolve_eegmmidb_root(args.data_dir)
        print(f"Resolved data root: {root}")
        print(f"EDF files found: {len(files)}")
        for k, v in counts.items():
            print(f"EDF scan {k}: {v}")
    except Exception as e:
        print(f"Warning: EDF scan failed: {e}")

    ds = PhysioNetEEGMMIDBDatasetCollection(**ds_cfg)
    ds.process_data_multi()

    def summarize(split_name, data):
        y = data["outputs"].reshape(-1)
        mask = data["active_entries"].reshape(-1).astype(bool)
        y = y[mask]
        values, counts = np.unique(y, return_counts=True)
        print(f"{split_name} window label counts:", dict(zip(values.tolist(), counts.tolist())))

        # Record-level: majority per sequence
        outputs = data["outputs"]
        active = data["active_entries"]
        rec_counts = {}
        for i in range(outputs.shape[0]):
            yi = outputs[i].reshape(-1)
            mi = active[i].reshape(-1).astype(bool)
            yi = yi[mi]
            if yi.size == 0:
                continue
            vals, cnts = np.unique(yi, return_counts=True)
            label = int(vals[np.argmax(cnts)])
            rec_counts[label] = rec_counts.get(label, 0) + 1
        print(f"{split_name} record label counts:", rec_counts)

    summarize("train", ds.train_f.data)
    summarize("val", ds.val_f.data)
    summarize("test", ds.test_f.data)

    feats = ds.train_f.data["current_covariates"]
    feat_dim = int(feats.shape[-1])
    print("Feature shape:", feats.shape)
    print("Feature dim:", feat_dim)
    if getattr(ds, "bandpower_include_quality", False):
        quality_dim = 8
        print("bandpower_include_quality: True (expect +8 dims)")
        if feat_dim >= quality_dim:
            print(f"Base feature dim (minus quality): {feat_dim - quality_dim}")
        if feat_dim >= quality_dim:
            tail = feats[:, :, -quality_dim:]
            std = float(np.std(tail))
            means = np.mean(tail, axis=(0, 1))
            print("Quality tail std:", std)
            print("Quality tail mean (rms_mean, ptp_mean, line_ratio):",
                  float(means[0]), float(means[2]), float(means[4]))
            if std <= 1e-8:
                raise RuntimeError("bandpower_include_quality=True but quality features look degenerate.")
        else:
            raise RuntimeError("bandpower_include_quality=True but features are too small to include quality tail.")
    else:
        print("bandpower_include_quality: False")

    report_path = getattr(ds, "labeling_report_path", None)
    if report_path:
        print(f"Labeling report path: {report_path}")
    else:
        print("Labeling report path: (not set)")

    print("Sanity checks complete.")


if __name__ == "__main__":
    main()
