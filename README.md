# CausalTransformer EEG — EEGMMIDB (PhysioNet) Quality Classification

This repository benchmarks **3-class EEG signal-quality classification** on the PhysioNet
**EEG Motor Movement/Imagery Dataset** (EEGMMIDB / EEGBCI) using the **Causal Transformer (CT)**.

> Important: EEGMMIDB does **not** provide clinical “signal quality” labels. The labels used here are
> **heuristic** (derived from objective signal metrics and quantiles) and must not be interpreted as
> medical ground truth.

Final report PDF : `report_latex/{french/english}/main.pdf`.

---

## Quickstart (copy/paste)

### 1) Install

If you use Conda on Windows, create/activate an environment first, then install:

```console
conda create -n causaltransformer python=3.10
conda activate causaltransformer
```

```console
pip install -r requirements.txt
pip install mne wfdb
```

Optional EDF backend (usually not required):

```console
pip install pyedflib
```

### 2) Point the code to the dataset

The default config uses `EEGMMIDB_DIR` if set, otherwise it falls back to `data/mne`
(see `config/dataset/eegmmidb.yaml`).

Windows (PowerShell):

```powershell
$env:EEGMMIDB_DIR="C:\\Projects\\CausalTransformer\\data\\eegmmidb"
$env:PYTHONPATH="."
```

Linux/macOS (bash):

```bash
export EEGMMIDB_DIR="$PWD/data/eegmmidb"
export PYTHONPATH=.
```

### 3) Run a smoke test (1 epoch)

```bash
PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_quick
```

### 4) Run CT baseline + 5-fold subject CV

```bash
PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_baseline fold_index=0
PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_baseline fold_index=1
PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_baseline fold_index=2
PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_baseline fold_index=3
PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_baseline fold_index=4
```

### 5) Run baselines

```bash
PYTHONPATH=. python runnables/train_baselines.py +experiment=eegmmidb_baselines
```

---

## Regenerating curated public results (`results_public/`)

Curated, audit-friendly artifacts live in `results_public/`. To rebuild them from exports:

```bash
python scripts/curate_public_results.py
```

By default, the script reads from `mlflow_exports/` and writes into `results_public/CT_eegmmidb`.
See `results_public/README.md` for details.

---

## What is not included in git

The following are intentionally **not** tracked:

- The EEGMMIDB dataset (`data/` or any path referenced by `EEGMMIDB_DIR`).
- Raw MLflow outputs (`mlruns/`, `mlartifacts/`, `mlflow_exports/`).
- Local run outputs / checkpoints (`outputs/`, `*.ckpt`, `*.pt`, `*.pth`, `*.npz`).

---

## Dataset: EEGMMIDB / EEGBCI (PhysioNet)

Official landing page: https://physionet.org/content/eegmmidb/1.0.0/  
Dataset DOI (v1.0.0): `10.13026/C28G6P`  
License: Open Data Commons Attribution License v1.0 (ODC-By)

### Expected directory layouts

The loader auto-detects multiple layouts (see `_fast_edf_scan` in `src/data/physionet_eegmmidb/dataset.py`), including:

```text
<EEGMMIDB_DIR>/S001/S001R01.edf
<EEGMMIDB_DIR>/files/eegmmidb/1.0.0/S001/S001R01.edf
<EEGMMIDB_DIR>/eegmmidb/1.0.0/S001/S001R01.edf
<EEGMMIDB_DIR>/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R01.edf
```

### Optional: download via MNE (large)

The dataset class can download EEGMMIDB through `mne.datasets.eegbci` when `dataset.download=True`.
Be careful: `dataset.subjects=all` and `dataset.runs=all` will attempt a full download.

---

## How labeling works (heuristic “quality”)

- Each EEG window gets a **quality score** from objective signal statistics (e.g., RMS, peak-to-peak,
  line-noise ratio, saturation/flatline indicators).
- Labels are derived by **quantile thresholds fit on the training split only** (per fold), then
  applied to validation/test splits (anti-leakage by design).
- The 3 classes correspond to `{bad, medium, excellent}` quality (the config’s `label_names` may use
  non-English strings; the intent is still 3 ordered quality bins).

Key config: `config/dataset/eegmmidb.yaml` and `config/experiment/eegmmidb_ct_*.yaml`.

---

## Training (CT) and baselines

### CT (classification)

- Main entrypoint: `runnables/train_multi.py`
- Recommended preset: `+experiment=eegmmidb_ct_baseline`
- Exported artifacts can include window/record/subject reports and confusion matrices, depending on
  the experiment preset.

Cross-validation note: with `fold_index=<0..n_folds-1>`, the test split is the selected fold; the
validation split is a **ratio-based** subset of the remaining subjects (not necessarily “fold+1”).

### Baselines

Baselines (EEGNet, ShallowConvNet, SimpleCNN1D, CSP+LDA) are implemented in `src/models/baselines.py`.

```bash
PYTHONPATH=. python runnables/train_baselines.py +experiment=eegmmidb_baselines
```

---

## Sanity checks (anti-leakage)

The CT pipeline supports “sanity” flags via environment variables (logged to MLflow):

- **Label shuffle (must drop to chance)**:
  ```bash
  CT_SHUFFLE_TRAIN_LABELS=1 PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_quick fold_index=0
  ```
- **Input shuffle (weak / order-only)**:
  ```bash
  CT_SHUFFLE_TRAIN_INPUTS=1 CT_SHUFFLE_TRAIN_INPUTS_MODE=intra_sequence_lockstep \
    PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_quick fold_index=0
  ```
- **Input/label decoupling (strong / should drop to chance)**:
  ```bash
  CT_SHUFFLE_TRAIN_INPUTS=1 CT_SHUFFLE_TRAIN_INPUTS_MODE=inter_batch_decouple \
    PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_quick fold_index=0
  ```
- **Split-only verification** (exits after asserting disjoint subject/record IDs):
  ```bash
  CT_ONLY_SPLIT_CHECK=1 PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_quick fold_index=0
  ```

There is also a lightweight, model-free QC script:

```bash
PYTHONPATH=. python runnables/quality_tests.py +experiment=quality_ablation fold_index=0
```

---

## Exporting results and building the report

- Training logs are written to local MLflow directories (`mlruns/`, `mlartifacts/`).
- Large local folders (`data/`, `mlruns/`, `mlartifacts/`, `mlflow_exports/`, `outputs/`) are not
  committed and are intended to be git-ignored.
- Use `scripts/export_results.py` to export a portable snapshot into `mlflow_exports/`.
- Use `python scripts/make_report_assets.py` to regenerate `report_latex/tables/*.tex` and
  `report_latex/figures/*.png` from exports.
- The LaTeX sources live in `report_latex/` and the compiled PDF is `report_latex/main.pdf`.

---

## Reproducibility and caching

- Set `exp.seed` for reproducibility (PyTorch Lightning seeding is used).
- Subject-disjoint splits are enabled via `dataset.split_by_subject=True`.
- Train-only normalization is enabled via `dataset.normalization_mode=fold_zscore` and
  `dataset.normalization_scope=train`.
- EEGMMIDB feature extraction can be cached to `"<EEGMMIDB_DIR>/processed/*.npz"`. If you change
  feature-related flags (e.g., `dataset.feature_set`, `dataset.bandpower_include_quality`), delete
  the old cache files to force a rebuild.

---

## Windows notes

- The EEG experiment presets set `exp.num_workers=0` (and disable `pin_memory`/persistent workers)
  to avoid common OpenMP / multiprocessing issues on Windows.
- Always run with `PYTHONPATH=.` (or set it in your shell) so `src/` imports resolve correctly.

---

## Troubleshooting

- “No valid EDF records found”: verify `EEGMMIDB_DIR` and one of the supported folder layouts.
- “CUDA OOM”: reduce `model.multi.batch_size`, or disable/scale down the frontend, or accumulate
  gradients (`exp.accumulate_grad_batches` if enabled in your config).
- Hydra config debugging:
  ```bash
  PYTHONPATH=. python runnables/train_multi.py +experiment=eegmmidb_ct_baseline --cfg job
  ```

---

## Local-only ignore (recommended)

This repo intentionally does **not** ignore certain local folders in the committed `.gitignore`.
If you want to ignore cleanup artifacts locally, copy the patterns from `gitignore_user_local.template` into:

- `.git/info/exclude` (per-repo), or
- your global gitignore file.

---

## Attribution / Upstream

This project builds on the upstream Causal Transformer implementation from:
https://github.com/Valentyn1997/CausalTransformer

---

## Tooling note

Prepared for release with assistance from the OpenAI Codex CLI.
