# results_public

## Why this folder exists
`results_public/` contains a **curated, lightweight** subset of MLflow artifacts needed to audit the
runs cited in `main.pdf` (tables/appendices). The goal is to make auditing reproducible without
versioning the full `mlflow_exports/` directory or heavy artifacts.

## How to regenerate results_public
From the repo root:

```bash
python scripts/curate_public_results.py
```

By default, the script reads:
`mlflow_exports/experiment_665075068361836799_CT_eegmmidb` and writes to
`results_public/CT_eegmmidb`. You can adjust `--src`, `--dst`, and the size limit
`--max_file_mb` (default 20 MB). Use `--include_large` only if needed.

## What is intentionally excluded
- **Window-level** predictions (e.g., `predictions_*_window*.csv`)
- Checkpoints / weights (`*.ckpt`, `*.pt`, `*.pth`) and heavy caches
- `.npz` and other large artifacts
- `features_cache*` inside `artifacts/`

The `data/` folder remains ignored and must never be versioned.

## Mapping to the PDF and included runs
Experiment: **CT EEGMMIDB** (`experiment_665075068361836799`)

Tables 1 & 5 (seed=600, folds 0–4):
- 1ebc9c06 (dazzling-pig-862)
- 16521b4c (bold-fawn-723)
- a8f5bd3a (masked-foal-764)
- b7f7e83a (fearless-shad-563)
- d5599a49 (dapper-hound-665)

Tables 1 & 5 (seed=700, folds 0–4):
- 54c985de
- b6694b46
- 7da77fac
- a974048a
- 07df9328

Table 9 (validity checks – label permutation, 5 runs):
- 691ce7bf
- ae0afe45
- 2bd040d2
- c6c94f0d
- 9cea104f

Table 9 (inter-batch-decouple, fold 0):
- baseline d45bb37f
- shuffle 47eda505

Appendix 9.2:
- See the same runs (record/subject artifacts + confusion matrices).

Full IDs and run names appear in the exported folder names under
`results_public/CT_eegmmidb/` after curation.
