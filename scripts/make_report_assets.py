#!/usr/bin/env python
"""
Generate LaTeX-ready tables and figures for the FINAL report.

Design goals:
- No fabrication: every number must come from local, auditable files:
  - mlflow exports: mlflow_exports/**/metrics.json, params.json, predictions_*.csv, artifacts/*.png
  - (for learning curves only) mlruns/**/metrics/* time-series
- Deterministic selection rules (no cherry-picking).

Outputs:
- report_latex/tables/*.tex
- report_latex/figures/*.png
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "report_latex"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

EXP_CT = REPO_ROOT / "mlflow_exports" / "experiment_665075068361836799_CT_eegmmidb"
EXP_BASELINES = REPO_ROOT / "mlflow_exports" / "experiment_886282226387861497_baselines_eegmmidb"

MLRUNS_CT = REPO_ROOT / "mlruns" / "665075068361836799"


CT_FINAL_SEED = 600
CT_N_FOLDS = 5
CT_CLASS_NAMES = ["mauvais", "moyen", "excellent"]
CT_OTHER_SEEDS = [700]


def _to_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def _fmt_float(x: Optional[float], ndigits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    return f"{float(x):.{ndigits}f}"


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan")
    return mean, std


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _parse_py_list(value: Optional[str]) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = literal_eval(value)
    except Exception:
        return None
    return parsed if isinstance(parsed, list) else None


def _is_ct_candidate_run(params: Dict[str, str]) -> bool:
    """
    Identify the *candidate* CT EEGMMIDB config for the FINAL report.

    We keep this explicit to avoid accidental mixing of runs with the same seed/fold
    but different presets, while remaining auditable (everything comes from params.json).
    """
    if params.get("dataset/name") != "eegmmidb":
        return False
    if params.get("dataset/feature_set") != "raw8":
        return False
    if params.get("dataset/normalization_scope") != "train":
        return False
    if params.get("dataset/label_strategy") != "quantile_fold":
        return False
    if params.get("exp/class_weights_mode") != "manual":
        return False
    if params.get("exp/record_level_agg") != "mean_logit":
        return False
    cw = _parse_py_list(params.get("exp/class_weights"))
    if cw != [1.0, 1.5, 1.0]:
        return False
    return True


def _load_ct_runs_index() -> List[Dict[str, object]]:
    """
    Return a lightweight index of runs in EXP_CT with (seed, fold, start_time, params).

    Selection rule used later:
    - For each (seed, fold): pick the most recent run by start_time.
    """
    summary_path = EXP_CT / "runs_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = pd.read_csv(summary_path, usecols=["run_id", "run_name", "start_time"])
    start_time_by_id = dict(zip(summary["run_id"], summary["start_time"]))

    rows: List[Dict[str, object]] = []
    for run_dir in EXP_CT.glob("run_*"):
        # Expected format: run_<run_id>_<run_name>
        parts = run_dir.name.split("_", 2)
        if len(parts) < 2:
            continue
        run_id = parts[1]
        params_json = run_dir / "params.json"
        metrics_json = run_dir / "metrics.json"
        artifacts_dir = run_dir / "artifacts"
        if not params_json.exists() or not metrics_json.exists():
            continue
        params = _load_json(params_json)

        seed = _to_int(params.get("exp/seed"))
        fold = _to_int(params.get("dataset/fold_index"))
        n_folds = _to_int(params.get("dataset/n_folds"))
        if seed is None or fold is None or n_folds is None:
            continue

        rows.append(
            {
                "run_id": run_id,
                "run_name": parts[2] if len(parts) >= 3 else run_dir.name,
                "run_dir": run_dir,
                "params": params,
                "seed": seed,
                "fold": fold,
                "n_folds": n_folds,
                "start_time": start_time_by_id.get(run_id, ""),
                "params_json": params_json,
                "metrics_json": metrics_json,
                "artifacts_dir": artifacts_dir,
            }
        )
    return rows


def _select_ct_cv_runs(seed: int, n_folds: int = CT_N_FOLDS) -> List[str]:
    """
    Select exactly one run per fold for a given seed, using a deterministic rule.
    """
    idx = _load_ct_runs_index()
    candidates = [r for r in idx if r["seed"] == seed and r["n_folds"] == n_folds and _is_ct_candidate_run(r["params"])]

    by_fold: Dict[int, Dict[str, object]] = {}
    for r in candidates:
        fold = int(r["fold"])
        st = str(r["start_time"])
        if fold not in by_fold or st > str(by_fold[fold]["start_time"]):
            by_fold[fold] = r

    missing = [f for f in range(n_folds) if f not in by_fold]
    if missing:
        folds_present = sorted(by_fold.keys())
        raise RuntimeError(
            f"Missing CT CV folds for seed={seed}: missing={missing}, present={folds_present}. "
            f"Check mlflow_exports (EXP_CT={EXP_CT})."
        )

    return [str(by_fold[f]["run_id"]) for f in range(n_folds)]


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_name: str
    run_dir: Path
    params_json: Path
    metrics_json: Path
    artifacts_dir: Path


def _find_run_dir(exp_dir: Path, run_id: str) -> Path:
    matches = list(exp_dir.glob(f"run_{run_id}_*"))
    if not matches:
        raise FileNotFoundError(f"Run directory not found for run_id={run_id} under {exp_dir}")
    if len(matches) > 1:
        # Deterministic tie-break: shortest name then lexicographic.
        matches = sorted(matches, key=lambda p: (len(p.name), p.name))
    return matches[0]


def _load_run(exp_dir: Path, run_id: str) -> RunPaths:
    run_dir = _find_run_dir(exp_dir, run_id)
    parts = run_dir.name.split("_", 2)
    run_name = parts[2] if len(parts) >= 3 else run_dir.name
    params_json = run_dir / "params.json"
    metrics_json = run_dir / "metrics.json"
    artifacts_dir = run_dir / "artifacts"
    if not params_json.exists():
        raise FileNotFoundError(f"Missing params.json for run_id={run_id}: {params_json}")
    if not metrics_json.exists():
        raise FileNotFoundError(f"Missing metrics.json for run_id={run_id}: {metrics_json}")
    return RunPaths(
        run_id=run_id,
        run_name=run_name,
        run_dir=run_dir,
        params_json=params_json,
        metrics_json=metrics_json,
        artifacts_dir=artifacts_dir,
    )


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_latex_table(path: Path, caption: str, label: str, header: List[str], rows: List[List[str]]) -> None:
    cols = "l" * len(header)
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{{_latex_escape(label)}}}")
    wide = len(header) > 6
    if wide:
        lines.append("\\scriptsize")
        lines.append("\\setlength{\\tabcolsep}{3pt}")
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_latex_escape(h) for h in header) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    if wide:
        lines.append("}")
    lines.append("\\end{table}")
    lines.append("")
    _write_text(path, "\n".join(lines))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def _per_class_prf(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Returns: support, precision, recall, f1
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return support, precision, recall, f1


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_png: Path, normalize: bool) -> None:
    import matplotlib.pyplot as plt

    if normalize:
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums > 0)
        mat = cm_norm
        fmt = "{:.2f}"
        title = "Matrice de confusion (normalisée par classe vraie)"
    else:
        mat = cm
        fmt = "{:d}"
        title = "Matrice de confusion (comptes)"

    fig = plt.figure(figsize=(5.5, 4.5), dpi=200)
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Vrai",
        xlabel="Prédit",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = mat.max() * 0.6 if mat.size else 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            text = fmt.format(val) if not normalize else fmt.format(float(val))
            ax.text(j, i, text, ha="center", va="center", color="white" if val > thresh else "black")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _plot_learning_curves(run_id: str, out_png: Path) -> None:
    """
    Plot train/val curves from mlruns time-series.
    (Needed because mlflow_exports/*/metrics.json contains only final scalars.)
    """
    import matplotlib.pyplot as plt

    run_dir = MLRUNS_CT / run_id
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"mlruns metrics directory not found for run_id={run_id}: {metrics_dir}")

    def load_metric(name: str) -> pd.DataFrame:
        p = metrics_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing mlruns metric file: {p}")
        df = pd.read_csv(p, sep=r"\s+", engine="python", header=None, names=["ts", "value", "step"])
        df = df[["step", "value"]]
        return df

    epoch_df = load_metric("epoch").drop_duplicates(subset=["step"], keep="last").set_index("step")

    def load_with_epoch(name: str) -> pd.DataFrame:
        df = load_metric(name).drop_duplicates(subset=["step"], keep="last").set_index("step")
        df = df.join(epoch_df, how="left", rsuffix="_epoch")
        df = df.rename(columns={"value": name, "value_epoch": "epoch"})
        df = df.dropna(subset=["epoch"])
        df["epoch"] = df["epoch"].astype(int)
        # one point per epoch
        df = df.sort_values("epoch").groupby("epoch", as_index=False).last()
        return df

    train_loss = load_with_epoch("multi_train_ce_loss")
    val_loss = load_with_epoch("multi_val_ce_loss")
    val_f1_record = load_with_epoch("multi_val_f1_macro_record")
    val_bal_subject = load_with_epoch("multi_val_balanced_acc_subject")

    fig = plt.figure(figsize=(7.2, 4.2), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(train_loss["epoch"], train_loss["multi_train_ce_loss"], label="Train CE", linewidth=1.5)
    ax1.plot(val_loss["epoch"], val_loss["multi_val_ce_loss"], label="Val CE", linewidth=1.5)
    ax1.set_title("Loss (Cross-Entropy)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CE")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(val_f1_record["epoch"], val_f1_record["multi_val_f1_macro_record"], label="Val Macro-F1 (record)", linewidth=1.5)
    ax2.plot(val_bal_subject["epoch"], val_bal_subject["multi_val_balanced_acc_subject"], label="Val BalAcc (subject)", linewidth=1.5)
    ax2.set_title("Validation (métriques)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _concat_predictions(run_ids: Sequence[str], level: str) -> pd.DataFrame:
    dfs = []
    for rid in run_ids:
        run = _load_run(EXP_CT, rid)
        pred_path = run.artifacts_dir / "predictions" / f"predictions_test_{level}.csv"
        if level == "window":
            pred_path = run.artifacts_dir / "predictions" / "predictions_test.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions for run_id={rid}: {pred_path}")
        df = pd.read_csv(pred_path)
        df["run_id"] = rid
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _write_class_table(df: pd.DataFrame, class_names: List[str], out_tex: Path, caption: str, label: str) -> None:
    cm = _confusion(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), n_classes=len(class_names))
    support, precision, recall, f1 = _per_class_prf(cm)
    rows = []
    for i, name in enumerate(class_names):
        rows.append([
            _latex_escape(name),
            f"{int(support[i])}",
            _fmt_float(float(precision[i]), 3),
            _fmt_float(float(recall[i]), 3),
            _fmt_float(float(f1[i]), 3),
        ])
    _write_latex_table(
        out_tex,
        caption=caption,
        label=label,
        header=["Classe", "Support", "Precision", "Recall", "F1"],
        rows=rows,
    )


def _generate_cv_table(run_ids: Sequence[str], out_tex: Path) -> Dict[str, float]:
    raise NotImplementedError("Use _generate_cv_table_split(...) instead.")


def _read_accuracy_from_class_report(run: RunPaths, split: str, level: str) -> float:
    """
    Read the scalar 'accuracy' from a classification_report_*.csv generated by post-analyze.
    split: 'val' or 'test'
    level: 'window' | 'record' | 'subject'
    """
    if split not in ("val", "test"):
        raise ValueError(f"Invalid split={split}")
    if level not in ("window", "record", "subject"):
        raise ValueError(f"Invalid level={level}")

    suffix = "" if level == "window" else f"_{level}"
    path = run.artifacts_dir / "predictions" / f"classification_report_{split}{suffix}.csv"
    df = pd.read_csv(path, index_col=0)
    # The export writes 'accuracy' as a row label.
    if "accuracy" not in df.index:
        raise ValueError(f"Missing 'accuracy' row in {path}")
    return float(df.loc["accuracy", "f1-score"])


def _generate_cv_table_split(run_ids: Sequence[str], *, split: str, seed: int, out_tex: Path) -> Dict[str, float]:
    if split not in ("val", "test"):
        raise ValueError(f"Invalid split={split}")

    rows = []
    vals_bal_w = []
    vals_f1_w = []
    vals_acc_w = []
    vals_bal_r = []
    vals_f1_r = []
    vals_acc_r = []
    vals_bal_s = []
    vals_f1_s = []

    for rid in run_ids:
        run = _load_run(EXP_CT, rid)
        params = _load_json(run.params_json)
        metrics = _load_json(run.metrics_json)

        fold = _to_int(params.get("dataset/fold_index", params.get("fold_index")))
        epoch = metrics.get("epoch")

        bal_w = float(metrics.get(f"multi_{split}_balanced_acc"))
        f1_w = float(metrics.get(f"multi_{split}_f1_macro"))
        acc_w = float(metrics.get(f"multi_{split}_acc"))

        bal_r = float(metrics.get(f"multi_{split}_balanced_acc_record"))
        f1_r = float(metrics.get(f"multi_{split}_f1_macro_record"))
        acc_r = _read_accuracy_from_class_report(run, split=split, level="record")

        bal_s = float(metrics.get(f"multi_{split}_balanced_acc_subject"))
        f1_s = float(metrics.get(f"multi_{split}_f1_macro_subject"))

        rows.append([
            f"{fold}",
            f"\\texttt{{{rid[:8]}}}",
            _latex_escape(run.run_name),
            _fmt_float(epoch, 0),
            _fmt_float(bal_w, 3),
            _fmt_float(f1_w, 3),
            _fmt_float(acc_w, 3),
            _fmt_float(bal_r, 3),
            _fmt_float(f1_r, 3),
            _fmt_float(acc_r, 3),
            _fmt_float(bal_s, 3),
            _fmt_float(f1_s, 3),
        ])

        vals_bal_w.append(bal_w)
        vals_f1_w.append(f1_w)
        vals_acc_w.append(acc_w)
        vals_bal_r.append(bal_r)
        vals_f1_r.append(f1_r)
        vals_acc_r.append(acc_r)
        vals_bal_s.append(bal_s)
        vals_f1_s.append(f1_s)

    # sort by fold index
    rows = sorted(rows, key=lambda r: int(r[0]))

    mean_bw, std_bw = _mean_std(vals_bal_w)
    mean_f1w, std_f1w = _mean_std(vals_f1_w)
    mean_aw, std_aw = _mean_std(vals_acc_w)
    mean_br, std_br = _mean_std(vals_bal_r)
    mean_f1r, std_f1r = _mean_std(vals_f1_r)
    mean_ar, std_ar = _mean_std(vals_acc_r)
    mean_bs, std_bs = _mean_std(vals_bal_s)
    mean_f1s, std_f1s = _mean_std(vals_f1_s)

    rows.append([
        "\\textbf{Moy $\\pm$ Std}",
        "",
        "",
        "",
        f"\\textbf{{{_fmt_float(mean_bw, 3)} $\\pm$ {_fmt_float(std_bw, 3)}}}",
        f"\\textbf{{{_fmt_float(mean_f1w, 3)} $\\pm$ {_fmt_float(std_f1w, 3)}}}",
        f"\\textbf{{{_fmt_float(mean_aw, 3)} $\\pm$ {_fmt_float(std_aw, 3)}}}",
        f"\\textbf{{{_fmt_float(mean_br, 3)} $\\pm$ {_fmt_float(std_br, 3)}}}",
        f"\\textbf{{{_fmt_float(mean_f1r, 3)} $\\pm$ {_fmt_float(std_f1r, 3)}}}",
        f"\\textbf{{{_fmt_float(mean_ar, 3)} $\\pm$ {_fmt_float(std_ar, 3)}}}",
        f"\\textbf{{{_fmt_float(mean_bs, 3)} $\\pm$ {_fmt_float(std_bs, 3)}}}",
        f"\\textbf{{{_fmt_float(mean_f1s, 3)} $\\pm$ {_fmt_float(std_f1s, 3)}}}",
    ])

    _write_latex_table(
        out_tex,
        caption=f"Validation croisée ({split}) — métriques par fold (CT, raw8, manualcw, seed={seed}).",
        label=f"tab:cv-{split}-seed{seed}",
        header=[
            "Fold",
            "Run ID",
            "Run name",
            "epoch",
            "BalAcc (window)",
            "Macro-F1 (window)",
            "Acc (window)",
            "BalAcc (record)",
            "Macro-F1 (record)",
            "Acc (record)",
            "BalAcc (subject)",
            "Macro-F1 (subject)",
        ],
        rows=rows,
    )

    return {
        "mean_bal_acc_subject": mean_bs,
        "std_bal_acc_subject": std_bs,
        "mean_macro_f1_record": mean_f1r,
        "std_macro_f1_record": std_f1r,
        "mean_macro_f1_subject": mean_f1s,
        "std_macro_f1_subject": std_f1s,
    }


def _generate_baselines_table(out_tex: Path) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Deterministic selection rule for duplicates:
    - Use runs_summary.csv as index.
    - For each (model, fold), keep the row with the latest start_time (string sort is OK: YYYY-MM-DD_HH-MM-SS).
    """
    csv_path = EXP_BASELINES / "runs_summary.csv"
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Select per (model, fold) the latest start_time
    selected: Dict[Tuple[str, int], Dict[str, str]] = {}
    for row in rows:
        run_name = row["run_name"]
        parts = run_name.split("_")
        model = parts[0]  # eegnet, shallowconv, simple, csp
        fold = None
        for p in parts:
            if p.startswith("fold"):
                fold = _to_int(p.replace("fold", ""))
        if fold is None:
            continue
        key = (model, fold)
        if key not in selected:
            selected[key] = row
            continue
        if row["start_time"] > selected[key]["start_time"]:
            selected[key] = row

    # Aggregate across folds
    by_model: Dict[str, List[Dict[str, str]]] = {}
    for (model, fold), row in selected.items():
        by_model.setdefault(model, []).append(row)

    class_names = {
        "eegnet": "EEGNet",
        "shallowconv": "ShallowConvNet",
        "simple": "SimpleCNN",
        "csp": "CSP+LDA",
    }

    rows_tex = []
    summary: Dict[str, Tuple[float, float, float, float]] = {}
    for model_key in ["eegnet", "shallowconv", "simple", "csp"]:
        model_rows = by_model.get(model_key, [])
        if len(model_rows) != 5:
            raise RuntimeError(f"Expected 5 folds for baseline {model_key}, got {len(model_rows)}")
        subj_vals = [float(r["metric.subject_balanced_acc"]) for r in model_rows]
        rec_f1_vals = [float(r["metric.record_macro_f1"]) for r in model_rows]
        subj_mean, subj_std = _mean_std(subj_vals)
        recf1_mean, recf1_std = _mean_std(rec_f1_vals)
        summary[model_key] = (subj_mean, subj_std, recf1_mean, recf1_std)
        rows_tex.append([
            _latex_escape(class_names[model_key]),
            f"{_fmt_float(subj_mean, 3)} $\\pm$ {_fmt_float(subj_std, 3)}",
            f"{_fmt_float(recf1_mean, 3)} $\\pm$ {_fmt_float(recf1_std, 3)}",
        ])

    _write_latex_table(
        out_tex,
        caption="Baselines EEG (test) — moyenne $\\pm$ std sur 5 folds.",
        label="tab:baselines-test",
        header=["Modèle", "Balanced Acc (subject)", "Macro-F1 (record)"],
        rows=rows_tex,
    )
    return summary


def _generate_benchmark_table(ct_summary: Dict[str, float], baseline_summary: Dict[str, Tuple[float, float, float, float]], out_tex: Path) -> None:
    ct_bal = ct_summary["mean_bal_acc_subject"]
    ct_bal_std = ct_summary["std_bal_acc_subject"]
    ct_f1r = ct_summary["mean_macro_f1_record"]
    ct_f1r_std = ct_summary["std_macro_f1_record"]

    rows = []
    rows.append([
        "\\textbf{CT (Raw8, manualcw)}",
        f"\\textbf{{{_fmt_float(ct_bal, 3)} $\\pm$ {_fmt_float(ct_bal_std, 3)}}}",
        f"\\textbf{{{_fmt_float(ct_f1r, 3)} $\\pm$ {_fmt_float(ct_f1r_std, 3)}}}",
        _latex_escape(f"Config candidate (seed={CT_FINAL_SEED}, 5 folds, train-only)."),
    ])

    order = [
        ("eegnet", "EEGNet"),
        ("shallowconv", "ShallowConvNet"),
        ("simple", "SimpleCNN"),
        ("csp", "CSP+LDA"),
    ]
    for key, name in order:
        subj_mean, subj_std, recf1_mean, recf1_std = baseline_summary[key]
        rows.append([
            _latex_escape(name),
            f"{_fmt_float(subj_mean, 3)} $\\pm$ {_fmt_float(subj_std, 3)}",
            f"{_fmt_float(recf1_mean, 3)} $\\pm$ {_fmt_float(recf1_std, 3)}",
            "",
        ])

    _write_latex_table(
        out_tex,
        caption="Benchmark interne — CT vs baselines (test).",
        label="tab:benchmark",
        header=["Modèle", "Balanced Acc (subject)", "Macro-F1 (record)", "Observations"],
        rows=rows,
    )


def _generate_multiseed_table(out_tex: Path) -> None:
    """
    Multi-seed stability for the *train-only* pipeline:
    - seed=CT_FINAL_SEED
    - seed in CT_OTHER_SEEDS
    """

    def summarize(run_ids: Sequence[str]) -> Tuple[float, float, float, float]:
        subj = []
        recf1 = []
        for rid in run_ids:
            run = _load_run(EXP_CT, rid)
            m = _load_json(run.metrics_json)
            subj.append(float(m["multi_test_balanced_acc_subject"]))
            recf1.append(float(m["multi_test_f1_macro_record"]))
        subj_mean, subj_std = _mean_std(subj)
        rec_mean, rec_std = _mean_std(recf1)
        return subj_mean, subj_std, rec_mean, rec_std

    rows = []
    for seed in [CT_FINAL_SEED, *CT_OTHER_SEEDS]:
        run_ids = _select_ct_cv_runs(seed)
        s = summarize(run_ids)
        rows.append([str(seed), f"{_fmt_float(s[0], 3)} $\\pm$ {_fmt_float(s[1], 3)}", f"{_fmt_float(s[2], 3)} $\\pm$ {_fmt_float(s[3], 3)}"])

    _write_latex_table(
        out_tex,
        caption="Robustesse multi-seed (test) — pipeline train-only (CT, raw8, manualcw, 5 folds).",
        label="tab:multiseed",
        header=["Seed", "Balanced Acc (subject)", "Macro-F1 (record)"],
        rows=rows,
    )


def _generate_sanity_table(out_tex: Path) -> None:
    """
    Sanity checks table: pulls values from specific runs (from results_report.md / audit_report.md).
    """
    # label-shuffle (folds 0..4, seed=601, 3 epochs)
    label_shuffle_runs = [
        "691ce7bfc9c4433997d5d6c41de619ca",
        "ae0afe452b39453b940fb4817719e3c1",
        "2bd040d284fb4a0082e1f4abac9191f1",
        "c6c94f0dd96446c197ac89efcd274ede",
        "9cea104ff9ea4d8d8cc59e3aca60da14",
    ]

    # inter-batch-decouple (seed=777, fold 0, epoch=2) — baseline vs shuffle
    baseline = "d45bb37fff884229a592a4c177ed677c"
    input_shuffle = "47eda505a4ce4363a01c2d14987df66c"

    def metric_triplet(run_id: str) -> Tuple[float, float, float]:
        m = _load_json(_load_run(EXP_CT, run_id).metrics_json)
        return float(m["multi_test_balanced_acc"]), float(m["multi_test_balanced_acc_record"]), float(m["multi_test_balanced_acc_subject"])

    # Label shuffle should be ~0.333
    ls_vals = [metric_triplet(r)[2] for r in label_shuffle_runs]
    ls_mean, ls_std = _mean_std(ls_vals)

    b = metric_triplet(baseline)
    s = metric_triplet(input_shuffle)
    delta = (s[0] - b[0], s[1] - b[1], s[2] - b[2])

    rows = [
        [
            _latex_escape("Label permutation (folds 0–4)"),
            _latex_escape("PASS"),
            f"BalAcc subject (moyenne $\\pm$ std) = {_fmt_float(ls_mean, 3)} $\\pm$ {_fmt_float(ls_std, 3)}",
            _latex_escape(", ".join(r[:8] for r in label_shuffle_runs)),
        ],
        [
            _latex_escape("Input-shuffle inter-batch-decouple (fold 0)"),
            _latex_escape("PASS"),
            _latex_escape(
                f"Delta BalAcc (window/record/subject) = {_fmt_float(delta[0], 3)} / {_fmt_float(delta[1], 3)} / {_fmt_float(delta[2], 3)}"
            ),
            _latex_escape(f"baseline {baseline[:8]}, shuffle {input_shuffle[:8]}"),
        ],
    ]

    _write_latex_table(
        out_tex,
        caption="Sanity checks (résumé) — résultats test issus des exports MLflow.",
        label="tab:sanity",
        header=["Test", "Statut", "Résumé métriques", "Run IDs"],
        rows=rows,
    )


def _copy_key_figures() -> None:
    """
    Copy a small set of existing figures to stable filenames:
    - MLflow-exported reports (confusion matrices, confidence histograms)
    - EEGMMIDB montage figure (bundled with the local dataset copy)
    """
    seed_runs = _select_ct_cv_runs(CT_FINAL_SEED)
    run0 = _load_run(EXP_CT, seed_runs[0])  # fold 0

    # Confusion matrices (subject level) and confidence histogram (window level)
    candidates = [
        (run0.artifacts_dir / "reports" / "confusion_matrix_test_subject_norm.png", f"cm_subject_norm_seed{CT_FINAL_SEED}_fold0.png"),
        (run0.artifacts_dir / "reports" / "confusion_matrix_test_record_norm.png", f"cm_record_norm_seed{CT_FINAL_SEED}_fold0.png"),
        (run0.artifacts_dir / "reports" / "confidence_hist_test_window.png", f"confidence_hist_window_seed{CT_FINAL_SEED}_fold0.png"),
        (
            REPO_ROOT
            / "data"
            / "eegmmidb"
            / "MNE-eegbci-data"
            / "files"
            / "64_channel_sharbrough.png",
            "eegmmidb_64_channel_montage.png",
        ),
    ]
    for src, dst_name in candidates:
        if src.exists():
            _copy(src, FIG_DIR / dst_name)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    ct_final_runs = _select_ct_cv_runs(CT_FINAL_SEED)

    # 1) Copy a few key figures from exports (no computation)
    _copy_key_figures()

    # 2) Learning curves (from mlruns time-series)
    _plot_learning_curves(ct_final_runs[0], FIG_DIR / f"learning_curves_seed{CT_FINAL_SEED}_fold0.png")

    # 3) Aggregated confusion matrices + class tables (concat folds, test)
    for level, suffix in [("window", "window"), ("record", "record"), ("subject", "subject")]:
        df = _concat_predictions(ct_final_runs, level=level)
        cm = _confusion(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), n_classes=len(CT_CLASS_NAMES))
        _plot_confusion_matrix(cm, CT_CLASS_NAMES, FIG_DIR / f"cm_{suffix}_agg_seed{CT_FINAL_SEED}.png", normalize=False)
        _plot_confusion_matrix(cm, CT_CLASS_NAMES, FIG_DIR / f"cm_{suffix}_agg_seed{CT_FINAL_SEED}_norm.png", normalize=True)
        _write_class_table(
            df,
            CT_CLASS_NAMES,
            out_tex=TABLE_DIR / f"class_report_{suffix}_agg_seed{CT_FINAL_SEED}.tex",
            caption=f"Rapport par classe (test, agrégé sur 5 folds) — niveau {suffix} (seed={CT_FINAL_SEED}).",
            label=f"tab:class-{suffix}-seed{CT_FINAL_SEED}",
        )

    # 4) CV fold tables (seed=CT_FINAL_SEED, candidate config)
    ct_summary = _generate_cv_table_split(ct_final_runs, split="test", seed=CT_FINAL_SEED, out_tex=TABLE_DIR / f"cv_test_seed{CT_FINAL_SEED}.tex")
    _generate_cv_table_split(ct_final_runs, split="val", seed=CT_FINAL_SEED, out_tex=TABLE_DIR / f"cv_val_seed{CT_FINAL_SEED}.tex")

    # 5) Baselines table + benchmark comparison table (CT vs baselines)
    baselines_summary = _generate_baselines_table(TABLE_DIR / "baselines_test.tex")
    _generate_benchmark_table(ct_summary, baselines_summary, TABLE_DIR / "benchmark_test.tex")

    # 6) Multi-seed stability (train-only seeds)
    _generate_multiseed_table(TABLE_DIR / "multiseed_trainonly.tex")

    # 7) Sanity checks
    _generate_sanity_table(TABLE_DIR / "sanity_checks.tex")


if __name__ == "__main__":
    main()
