#!/usr/bin/env python3
"""
Export MLflow experiments/runs into a portable, Git-friendly folder structure.

This script snapshots runs from:
- a local MLflow tracking server (HTTP/S), or
- a local `mlruns/` directory (file-based tracking),
into a directory like `mlflow_exports/<experiment_name>/run_<id>_<name>/`.

The goal is reproducibility and auditing (metrics/params/artifacts on disk) without committing
large MLflow working directories to Git.
"""

import argparse
import csv
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote

import mlflow
from mlflow.tracking import MlflowClient

try:
    from mlflow.artifacts import download_artifacts as mlflow_download_artifacts
except Exception:
    mlflow_download_artifacts = None

DEFAULT_MLFLOW_URI = "http://127.0.0.1:5000"


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "t"}:
        return True
    if v in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _sanitize_name(name: str) -> str:
    if not name:
        return "unnamed"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return safe or "unnamed"


def _format_time_ms(ms: int) -> str:
    if ms is None:
        return "unknown_time"
    return datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d_%H-%M-%S")


def _get_run_name(run) -> str:
    run_name = None
    if hasattr(run.info, "run_name") and run.info.run_name:
        run_name = run.info.run_name
    if not run_name:
        run_name = run.data.tags.get("mlflow.runName") if run.data and run.data.tags else None
    if not run_name:
        run_name = run.data.tags.get("mlflow.run_name") if run.data and run.data.tags else None
    if not run_name:
        run_name = _format_time_ms(run.info.start_time)
    return run_name


def normalize_tracking_uri(uri: str) -> str:
    """Validate and normalize a user-provided MLflow tracking URI (HTTP/S URL or file path)."""
    if uri is None:
        raise ValueError("MLflow tracking URI is None.")
    uri = uri.strip()
    if not uri:
        raise ValueError("MLflow tracking URI is empty.")
    if re.search(r"\\s", uri):
        raise ValueError(
            "MLflow tracking URI contains whitespace. This often happens on Windows when "
            "a command is appended to the URI (e.g. 'http://127.0.0.1:5000 python ...'). "
            "Set MLFLOW_TRACKING_URI to the URL only, or pass --mlflow-uri explicitly."
        )
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        if not parsed.netloc:
            raise ValueError(f"Invalid MLflow tracking URI (missing host): {uri}")
        return uri
    if parsed.scheme == "file":
        return uri
    if parsed.scheme == "":
        # Allow bare file paths (mlruns) or relative paths
        return uri
    raise ValueError(f"Unsupported MLflow tracking URI scheme: {parsed.scheme}")


def _get_tracking_uri(args) -> str:
    if args.mlruns_path:
        path = Path(args.mlruns_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"mlruns path not found: {path}")
        return path.as_uri()
    if args.mlflow_uri:
        return normalize_tracking_uri(args.mlflow_uri)
    env_uri = os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("MLFLOW_TRACKING_URL")
    if env_uri:
        return normalize_tracking_uri(env_uri)
    return normalize_tracking_uri(DEFAULT_MLFLOW_URI)


def _artifact_uri_to_path(uri: str):
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file":
            path = unquote(parsed.path)
            if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
                path = path[1:]
            return path
        return uri
    return None


def _match_globs(path: str, globs):
    if not globs:
        return True
    norm = path.replace(os.sep, "/")
    base = os.path.basename(path)
    for g in globs:
        g = g.strip()
        if not g:
            continue
        if fnmatch.fnmatch(norm, g) or fnmatch.fnmatch(base, g):
            return True
    return False


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _copy_file(src: str, dest: str, copy_mode: str):
    _ensure_dir(os.path.dirname(dest))
    if copy_mode == "symlink":
        try:
            if os.path.lexists(dest):
                os.remove(dest)
            os.symlink(src, dest)
            return
        except Exception:
            pass
    shutil.copy2(src, dest)


def _copy_artifacts_from_local(local_root: str, dest_root: str, globs, copy_mode: str):
    if not os.path.isdir(local_root):
        raise FileNotFoundError(f"Local artifact directory not found: {local_root}")
    for root, _, files in os.walk(local_root):
        for fname in files:
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, local_root)
            if not _match_globs(rel, globs):
                continue
            dest = os.path.join(dest_root, rel)
            _copy_file(src, dest, copy_mode)


def _list_artifact_files(client: MlflowClient, run_id: str, path: str = ""):
    files = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            files.extend(_list_artifact_files(client, run_id, item.path))
        else:
            files.append(item.path)
    return files


def _safe_list_artifacts_root(client: MlflowClient, run_id: str):
    try:
        items = client.list_artifacts(run_id, "")
        return list(items or []), None
    except Exception as exc:
        return [], exc


def _local_has_any_files(root: str) -> bool:
    try:
        for _, _, files in os.walk(root):
            if files:
                return True
        return False
    except Exception:
        return False


def _download_artifacts(client: MlflowClient, run_id: str, artifact_uri: str, dest_root: str, globs):
    if not globs:
        if mlflow_download_artifacts is not None:
            try:
                mlflow_download_artifacts(run_id=run_id, dst_path=dest_root)
                return
            except TypeError:
                try:
                    mlflow_download_artifacts(artifact_uri=artifact_uri, dst_path=dest_root)
                    return
                except Exception:
                    pass
            except Exception:
                pass
        client.download_artifacts(run_id, "", dest_root)
        return

    files = _list_artifact_files(client, run_id)
    for path in files:
        if not _match_globs(path, globs):
            continue
        client.download_artifacts(run_id, path, dest_root)


def _promote_prediction_files(artifacts_dir: str, run_dir: str):
    targets = {
        "predictions_val.csv",
        "predictions_test.csv",
        "predictions_val_record.csv",
        "predictions_test_record.csv",
        "classification_report_val.csv",
        "classification_report_val.txt",
        "classification_report_test.csv",
        "classification_report_test.txt",
        "classification_report_val_record.csv",
        "classification_report_val_record.txt",
        "classification_report_test_record.csv",
        "classification_report_test_record.txt",
        "confusion_matrix_val_record.png",
        "confusion_matrix_test_record.png",
        "confusion_matrix_val_subject.png",
        "confusion_matrix_test_subject.png",
        "confusion_matrix_val_window_norm.png",
        "confusion_matrix_test_window_norm.png",
        "confusion_matrix_val_record_norm.png",
        "confusion_matrix_test_record_norm.png",
        "confusion_matrix_val_subject_norm.png",
        "confusion_matrix_test_subject_norm.png",
        "confidence_hist_val_window.png",
        "confidence_hist_test_window.png",
        "confidence_hist_val_record.png",
        "confidence_hist_test_record.png",
        "confidence_hist_val_subject.png",
        "confidence_hist_test_subject.png",
        "quality_report_val.json",
        "quality_report_val.md",
        "quality_report_test.json",
        "quality_report_test.md",
    }
    for root, _, files in os.walk(artifacts_dir):
        for fname in files:
            if fname in targets:
                src = os.path.join(root, fname)
                dest = os.path.join(run_dir, fname)
                try:
                    shutil.copy2(src, dest)
                except Exception:
                    pass


def _write_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _get_fold_index(run):
    if not run.data or not run.data.params:
        return None
    
    # Primary sources for fold index
    fold_keys = [
        "fold_index", 
        "fold", 
        "data.fold_index", 
        "dataset.fold_index"
    ]
    for key in fold_keys:
        if key in run.data.params:
            return int(run.data.params[key])

    # Fallback: check for "foldX" in the run name as a last resort
    run_name = _get_run_name(run)
    match = re.search(r"fold(\d+)", run_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
        
    return None


def _write_manifest(path: str, exp: 'Experiment', export_dir: str, exported_runs_info: list):
    manifest = {
        "createdAt": datetime.now().isoformat(),
        "experimentId": exp.experiment_id,
        "experimentName": exp.name,
        "exportDir": export_dir,
        "runs": exported_runs_info,
    }
    _write_json(path, manifest)




def _write_runs_summary(path: str, runs):
    base_fields = [
        "run_id",
        "run_name",
        "status",
        "start_time",
        "end_time",
        "artifact_uri",
    ]
    metric_keys = set()
    rows = []
    for run in runs:
        metrics = run.data.metrics if run.data else {}
        metric_keys.update(metrics.keys())
        row = {
            "run_id": run.info.run_id,
            "run_name": _get_run_name(run),
            "status": run.info.status,
            "start_time": _format_time_ms(run.info.start_time),
            "end_time": _format_time_ms(run.info.end_time),
            "artifact_uri": run.info.artifact_uri,
        }
        for k, v in metrics.items():
            row[f"metric.{k}"] = v
        rows.append(row)

    metric_fields = [f"metric.{k}" for k in sorted(metric_keys)]
    fieldnames = base_fields + metric_fields
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _select_runs(runs, mode: str, metric: str):
    if not runs:
        return []
    mode = mode.lower()
    if mode == "all":
        return runs
    if mode == "latest":
        return [max(runs, key=lambda r: r.info.start_time or 0)]
    if mode == "best":
        scored = []
        for run in runs:
            val = run.data.metrics.get(metric) if run.data else None
            if val is None:
                continue
            scored.append((val, run))
        if not scored:
            print(f"WARNING: No runs have metric '{metric}'. Falling back to latest run.")
            return [max(runs, key=lambda r: r.info.start_time or 0)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [scored[0][1]]
    raise ValueError(f"Unknown runs mode: {mode}")


def _search_runs_paginated(client: MlflowClient, experiment_id: str, order_by, page_size: int, max_runs: int):
    runs = []
    page_token = None
    while True:
        batch = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=order_by,
            max_results=page_size,
            page_token=page_token,
        )
        runs.extend(batch)
        if max_runs and len(runs) >= max_runs:
            break
        page_token = getattr(batch, "token", None) or getattr(batch, "next_page_token", None)
        if not page_token:
            break
    if max_runs:
        return runs[:max_runs]
    return runs


def _export_run(client: MlflowClient, run, run_dir: str, include_artifacts: bool, globs, copy_mode: str):
    _ensure_dir(run_dir)
    params_path = os.path.join(run_dir, "params.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    tags_path = os.path.join(run_dir, "tags.json")

    _write_json(params_path, run.data.params if run.data else {})
    _write_json(metrics_path, run.data.metrics if run.data else {})
    _write_json(tags_path, run.data.tags if run.data else {})

    export_status = {
        "has_artifacts": False,
        "artifacts_downloaded": False,
        "artifact_error": None,
        "paramsPath": params_path,
        "tagsPath": tags_path,
    }

    if not include_artifacts:
        return export_status

    run_id = run.info.run_id
    artifact_uri = run.info.artifact_uri
    local_root = _artifact_uri_to_path(artifact_uri)

    # Detect artifacts before attempting download/copy (and tolerate missing/errored endpoints).
    if local_root and os.path.exists(local_root):
        if not _local_has_any_files(local_root):
            print(f"INFO: No artifacts for run {run_id}, skipping artifact download.")
            return export_status
        export_status["has_artifacts"] = True
    else:
        root_items, list_err = _safe_list_artifacts_root(client, run_id)
        if list_err is not None:
            export_status["artifact_error"] = f"list_artifacts failed: {list_err}"
            print(f"INFO: No artifacts for run {run_id}, skipping artifact download.")
            return export_status
        if not root_items:
            print(f"INFO: No artifacts for run {run_id}, skipping artifact download.")
            return export_status
        export_status["has_artifacts"] = True

    artifacts_dir = os.path.join(run_dir, "artifacts")
    _ensure_dir(artifacts_dir)

    try:
        if local_root and os.path.exists(local_root):
            _copy_artifacts_from_local(local_root, artifacts_dir, globs, copy_mode)
        else:
            _download_artifacts(client, run_id, artifact_uri, artifacts_dir, globs)
        export_status["artifacts_downloaded"] = True
    except Exception as exc:
        export_status["artifact_error"] = str(exc)
        print(f"WARNING: Failed to export artifacts for run {run_id}: {exc}")
        # Continue: params/tags/metrics are already written; analysis/promotion should not crash.

    try:
        _promote_prediction_files(artifacts_dir, run_dir)
    except Exception:
        pass

    return export_status


def main():
    """CLI entrypoint for exporting MLflow experiments/runs into `mlflow_exports/`."""
    parser = argparse.ArgumentParser(
        description="Export MLflow experiment results per experiment and per run.",
        epilog="""\
Examples (Windows CMD):
  :: Export all runs from an experiment
  python scripts\\export_results.py --experiment-name "CT/eegmmidb"
  
  :: Export the last 5 runs from an experiment
  python scripts\\export_results.py --experiment-name "CT/eegmmidb" --last-n 5

  :: Export specific runs by ID
  python scripts\\export_results.py --experiment-name "CT/eegmmidb" --run-ids "b2a140...,dc9a39...,e40b59..."
"""
    )
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI (default: env or http://127.0.0.1:5000)")
    parser.add_argument("--mlruns-path", type=str, default=None, help="Path to local mlruns folder for offline export")
    parser.add_argument("--experiment-id", type=str, default=None, help="Filter by experiment ID")
    parser.add_argument("--experiment-name", type=str, default=None, help="Filter by experiment name")
    parser.add_argument("--output-dir", type=str, default="mlflow_exports", help="Root output directory")
    
    # New arguments
    parser.add_argument("--last-n", type=int, default=None, help="Export the N most recent runs.")
    parser.add_argument("--run-ids", type=str, default=None, help="Comma-separated list of specific run IDs to export.")

    parser.add_argument("--runs", type=str, default="all", choices=["all", "latest", "best"], help="Which runs to export (legacy, prefer --limit or --run-ids)")
    parser.add_argument("--metric", type=str, default="multi_val_f1_macro", help="Metric for selecting best run")
    parser.add_argument("--include-artifacts", type=_parse_bool, default=True, help="Whether to export artifacts (true/false)")
    parser.add_argument("--artifact-globs", type=str, default=None, help="Comma-separated glob filters (e.g. '*.png,*.csv')")
    parser.add_argument("--copy-mode", type=str, default="copy", choices=["copy", "symlink"], help="Copy mode for local artifacts")
    parser.add_argument("--max-runs", type=int, default=200, help="Max runs to fetch per experiment (pagination-aware).")
    parser.add_argument("--page-size", type=int, default=1000, help="Page size for MLflow search_runs pagination.")
    parser.add_argument("--write-manifest", type=_parse_bool, default=True, help="Write a manifest.json file in the root of the experiment export directory.")
    parser.add_argument("--post-analyze", type=_parse_bool, default=False, help="Run post-export analysis script after export is complete.")
    parser.add_argument("--post-analyze-metric", type=str, default="subject_balanced_acc", help="Metric to use for ranking in post-analysis.")

    args = parser.parse_args()

    if args.experiment_id and args.experiment_name:
        print("Please provide only one of --experiment-id or --experiment-name.")
        sys.exit(1)
    if args.mlflow_uri and args.mlruns_path:
        print("ERROR: Provide only one of --mlflow-uri or --mlruns-path. Use --mlruns-path for offline exports.")
        sys.exit(1)

    try:
        tracking_uri = _get_tracking_uri(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    print(f"INFO: Tracking URI: {tracking_uri}")

    experiments = []
    if args.experiment_id:
        exp = client.get_experiment(args.experiment_id)
        if exp is None:
            print(f"ERROR: Experiment id {args.experiment_id} not found.")
            sys.exit(1)
        experiments = [exp]
    elif args.experiment_name:
        exp = client.get_experiment_by_name(args.experiment_name)
        if exp is None:
            print(f"ERROR: Experiment name '{args.experiment_name}' not found.")
            sys.exit(1)
        experiments = [exp]
    else:
        experiments = client.list_experiments()

    globs = None
    if args.artifact_globs:
        globs = [g.strip() for g in args.artifact_globs.split(",") if g.strip()]

    output_root = os.path.abspath(args.output_dir)
    _ensure_dir(output_root)

    for exp in experiments:
        exp_name = _sanitize_name(exp.name)
        exp_dir = os.path.join(output_root, f"experiment_{exp.experiment_id}_{exp_name}")
        _ensure_dir(exp_dir)

        runs_to_export = []
        if args.run_ids:
            print(f"INFO: Selecting specific runs by ID for experiment '{exp.name}'...")
            run_ids = [rid.strip() for rid in args.run_ids.split(",") if rid.strip()]
            for run_id in run_ids:
                try:
                    run = client.get_run(run_id)
                    if run.info.experiment_id != exp.experiment_id:
                        print(f"WARNING: Run {run_id} does not belong to experiment {exp.name}. Skipping.")
                        continue
                    runs_to_export.append(run)
                except Exception as e:
                    print(f"WARNING: Failed to get run {run_id}: {e}. Skipping.")
            # Also write a summary for the runs we are exporting
            summary_path = os.path.join(exp_dir, "runs_summary.csv")
            _write_runs_summary(summary_path, runs_to_export)

        elif args.last_n:
            print(f"INFO: Selecting latest {args.last_n} runs for experiment '{exp.name}'...")
            runs_to_export = _search_runs_paginated(
                client,
                exp.experiment_id,
                order_by=["attribute.start_time DESC"],
                page_size=args.last_n,
                max_runs=args.last_n,
            )
            summary_path = os.path.join(exp_dir, "runs_summary.csv")
            _write_runs_summary(summary_path, runs_to_export)

        else: # Legacy mode
            print(f"INFO: Using legacy --runs='{args.runs}' mode for experiment '{exp.name}'...")
            fetch_n = args.max_runs
            if args.runs == "latest":
                fetch_n = min(fetch_n, 50)
            elif args.runs == "best":
                fetch_n = min(fetch_n, 5000)

            all_runs = _search_runs_paginated(
                client,
                exp.experiment_id,
                order_by=["attribute.start_time DESC"],
                page_size=args.page_size,
                max_runs=fetch_n,
            )
            if not all_runs:
                print(f"WARNING: No runs found for experiment {exp.name} ({exp.experiment_id}).")
                continue
            
            summary_path = os.path.join(exp_dir, "runs_summary.csv")
            _write_runs_summary(summary_path, all_runs)
            runs_to_export = _select_runs(all_runs, args.runs, args.metric)

        print(f"INFO: Found {len(runs_to_export)} run(s) to export for experiment '{exp.name}'.")
        exported_runs_info = []
        for run in runs_to_export:
            run_name = _sanitize_name(_get_run_name(run))
            run_dir = os.path.join(exp_dir, f"run_{run.info.run_id}_{run_name}")
            print(f"INFO: Exporting run {run.info.run_id} -> {run_dir}")
            export_status = _export_run(client, run, run_dir, args.include_artifacts, globs, args.copy_mode)
            
            # Prepare run manifest entry
            metrics_path = os.path.join(run_dir, "metrics.json")
            predictions_path = os.path.join(run_dir, "predictions_test.csv")
            params_path = os.path.join(run_dir, "params.json")
            tags_path = os.path.join(run_dir, "tags.json")
            
            run_info = {
                "runId": run.info.run_id,
                "runName": run_name,
                "foldIndex": _get_fold_index(run),
                "exportedRunDir": run_dir,
                "metricsPath": metrics_path if os.path.exists(metrics_path) else None,
                "paramsPath": params_path,
                "tagsPath": tags_path,
                "predictionsTestPath": predictions_path if os.path.exists(predictions_path) else None,
                "keyMetrics": run.data.metrics if run.data else {},
                "has_artifacts": bool(export_status.get("has_artifacts")) if export_status else False,
                "artifacts_downloaded": bool(export_status.get("artifacts_downloaded")) if export_status else False,
                "artifact_error": export_status.get("artifact_error") if export_status else None,
            }
            exported_runs_info.append(run_info)

        if args.write_manifest:
            manifest_path = os.path.join(exp_dir, "manifest.json")
            print(f"INFO: Writing manifest to {manifest_path}")
            _write_manifest(manifest_path, exp, exp_dir, exported_runs_info)

        if args.post_analyze:
            print("INFO: Triggering post-export analysis...")
            # We need to find the script relative to this one's location
            script_dir = os.path.dirname(os.path.realpath(__file__))
            post_analyze_script = os.path.join(script_dir, "post_export_analyze.py")
            
            if not os.path.exists(post_analyze_script):
                print(f"WARNING: Post-analysis script not found at '{post_analyze_script}'. Skipping.")
            else:
                cmd = [
                    sys.executable,
                    post_analyze_script,
                    "--export-dir",
                    exp_dir,
                    "--metric",
                    args.post_analyze_metric,
                ]
                print(f"INFO: Running command: {' '.join(cmd)}")
                try:
                    # Use subprocess.run, but don't fail the export if analysis fails
                    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"WARNING: Post-analysis script failed with exit code {result.returncode}.")
                        print("--- STDOUT ---\n", result.stdout)
                        print("--- STDERR ---\n", result.stderr)
                    else:
                        print("INFO: Post-analysis script completed successfully.")
                        print("--- STDOUT ---\n", result.stdout)

                except Exception as e:
                    print(f"WARNING: An error occurred while running the post-analysis script: {e}")

    print(f"INFO: Export complete: {output_root}")



if __name__ == "__main__":
    main()
