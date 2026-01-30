#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
from datetime import datetime

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


def _find_script(script_name: str) -> str:
    """Finds a script in common locations relative to this script."""
    this_dir = pathlib.Path(__file__).parent.resolve()
    
    # Check in current dir (scripts/)
    path = this_dir / script_name
    if path.is_file():
        return str(path)
        
    # Check in parent dir (project root)
    path = this_dir.parent / script_name
    if path.is_file():
        return str(path)
        
    # Check in runnable/
    path = this_dir.parent / "runnables" / script_name
    if path.is_file():
        return str(path)

    return None


def _run_subprocess(cmd: list, cwd: str = None):
    """Runs a command, captures output, and returns success status."""
    print(f"INFO: Running command: {' '.join(map(shlex.quote, cmd))}")
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode != 0:
            print(f"WARNING: Subprocess failed with exit code {result.returncode}.")
            print("--- STDOUT ---\\n", result.stdout)
            print("--- STDERR ---\\n", result.stderr)
            return False, result.stdout, result.stderr
        print("--- STDOUT ---\\n", result.stdout)
        return True, result.stdout, result.stderr
    except FileNotFoundError:
        print(f"ERROR: Command not found: {cmd[0]}. Make sure it is in your PATH.", file=sys.stderr)
        return False, "", "Command not found"
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}", file=sys.stderr)
        return False, "", str(e)


def _load_manifest(export_dir: str):
    """Loads the manifest.json file from the export directory."""
    manifest_path = os.path.join(export_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _scan_for_runs(export_dir: str):
    """Scans the export directory to build a run list if manifest is missing."""
    print("INFO: manifest.json not found. Scanning directory for runs...")
    runs = []
    for item in os.listdir(export_dir):
        if item.startswith("run_"):
            run_dir = os.path.join(export_dir, item)
            if os.path.isdir(run_dir):
                run_id_match = item.split('_')[1]
                run_info = {
                    "runId": run_id_match,
                    "runName": '_'.join(item.split('_')[2:]),
                    "exportedRunDir": run_dir,
                    "metricsPath": os.path.join(run_dir, "metrics.json"),
                    "keyMetrics": {},
                }
                # Try to load key metrics
                if os.path.exists(run_info["metricsPath"]):
                    with open(run_info["metricsPath"], "r") as f:
                        run_info["keyMetrics"] = json.load(f)
                runs.append(run_info)
    return {"runs": runs}


def main():
    parser = argparse.ArgumentParser(description="Post-export analysis of MLflow runs.")
    parser.add_argument("--export-dir", type=str, required=True, help="Path to the MLflow export directory (e.g., 'mlflow_exports/experiment_...')")
    parser.add_argument("--metric", type=str, default="multi_test_balanced_acc_subject", help="Metric to use for ranking best/worst runs.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of best/worst runs to compare.")
    parser.add_argument("--skip-existing", type=_parse_bool, default=True, help="Skip analysis if 'analysis_report.md' already exists.")
    parser.add_argument("--dry-run", type=_parse_bool, default=False, help="Print what would be done without executing.")
    args = parser.parse_args()

    # --- 1. Load manifest or scan directory ---
    manifest = _load_manifest(args.export_dir)
    if manifest is None:
        manifest = _scan_for_runs(args.export_dir)
    
    if not manifest.get("runs"):
        print("ERROR: No runs found in the export directory.", file=sys.stderr)
        sys.exit(1)

    print(f"INFO: Found {len(manifest['runs'])} runs to process.")

    # --- 2. Analyze each run ---
    analyze_script_path = _find_script("analyze_predictions.py")
    if not analyze_script_path:
        print("WARNING: 'analyze_predictions.py' script not found. Skipping individual run analysis.", file=sys.stderr)

    analysis_failures = []
    for run in manifest["runs"]:
        run_dir = run["exportedRunDir"]
        report_path = os.path.join(run_dir, "artifacts", "analysis_report.md") # As per analyze_predictions.py

        if args.skip_existing and os.path.exists(report_path):
            print(f"INFO: Analysis report already exists for run {run['runId']}. Skipping.")
            continue

        if not analyze_script_path:
            continue
            
        # Check for prediction files
        preds_path = run.get("predictionsTestPath") or os.path.join(run_dir, "predictions_test.csv")
        if not os.path.exists(preds_path):
            print(f"INFO: No predictions file, skipping analysis for run {run['runId']}.")
            continue

        print(f"--- Analyzing Run: {run['runId']} ---")
        cmd = [sys.executable, analyze_script_path, run_dir]

        if args.dry_run:
            print(f"DRY-RUN: Would execute: {' '.join(cmd)}")
        else:
            success, _, _ = _run_subprocess(cmd)
            if not success:
                analysis_failures.append(run['runId'])

    # --- 3. Find best and worst runs ---
    scored_runs = []
    for run in manifest["runs"]:
        metric_val = run.get("keyMetrics", {}).get(args.metric)
        if metric_val is not None:
            scored_runs.append((metric_val, run))
    
    scored_runs.sort(key=lambda x: x[0], reverse=True)

    best_runs = scored_runs[:args.top_k]
    worst_runs = scored_runs[-args.top_k:]

    print("\n--- Ranking ---")
    if not scored_runs:
        print(f"WARNING: No runs found with the specified metric '{args.metric}'. Cannot determine best/worst runs.")
    else:
        print(f"Metric for ranking: {args.metric}")
        if best_runs:
             print(f"Best Run ({args.metric}={best_runs[0][0]:.4f}): {best_runs[0][1]['runId']}")
        if worst_runs:
             print(f"Worst Run ({args.metric}={worst_runs[0][0]:.4f}): {worst_runs[0][1]['runId']}")

    # --- 4. Compare best and worst run ---
    compare_script_path = _find_script("compare_runs.py")
    if not compare_script_path:
        print("WARNING: 'compare_runs.py' not found. Skipping comparison.", file=sys.stderr)
    elif len(scored_runs) < 2:
        print("INFO: Fewer than 2 scored runs available. Skipping comparison.")
    else:
        best_run_dir = best_runs[0][1]["exportedRunDir"]
        worst_run_dir = worst_runs[0][1]["exportedRunDir"]
        output_path = os.path.join(args.export_dir, "compare_report_best_vs_worst.md")
        
        print("\n--- Comparing Best vs Worst ---")
        cmd = [sys.executable, compare_script_path, best_run_dir, worst_run_dir, "--output", output_path]
        if args.dry_run:
            print(f"DRY-RUN: Would execute: {' '.join(cmd)}")
        else:
            _run_subprocess(cmd)

    # --- 5. Generate summary report ---
    summary_path = os.path.join(args.export_dir, "summary_report.md")
    print(f"\n--- Generating Summary Report ---")

    report_lines = [
        f"# Analysis Summary Report",
        f"- **Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Ranking Metric:** `{args.metric}`",
        f"- **Total Runs Processed:** {len(manifest['runs'])}",
        f"- **Analysis Failures:** {len(analysis_failures)}",
    ]

    # Stats table
    metric_values = [m for m, r in scored_runs]
    if metric_values:
        report_lines.extend([
            "\n## Metric Statistics",
            f"- **Mean:** {statistics.mean(metric_values):.4f}",
            f"- **Std Dev:** {statistics.stdev(metric_values) if len(metric_values) > 1 else 0:.4f}",
            f"- **Min:** {min(metric_values):.4f} (Run: {worst_runs[0][1]['runId']})",
            f"- **Max:** {max(metric_values):.4f} (Run: {best_runs[0][1]['runId']})",
        ])

    # Runs table
    report_lines.append("\n## Run Details")
    report_lines.append("| Run Name | Fold | Metric Value | Notes |")
    report_lines.append("|---|---|---|---|")

    for run in manifest["runs"]:
        metric_val = run.get("keyMetrics", {}).get(args.metric)
        metric_str = f"{metric_val:.4f}" if metric_val is not None else "N/A"
        fold_str = str(run.get("foldIndex")) if run.get("foldIndex") is not None else "N/A"
        notes = "Analysis Failed" if run["runId"] in analysis_failures else ""
        report_lines.append(f"| {run['runName']} | {fold_str} | {metric_str} | {notes} |")

    if args.dry_run:
        print("DRY-RUN: Would write the following content to summary_report.md:")
        print("\n".join(report_lines))
    else:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"INFO: Summary report saved to {summary_path}")


if __name__ == "__main__":
    main()
