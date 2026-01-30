#!/usr/bin/env python3
"""
Curate audit-friendly artifacts from mlflow_exports into results_public.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


RUN_ID_PREFIXES = [
    # CT EEGMMIDB, seed=600, folds 0–4
    "1ebc9c06",
    "16521b4c",
    "a8f5bd3a",
    "b7f7e83a",
    "d5599a49",
    # CT EEGMMIDB, seed=700, folds 0–4
    "54c985de",
    "b6694b46",
    "7da77fac",
    "a974048a",
    "07df9328",
    # Validity checks (Table 9): label permutation
    "691ce7bf",
    "ae0afe45",
    "2bd040d2",
    "c6c94f0d",
    "9cea104f",
    # Validity checks: inter-batch-decouple (fold 0)
    "d45bb37f",
    "47eda505",
]


def _bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def _human_mb(num_bytes: int) -> str:
    return f"{_bytes_to_mb(num_bytes):.2f} MB"


def _collect_run_dirs(src_dir: Path) -> dict[str, tuple[str | None, Path]]:
    run_dirs: dict[str, tuple[str | None, Path]] = {}
    pattern = re.compile(r"^run_([0-9a-f]{32})(?:_(.+))?$")
    for entry in src_dir.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if not match:
            continue
        run_id = match.group(1)
        run_name = match.group(2)
        run_dirs[run_id] = (run_name, entry)
    return run_dirs


def _should_include_file(path: Path) -> bool:
    name = path.name.lower()

    if name in {"metrics.json", "params.json"}:
        return True

    if name.endswith(".csv") and "predictions" in name:
        if "window" in name:
            return False
        if "record" in name or "subject" in name:
            return True
        return False

    if name.endswith((".png", ".pdf")):
        if "window" in name:
            return False
        if "record" in name or "subject" in name:
            return True
        return False

    if name.endswith((".txt", ".md", ".json")):
        if "classification_report" in name or "quality_report" in name:
            return True

    return False


def _iter_candidate_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in run_dir.rglob("*"):
        if path.is_file() and _should_include_file(path):
            files.append(path)
    return files


def _copy_file(
    src: Path,
    dst: Path,
    max_bytes: int,
    include_large: bool,
    skipped: list[dict],
    copied: list[dict],
) -> None:
    size = src.stat().st_size
    if size > max_bytes and not include_large:
        skipped.append(
            {
                "path": dst.as_posix(),
                "bytes": size,
                "reason": f"exceeds max_file_mb ({_human_mb(size)})",
            }
        )
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"path": dst.as_posix(), "bytes": size})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Curate audit-friendly MLflow export artifacts into results_public."
    )
    parser.add_argument(
        "--src",
        default="mlflow_exports/experiment_665075068361836799_CT_eegmmidb",
        help="Source MLflow export experiment directory.",
    )
    parser.add_argument(
        "--dst",
        default="results_public/CT_eegmmidb",
        help="Destination directory inside results_public.",
    )
    parser.add_argument(
        "--max_file_mb",
        type=float,
        default=20.0,
        help="Max file size to copy (MB). Larger files are skipped unless --include_large is set.",
    )
    parser.add_argument(
        "--include_large",
        action="store_true",
        help="Allow copying files larger than --max_file_mb.",
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    results_root = dst_dir.parent
    max_bytes = int(args.max_file_mb * 1024 * 1024)

    if not src_dir.exists():
        print(f"ERROR: source directory not found: {src_dir}", file=sys.stderr)
        return 1

    run_dirs = _collect_run_dirs(src_dir)
    if not run_dirs:
        print(f"ERROR: no run directories found in {src_dir}", file=sys.stderr)
        return 1

    results_root.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": src_dir.as_posix(),
        "destination": results_root.as_posix(),
        "max_file_mb": args.max_file_mb,
        "include_large": args.include_large,
        "experiment_files": [],
        "experiment_skipped": [],
        "runs": [],
    }

    copied_all: list[dict] = []
    total_bytes = 0
    total_files = 0

    # Copy experiment-level summary if present.
    experiment_summary = src_dir / "runs_summary.csv"
    if experiment_summary.exists():
        skipped: list[dict] = []
        copied: list[dict] = []
        dst_path = dst_dir / experiment_summary.name
        _copy_file(experiment_summary, dst_path, max_bytes, args.include_large, skipped, copied)
        if copied:
            manifest["experiment_files"].extend(copied)
            copied_all.extend(copied)
            total_files += len(copied)
            total_bytes += sum(item["bytes"] for item in copied)
        if skipped:
            manifest["experiment_skipped"].extend(skipped)

    missing_prefixes: list[str] = []
    ambiguous_prefixes: list[str] = []

    for prefix in RUN_ID_PREFIXES:
        matches = [run_id for run_id in run_dirs if run_id.startswith(prefix)]
        if len(matches) == 0:
            missing_prefixes.append(prefix)
            continue
        if len(matches) > 1:
            ambiguous_prefixes.append(prefix)
            continue

        run_id = matches[0]
        run_name, run_path = run_dirs[run_id]
        run_entry = {
            "run_id": run_id,
            "run_name": run_name,
            "files": [],
            "skipped": [],
            "total_bytes": 0,
        }

        run_dst = dst_dir / run_path.name
        candidates = _iter_candidate_files(run_path)
        for src_path in candidates:
            rel = src_path.relative_to(run_path)
            dst_path = run_dst / rel
            _copy_file(
                src_path,
                dst_path,
                max_bytes,
                args.include_large,
                run_entry["skipped"],
                run_entry["files"],
            )

        run_entry["total_bytes"] = sum(item["bytes"] for item in run_entry["files"])
        if run_entry["files"] or run_entry["skipped"]:
            manifest["runs"].append(run_entry)
            copied_all.extend(run_entry["files"])
            total_files += len(run_entry["files"])
            total_bytes += run_entry["total_bytes"]

    if missing_prefixes:
        print(
            f"WARNING: missing run prefixes: {', '.join(missing_prefixes)}",
            file=sys.stderr,
        )
    if ambiguous_prefixes:
        print(
            f"ERROR: ambiguous run prefixes (multiple matches): {', '.join(ambiguous_prefixes)}",
            file=sys.stderr,
        )
        return 2

    manifest["summary"] = {
        "total_files": total_files,
        "total_bytes": total_bytes,
    }

    # Top 10 largest copied files (by size).
    top_files = sorted(copied_all, key=lambda item: item["bytes"], reverse=True)[:10]
    manifest["summary"]["top_files"] = top_files

    manifest_path = results_root / "MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Copied {total_files} files to {results_root} ({_human_mb(total_bytes)} total).")
    if top_files:
        print("Top 10 largest copied files:")
        for item in top_files:
            print(f"  {_human_mb(item['bytes'])}  {item['path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
