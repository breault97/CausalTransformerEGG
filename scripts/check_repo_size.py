#!/usr/bin/env python3
"""
Check tracked file sizes to keep the repo git-safe.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def _human_mb(num_bytes: int) -> str:
    return f"{_bytes_to_mb(num_bytes):.2f} MB"


def _get_repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: not a git repository.", file=sys.stderr)
        raise RuntimeError("not a git repository")
    return Path(result.stdout.strip())


def _git_ls_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: failed to run git ls-files.", file=sys.stderr)
        raise RuntimeError("git ls-files failed")
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check repository size safety.")
    parser.add_argument(
        "--count-objects",
        action="store_true",
        help="Run 'git count-objects -vH' and display pack size.",
    )
    args = parser.parse_args()

    try:
        repo_root = _get_repo_root()
    except RuntimeError:
        return 1

    try:
        files = _git_ls_files(repo_root)
    except RuntimeError:
        return 1

    sizes: list[tuple[int, str]] = []
    for rel_path in files:
        full_path = repo_root / rel_path
        if not full_path.exists() or full_path.is_symlink():
            continue
        try:
            size = os.path.getsize(full_path)
        except OSError:
            continue
        sizes.append((size, rel_path))

    sizes.sort(reverse=True, key=lambda item: item[0])
    top = sizes[:20]

    print("Top 20 tracked files by size:")
    for size, rel_path in top:
        print(f"  {_human_mb(size)}  {rel_path}")

    warnings = [(size, path) for size, path in sizes if size > 50 * 1024 * 1024]
    errors = [(size, path) for size, path in sizes if size > 90 * 1024 * 1024]

    if warnings:
        print("\nWarnings (>50 MB):")
        for size, rel_path in warnings:
            print(f"  {_human_mb(size)}  {rel_path}")

    if errors:
        print("\nErrors (>90 MB):")
        for size, rel_path in errors:
            print(f"  {_human_mb(size)}  {rel_path}")

    if args.count_objects:
        result = subprocess.run(
            ["git", "count-objects", "-vH"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("\nGit object stats:")
            print(result.stdout.rstrip())
        else:
            print("WARNING: failed to run git count-objects -vH.", file=sys.stderr)

    if errors:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
