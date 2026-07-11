#!/usr/bin/env python3
"""Export a small health snapshot of the latest registered model.

Reads the most recent entry of ``artifacts/registry/`` (the same registry the
training pipeline writes to) and combines it with the test results passed in
from the caller to produce ``health/latest.json``. That file is what the
website reads to show that the model is checked on a schedule, without a
human involved.

Standard library only, on purpose: this script has to run in a minimal CI
job that only installed the project's own dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "artifacts" / "registry"
HEALTH_PATH = ROOT / "health" / "latest.json"


def find_latest_entry(registry_dir: Path) -> Path:
    """Return the directory of the most recently registered model.

    Prefers ``registry_dir/latest.txt`` (written by the training pipeline,
    see ``credit_scoring.models.registry``). Falls back to the
    lexicographically-last subdirectory that actually contains a
    ``model.joblib``, since the registry's version strings are sortable
    timestamps (``YYYYMMDD-HHMMSS``).
    """
    latest_pointer = registry_dir / "latest.txt"
    if latest_pointer.is_file():
        version = latest_pointer.read_text(encoding="utf-8").strip()
        candidate = registry_dir / version
        if candidate.is_dir() and (candidate / "model.joblib").is_file():
            return candidate

    candidates = sorted(
        entry
        for entry in registry_dir.iterdir()
        if entry.is_dir() and (entry / "model.joblib").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"no registered model found under {registry_dir}")
    return candidates[-1]


def load_metadata(entry_dir: Path) -> dict:
    metadata_path = entry_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata.json in {entry_dir}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_metrics(metadata: dict) -> dict:
    """Flatten the bits of the metadata that are worth showing publicly.

    The registry stores ranking metrics under ``metrics`` and the chosen
    decision threshold at the top level; both are folded into a single
    ``metrics`` object here, using whatever the metadata actually has
    rather than a hardcoded list.
    """
    metrics = dict(metadata.get("metrics", {}))
    if "threshold" in metadata:
        metrics["threshold"] = metadata["threshold"]
    return metrics


def build_snapshot(entry_dir: Path, metadata: dict, tests_passed: int, tests_total: int) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": metadata.get("version", entry_dir.name),
        "metrics": build_metrics(metadata),
        "tests": {
            "passed": tests_passed,
            "total": tests_total,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests-passed",
        type=int,
        required=True,
        help="Number of tests that passed in the last full test run.",
    )
    parser.add_argument(
        "--tests-total",
        type=int,
        required=True,
        help="Total number of tests collected in the last full test run.",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=REGISTRY_DIR,
        help="Override the model registry location (mainly for testing).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HEALTH_PATH,
        help="Where to write the health snapshot.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    entry_dir = find_latest_entry(args.registry_dir)
    metadata = load_metadata(entry_dir)
    snapshot = build_snapshot(entry_dir, metadata, args.tests_passed, args.tests_total)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
