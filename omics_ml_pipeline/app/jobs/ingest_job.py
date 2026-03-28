"""
ingest_job.py — Validate that required input files exist before the pipeline runs.
"""

import os
from rich.progress import track
from app.utils.logging_utils import get_logger, console

log = get_logger("ingest_job")


def run(config: dict) -> None:
    log.info("📂 Checking input files...")

    required = [
        config["paths"]["series_matrix"],
        config["paths"]["soft_file"],
    ]

    for path in track(required, description="Ingest", console=console):
        assert os.path.exists(path), f"Missing input file: {path}"
        size_mb = os.path.getsize(path) / 1_000_000
        log.info(f"  ✅ Found: {path}  ({size_mb:.1f} MB)")

    log.info("✅ Ingest check passed.")
