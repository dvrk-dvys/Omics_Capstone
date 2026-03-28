"""
preprocess_job.py — Orchestrates preprocess.py with config-driven paths.
"""

import os
from rich.progress import Progress

from app.utils.preprocess import load_parsed, filter_probes, check_normalization
from app.utils.logging_utils import get_logger, console

log = get_logger("preprocess_job")


def run(config: dict) -> None:
    input_path  = config["paths"]["parsed_csv"]
    output_path = config["paths"]["preprocessed_csv"]
    threshold   = config["preprocessing"]["iqr_threshold"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log.info(f"⚙️  Preprocessing: {input_path}")

    with Progress(console=console) as progress:
        task = progress.add_task("Preprocess matrix", total=3)

        df = load_parsed(input_path)
        progress.advance(task)

        df = filter_probes(df, iqr_threshold=threshold)
        progress.advance(task)

        df = check_normalization(df)
        progress.advance(task)

    df.to_csv(output_path)
    log.info(f"💾 Saved preprocessed matrix: {output_path}  shape={df.shape}")
