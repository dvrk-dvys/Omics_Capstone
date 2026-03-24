"""
preprocess_job.py — Orchestrates preprocess.py with config-driven paths.
"""

import os

from app.utils.preprocess import load_parsed, filter_probes, check_normalization
from app.utils.logging_utils import get_logger

log = get_logger("preprocess_job")


def run(config: dict) -> None:
    input_path  = config["paths"]["parsed_csv"]
    output_path = config["paths"]["preprocessed_csv"]
    threshold   = config["preprocessing"]["iqr_threshold"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log.info(f"Preprocessing: {input_path}")

    df = load_parsed(input_path)
    df = filter_probes(df, iqr_threshold=threshold)
    df = check_normalization(df)

    df.to_csv(output_path)
    log.info(f"Saved preprocessed matrix: {output_path}  shape={df.shape}")
