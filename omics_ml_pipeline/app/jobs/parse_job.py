"""
parse_job.py — Orchestrates parse_series_matrix.py with config-driven paths.

Imports the parsing functions directly and calls them with app paths,
bypassing the hardcoded defaults in the standalone script.
"""

import os
from rich.progress import Progress

from app.utils.logging_utils import get_logger, console
from app.utils.parse_series_matrix import (
    parse_header,
    parse_data_matrix,
    assign_classes,
    build_sample_matrix,
)
log = get_logger("parse_job")


def run(config: dict) -> None:
    input_path  = config["paths"]["series_matrix"]
    output_path = config["paths"]["parsed_csv"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log.info(f"🔍 Parsing: {input_path}")

    project       = config.get("project", {})
    disease_label = project.get("disease_label", "SONFH")
    control_label = project.get("control_label", "control")

    with Progress(console=console) as progress:
        task = progress.add_task("Parse GEO series matrix", total=4)

        gsm_ids, titles, disease_vals = parse_header(input_path)
        progress.advance(task)

        data_df = parse_data_matrix(input_path)
        progress.advance(task)

        class_map = assign_classes(gsm_ids, disease_vals, disease_label=disease_label, control_label=control_label)
        progress.advance(task)

        df = build_sample_matrix(data_df, class_map)
        progress.advance(task)

    df.to_csv(output_path)
    log.info(f"💾 Saved parsed matrix: {output_path}  shape={df.shape}")
