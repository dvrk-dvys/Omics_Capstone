"""
main.py — Omics ML Pipeline entry point.

Runs the full automated pipeline in order:
  1. ingest     — validate input files exist
  2. parse      — extract expression matrix from GEO series matrix
  3. preprocess — IQR filtering of low-variance probes
  4. feature_select — rank probes by fold-change, select top N
  5. train_eval — baseline models + hyperopt (replaces Weka)
  6. biomarker  — generate biomarker shortlist from RF importance + fold-change
  7. llm        — (scaffold) biological interpretation via LLM

Usage:
  cd omics_ml_pipeline
  python -m app.main
  python -m app.main --config app/config/pipeline.yaml
  python -m app.main --skip-parse   (if parsed/preprocessed data already exists)
"""

import os
import argparse

from app.utils.io_utils import load_config


def _next_run_id(config: dict) -> str:
    """
    Increment a persistent counter and return a zero-padded run prefix.
    Stored at app/data/.run_counter so every pipeline execution gets a unique ID.
    Example: 'r001_', 'r002_', ...
    """
    data_dir = os.path.dirname(config["paths"]["model_output_dir"])
    os.makedirs(data_dir, exist_ok=True)
    counter_path = os.path.join(data_dir, ".run_counter")

    count = 1
    if os.path.exists(counter_path):
        with open(counter_path) as f:
            count = int(f.read().strip()) + 1
    with open(counter_path, "w") as f:
        f.write(str(count))

    return f"r{count:03d}_"
from app.utils.logging_utils import get_logger
from app.jobs import (
    ingest_job,
    parse_job,
    preprocess_job,
    feature_select_job,
    train_eval_job,
    biomarker_job,
    llm_job,
)

log = get_logger("main")

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "pipeline.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Omics ML Pipeline")
    parser.add_argument("--config",      default=DEFAULT_CONFIG, help="Path to pipeline.yaml")
    parser.add_argument("--skip-parse",  action="store_true",    help="Skip parse + preprocess if outputs already exist")
    parser.add_argument("--skip-train",  action="store_true",    help="Skip model training")
    parser.add_argument("--llm",         action="store_true",    help="Run LLM job (scaffold)")
    parser.add_argument("--shortlist",   default=None,           help="Override shortlist path for LLM job")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    run_id = _next_run_id(config)

    log.info("=" * 60)
    log.info("Omics ML Pipeline — SONFH Biomarker Discovery")
    log.info(f"Config: {args.config}")
    log.info(f"Run ID: {run_id.rstrip('_')}")
    log.info("=" * 60)

    # 1. Ingest
    ingest_job.run(config)

    # 2. Parse + Preprocess (skippable if outputs already exist)
    if not args.skip_parse:
        parse_job.run(config)
        preprocess_job.run(config)
    else:
        log.info("Skipping parse/preprocess (--skip-parse)")

    # 3. Feature selection
    selected_df, fc_ranking, gene_map = feature_select_job.run(config)

    # 4. Train + evaluate
    if not args.skip_train:
        comparison_df = train_eval_job.run(config, selected_df, run_id)
    else:
        log.info("Skipping training (--skip-train)")

    # 5. Biomarker shortlist
    shortlist = biomarker_job.run(config, selected_df, fc_ranking, gene_map, run_id)

    # 6. LLM (optional)
    if args.llm:
        llm_job.run(config, shortlist_path=args.shortlist)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info(f"  Biomarker shortlist : {config['paths']['biomarker_shortlist']}")
    log.info(f"  Model comparison    : {config['paths']['model_comparison']}")
    log.info(f"  MLflow UI           : {config['mlflow']['tracking_uri']}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
