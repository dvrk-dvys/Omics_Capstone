"""
main.py — Omics ML Pipeline entry point.

Runs the full automated pipeline in order:
  1. ingest     — validate input files exist
  2. parse      — extract expression matrix from GEO series matrix
  3. preprocess — IQR filtering of low-variance probes
  4. feature_select — rank probes by fold-change, select top N
  5. train_eval — baseline models + hyperopt (replaces Weka)
  6. biomarker  — generate biomarker shortlist from RF importance + fold-change
  7. llm        — biological interpretation via LLM

Usage:
  cd omics_ml_pipeline
  docker compose up -d
  python -m app.main                              # full pipeline
  python -m app.main --skip-pre                   # skip ingest/parse/preprocess
  python -m app.main --skip-pre --llm             # skip pre + run LLM
  python -m app.main --skip-pre --skip-train --llm  # LLM only
"""

###-----------------------------
#TEMPORARY FIX
#! WHY DOES THREADING NOT WORK AT ALL?
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
###-----------------------------


import os
import argparse
import pathlib
import time
import traceback
import uuid
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.utils.io_utils import load_config
from app.utils.logging_utils import get_logger, log_duration
from app.jobs import (
    ingest_job,
    parse_job,
    preprocess_job,
    feature_select_job,
    train_eval_job,
    biomarker_job,
    llm_job,
    univariate_ann_job,
)

log = get_logger("main")
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "pipeline.yaml")


def _next_run_id(config: dict) -> str:
    """
    Increment a persistent counter and return a zero-padded run prefix.
    Stored at app/data/.run_counter so every pipeline execution gets a unique ID.
    Example: 'r001_', 'r002_', ...
    """
    # Two levels up from app/data/output/models → app/data/
    data_dir = os.path.dirname(os.path.dirname(config["paths"]["model_output_dir"]))
    os.makedirs(data_dir, exist_ok=True)
    counter_path = os.path.join(data_dir, ".run_counter")

    count = 1
    if os.path.exists(counter_path):
        with open(counter_path) as f:
            count = int(f.read().strip()) + 1
    with open(counter_path, "w") as f:
        f.write(str(count))

    return f"r{count:03d}_"

def parse_args():
    parser = argparse.ArgumentParser(description="Omics ML Pipeline")
    parser.add_argument("--config",      default=DEFAULT_CONFIG, help="Path to pipeline.yaml")
    parser.add_argument("--skip-pre",    action="store_true",    help="Skip ingest, parse, and preprocess (use existing outputs)")
    parser.add_argument("--skip-train",  action="store_true",    help="Skip model training")
    parser.add_argument("--llm",         action="store_true",    help="Run LLM job (scaffold)")
    parser.add_argument("--shortlist",   default=None,           help="Override shortlist path for LLM job")
    parser.add_argument(
        "-m", "--mode",
        default="univariate",
        choices=["univariate", "multivariate"],
        help="Feature selection mode: univariate (ANN-based ranking) or multivariate (hybrid statistical ranking)",
    )
    parser.add_argument(
        "--univariate-rerank",
        action="store_true",
        default=False,
        help=(
            "Univariate mode only. When set, rerank the ANN shortlist by "
            "univariate_score = 0.7*z(Median_TestAUC) + 0.3*z(abs_fold_change) "
            "before writing biomarker_shortlist.csv. "
            "Default: OFF (baseline ANN-rank-only behavior)."
        ),
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    # Ensure all output directories exist — safe to run on every startup
    for dir_key in ("feature_select_dir", "model_output_dir", "plots_dir", "llm_outputs_dir"):
        dir_path = config["paths"].get(dir_key)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    for file_key in ("parsed_csv", "preprocessed_csv", "biomarker_shortlist"):
        file_path = config["paths"].get(file_key)
        if file_path:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

    run_id = _next_run_id(config)
    crash_id = uuid.uuid4().hex[:8]
    pipeline_start = time.perf_counter()
    durations = {}

    log.info("=" * 60)
    log.info(f"🧬 Omics ML Pipeline — {config['project'].get('disease', 'Biomarker Discovery')}")
    log.info(f"📄 Config : {args.config}")
    log.info(f"🔖 Run ID : {run_id.rstrip('_')}")
    log.info(f"⚙️  Mode   : {args.mode}  (top_n={config['feature_selection']['top_n_feats']})")
    log.info("=" * 60)

    try:
        if not args.skip_pre:
            # 1. Ingest
            log.info("📂 [1/6] Ingest")
            try:
                with log_duration(log, "Ingest"):
                    t0 = time.perf_counter(); ingest_job.run(config)
                    durations["Ingest"] = time.perf_counter() - t0
            except Exception as e:
                log.error(f"❌ Ingest failed: {e}")
                raise

            # 2. Parse
            log.info("🔍 [2/6] Parse")
            try:
                with log_duration(log, "Parse"):
                    t0 = time.perf_counter(); parse_job.run(config)
                    durations["Parse"] = time.perf_counter() - t0
            except Exception as e:
                log.error(f"❌ Parse failed: {e}")
                raise

            # 3. Preprocess
            log.info("⚙️  [3/6] Preprocess")
            try:
                with log_duration(log, "Preprocess"):
                    t0 = time.perf_counter(); preprocess_job.run(config)
                    durations["Preprocess"] = time.perf_counter() - t0
            except Exception as e:
                log.error(f"❌ Preprocess failed: {e}")
                raise
        else:
            log.info("⏭️  Skipping ingest/parse/preprocess (--skip-pre)")

        if args.skip_train and args.shortlist:
            # Steps 4–6 skipped: external shortlist provided and training disabled.
            # LLM job will consume args.shortlist directly.
            log.info("⏭️  Skipping feature selection, ANN, and biomarker shortlist (--skip-train + --shortlist)")
            selected_df = fc_ranking = gene_map = None
        else:
            # 4. Feature selection — runs for fc_ranking, gene_map, EDA plots, and biomarker shortlist.
            #    In multivariate mode selected_df is used downstream.
            #    In univariate mode selected_df is replaced after the ANN step.
            log.info("📊 [4/7] Feature selection  (rankings + EDA)")
            try:
                with log_duration(log, "Feature selection"):
                    t0 = time.perf_counter()
                    selected_df, fc_ranking, gene_map = feature_select_job.run(config)
                    durations["Feature selection"] = time.perf_counter() - t0
            except Exception as e:
                log.error(f"❌ Feature selection failed: {e}")
                raise

            # 5. Mode-specific feature source
            top_n = config["feature_selection"]["top_n_feats"]
            if args.mode == "univariate":
                log.info("🧬 [5/7] Univariate ANN  (mode=univariate)")
                try:
                    with log_duration(log, "Univariate ANN"):
                        t0 = time.perf_counter()
                        univariate_ann_job.run(config, run_id=run_id)
                        durations["Univariate ANN"] = time.perf_counter() - t0
                    uni_csv = os.path.join(
                        config["paths"]["univariate_ann_dir"],
                        f"top{top_n}_features_univariate_ann.csv",
                    )
                    selected_df = pd.read_csv(uni_csv, index_col="sample")
                    log.info(
                        f"  selected_df → ANN top-{top_n}  "
                        f"({selected_df.shape[0]} samples × {selected_df.shape[1] - 1} probes)"
                    )
                    log.info(f"  Source : {uni_csv}")
                except Exception as e:
                    log.error(f"❌ Univariate ANN failed — falling back to hybrid selected_df: {e}")
            else:
                log.info("⏭️  [5/7] Univariate ANN skipped  (mode=multivariate)")
                log.info(
                    f"  selected_df → hybrid top-{top_n}  "
                    f"({selected_df.shape[0]} samples × {selected_df.shape[1] - 1} probes)"
                )

            # Inject mode + top_n into config so downstream plot functions can label outputs
            config["_mode"]             = args.mode
            config["_top_n"]            = top_n
            config["_univariate_rerank"] = args.univariate_rerank

            # 6. Biomarker shortlist — runs before train so composite plot has the CSV
            log.info("🎯 [6/7] Biomarker shortlist")
            try:
                with log_duration(log, "Biomarker shortlist"):
                    t0 = time.perf_counter()
                    shortlist = biomarker_job.run(config, selected_df, fc_ranking, gene_map, run_id)
                    durations["Biomarker shortlist"] = time.perf_counter() - t0
            except Exception as e:
                log.error(f"❌ Biomarker shortlist failed: {e}")
                raise

        # 7. Train + evaluate (non-critical — failure does not abort pipeline)
        if not args.skip_train:
            log.info("🤖 [7/7] Train + evaluate")
            try:
                with log_duration(log, "Train + evaluate"):
                    t0 = time.perf_counter()
                    comparison_df = train_eval_job.run(config, selected_df, run_id, gene_map=gene_map)
                    durations["Train + evaluate"] = time.perf_counter() - t0
            except Exception as e:
                log.error(f"❌ Train + evaluate failed (continuing): {e}")
        else:
            log.info("⏭️  Skipping training (--skip-train)")

        # LLM (opt-in, non-critical — failure does not abort pipeline)
        if args.llm:
            log.info("🧠 [+LLM] Biological interpretation")
            try:
                with log_duration(log, "LLM interpretation"):
                    t0 = time.perf_counter()
                    llm_job.run(config, biomarker_path=args.shortlist)
                    durations["LLM interpretation"] = time.perf_counter() - t0
            except Exception as e:
                log.error(f"❌ LLM interpretation failed (continuing): {e}")

        total_elapsed = time.perf_counter() - pipeline_start
        durations["TOTAL"] = total_elapsed
        total_str = f"{total_elapsed / 60:.1f} min" if total_elapsed >= 60 else f"{total_elapsed:.1f}s"

        # Duration bar chart
        _plot_pipeline_duration(durations, config["paths"]["plots_dir"], run_id,
                                mode_suffix=f"_{args.mode}_top{top_n}")

        log.info("=" * 60)
        log.info(f"🏁 Pipeline complete  [{total_str}]  mode={args.mode}  top_n={top_n}")
        log.info(f"  📋 Biomarker shortlist : {config['paths']['biomarker_shortlist']}")
        log.info(f"  📈 Model comparison    : {config['paths']['model_comparison']}")
        log.info(f"  🌐 MLflow UI           : {config['mlflow']['tracking_uri']}")
        log.info("=" * 60)

    except Exception as exc:
        tb = traceback.format_exc()
        crash_path = pathlib.Path("crash.log")
        with open(crash_path, "w") as f:
            f.write(f"run_id   : {run_id.rstrip('_')}\n")
            f.write(f"crash_id : {crash_id}\n")
            f.write(f"time     : {datetime.now().isoformat()}\n")
            f.write(f"error    : {exc}\n\n")
            f.write(tb)
        log.error(f"💥 Unhandled crash — see {crash_path.resolve()}  (crash_id={crash_id})")
        raise


def _plot_pipeline_duration(durations: dict, plots_dir: str, run_id: str = "", mode_suffix: str = "") -> None:
    """Horizontal bar chart of per-job and total pipeline duration."""
    os.makedirs(plots_dir, exist_ok=True)

    # Separate total from per-job bars
    jobs   = {k: v for k, v in durations.items() if k != "TOTAL"}
    total  = durations.get("TOTAL", sum(jobs.values()))

    labels = list(jobs.keys())
    values = [v / 60 for v in jobs.values()]   # convert to minutes

    colours = ["#2196F3" if v * 60 < 60 else "#FF5722" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.55 + 1.5)))
    bars = ax.barh(labels[::-1], values[::-1], color=colours[::-1],
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values[::-1]):
        label = f"{val * 60:.1f}s" if val < 1 else f"{val:.1f} min"
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)

    total_str = f"{total / 60:.1f} min" if total >= 60 else f"{total:.1f}s"
    ax.set_xlabel("Duration (minutes)", fontsize=10)
    ax.set_title(
        f"Pipeline Job Duration — {run_id.rstrip('_') if run_id else 'latest'}\n"
        f"Total: {total_str}",
        fontsize=11
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()

    out_path = os.path.join(plots_dir, f"pipeline_duration{mode_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Duration plot saved: {out_path}")


if __name__ == "__main__":
    main()
