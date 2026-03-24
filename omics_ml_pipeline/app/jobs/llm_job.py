"""
llm_job.py — LLM-assisted biological interpretation of biomarker candidates.

TODO: Implement in Phase 6.

Designed to accept either:
  - app-produced biomarker shortlist (app/data/biomarker_shortlist.csv)
  - manually produced Weka shortlist (data/femoral_head_necrosis/feature_selection/biomarker_shortlist.csv)

The shortlist_path is configurable so both pipelines can feed into this stage.
"""

from app.utils.logging_utils import get_logger

log = get_logger("llm_job")


def run(config: dict, shortlist_path: str = None) -> None:
    if shortlist_path is None:
        shortlist_path = config["paths"]["biomarker_shortlist"]

    log.info(f"LLM job: shortlist input = {shortlist_path}")
    log.info("TODO: implement Phase 6 — PubMed retrieval + Claude API interpretation")
    log.info("  Step 1: load shortlist CSV")
    log.info("  Step 2: for each gene, query PubMed (gene + 'osteonecrosis OR bone ischemia')")
    log.info("  Step 3: verify PMIDs resolve")
    log.info("  Step 4: build structured prompt (role + gene context + constraints)")
    log.info("  Step 5: call Claude API")
    log.info("  Step 6: write interpretation results to app/data/")
