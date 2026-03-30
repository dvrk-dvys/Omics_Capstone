# omics_ml_pipeline

Python ML pipeline for SONFH biomarker discovery — parallel to the manual Weka capstone workflow.

Turns a classroom no-code Weka exercise into a reproducible, experiment-tracked Python ML pipeline.

---

## What this is

The capstone course uses Weka (GUI-based) for classification. This app replicates and extends that workflow in Python:

| Capstone (manual) | This app |
|---|---|
| Weka GUI | scikit-learn + XGBoost |
| Manual classifier runs | Automated pipeline via `app/main.py` |
| No experiment tracking | MLflow with full web UI |
| Single 10-fold CV run per model | RepeatedStratifiedKFold (50 evaluations per model) |
| No hyperparameter tuning | Hyperopt TPE search (50 trials × 2 models) |
| Manual biomarker shortlist | Auto-generated `biomarker_shortlist.csv` |
| No EDA plots from pipeline | 7 EDA plots generated automatically (6 individual + composite multi-panel) |

Both pipelines share the same preprocessing logic (`app/utils/`). The automated pipeline feeds the same LLM interpretation stage (Phase 6).

---

## Feature selection method (updated)

Probes are now ranked by a **hybrid score** instead of fold change alone:

```
hybrid_score = zscore(|fold change|) + zscore(|Welch t-statistic|)
```

Both components are z-scored before summing so neither dominates. The IQR variance filter threshold was also loosened from `IQR < 0.5` to `IQR < 0.2` to retain moderate-effect probes earlier in the pipeline.

The absolute fold change column is still included in all outputs for biological interpretability.

---

## Structure

```
omics_ml_pipeline/
├── app/
│   ├── main.py               ← pipeline entry point
│   ├── config/
│   │   └── pipeline.yaml     ← all paths, params, MLflow config
│   ├── data/
│   │   ├── input/            ← raw dataset files (drop any GEO dataset here)
│   │   │   ├── GSE123568_series_matrix.txt.gz
│   │   │   ├── GSE123568_family.soft.gz
│   │   │   └── GSE123568_abstract.txt
│   │   └── output/           ← all generated files (safe to delete & regenerate)
│   │       ├── parsed/
│   │       ├── feature_selection/
│   │       │   ├── top100_features.csv      ← Weka-comparable branch (train/eval default)
│   │       │   ├── top500_features.csv      ← broader discovery branch
│   │       │   ├── gene_rankings.csv        ← all probes ranked: hybrid_score, FC, t-stat, p-value, IQR
│   │       │   ├── gene_level_summary.csv   ← selected probes grouped by gene symbol
│   │       │   ├── gene_level_rankings.csv  ← one best probe per gene (deduped, all genes)
│   │       │   └── top100_genes.csv         ← top 100 genes from gene_level_rankings.csv
│   │       ├── models/
│   │       │   └── model_comparison.csv
│   │       ├── plots/
│   │       │   ├── volcano_plot.png
│   │       │   ├── fold_change_top20.png
│   │       │   ├── boxplots_top6.png
│   │       │   ├── pca_plot.png
│   │       │   ├── sample_correlation.png
│   │       │   ├── heatmap_top20.png
│   │       │   └── eda_composite.png     ← 2×3 multi-panel composite (report Figure 1)
│   │       ├── llm_outputs/  ← one JSON per gene (Phase 6)
│   │       └── biomarker_shortlist.csv
│   ├── jobs/                 ← orchestration layer
│   │   ├── ingest_job.py
│   │   ├── parse_job.py
│   │   ├── preprocess_job.py
│   │   ├── feature_select_job.py
│   │   ├── train_eval_job.py
│   │   ├── biomarker_job.py
│   │   └── llm_job.py        ← scaffold (Phase 6)
│   ├── models/
│   │   └── baseline_models.py
│   └── utils/                ← shared logic (also used by manual Weka workflow)
│       ├── parse_series_matrix.py
│       ├── preprocess.py
│       ├── feature_select.py
│       ├── io_utils.py
│       ├── mlflow_utils.py
│       └── logging_utils.py
├── docker-compose.yml        ← MLflow service
├── Dockerfile
└── requirements.txt
```

---

## Quickstart

**All commands run from `omics_ml_pipeline/`.**

### 1. Start MLflow (required before running the pipeline)

```bash
docker compose up -d
```

MLflow UI → http://localhost:5002

Stop when done:
```bash
docker compose down
```

### 2. Create environment and install dependencies

```bash
conda create -n omics python=3.13
conda activate omics
pip install -r requirements.txt
```

### 3. Run the full pipeline — from scratch

Run this once to parse, preprocess, select features, and train:

```bash
# Full pipeline (ingest → parse → preprocess → feature_select → train → biomarker)
python -m app.main
```

This runs on **top 100 probes** (the default, matching the Weka branch).

It produces all outputs including `top500_features.csv` and `gene_level_rankings.csv` automatically — no extra step needed.

---

### 4. Re-run feature selection + training only (skip parse/preprocess)

Use `--skip-pre` on every subsequent run — parse and preprocess only need to run once:

```bash
# Re-run feature selection, train, biomarker (use existing preprocessed matrix)
python -m app.main --skip-pre
```

---

### 5. Run A vs Run B — controlled comparison

#### Run A — top 100 (default, Weka-comparable)

This is already the default config. Nothing to change:

```bash
python -m app.main --skip-pre
```

- Trains on: `top100_features.csv` (100 probes × 40 samples)
- Comparable to the Weka top-100 branch
- Results go to: `app/data/output/models/model_comparison.csv`

---

#### Run B — top 500 (discovery branch)

Edit `app/config/pipeline.yaml` and change two lines:

```yaml
feature_selection:
  top_n: 500

paths:
  top_features_csv: app/data/output/feature_selection/top500_features.csv
```

Then run:

```bash
python -m app.main --skip-pre
```

- Trains on: `top500_features.csv` (500 probes × 40 samples)
- More features = better biomarker recovery, potentially different model behaviour
- Compare results to Run A in `model_comparison.csv`

**To switch back to Run A afterward**, restore the original values:
```yaml
feature_selection:
  top_n: 100

paths:
  top_features_csv: app/data/output/feature_selection/top100_features.csv
```

---

### 6. Full pipeline + LLM interpretation

```bash
export OPENAI_API_KEY=sk-...
python -m app.main --skip-pre --llm
```

### 7. Running long jobs (Mac local)

```bash
export OPENAI_API_KEY=sk-...
caffeinate -i python -u -m app.main --skip-pre --llm 2>&1 | tee llm_run_$(date +%Y%m%d_%H%M%S).log
```

| Flag | Effect |
|---|---|
| `caffeinate -i` | Prevents the Mac from sleeping |
| `python -u` | Unbuffered stdout — logs appear immediately |
| `2>&1 \| tee ...` | Shows logs live and saves to a timestamped file |

Monitor:
```bash
tail -f llm_run_*.log
grep -a "\[ITER\|\[START\|\[DONE\|\[FAIL\|\[WRITE\|\[AGENT" llm_run_*.log
```

---

### All flags

| Flag | Effect |
|---|---|
| *(none)* | Full pipeline from scratch |
| `--skip-pre` | Skip ingest, parse, preprocess (use existing outputs) |
| `--skip-train` | Skip model training |
| `--llm` | Run LLM biological interpretation (opt-in) |

---

### 8. View results

- MLflow UI: http://localhost:5002 → experiment `sonfh_classification`
- Model comparison CSV: `app/data/output/models/model_comparison.csv`
- Biomarker shortlist: `app/data/output/biomarker_shortlist.csv`
- Gene-level deduped rankings: `app/data/output/feature_selection/gene_level_rankings.csv`
- Top 100 genes: `app/data/output/feature_selection/top100_genes.csv`
- EDA plots: `app/data/output/plots/` (7 plots: volcano, FC bar, boxplots, PCA, sample correlation, heatmap, eda_composite)

---

## Outputs

| File | Description |
|---|---|
| `feature_selection/top100_features.csv` | Top 100 probes by hybrid score + class column — default train/eval input |
| `feature_selection/top500_features.csv` | Top 500 probes by hybrid score + class column — discovery branch |
| `feature_selection/gene_rankings.csv` | All probes ranked by hybrid score; includes abs_fold_change, t_stat, p_value, IQR |
| `feature_selection/gene_level_summary.csv` | Selected probes grouped by gene symbol (direction consistency, probe count) |
| `feature_selection/gene_level_rankings.csv` | One best probe per gene (deduped); all genes ranked by hybrid score |
| `feature_selection/top100_genes.csv` | Top 100 rows of gene_level_rankings.csv — for literature comparison |
| `models/model_comparison.csv` | AUC / F1 / balanced-acc for all baseline + tuned models |
| `biomarker_shortlist.csv` | Top candidates ranked by combined RF importance + fold-change score |
| `llm_outputs/*.json` | Per-gene LLM interpretation (Phase 6) |
| `plots/volcano_plot.png` | Volcano plot — all filtered probes, top N highlighted |
| `plots/fold_change_top20.png` | Top 20 probes by absolute fold change |
| `plots/boxplots_top6.png` | Box plots — top 6 probes, SONFH vs control |
| `plots/pca_plot.png` | PCA — PC1/PC2 variance explained |
| `plots/sample_correlation.png` | 40×40 sample Pearson correlation heatmap |
| `plots/heatmap_top20.png` | Expression heatmap — top 20 probes × 40 samples |
| `plots/eda_composite.png` | 2×3 multi-panel composite — **report Figure 1** |
| MLflow UI | All run params, metrics, tags, and artifacts |

---

## Models

| Model | Stage | Notes |
|---|---|---|
| Logistic (Elastic Net) | Baseline | L1+L2 penalty, class_weight=balanced |
| RandomForest | Baseline + Hyperopt | Feature importance for biomarker ranking |
| LinearSVC | Baseline | class_weight=balanced |
| GaussianNB | Baseline | Probabilistic baseline |
| KNN (k=5) | Baseline | Instance-based |
| MLP | Baseline | 2-layer neural net (64→32) |
| XGBoost | Baseline + Hyperopt | scale_pos_weight=3 for class imbalance |

CV: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` = 50 evaluations per model.

MLflow run taxonomy (filterable by tag in the UI):

| Tag `stage` | Tag `run_kind` | What it is |
|---|---|---|
| `baseline` | `evaluation` | Baseline model result — directly reportable |
| `tuned` | `evaluation` | Tuned model result — directly reportable |
| `hyperopt_search` | `search` | Parent run for a hyperopt search |
| `hyperopt_trial` | `search` | Nested child trial run |
| `biomarker` | `artifact_generation` | Biomarker shortlist generation |

Each pipeline execution is prefixed with a run ID (`r001_`, `r002_`, ...) so all runs from the same execution are visually grouped in the MLflow UI.

---

## Manual utils scripts

The shared preprocessing scripts in `app/utils/` can still be run standalone. These produce the Weka-compatible ARFF output and all EDA plots.

**Run from the repo root (`Omics_Capstone/`):**

```bash
python omics_ml_pipeline/app/utils/parse_series_matrix.py
python omics_ml_pipeline/app/utils/preprocess.py
python omics_ml_pipeline/app/utils/feature_select.py
```

`feature_select.py` accepts an `--outdir` argument (defaults to `data/femoral_head_necrosis/`). It derives two subdirectories from that root automatically:
- `{outdir}/EDA/` — 7 EDA plots including `eda_composite.png`
- `{outdir}/feature_selection/` — CSV/ARFF outputs

```bash
# Custom output root (optional)
python omics_ml_pipeline/app/utils/feature_select.py --outdir path/to/output_root
```

**After both pipelines have run, generate the model comparison figures:**

```bash
# From repo root — reads Weka .txt results + Python model_comparison.csv
python generate_report_figures.py
```

Outputs to `report/figures/`:
- `fig_weka_model_comparison.png` — report Figure 2
- `fig_python_model_comparison.png` — report Figure 3
- `weka_model_comparison.csv` — Weka results in same schema as Python's model_comparison.csv

---

## LLM Integration (Phase 6)

Triggered with `--llm` flag. Wired into `main.py` — runs after the biomarker shortlist is generated.

```bash
# Set API key (once per session)
export OPENAI_API_KEY=sk-...

# Run full pipeline including LLM step
python -m app.main --skip-pre --llm
```

**Architecture:** PubMed retrieval → semantic ranking → constrained GPT-4o prompt → structured output → human validation → Phase 7 report input.

| Step | What it does |
|---|---|
| 1 | Load `biomarker_shortlist.csv`, deduplicate to ~14 unique genes |
| 2 | For each gene, query PubMed: `"{gene} osteonecrosis OR bone ischemia OR avascular necrosis"` |
| 3 | Retrieve abstracts, chunk + cosine-rank by semantic similarity |
| 4 | Build structured prompt: researcher role + gene + ranked abstracts + constraints |
| 5 | Call OpenAI API (`gpt-4o`) — interpret only what the abstracts contain |
| 6 | Write `app/data/output/llm_outputs/{gene}.json` — interpretation, citations, token usage per gene |

Config in `pipeline.yaml` under `llm:` — model, query template, abstracts per gene, output path all configurable.

Output `app/data/output/llm_outputs/` is the direct input to the Phase 7 written report Discussion section.
