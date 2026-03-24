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
| Single run per model | RepeatedStratifiedKFold (50 evaluations per model) |
| No hyperparameter tuning | Hyperopt for XGBoost + RandomForest |
| Manual biomarker shortlist | Auto-generated `biomarker_shortlist.csv` |

Both pipelines share the same preprocessing logic (`app/utils/`). The automated pipeline feeds the same LLM interpretation stage (Phase 6).

---

## Structure

```
omics_ml_pipeline/
├── app/
│   ├── main.py               ← pipeline entry point
│   ├── config/
│   │   └── pipeline.yaml     ← all paths, params, MLflow config
│   ├── data/                 ← all inputs and outputs
│   │   ├── GSE123568_series_matrix.txt.gz
│   │   ├── GSE123568_family.soft.gz
│   │   ├── parsed/
│   │   ├── feature_selection/
│   │   ├── models/
│   │   └── plots/
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
│   └── utils/                ← shared logic (also used by manual workflow)
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

### 3. Run the pipeline

```bash
# Full run (parse + preprocess + feature select + train + biomarker)
python -m app.main

# Skip parse/preprocess on subsequent runs (outputs already saved)
python -m app.main --skip-parse
```

### 4. View results

- MLflow UI: http://localhost:5002 → experiment `sonfh_classification`
- Model comparison CSV: `app/data/models/model_comparison.csv`
- Biomarker shortlist: `app/data/biomarker_shortlist.csv`

---

## Models

| Model | Stage | Notes |
|---|---|---|
| Logistic (Elastic Net) | Baseline | L1+L2 penalty, class_weight=balanced |
| RandomForest | Baseline + Hyperopt | Feature importance for biomarker ranking |
| LinearSVC | Baseline | class_weight=balanced |
| GaussianNB | Baseline | Probabilistic baseline |
| KNN (k=5) | Baseline | Instance-based |
| MLP | Baseline | 2-layer neural net |
| XGBoost | Baseline + Hyperopt | scale_pos_weight=3 for class imbalance |

CV: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` = 50 evaluations per model.

---

## Outputs

| File | Description |
|---|---|
| `app/data/feature_selection/top_features.csv` | Top 100 probes by fold-change |
| `app/data/feature_selection/gene_rankings.csv` | Full probe ranking with gene symbols |
| `app/data/models/model_comparison.csv` | Accuracy/AUC/F1 across all models |
| `app/data/biomarker_shortlist.csv` | Top candidates ranked by RF importance + fold-change |
| MLflow UI | All run params, metrics, and artifacts |

---

## Manual utils scripts

The shared preprocessing scripts in `app/utils/` can still be run standalone:

```bash
python omics_ml_pipeline/app/utils/parse_series_matrix.py
python omics_ml_pipeline/app/utils/preprocess.py
python omics_ml_pipeline/app/utils/feature_select.py
```

These use their own hardcoded default paths pointing to `data/femoral_head_necrosis/` and produce the Weka-compatible ARFF output alongside the app outputs.

---

## LLM Integration (Phase 6 — TODO)

`app/jobs/llm_job.py` is scaffolded to accept a configurable shortlist path:

```python
# Use app-produced shortlist
llm_job.run(config)

# Or point at the Weka-produced shortlist
llm_job.run(config, shortlist_path="data/femoral_head_necrosis/feature_selection/biomarker_shortlist.csv")
```

Implementation follows the RAG pattern: PubMed retrieval → verified abstracts → constrained Claude API prompt → human validation.
