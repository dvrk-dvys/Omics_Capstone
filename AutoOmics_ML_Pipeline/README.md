# AutoOmics_ML_Pipeline

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
AutoOmics_ML_Pipeline/
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

**All commands run from `AutoOmics_ML_Pipeline/`.**

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

### 9. Multi-run comparison — archiving outputs

After each pipeline run, archive the output directory with a descriptive name before starting the next run:

```bash
# Archive completed run output (name by mode + top_n + any run notes)
mv app/data/output app/data/output_<mode>_top<n>_<notes>
mkdir app/data/output
```

**Example — full 4-run comparison sequence (run from `AutoOmics_ML_Pipeline/`):**

```bash
# Run 1 — multivariate, top 100
python -m app.main --skip-pre --mode multivariate
mv app/data/output app/data/output_multivariate_top100
mkdir app/data/output

# Run 2 — multivariate, top 500
# (edit pipeline.yaml: top_n_feats: 500)
python -m app.main --skip-pre --mode multivariate
mv app/data/output app/data/output_multivariate_top500_min_s_45
mkdir app/data/output

# Run 3 — univariate, top 100
# (edit pipeline.yaml: top_n_feats: 100)
python -m app.main --skip-pre --mode univariate
mv app/data/output app/data/output_univariate_top100_min_s_50
mkdir app/data/output
```

Each archived directory is self-contained — it holds all plots, CSVs, and the biomarker shortlist for that run.

> **Tip:** Run the gene audit (below) before archiving to avoid symlink gymnastics. If you've already archived, see the symlink approach in the audit section.

---

### 10. Gene audit — known SONFH biomarker overlap

Cross-references each pipeline run against the known gene reference list (`report/sonfh_known_genes.csv`).

**Run from `Omics_Capstone/` (project root):**

```bash
# Multivariate audit — reads from data/femoral_head_necrosis/feature_selection/ (standalone Weka prep path)
python report/generate_sonfh_gene_audit.py --mode multivariate
# Output: report/sonfh_gene_audit.csv

# Univariate audit — reads from AutoOmics_ML_Pipeline/app/data/output/feature_selection/univariate_ann/
python report/generate_sonfh_gene_audit.py --mode univariate
# Output: report/sonfh_gene_audit_univariate.csv
```

**If the output has already been archived**, use a symlink to point the script at the right directory:

```bash
# Univariate audit on archived output_univariate_top100_min_s_50
ln -sfn output_univariate_top100_min_s_50 AutoOmics_ML_Pipeline/app/data/output
python report/generate_sonfh_gene_audit.py --mode univariate
cp report/sonfh_gene_audit_univariate.csv report/sonfh_gene_audit_univariate_top100.csv
rm AutoOmics_ML_Pipeline/app/data/output   # remove symlink

# Multivariate audit on archived output_multivariate_top500_min_s_45
# (reads standalone Weka prep paths — no symlink needed)
python report/generate_sonfh_gene_audit.py --mode multivariate
cp report/sonfh_gene_audit.csv report/sonfh_gene_audit_multivariate_top500.csv
```

---

### All flags

| Flag | Effect |
|---|---|
| *(none)* | Full pipeline from scratch |
| `--skip-pre` | Skip ingest, parse, preprocess (use existing outputs) |
| `--skip-train` | Skip model training |
| `--llm` | Run LLM biological interpretation (opt-in) |
| `-m univariate` | Use ANN MCCV probe ranking for feature selection (default) |
| `-m multivariate` | Use hybrid fold-change + t-stat ranking for feature selection |

---

### 8. View results

- MLflow UI: http://localhost:5002 → experiment `sonfh_classification`
- Model comparison CSV: `app/data/output/models/model_comparison.csv`
- Biomarker shortlist: `app/data/output/biomarker_shortlist.csv`
- Gene-level deduped rankings: `app/data/output/feature_selection/gene_level_rankings.csv`
- Top 100 genes: `app/data/output/feature_selection/top100_genes.csv`
- EDA plots: `app/data/output/plots/` (7 plots: volcano, FC bar, boxplots, PCA, sample correlation, heatmap, eda_composite)

---

## Results

### LLM-Validated Biomarker Findings (Phase 6)

All pipeline runs were filtered to direct-evidence genes (Tier 1–3, `evidence_relation=direct`).
Results stored in `report/best_runs/` — each file represents one pipeline configuration.

#### Aligned Gene Table (Tier + LLM Score)

```
Gene        NewMulti100     NewUni100       UniRerank       WekaMulti       WekaUni         |FC|
------------------------------------------------------------------------------------------------
NLRP1       —               T1(1.00)        T1(1.26)        —               T1(0.64)        ~1.27
BPGM        T2(0.68)        —               —               T1(0.65)        —               ~3.39
CISD2       T2(0.45)        —               —               T2(0.71)        —               ~2.26
GYPA        T1(0.54)        —               —               T1(0.73)        —               ~2.88
HEMGN       T1(0.46)        —               —               T2(0.61)        —               ~2.94
P2RY13      —               —               T2(1.29)        —               T2(0.60)        ~1.29
PIP5K1B     T2(0.56)        —               —               T2(0.65)        —               ~2.02
TSTA3       T1(0.74)        —               —               T1(0.71)        —               ~1.74
CBL         —               T1(1.00)        —               —               —               ~1.01
LCP1        —               T1(1.00)        —               —               —               ~0.93
RUNX2       —               —               —               —               T2(0.56)        ~1.22
SETD1B      —               T2(1.00)        —               —               —               ~0.72
STOM        T2(0.39)        —               —               —               —               ~1.63
TMCC2       —               —               —               T2(0.64)        —               ~2.76
```

> **Note:** UniRerank (2 genes) and NewUni100 (4 genes) are incomplete — LLM runs cut early on those configurations. Counts will increase when those jobs finish.

#### Tier Distribution Summary

```
Run            | T1 | T2 | T3 | Total
----------------------------------------
NewMulti100    |  3 |  4 |  0 |    7
NewUni100      |  3 |  1 |  0 |    4
UniRerank      |  1 |  1 |  0 |    2
WekaMulti      |  3 |  4 |  0 |    7
WekaUni        |  1 |  2 |  0 |    3
```

The multivariate configurations (NewMulti100, WekaMulti) converge on almost identical gene sets — strong agreement between the Weka and Python pipelines despite completely independent implementations.

---

### Cross-Reference: Known Literature + Jia et al. 2023 (GSE123568)

```
Gene       Runs Found                    Known CSV   Category               Paper (Jia23)   |FC|
------------------------------------------------------------------------------------------------
NLRP1      NU100:T1  UR100:T1  WU:T1    T1 ✓        pyroptosis             —               1.27
BPGM       NM100:T2  WM:T1              T1 ✓        erythroid metabolism   —               3.39
GYPA       NM100:T1  WM:T1              T1 ✓        erythroid surface      —               2.45
HEMGN      NM100:T1  WM:T2              T2 ✓        erythroid TF           —               2.94
TMCC2      WM:T2                        T1 ✓        erythroid metabolism   —               2.76
RUNX2      WU:T2                        T2 ✓        osteoblast TF          —               1.22
TSTA3      NM100:T1  WM:T1              —           (other SONFH papers)   —               1.74
PIP5K1B    NM100:T2  WM:T2              —           —                      —               2.02
CISD2      NM100:T2  WM:T2              —           —                      —               2.10
STOM       NM100:T2                     —           —                      —               1.63
P2RY13     UR100:T2  WU:T2              —           —                      —               1.29
CBL        NU100:T1                     —           —                      —               1.01
LCP1       NU100:T1                     —           —                      —               0.93
SETD1B     NU100:T2                     —           —                      —               0.72
------------------------------------------------------------------------------------------------
Known gene match:  6 / 14  (NLRP1, BPGM, GYPA, HEMGN, TMCC2, RUNX2)
Paper gene match:  0 / 14  — no direct overlap with Jia et al. 2023
```

**Gene classification by source:**

| Group | Genes |
|---|---|
| Found by pipeline + in known literature | NLRP1, BPGM, GYPA, HEMGN, TMCC2, RUNX2 |
| Found by pipeline + novel (not in known list or paper) | TSTA3, PIP5K1B, CISD2, STOM, P2RY13, CBL, LCP1, SETD1B |
| Found by paper (Jia23) + not found by pipeline | PAK2, CD28, CD4, PIK3CD, PLCG1, PRKCA, VCL, IQGAP1, ACTN4, RAC1, XIAP, PIK3CB, TRAF6, PSEN1, TUBA1A, YWHAZ, HDAC4, CSNK1D, STK11, STAT2, STAT1, ERBB2, CXCR4, AGT, LPAR1 |

> **TSTA3** has `—` in the known CSV but is cited directly in other SONFH literature (the LLM retrieved a 2021 paper identifying it as a diagnostic marker alongside ARG2 and MAP4K5). Not truly novel — just not yet added to the curated list.

---

### Why Zero Overlap with Jia et al. 2023 — and Why That's Expected

The paper runs a completely different methodology on the same dataset (GSE123568):

| Dimension | Jia et al. 2023 | Weka Pipeline | Python Pipeline |
|---|---|---|---|
| **Problem framing** | Stage prediction (Early / Mid / Late NONFH) | Binary: SONFH vs control | Binary: SONFH vs control |
| **Feature selection** | STRING PPI network topology (degree, betweenness, closeness centrality via Cytoscape) | Fold-change ranking → Weka attribute selection | Hybrid score Z(FC)+Z(t-stat) or ANN wrapper |
| **Gene scoring** | Network hub centrality + symptom participation rate (HPO/DisGeNET) | RF attribute importance + J48 split nodes | RF/XGB importance + combined_score |
| **Validation** | Separate holdout cohort (n=64) + RT-qPCR wet lab validation | 10-fold stratified CV only | RepeatedStratKFold (5×10) + MLflow |
| **Biological enrichment** | Clinical phenomics (symptom linkage) | None | LLM agentic RAG (PubMed + UniProt + OpenTargets) |
| **Output** | Stage-specific biomarkers (25 genes, 3 sets) | Single ranked shortlist | Single ranked shortlist |

The paper's approach surfaces **network hubs** — genes centrally wired in PPI space that coordinate signalling, not necessarily the most differentially expressed. This pipeline surfaces **statistically discriminating probes** — genes that maximally separate SONFH from control by expression magnitude, RF splitting power, or ANN classification accuracy.

These two methods surface genuinely different biology. The paper's hubs (IQGAP1, STAT2, CXCR4) are signalling coordinators with modest fold-changes (~1.0–1.1) and near-zero RF importance — they sink to the bottom of a fold-change or importance ranking even though they're biologically central. Confirming this: PIK3CD ranked **493/500** and IQGAP1 ranked **89/100** in the respective shortlists, with RF importance of 0.0 and 0.0 respectively.

Conversely, the loudest genes this pipeline found — GYPA, BPGM, HEMGN, TMCC2 (FC 2.7–3.4, all erythroid) — are the downstream wreckage of ischemia, not its cause. When the femoral head collapses and bone marrow dies, a massive erythroid distress signal floods peripheral blood. These are the fire alarm, not the arsonist. The paper's staging approach explicitly filters these out by requiring genes to appear *before* structural collapse.

The one exception worth highlighting: **ELOVL6** (FC=1.49, T1, found in UniRerank) sits earlier in the causal chain — a lipid elongation enzyme linked to the fatty infiltration mechanism through which glucocorticoids reduce blood flow to the femoral head. It is a plausible upstream signal and warrants further investigation.

```
Glucocorticoids
    → fat embolism / lipid metabolism shift   (PPARG, ELOVL6 ← found by this pipeline)
    → endothelial dysfunction / NO reduction  (NOS3, VEGFA)
    → microvascular occlusion
    → ischemia of femoral head
    → bone cell death
    → structural collapse
    → erythroid/inflammatory chaos            ← GYPA, BPGM, HEMGN, TMCC2 live here
```

The zero overlap is methodologically expected and is not a contradiction. It reflects a genuine difference in what each approach is designed to find.

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
python AutoOmics_ML_Pipeline/app/utils/parse_series_matrix.py
python AutoOmics_ML_Pipeline/app/utils/preprocess.py
python AutoOmics_ML_Pipeline/app/utils/feature_select.py
```

`feature_select.py` accepts an `--outdir` argument (defaults to `data/femoral_head_necrosis/`). It derives two subdirectories from that root automatically:
- `{outdir}/EDA/` — 7 EDA plots including `eda_composite.png`
- `{outdir}/feature_selection/` — CSV/ARFF outputs

```bash
# Custom output root (optional)
python AutoOmics_ML_Pipeline/app/utils/feature_select.py --outdir path/to/output_root
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

---

## TODO

- [ ] **Save Auto-WEKA `.txt` output for multivariate Weka run** — re-run Auto-WEKA on `top100_features.arff`, copy the full results text from the Weka GUI and save to `data/femoral_head_necrosis/weka_models/multivariate/auto_weka.txt`
- [ ] **Save Auto-WEKA `.txt` output for univariate ANN Weka run** — re-run Auto-WEKA on `top100_features_univariate_ann.arff`, save to `data/femoral_head_necrosis/weka_models/univariate_ann/auto_weka.txt`
- Screenshots already exist at `data/screenshots/weka/auto_weka.png` and `data/screenshots/weka_old/auto_weka.png` for reference
