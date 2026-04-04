<h1 align="center">AutoOmics_ML_Pipeline</h1>

<p align="center">
  <img src="ASSETS/f49cee5ecb1b78a02ddbfea596afeead.jpg" alt="Microarray Gene Expression Heatmap" width="700">
</p>

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

Both pipelines share the same preprocessing logic (`app/utils/`). The automated pipeline feeds the same LLM interpretation stage.

**Example LLM output — `BPGM.json`:**

```json
{
  "file": "BPGM.json",
  "probe_id": "11719581_a_at",
  "gene_symbol": "BPGM",
  "evidence_tier": "Tier 1",
  "evidence_relation": "direct",
  "evidence_confidence": "high",
  "biomarker_potential": "strong",
  "score": 0.6801,
  "abs_fold_change": 3.3879618999999996,
  "relevance_summary": "BPGM is implicated in the pathophysiology of steroid-induced osteonecrosis of the femoral head and has been identified as a potential biomarker for early detection of this condition.",
  "citations": [
    "BPGM (P07738). https://www.uniprot.org/uniprotkb/P07738",
    "BPGM — BPGM. https://platform.opentargets.org/target/ENSG00000172331",
    "BPGM ↔ hemolytic anemia due to diphosphoglycerate mutase deficiency. https://platform.opentargets.org/target/ENSG00000172331/associations",
    "BPGM ↔ autosomal recessive secondary polycythemia not associated with VHL gene. https://platform.opentargets.org/target/ENSG00000172331/associations",
    "BPGM ↔ placenta praevia. https://platform.opentargets.org/target/ENSG00000172331/associations",
    "BPGM ↔ vertebral column disorder. https://platform.opentargets.org/target/ENSG00000172331/associations",
    "Screening of Potential Biomarkers in the Peripheral Serum for Steroid-Induced Osteonecrosis of the Femoral Head Based on WGCNA and Machine Learning Algorithms. (2022). https://pubmed.ncbi.nlm.nih.gov/35154510/",
    "BPGM — bisphosphoglycerate mutase (NCBI Gene). https://www.ncbi.nlm.nih.gov/gene/",
    "AKT1 — AKT serine/threonine kinase 1 (NCBI Gene). https://www.ncbi.nlm.nih.gov/gene/",
    "GRB2 — growth factor receptor bound protein 2 (NCBI Gene). https://www.ncbi.nlm.nih.gov/gene/",
    "GAPDH — glyceraldehyde-3-phosphate dehydrogenase (NCBI Gene). https://www.ncbi.nlm.nih.gov/gene/",
    "ATF6 — activating transcription factor 6 (NCBI Gene). https://www.ncbi.nlm.nih.gov/gene/",
    "Bioinformatics analysis and identification of genes and molecular pathways in steroid-induced osteonecrosis of the femoral head (PMC full text). https://pmc.ncbi.nlm.nih.gov/articles/PMC8136174/",
    "Advances in the mechanism for steroid-induced osteonecrosis of the femoral head (PMC full text). https://pmc.ncbi.nlm.nih.gov/articles/PMC12902040/",
    "Pathological mechanisms and related markers of steroid-induced osteonecrosis of the femoral head (PMC full text). https://pmc.ncbi.nlm.nih.gov/articles/PMC11559024/",
    "Transcriptomic analysis reveals genetic factors regulating early steroid-induced osteonecrosis of the femoral head (PMC full text). https://pmc.ncbi.nlm.nih.gov/articles/PMC9478254/",
    "MEF cells, siBPGM control 5 [GSM6037560] (GEO). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6037560",
    "MEF cells, siBPGM control 2 [GSM6037557] (GEO). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6037557",
    "MEF cells, siBPGM control 4 [GSM6037559] (GEO). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6037559",
    "MEF cells, siBPGM control 3 [GSM6037558] (GEO). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6037558",
    "A renal function for 2,3-bisphosphoglycerate mutase (BPGM) [GSE200544] (GEO). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200544"
  ]
}
```

---

## Feature selection method

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
│   │   └── output_<run_name>/   ← each run archived under a descriptive name
│   │       │                       e.g. output_new_multivariate_top100/
│   │       │                            output_new_multivariate_top500/
│   │       │                            output_new_univariate_top100/
│   │       │                            output_univariate_rerank_top100/
│   │       │                            output_weka_multivariate/
│   │       │                            output_weka_univariate_ann/
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
│   │       │   ├── eda_composite.png        ← 2×3 multi-panel composite (report Figure 1)
│   │       │   ├── fig_2_model_comparison_<run>.png
│   │       │   ├── fig_3_model_eval_<run>.png
│   │       │   ├── fig_4_stat_vs_model_importance_<run>.png
│   │       │   └── fig_biomarker_summary_composite_<run>.png
│   │       ├── llm_outputs/  ← one JSON per gene
│   │       └── biomarker_shortlist.csv
│   ├── jobs/                 ← orchestration layer
│   │   ├── ingest_job.py
│   │   ├── parse_job.py
│   │   ├── preprocess_job.py
│   │   ├── feature_select_job.py
│   │   ├── train_eval_job.py
│   │   ├── biomarker_job.py
│   │   └── llm_job.py        ← LLM enrichment job
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
| `-m univariate` | **Mode:** Use ANN MCCV probe ranking for feature selection (default) |
| `-m multivariate` | **Mode:** Use hybrid fold-change + t-stat ranking for feature selection |

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
| `llm_outputs/*.json` | Per-gene LLM interpretation |
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

## LLM Integration

Triggered with `--llm` flag. Wired into `main.py` — runs after the biomarker shortlist is generated.

```bash
# Set API key (once per session)
export OPENAI_API_KEY=sk-...

# Run full pipeline including LLM step
python -m app.main --skip-pre --llm
```

**Architecture:** Per-gene agentic retrieval loop → semantic ranking → constrained GPT-4o-mini prompt → structured JSON output.

**Agentic loop:** For each gene the agent runs up to 4 iterations, autonomously selecting tools at each step without a hard-coded call order. Tools are registered as typed JSON function schemas (OpenAI tool-use format), with descriptions written as instructions so the LLM decides what to call and when. On iteration 1, `ncbi_gene_search` is always called first to canonicalise the gene symbol and resolve aliases before any disease-specific queries are issued.

**Registered tools (8 across 5 sources):**

| Tool | Source | Purpose |
|---|---|---|
| `ncbi_gene_search` | NCBI Entrez | Canonicalise gene symbol + resolve aliases (always first) |
| `uniprot_search` | UniProt | Curated protein function |
| `opentargets_search` | Open Targets | Gene–disease association scores |
| `pubmed_search` | PubMed Entrez | Retrieve biomedical abstracts |
| `pubmed_fetch_by_id` | PubMed Entrez | Hydrate specific abstracts by PMID |
| `pmc_fulltext_search` | PMC Entrez | Escalation path when abstract-level evidence is absent or weak |
| `geo_search` | GEO Entrez | Dataset-level metadata to validate disease context + tissue relevance |
| `wikipedia_search` | MediaWiki | General-biology fallback when all domain-specific tools return insufficient evidence |

Before any retrieved text is passed to the LLM, all chunks are cosine-ranked by semantic similarity to the query using `sentence-transformers`, ensuring the most relevant passages fill the context window.

**Evidence tiers — how they are decided:**

The LLM assigns a tier based on what the retrieved evidence actually contains, not on model recall. The prompt explicitly constrains it to grade only retrieved material:

| Tier | Criteria |
|---|---|
| **Tier 1** | Retrieved source directly links this gene to SONFH in human peripheral blood or matching modality |
| **Tier 2** | Human disease-specific mechanistic, pathway, or association evidence for SONFH or close ONFH variants |
| **Tier 3** | Human evidence in disease-adjacent biology (e.g. osteonecrosis of other sites, related bone/vascular pathology) |
| **Tier 4** | Animal, cell-line, speculative, or weakly grounded support only |

Relation type (`direct` / `indirect` / `inferred`) is assigned separately: `direct` means the source explicitly names the gene in a SONFH context; `indirect` means biologically relevant pathway support without a direct disease paper; `inferred` means general gene biology with weak disease-specific grounding.

| Step | What it does |
|---|---|
| 1 | Load `biomarker_shortlist.csv`, deduplicate to unique genes |
| 2 | Agent begins iteration loop — calls tools autonomously up to 4 times per gene |
| 3 | All retrieved chunks cosine-ranked by semantic similarity before LLM sees them |
| 4 | Constrained prompt: interpret only retrieved evidence, flag unsupported claims |
| 5 | Write `app/data/output/llm_outputs/{gene}.json` — tier, relation, confidence, citations, score |

Config in `pipeline.yaml` under `llm:` — model, query template, abstracts per gene, output path all configurable.

Output `app/data/output/llm_outputs/` feeds directly into the report Discussion section.

---

