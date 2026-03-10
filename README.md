# Capstone: RNA-Seq Analysis with Weka
## Femoral Head Necrosis (scRNA-seq) – GSE316957
### Due: March 21 — Target completion: March 19

---

## Before Anything Else

- [x] **EMAIL THE PROFESSOR (Mar 6)** — asked whether prostate cancer is mandatory; professor
  approved femoral head necrosis (GSE316957) as the dataset for this project.

---

## Dataset Decision

**Using GSE316957** (Femoral Head Necrosis, scRNA-seq)

Notes:
- scRNA-seq data (sparse MTX/TSV format, 336 MB) — will require additional preprocessing to
  aggregate/pseudobulk into a tabular format suitable for Weka
- Confirm class labels and sample groupings from the series matrix before starting Phase 3
- GSE305522 (prostate cancer bulk RNA-seq) remains downloaded as a fallback if needed

---

## Getting Started — Data Download

> **Data is not included in this repository.** All raw and processed data files are gitignored
> due to file size. Follow the steps below to download the necessary datasets before running
> any scripts.

---

### Primary Dataset — Femoral Head Necrosis (scRNA-seq)

**GEO Accession:** [GSE316957](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE316957)

**Study:** *Single-Cell RNA Sequencing Reveals Distinct Cellular and Molecular Features of
Steroid-Induced Osteonecrosis of the Femoral Head Compared to Alcohol-Induced Osteonecrosis
and Hip Osteoarthritis*

**Download steps:**
1. Go to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE316957
2. Scroll to the bottom of the page to the **"Download family"** section
3. Click **"(custom)"** or use the **HTTP** link next to `GSE316957_RAW.tar` to download the raw data archive
4. Also download `GSE316957_series_matrix.txt.gz` (sample metadata / condition labels)
5. Place the downloaded files into: `data/femoral_head_necrosis/`
6. Extract the tar archive:
   ```bash
   cd data/femoral_head_necrosis/
   tar -xf GSE316957_RAW.tar
   ```
   This will produce 5 sample folders (`GSM9463148_onfh_1/` through `GSM9463152_oa_3/`),
   each containing `barcodes.tsv.gz`, `features.tsv.gz`, and `matrix.mtx.gz`.

**Expected structure after extraction:**
```
data/femoral_head_necrosis/
├── GSE316957_series_matrix.txt.gz
├── GSM9463148_onfh_1/
│   ├── barcodes.tsv.gz
│   ├── features.tsv.gz
│   └── matrix.mtx.gz
├── GSM9463149_onfh_2/  ...
├── GSM9463150_oa_1/    ...
├── GSM9463151_oa_2/    ...
└── GSM9463152_oa_3/    ...
```

---

### Fallback / Secondary Dataset — Prostate Adenocarcinoma (bulk RNA-seq)

**GEO Accession:** [GSE305522](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305522)

**Study:** *DHODH regulates mitochondrial bioenergetics, methylation cycle, and DNA repair in
Prostate Adenocarcinoma*

**Download steps:**
1. Go to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305522
2. Scroll to the **"Download family"** section at the bottom
3. Download the supplementary files (typically a counts matrix or series matrix)
4. Place downloaded files into: `data/soulaan_prostate_cancer/`

> This dataset is bulk RNA-seq, so no pseudobulk aggregation is needed. The samples come
> pre-aggregated — one row per sample. The transpose step may be required depending on
> the orientation of the downloaded matrix.

---

### After Downloading — Run the Pipeline

Once the primary dataset is in place, run the scripts in this order:

```bash
# Step 1: Aggregate 46,891 cells → 5 rows (one per patient)
python pseudobulk.py

# Step 2: Filter low-expression genes + normalize (log-CPM)
python preprocess.py

# Output: preprocessed_matrix.csv — this is the Weka-ready file
```

> `preprocessed_matrix.csv` and `pseudobulk_matrix.csv` are also gitignored (large generated
> files). They are fully reproducible by running the two scripts above on the raw GEO data.

---

## Understanding the Raw Data Files

The original download consisted of 3 files:

1. **`GSE316957_RAW_femoral_head_necrosis_alcohol.tar`** — a tar archive, which is like a folder
   zipped into one file. Think of it as a box containing many files. When you extract it, the
   box opens. Inside were 15 files: 5 patient samples × 3 files each.

2. **`GSE316957_family.soft.gz`** — metadata about the study (study title, summary, methods).

3. **`GSE316957_series_matrix.txt.gz`** — metadata about each sample (sample names, conditions,
   which patient is ONFH vs osteoarthritis).

> `.gz` = a single compressed file (like `.zip`). Decompress it to get the original file back.

### What's inside each sample folder

Every sample produces exactly 3 files from the sequencing machine:

| File | What it is |
|------|-----------|
| `barcodes.tsv.gz` | A list of individual cells — one line per cell. Each barcode (e.g. `AAACCCAAGCGACATG-1`) is a unique ID assigned to one cell during the experiment. |
| `features.tsv.gz` | A list of every gene measured — 33,538 genes total. Each line = gene ID, gene name, type. Same list for every sample. |
| `matrix.mtx.gz` | The actual data — how much of each gene was detected in each cell. Stored as triplets: `gene# cell# count`. Only non-zero values are stored to save space (called "sparse" format). |

### The 5 samples

| Folder | Condition | # Cells |
|--------|-----------|---------|
| `GSM9463148_onfh_1/` | Steroid-induced ONFH | 8,160 |
| `GSM9463149_onfh_2/` | Steroid-induced ONFH | 2,128 |
| `GSM9463150_oa_1/` | Hip Osteoarthritis | 8,164 |
| `GSM9463151_oa_2/` | Hip Osteoarthritis | 16,489 |
| `GSM9463152_oa_3/` | Hip Osteoarthritis | 11,950 |

33,538 genes × ~46,891 total cells across all 5 samples.

### What pseudobulk.py actually compressed

Each patient's folder contained thousands of individual cell measurements.
`pseudobulk.py` summed all of those cells into one row per patient:

| Patient folder | Condition | Cells in raw data | After pseudobulk |
|----------------|-----------|------------------:|-----------------|
| GSM9463148_onfh_1 | ONFH | 8,160 cells | 1 row |
| GSM9463149_onfh_2 | ONFH | 2,128 cells | 1 row |
| GSM9463150_oa_1   | OA   | 8,164 cells | 1 row |
| GSM9463151_oa_2   | OA   | 16,489 cells | 1 row |
| GSM9463152_oa_3   | OA   | 11,950 cells | 1 row |
| **Total** | | **46,891 cells** | **5 rows** |

Result: `pseudobulk_matrix.csv` — 5 rows × 33,539 columns (33,538 genes + class label).

### What preprocess.py did to the genes

Of the 33,538 genes measured, many were silent in nearly every sample —
measuring them adds noise without adding signal. `preprocess.py` did two things:

**Filter:** kept only genes expressed (count > 0) in at least 2 of 5 samples.
- Genes before : 33,538
- Genes removed: 3,909  (zero in 4 or 5 samples — no useful signal)
- Genes kept   : **29,629**

**Normalize (log-CPM):** corrected for the fact that different patients had very
different total read counts (23M to 70M). Without this, a gene could appear more
active in one patient just because that sample has more total data — not biology.
- Raw count range after summing: 0 – 3,151,398
- After log-CPM: 0.000 – 10.788  (compressed to a manageable scale)

Result: `preprocessed_matrix.csv` — 5 rows × 29,630 columns, ready for feature selection.

### Which files to look at first (and which to skip)

Look at these first, in this order:

1. **`features.tsv.gz`** (any sample — they're all identical) — this is your gene list. Easy to
   read, just a text file. Tells you what 33,538 genes were measured.
2. **`GSE316957_series_matrix.txt.gz`** — confirms which sample is ONFH vs OA. You already know
   from the folder names but this is the official record.
3. **`matrix.mtx.gz`** — the actual numbers. Hard to read raw, but you can peek at it to
   understand the format.

Don't bother opening `barcodes.tsv.gz` — it's just a list of cell IDs, nothing useful to look at manually.

---

## Repository Contents

```
Omics_Capstone/
├── Screenshots from course (Feb 27)        ← Weka/workflow reference
├── Tranpose_Function.R                     ← Professor's transpose utility (R)
├── transpose.py                            ← Python equivalent of above (BUILT)
├── file_splitter.py                        ← Python single-gene split utility (BUILT)
├── simple ANN wrapper and filter.R         ← Professor's ANN + filter/wrapper
├── TCGA_download_from_Manifest*.R          ← Professor's TCGA download scripts
├── tcga merge by and align by geneid tpm.R ← Professor's merge/align utility
├── data/
│   ├── data_for_course.csv                 ← Course example (already post-transpose)
│   ├── data_for_courseweka.csv             ← Same + case ID column
│   └── test_split/                         ← Output of file_splitter.py on course data
├── pseudobulk.py                           ← Step 1: collapse 46,891 cells → 5 rows (BUILT)
├── preprocess.py                           ← Step 2: filter + normalize genes (BUILT)
├── pseudobulk_matrix.csv                   ← Output of pseudobulk.py (5 × 33,538)
├── preprocessed_matrix.csv                 ← Output of preprocess.py (5 × 29,629) ← USE THIS
├── soulaan_prostate_cancer/                ← Fallback dataset (bulk RNA-seq, not used)
└── femoral_head_necrosis/                  ← PRIMARY DATASET
    ├── GSE316957_RAW_...tar                ← Original archive (can delete after extraction)
    ├── GSE316957_family.soft.gz            ← Study-level metadata
    ├── GSE316957_series_matrix.txt.gz      ← Sample conditions/labels
    ├── GSM9463148_onfh_1/                  ← Steroid ONFH patient 1 (8,160 cells)
    ├── GSM9463149_onfh_2/                  ← Steroid ONFH patient 2 (2,128 cells)
    ├── GSM9463150_oa_1/                    ← Osteoarthritis patient 1 (8,164 cells)
    ├── GSM9463151_oa_2/                    ← Osteoarthritis patient 2 (16,489 cells)
    └── GSM9463152_oa_3/                    ← Osteoarthritis patient 3 (11,950 cells)
```

---

## Schedule Overview

| Phase | Task | Target Dates |
|-------|------|-------------|
| 0 | Orientation — recordings, scripts, example data | Mar 5–7 |
| 1 | R vs Python decision | Mar 7 |
| 3 | Data Acquisition & Verification | Mar 7–8 |
| 4 | Preprocessing | Mar 8–10 |
| 5 | Feature Selection | Mar 10–11 |
| 6 | Weka Analysis | Mar 11–13 |
| 2 | LLM Agent build & test | Mar 13–14 |
| 7 | Run LLM Interpretation | Mar 14–15 |
| 8 | Report Writing (draft all sections) | Mar 14–19 |
| 9 | Polish & final checks | Mar 19 |
| — | **Buffer / submission** | Mar 19–21 |

> Start drafting the Introduction and Methods sections **while** running the analysis (Phase 4 onward).
> Don't wait until everything is done to start writing.

---

## To-Do Checklist

### Phase 0 — Orientation `Mar 5–7`

- [x] **Email professor (Mar 6)** — asked whether prostate cancer is mandatory; proposed femoral head necrosis (GSE316957) as alternative; awaiting reply
- [x] **Watch course recording 4 (Mar 5–6)** — Weka GUI walkthrough, data import, classifier setup
- [x] **Watch course recording 5 (Mar 6–7)** — feature selection, evaluation metrics, output interpretation
- [x] While watching, note:
  - [x] Which steps the professor does in R (use his scripts as-is or adapt)
  - [x] Which steps could be done in Python instead
  - [x] Any specific Weka settings, file formats, or workflow order he recommends
  - [x] Whether he demonstrates LLM interpretation — what tool/API does he use? ChatGPT UI or API?
- [x] Study `data_for_courseweka.csv` — understand format (rows = samples, cols = genes + class label);
  confirmed: course example is already post-transpose (samples as rows, genes as columns)
- [x] Skim `Tranpose_Function.R` — understood; built Python equivalent (`transpose.py`)
- [x] Understand split pattern from `simple ANN wrapper and filter.R` — built Python equivalent (`file_splitter.py`); tested on course data → 4 split files produced correctly

### Phase 1 — Decide R vs Python for Each Step `Mar 7`

Fill in after watching recordings:

| Step | Professor's Tool | Your Option | Decision |
|------|-----------------|-------------|----------|
| Data loading & merging | R (tcga merge script) | Python (pandas) | **Python** (`pseudobulk.py` — DONE) |
| Normalization (DESeq2/TMM) | R (DESeq2/edgeR) | Python (log-CPM) | **Python** (`preprocess.py` — DONE) |
| Transpose for Weka | R (`Tranpose_Function.R`) | Python (df.T) | **Not needed** — pseudobulk already outputs correct orientation |
| Feature selection (filter/split) | R (`simple ANN wrapper`) | Python (sklearn) | **Python** (`file_splitter.py`) — run after Phase 5 on top genes only |
| ARFF export | R (RWeka / foreign) | Python (manual ARFF) | TBD |
| Weka classification | Weka GUI | Weka GUI | Weka GUI |
| Visualization / figures | R (ggplot2, pheatmap) | Python (seaborn, matplotlib) | TBD |
| LLM result interpretation | ChatGPT (per lecture) | Claude API (ResidentRAG pattern) | TBD |

**Guideline:** Use professor's R scripts for steps he demonstrates. Python for everything else.

### Phase 3 — Data Acquisition & Verification `Mar 7–8`

- [x] Extract `GSE316957_RAW_femoral_head_necrosis.tar` and inspect file structure
- [x] Record all GSM IDs and their condition labels from the series matrix
- [x] Determine cell type / condition groupings for class labels (ONFH vs OA, 2 vs 3 samples)
- [x] Verify sparse MTX/TSV format; plan pseudobulk aggregation strategy for Weka input

### Phase 4 — Preprocessing `Mar 8–10`

> **scRNA-seq note:** The README was originally written assuming bulk RNA-seq. For scRNA-seq
> there is an extra step not in the original plan: collapsing ~46,000 cells into 5 rows (one
> per patient sample) before any other preprocessing can happen. This is called pseudobulk
> aggregation and it is the first thing Phase 4 needs to tackle. Everything else flows from it.
>
> Pipeline order for this dataset:
> pseudobulk → filter genes → normalize → [no transpose needed] → feature select → file splitter → Weka

- [x] **pseudobulk.py** — loaded all 5 sample folders, summed counts across cells per sample → `pseudobulk_matrix.csv` (5 × 33,538)
- [x] Filter lowly expressed genes — removed 3,909 genes expressed in fewer than 2 samples; kept 29,629
- [x] Normalize — log-CPM applied; count range compressed from 0–3,151,398 to 0–10.788
- [x] QC: all 5 sample totals confirmed > 0; no missing values
- [x] **transpose.py NOT needed** — pseudobulk.py already outputs samples as rows (Weka format). Only needed for bulk RNA-seq where genes come out as rows.
- [x] Output: `preprocessed_matrix.csv` (5 rows × 29,630 cols) — this is the Weka-ready file
- [ ] **START WRITING: Methods — dataset and preprocessing sections**
  - Introduce the dataset: GEO accession GSE316957, single-cell RNA sequencing of femoral head
    tissue from patients undergoing total hip arthroplasty
  - Explain the two conditions being compared: steroid-induced osteonecrosis of the femoral head
    (ONFH, n=2) vs hip osteoarthritis (OA, n=3) — 5 patient samples total
  - Explain pseudobulk aggregation: 46,891 individual cells across 5 patients were summed per
    sample to produce one gene expression row per patient (the step that makes scRNA-seq usable
    in a tabular ML classifier like Weka)
  - Report the real numbers: started with 33,538 genes × 46,891 cells; after pseudobulk → 5 rows
    × 33,538 genes; after filtering genes expressed in fewer than 2 samples → 29,629 genes kept
  - Explain normalization: log-CPM applied to correct for differences in total read counts between
    patients (ranged from 23M to 70M); brings all samples to a comparable scale
  - Note the limitation: n=5 is very small; results are exploratory and hypothesis-generating,
    not clinically definitive; future work requires larger cohorts

### Phase 5 — Feature Selection / Dimensionality Reduction `Mar 10–11`

- [ ] Write `feature_select.py` — rank genes by variance or mean expression difference between
  ONFH and OA; select top 50–200 genes from `preprocessed_matrix.csv`
- [ ] **file_splitter.py** — run on the feature-selected output only (top 50–200 genes);
  do NOT run on the full 29,629-gene matrix (would produce ~29,000 files, too many)
- [ ] Document gene count at each filtering step (for Methods)
- [ ] Optionally: PCA plot to visualize whether the 5 samples separate by condition
- [ ] **START WRITING: Introduction — background on femoral head necrosis and disease biology**

### Phase 6 — Weka Analysis `Mar 11–13`

- [ ] Import `.arff` into Weka Explorer; set class attribute to condition label
- [ ] Run and compare multiple classifiers:
  - [ ] Naive Bayes
  - [ ] J48 Decision Tree
  - [ ] Random Forest
  - [ ] SVM (SMO)
  - [ ] k-NN (IBk)
- [ ] Evaluation: choose LOOCV or 10-fold CV based on final sample count — justify in Methods
- [ ] Record per classifier: Accuracy, AUC, Confusion Matrix, Precision/Recall/F1
- [ ] Save Weka output logs / screenshots
- [ ] **Collect the top contributing genes/features** — this list is the input to Phase 2
- [ ] **START WRITING: Methods — classifiers and evaluation strategy**

### Phase 2 — LLM Agent for Literature Search & Result Interpretation `Mar 13–14`

> **This targets the bonus 5 marks** ("novel insights or particularly well-executed analysis")
> and is exactly what the professor means by "use LLMs to help interpret results."

**What to build:** `gene_interpreter.py` — a Python script that takes your top genes from Weka,
searches PubMed live, and asks Claude API to interpret results and find supporting papers.

**Reuse from `/Users/jordanharris/Code/ResidentRAG`:**

| ResidentRAG file | What it does | How to reuse |
|---|---|---|
| `app/tools/search_pubmed.py` (`PubMedTool`) | Searches PubMed via Biopython Entrez, fetches abstracts | **Extract `search_pmids()` + `get_title_and_abstract()`** — these are standalone. Remove the embedding/sentence-transformer dependency for simplicity. |
| `app/llm/openai_client.py` (`agentic_llm`) | Iterative tool-calling agent loop | **Reuse the loop pattern.** Swap `OpenAI()` → `anthropic.Anthropic()` and adapt to Claude's tool-use API format. Loop structure (accumulate results → rebuild prompt → iterate) is identical. |
| `app/tools/registry.py` | Defines `pubmed_search` as a JSON-schema LLM tool | **Copy the tool JSON schema** for `pubmed_search` — Claude's tool-use API uses the same format. |

**What you do NOT need from ResidentRAG:**
- Elasticsearch, Qdrant, Docker, hybrid search — those are for a local document corpus.
  Your use case is live PubMed API. None of that infrastructure is needed.

**Script workflow:**
```
Input:  - Top N genes from Weka (e.g., ["DHODH", "SHMT2", "MTHFD1"])
        - Weka summary (accuracy, AUC, confusion matrix as text)
        - Context: femoral head necrosis, scRNA-seq cell type/condition labels

Agent loop (from agentic_llm pattern):
  Iter 0: Claude sees gene list + Weka results → decides which genes to search
  Iter 1: Claude calls pubmed_search tool for top genes
  Iter 2: Claude synthesizes abstracts + metrics → produces interpretation

Output: - Biological interpretation of top classifying genes
        - PubMed citations (PMID + title + year) supporting results
        - Suggested pathways/cell types/interactions for Discussion
        - Draft paragraph for the Discussion section
```

**To-dos:**
- [ ] After watching recordings — confirm whether professor uses ChatGPT API or just the UI;
  this changes how much you need to build
- [ ] Extract `search_pmids()` + `get_title_and_abstract()` into standalone `pubmed_utils.py`
  (remove the embedding model dependency — not needed for keyword search)
- [ ] Build `gene_interpreter.py` using Claude API tool-use
- [ ] Test on known gene first related to femoral head necrosis → verify relevant papers return
- [ ] Run on actual Weka top features

### Phase 7 — Run LLM Interpretation `Mar 14–15`

- [ ] Run `gene_interpreter.py` with actual Weka gene list + results
- [ ] Read the returned PubMed abstracts — do not cite blindly; verify relevance
- [ ] Note citations (PMID, title, year) to include in References
- [ ] Use interpretation output to draft the Discussion section
- [ ] Add a short Methods note:
  *"LLM-assisted literature search was performed using the PubMed Entrez API and Claude API
  to identify supporting publications for top classifying features identified by Weka."*

### Phase 8 — Report Writing (all sections) `Mar 14–19`

> Start earlier sections (Intro, Methods) while analysis is still running — don't wait.

#### Title & Abstract (5 marks)
- [ ] Draft concise, informative title
- [ ] Abstract: background, objective, methods, results, conclusions (~250 words)

#### Introduction (15 marks)
- [ ] Femoral head necrosis background, disease biology, clinical relevance
- [ ] Rationale for scRNA-seq (transcriptomic profiling of cell type/condition changes)
- [ ] Justification for Weka and ML classification
- [ ] Clear stated aim and objectives

#### Materials & Methods (20 marks)
- [ ] Dataset: GEO accession GSE316957, femoral head necrosis scRNA-seq — confirm sample count and conditions
- [ ] Pseudobulk aggregation strategy for converting scRNA-seq to Weka-compatible format
- [ ] Preprocessing pipeline (tools, versions, filtering thresholds, normalization)
- [ ] Feature selection: DE analysis approach + reference professor's filter/wrapper
- [ ] Weka classifiers: brief description of each + justification for choosing them
- [ ] Evaluation: LOOCV, metrics (Accuracy, AUC, F1, Confusion Matrix), justify LOOCV for n=12
- [ ] (If applicable) LLM-assisted literature search methodology

#### Results (20 marks)
- [ ] Table of classifier performance (Accuracy, AUC, F1 per classifier)
- [ ] ROC curves or accuracy bar chart
- [ ] Confusion matrices for best model(s)
- [ ] PCA or heatmap of normalized expression (class separation)
- [ ] Narrative commentary on observed patterns

#### Discussion (20 marks) ← LLM agent output most useful here
- [ ] Interpret top classifying genes in femoral head necrosis biology context (use agent output)
- [ ] Compare to PubMed-retrieved papers on relevant pathways/cell types
- [ ] Limitations: scRNA-seq sparsity, pseudobulk assumptions, no external validation
- [ ] Future work: larger cohorts, patient samples, multi-omics, deep learning

#### Conclusion (5 marks)
- [ ] Summarize key findings; tie back to stated objectives (~150 words)

#### References (5 marks)
- [ ] Key femoral head necrosis papers (identify via LLM agent search)
- [ ] DESeq2 / edgeR or Seurat/pseudobulk methods papers
- [ ] Weka (Hall et al. 2009, SIGKDD Explorations)
- [ ] PubMed papers returned by LLM agent (verify each before citing)
- [ ] Consistent APA or Harvard throughout

### Phase 9 — Polish & Final Checks `Mar 19`

- [ ] All figures have captions and are referenced in-text
- [ ] Proofread for scientific tone and grammar
- [ ] Methods section is reproducible (all tools, versions, parameters documented)
- [ ] Consider bonus: run both 4-class (NT/KD1/KD2/Rescue) and binary (NT vs KD pooled);
  discuss which classifier performs better and what that implies biologically

---

## Key Reference

> To be identified — run LLM agent on top Weka features to surface relevant femoral head necrosis
> literature from PubMed.

---

## Suggested Tools

| Task | Tool |
|------|------|
| Preprocessing & DE | R (DESeq2, edgeR, ggplot2) or Python (pydeseq2, pandas) |
| ARFF export | R `RWeka` or Python manual ARFF writer |
| Classification | Weka 3.8 (GUI Explorer) |
| Figures | R ggplot2, pheatmap or Python seaborn/matplotlib |
| LLM interpretation | Claude API (tool-use) + Biopython Entrez |
| Writing | Overleaf or Word |

---

## Data Format & Workflow Order

### Transpose → Split (this order is mandatory)

**Step 1 — Transpose** (`transpose.py` / `Tranpose_Function.R`)

RNA-seq data comes out of preprocessing in bioinformatics standard orientation:
- Rows = genes (~20,000+)
- Columns = samples (12 GSM IDs)

Weka requires the opposite (ML standard orientation):
- Rows = samples (one observation per row)
- Columns = genes (features) + class label as the last column

Run transpose first to flip the matrix.

**Step 2 — Split** (`file_splitter.py`)

After transposing, split the full matrix into one file per gene for single-gene
wrapper/ANN feature selection. Each split file contains:
`[TNFAlpha, <one_gene>, class_label]`

**Why the course example data (`data_for_course.csv`) looks different:**
The professor provided that file already in post-transpose format (samples as rows).
So splitting it directly produced correct output — but the general rule is always
transpose first, then split.

---

## Notes

- Choose CV strategy (LOOCV vs 10-fold) based on final pseudobulk sample count — justify in Methods
- Consider whether class labels can be framed as binary (disease vs control) in addition to
  multi-class — run both and compare; good bonus insight
- Keep all preprocessing scripts in this directory for reproducibility marks
- Professor's R scripts assume Windows paths (`C:/Graham/...`) — update paths before running locally
- **Do not cite LLM-generated text directly** — use the agent to *find* papers and *suggest*
  interpretations, but read the actual papers and write Discussion in your own words
