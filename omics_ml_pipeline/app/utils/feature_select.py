"""
feature_select.py — Select top differentially expressed probes for Weka

WHY THIS SCRIPT EXISTS:
  preprocess.py produced a 40-sample × 11,687-probe matrix for the GSE123568
  microarray dataset (SONFH vs control). That is still too many features for
  Weka to handle efficiently. This script reduces to a manageable probe set.

  DATASET CONTEXT:
    - 40 samples: 30 SONFH (steroid-induced osteonecrosis) + 10 control
    - Affymetrix PrimeView microarray, GPL15207 platform
    - Values are log2 RMA-normalized intensities — already on a comparable scale
    - After IQR filtering: 11,687 probes remain from the original 49,293

  RANKING STRATEGY:
    With n=40, formal statistical tests (t-test) have reasonable power, but
    we use simpler proxy metrics that are robust and interpretable:

    PRIMARY:  |fold change| — absolute difference of group means in log2 space.
              mean(SONFH) − mean(control) is the log2-fold-change.
              Probes with large |FC| are the most biologically distinct between
              conditions and the most informative for classification.

    SECONDARY: variance across all 40 samples — used for comparison and the
               volcano-style plot, but FC is the primary ranking criterion.

  SELECTION IS PROBE-LEVEL (not gene-level):
    Multiple probes can target the same gene (e.g. _at, _s_at, _x_at suffixes).
    Selection is intentionally probe-level — multiple probes for the same gene
    may carry distinct isoform- or region-specific signal. Deduplication to one
    probe per gene is NOT applied here; see gene_level_summary.csv for a
    post-selection view grouped by gene symbol.

OUTPUTS:
  data/femoral_head_necrosis/feature_selection/
  ├── top100_features.csv       ← 40 rows × 101 cols (100 probes + class) — for Weka
  ├── top100_features.arff      ← Weka ARFF format — import directly into Weka Explorer
  ├── gene_rankings.csv         ← full ranking of all 11,687 probes by |FC| and variance
  └── gene_level_summary.csv   ← post-selection: probes grouped by gene symbol

  data/femoral_head_necrosis/EDA/
  ├── volcano_plot.png          ← all 11,687 probes (FC vs variance), top 100 highlighted
  ├── fold_change_top20.png     ← top 20 probes ranked by |FC|
  ├── boxplots_top6.png         ← top 6 probes, SONFH vs control distributions
  ├── sample_correlation.png    ← 40×40 patient similarity heatmap
  ├── heatmap_top20.png         ← top 20 probes × 40 samples expression heatmap
  └── pca_plot.png              ← 2D PCA of top 100 probes

Usage:
  python3 feature_select.py
  python3 feature_select.py --top 50    (select top 50 probes instead)
  python3 feature_select.py --top 100 --method fc   (default)
  python3 feature_select.py --top 100 --method var  (rank by variance)
"""

import sys
import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# PATHS  (relative to app/ — works from any working directory)
# ---------------------------------------------------------------------------
_APP_DIR   = pathlib.Path(__file__).resolve().parent.parent
INPUT_CSV  = str(_APP_DIR / "data" / "output" / "parsed"            / "preprocessed_matrix.csv")
OUTPUT_DIR = str(_APP_DIR / "data" / "output" / "feature_selection")
EDA_DIR    = str(_APP_DIR / "data" / "output" / "plots")
SOFT_GZ    = str(_APP_DIR / "data" / "input"  / "GSE123568_family.soft.gz")


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def load_preprocessed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded: {df.shape[0]} samples × {df.shape[1] - 1} probes (+class)")
    return df


def load_probe_annotation(soft_gz_path: str) -> pd.Series:
    """
    Read the platform annotation table directly from the SOFT .gz file on the fly.
    Extracts rows between !platform_table_begin and !platform_table_end.
    Returns a Series mapping probe_id → gene_symbol.

    No pre-extracted file needed — reads from the compressed SOFT file directly,
    same pattern as parse_series_matrix.py reads the series matrix .gz.
    """
    if not os.path.exists(soft_gz_path):
        print(f"  Note: SOFT file not found at {soft_gz_path} — gene names unavailable")
        return pd.Series(dtype=str)

    import gzip, io
    rows = []
    header = None
    in_table = False

    with gzip.open(soft_gz_path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            elif line == "!platform_table_end":
                break
            elif in_table:
                parts = line.split("\t")
                if header is None:
                    header = parts
                else:
                    rows.append(parts)

    if not header or not rows:
        print("  Note: could not parse platform table from SOFT file")
        return pd.Series(dtype=str)

    ann = pd.DataFrame(rows, columns=header).set_index("ID")
    col = "Gene Symbol" if "Gene Symbol" in ann.columns else ann.columns[0]
    return ann[col].fillna("---")


# ---------------------------------------------------------------------------
# RANK PROBES
# ---------------------------------------------------------------------------
def rank_by_fold_change(
    df: pd.DataFrame,
    disease_label: str = "SONFH",
    control_label: str = "control",
) -> pd.Series:
    """
    Rank probes by absolute log2-fold-change between disease and control groups.

    Values are already log2 RMA intensities, so the difference of group means
    is the log2-fold-change. Absolute value captures both up- and down-regulated
    probes relative to control.

    Returns a Series of |FC| values indexed by probe ID, sorted descending.
    """
    probe_cols = df.columns[:-1]
    sonfh   = df[df["class"] == disease_label][probe_cols]
    control = df[df["class"] == control_label][probe_cols]

    fc = (sonfh.mean() - control.mean()).abs()
    fc.name = "abs_log_fold_change"
    return fc.sort_values(ascending=False)


def rank_by_variance(df: pd.DataFrame) -> pd.Series:
    """
    Rank probes by variance across all 40 samples.
    High-variance probes are informative regardless of direction.
    """
    probe_cols = df.columns[:-1]
    var = df[probe_cols].var(axis=0)
    var.name = "variance"
    return var.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# SELECT TOP N PROBES
# ---------------------------------------------------------------------------
def select_top_probes(df: pd.DataFrame, top_n: int, method: str) -> pd.DataFrame:
    """
    Select the top_n probes by the chosen ranking method and return a
    reduced DataFrame with those probes + the class column.

    Selection is probe-level: multiple probes for the same gene may be
    selected if they all rank highly. See build_gene_level_summary() for
    a post-selection view grouped by gene symbol.
    """
    if method == "fc":
        ranking = rank_by_fold_change(df)
        print(f"\nTop {top_n} probes by |fold change|:")
    elif method == "var":
        ranking = rank_by_variance(df)
        print(f"\nTop {top_n} probes by variance:")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fc' or 'var'.")

    top_probes = ranking.head(top_n).index.tolist()
    annotation = load_probe_annotation(SOFT_GZ)

    # Print top 20 for inspection — show probe ID and gene symbol side by side
    print(f"  {'Rank':<5} {'Probe ID':<18} {'Gene Symbol':<16} {'|FC|'}")
    print(f"  {'-'*55}")
    for i, probe in enumerate(top_probes[:20]):
        gene_sym = annotation.get(probe, "---") if len(annotation) else "---"
        print(f"  {i+1:<5} {probe:<18} {gene_sym:<16} {ranking.loc[probe]:.4f}")
    if top_n > 20:
        print(f"  ... and {top_n - 20} more")

    selected = df[top_probes + ["class"]].copy()
    print(f"\nSelected matrix: {selected.shape[0]} rows × {selected.shape[1] - 1} probes (+class)")
    return selected, ranking


# ---------------------------------------------------------------------------
# ARFF EXPORT
# ---------------------------------------------------------------------------
def write_arff(df: pd.DataFrame, relation_name: str, path: str) -> None:
    """
    Write a DataFrame to Weka ARFF format.

    Format:
      @RELATION name
      @ATTRIBUTE gene NUMERIC
      ...
      @ATTRIBUTE class {ONFH,OA}
      @DATA
      val,val,...,class_label
    """
    gene_cols = df.columns[:-1]
    class_values = sorted(df["class"].unique().tolist())  # e.g. ['ONFH', 'OA']

    with open(path, "w") as f:
        f.write(f"@RELATION {relation_name}\n\n")

        for gene in gene_cols:
            # ARFF attribute names cannot have special chars — replace dots/dashes/slashes
            safe_name = str(gene).replace("-", "_").replace(".", "_")
            f.write(f"@ATTRIBUTE {safe_name} NUMERIC\n")

        class_str = ",".join(class_values)
        f.write(f"@ATTRIBUTE class {{{class_str}}}\n")
        f.write("\n@DATA\n")

        for _, row in df.iterrows():
            gene_vals = ",".join(f"{v:.6f}" for v in row[gene_cols])
            class_val = row["class"]
            f.write(f"{gene_vals},{class_val}\n")

    print(f"ARFF written: {path}")
    print(f"  {len(gene_cols)} features + class {{{class_str}}}")
    print(f"  {len(df)} instances")


# ---------------------------------------------------------------------------
# PCA PLOT
# ---------------------------------------------------------------------------
def plot_pca(
    df: pd.DataFrame,
    output_path: str,
    disease_label: str = "SONFH",
    control_label: str = "control",
    dataset: str = "",
) -> None:
    """
    Generate a 2D PCA plot of samples coloured by condition.
    Uses the top selected probes as features. Separation in PC space
    confirms the selected probes capture disease-state signal.
    """
    gene_cols = df.columns[:-1]
    X = df[gene_cols].values
    y = df["class"].values
    sample_names = df.index.tolist()

    # Standardize before PCA (center + scale each gene)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    # Colour map
    colour_map = {disease_label: "#d62728", control_label: "#1f77b4"}
    colours = [colour_map[c] for c in y]

    n_disease = int((df["class"] == disease_label).sum())
    n_control = int((df["class"] == control_label).sum())

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (x1, x2) in enumerate(coords):
        ax.scatter(x1, x2, c=colours[i], s=120, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate(
            sample_names[i],
            (x1, x2), textcoords="offset points", xytext=(8, 4), fontsize=7
        )

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colour_map[disease_label],
               markeredgecolor="k", markersize=9, label=f"{disease_label} (n={n_disease})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colour_map[control_label],
               markeredgecolor="k", markersize=9, label=f"{control_label} (n={n_control})"),
    ]
    ax.legend(handles=handles, framealpha=0.9)

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance explained)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance explained)", fontsize=11)
    ax.set_title(
        f"PCA of {len(df)} Samples — Top {len(gene_cols)} Features (log2 microarray)\n"
        f"{disease_label} vs {control_label} — {dataset}",
        fontsize=11
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPCA plot saved: {output_path}")
    print(f"  PC1: {var_exp[0]:.1f}%  PC2: {var_exp[1]:.1f}%  Total: {sum(var_exp):.1f}%")


# ---------------------------------------------------------------------------
# EXPRESSION HEATMAP (top 20 genes)
# ---------------------------------------------------------------------------
def plot_heatmap(
    df: pd.DataFrame,
    output_path: str,
    disease_label: str = "SONFH",
    control_label: str = "control",
    dataset: str = "",
) -> None:
    """
    Heatmap of the top 20 probes × samples, annotated by condition.
    Uses seaborn clustermap for automatic row/column ordering.
    """
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn not installed — skipping heatmap (pip install seaborn)")
        return

    gene_cols = df.columns[:-1][:20]    # top 20 probes only
    heatmap_data = df[gene_cols].T      # probes × samples

    # Colour bar for condition labels
    condition_colours = df["class"].map({disease_label: "#d62728", control_label: "#1f77b4"})
    condition_colours.index = df.index

    g = sns.clustermap(
        heatmap_data,
        col_colors=condition_colours,
        cmap="RdYlBu_r",
        figsize=(8, 9),
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    g.ax_heatmap.set_xlabel("Sample", fontsize=10)
    g.ax_heatmap.set_ylabel("Gene", fontsize=10)
    g.fig.suptitle(
        f"Top 20 Differentially Expressed Probes — {disease_label} vs {control_label}\n"
        f"(log2 microarray, hierarchical clustering) — {dataset}",
        y=1.02, fontsize=11
    )

    # Add condition legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d62728", label=disease_label),
        Patch(facecolor="#1f77b4", label=control_label),
    ]
    g.ax_col_dendrogram.legend(
        handles=legend_handles, loc="upper left",
        ncol=2, bbox_to_anchor=(0, 1), fontsize=9
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {output_path}")


# ---------------------------------------------------------------------------
# FOLD CHANGE BAR CHART (top 20 genes — explains why features were selected)
# ---------------------------------------------------------------------------
def plot_fold_change_bar(
    ranking: pd.Series,
    output_path: str,
    top_n: int = 20,
    disease_label: str = "SONFH",
    control_label: str = "control",
    dataset: str = "",
) -> None:
    """
    Horizontal bar chart of the top N probes by |log2-fold-change|.
    Colour-codes by rank tier (top 10 vs 11–20).
    This is the key EDA plot explaining the probe selection decision.
    """
    top = ranking.head(top_n)
    genes  = top.index.tolist()
    values = top.values.tolist()

    # We'll colour by a simple rule: load signed FC from the ranking Series name
    # Since ranking is |FC|, use a neutral colour scheme with magnitude only
    colours = ["#c0392b" if i < top_n // 2 else "#2980b9" for i in range(len(genes))]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(genes[::-1], values[::-1], color=colours[::-1],
                   edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8, color="#333333")

    ax.set_xlabel(f"|Log Fold Change| ({disease_label} mean − {control_label} mean, log2 scale)", fontsize=10)
    ax.set_ylabel("Gene", fontsize=10)
    ax.set_title(
        f"Top {top_n} Probes by Absolute Fold Change — {disease_label} vs {control_label}\n"
        f"Feature selection basis: probes most different between conditions ({dataset})",
        fontsize=11, pad=12
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Legend explaining the ranking
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#c0392b", label="Top 10 — highest |FC|"),
        Patch(facecolor="#2980b9", label="Ranks 11–20"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fold-change bar chart saved: {output_path}")


# ---------------------------------------------------------------------------
# BOX PLOTS — top N genes, expression split by class
# ---------------------------------------------------------------------------
def plot_boxplots(
    df: pd.DataFrame,
    annotation: pd.Series,
    output_path: str,
    top_n: int = 6,
    disease_label: str = "SONFH",
    control_label: str = "control",
    dataset: str = "",
) -> None:
    """
    Box plots of the top N probes (by fold change — already ordered in df columns).
    Each subplot shows the log2 expression distribution for disease vs control.
    """
    top_probes = df.columns[:-1][:top_n].tolist()

    ncols = 3
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = axes.flatten()

    colour_map = {disease_label: "#d62728", control_label: "#1f77b4"}
    n_control = int((df["class"] == control_label).sum())
    n_disease = int((df["class"] == disease_label).sum())

    for i, probe in enumerate(top_probes):
        ax = axes[i]
        groups = [
            df[df["class"] == control_label][probe].values,
            df[df["class"] == disease_label][probe].values,
        ]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2))
        bp["boxes"][0].set_facecolor(colour_map[control_label])
        bp["boxes"][1].set_facecolor(colour_map[disease_label])

        gene_sym = annotation.get(probe, probe) if len(annotation) else probe
        ax.set_title(f"{gene_sym}\n({probe})", fontsize=9)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"{control_label}\n(n={n_control})", f"{disease_label}\n(n={n_disease})"], fontsize=9)
        ax.set_ylabel("log2 expression", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Hide unused axes
    for j in range(top_n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Top {top_n} Probes — Expression by Condition ({disease_label} vs {control_label})\n"
        f"{dataset} | log2 RMA microarray",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Box plots saved: {output_path}")


# ---------------------------------------------------------------------------
# SAMPLE CORRELATION HEATMAP — 40×40 patient similarity matrix
# ---------------------------------------------------------------------------
def plot_sample_correlation(
    df: pd.DataFrame,
    output_path: str,
    disease_label: str = "SONFH",
    control_label: str = "control",
    dataset: str = "",
) -> None:
    """
    Pearson correlation between all patient samples across the top 100 probes.
    If disease patients cluster together and controls cluster together, the disease
    signal is real and not just noise.
    """
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn not installed — skipping sample correlation heatmap")
        return

    gene_cols = df.columns[:-1]
    corr = df[gene_cols].T.corr()   # samples × samples

    n_disease = int((df["class"] == disease_label).sum())
    n_control = int((df["class"] == control_label).sum())
    condition_colours = df["class"].map({disease_label: "#d62728", control_label: "#1f77b4"})
    condition_colours.index = df.index

    g = sns.clustermap(
        corr,
        col_colors=condition_colours,
        row_colors=condition_colours,
        cmap="RdYlBu_r",
        figsize=(11, 10),
        xticklabels=False,
        yticklabels=False,
        dendrogram_ratio=0.12,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        vmin=0.7, vmax=1.0,
    )
    g.fig.suptitle(
        "Sample-Level Pearson Correlation — Top 100 Probes\n"
        f"Red = {disease_label}  |  Blue = {control_label}  |  {dataset}",
        y=1.02, fontsize=11
    )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d62728", label=f"{disease_label} (n={n_disease})"),
        Patch(facecolor="#1f77b4", label=f"{control_label} (n={n_control})"),
    ]
    g.ax_col_dendrogram.legend(
        handles=legend_handles, loc="upper left",
        ncol=2, bbox_to_anchor=(0, 1), fontsize=9
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample correlation heatmap saved: {output_path}")


# ---------------------------------------------------------------------------
# VOLCANO-STYLE PLOT — fold change vs variance, global feature landscape
# ---------------------------------------------------------------------------
def plot_volcano(
    fc_ranking: pd.Series,
    var_ranking: pd.Series,
    top_probe_ids: list,
    annotation: pd.Series,
    output_path: str,
    disease_label: str = "SONFH",
    control_label: str = "control",
    dataset: str = "",
) -> None:
    """
    All 11,687 filtered probes plotted as |fold change| (x) vs variance (y).
    Top 100 selected probes highlighted in red.
    Top 10 labeled with gene names.

    This shows the global feature landscape and visually justifies why the
    top 100 were chosen — they sit in the high-FC, high-variance corner.
    Note: this uses variance as a proxy for significance (no p-values available
    from fold-change ranking alone). Described in report as a 'volcano-style plot.'
    """
    # Align both series to the same probe index
    common = fc_ranking.index.intersection(var_ranking.index)
    fc  = fc_ranking.loc[common]
    var = var_ranking.loc[common]

    top_set = set(top_probe_ids)

    is_top  = [p in top_set for p in common]
    colours = ["#d62728" if t else "#cccccc" for t in is_top]
    sizes   = [25 if t else 5 for t in is_top]
    alphas  = [0.85 if t else 0.3 for t in is_top]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Background probes first, then top 100 on top
    bg_mask  = [not t for t in is_top]
    top_mask = is_top

    ax.scatter(fc[bg_mask],  var[bg_mask],  c="#cccccc", s=5,  alpha=0.3, zorder=1)
    ax.scatter(fc[top_mask], var[top_mask], c="#d62728", s=25, alpha=0.85, zorder=2,
               label=f"Top {len(top_probe_ids)} selected probes")

    # Label top 10 with gene names
    for probe in top_probe_ids[:10]:
        if probe in fc.index:
            gene_sym = annotation.get(probe, probe) if len(annotation) else probe
            ax.annotate(
                gene_sym, (fc[probe], var[probe]),
                textcoords="offset points", xytext=(6, 3),
                fontsize=7, color="#222222",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5)
            )

    ax.set_xlabel(f"|Fold Change| ({disease_label} mean − {control_label} mean, log2)", fontsize=11)
    ax.set_ylabel(f"Variance across all {len(fc_ranking)} probes", fontsize=11)
    ax.set_title(
        "Volcano-Style Plot: Effect Size vs Variability\n"
        f"All {len(common):,} filtered probes — top 100 selected highlighted in red | {dataset}",
        fontsize=11
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Volcano-style plot saved: {output_path}")


# ---------------------------------------------------------------------------
# GENE-LEVEL SUMMARY — post-selection interpretation layer
# ---------------------------------------------------------------------------
def build_gene_level_summary(
    selected_df: pd.DataFrame,
    fc_ranking: pd.Series,
    annotation: pd.Series,
    output_path: str,
    disease_label: str = "SONFH",
    control_label: str = "control",
) -> pd.DataFrame:
    """
    Group the selected probes by gene symbol and produce a summary table.

    WHY THIS EXISTS:
      Feature selection is probe-level — multiple probes for the same gene can
      all rank in the top 100. This function provides a post-selection view that
      answers: which genes are actually represented, by how many probes, and are
      those probes consistent in their direction?

    DIRECTION CONSISTENCY:
      'consistent' — all probes for this gene were selected in the same direction
                     (all higher in SONFH, or all higher in control)
      'mixed'      — probes disagree in direction (possible cross-hybridisation
                     artefact or real isoform differences; interpret with care)
      'single'     — only one probe selected, direction not independently confirmed

    NOTE ON PROBE TYPES:
      _at    = standard (one gene target)   — most specific
      _s_at  = shared (multiple transcripts of same gene)
      _x_at  = cross-hybridising (may target multiple genes) — least specific

    Parameters:
      selected_df : DataFrame — top N probes × 40 samples + class column
      fc_ranking  : Series    — abs fold change for ALL probes (probe_id index)
      annotation  : Series    — probe_id → gene_symbol (from SOFT file)
      output_path : str       — where to save gene_level_summary.csv

    Returns:
      pd.DataFrame — one row per unique gene symbol, columns described above
    """
    probe_cols = selected_df.columns[:-1].tolist()

    # Compute signed FC (needed for direction consistency check)
    sonfh_mean   = selected_df[selected_df["class"] == disease_label][probe_cols].mean()
    control_mean = selected_df[selected_df["class"] == control_label][probe_cols].mean()
    signed_fc    = sonfh_mean - control_mean   # positive = higher in SONFH

    def probe_type(pid: str) -> str:
        if pid.endswith("_x_at"):  return "_x_at"
        if pid.endswith("_s_at"):  return "_s_at"
        if pid.endswith("_a_at"):  return "_a_at"
        if pid.endswith("_at"):    return "_at"
        return "other"

    # Build per-probe rows
    rows = []
    for probe in probe_cols:
        gene_sym = annotation.get(probe, "---") if len(annotation) else "---"
        rows.append({
            "probe_id":        probe,
            "gene_symbol":     gene_sym,
            "probe_type":      probe_type(probe),
            "abs_fold_change": fc_ranking.get(probe, float("nan")),
            "signed_fc":       signed_fc.get(probe, float("nan")),
        })

    probe_df = pd.DataFrame(rows)

    # Group by gene symbol
    records = []
    for gene, grp in probe_df.groupby("gene_symbol", sort=False):
        n = len(grp)
        probe_ids  = " | ".join(grp["probe_id"].tolist())
        types      = " | ".join(grp["probe_type"].tolist())
        max_fc     = grp["abs_fold_change"].max()
        mean_fc    = grp["abs_fold_change"].mean()

        if n == 1:
            consistency = "single"
        elif (grp["signed_fc"] > 0).all() or (grp["signed_fc"] < 0).all():
            consistency = "consistent"
        else:
            consistency = "mixed"

        # Notes
        has_x_at = "_x_at" in types
        note_parts = []
        if n > 2:
            note_parts.append(f"{n} probes selected — high representation")
        if has_x_at:
            note_parts.append("contains _x_at probe (cross-hybridising — interpret with care)")
        if consistency == "mixed":
            note_parts.append("mixed FC direction — possible isoform or cross-hybridisation effect")
        notes = "; ".join(note_parts) if note_parts else ""

        records.append({
            "gene_symbol":           gene,
            "selected_probe_count":  n,
            "probe_ids":             probe_ids,
            "probe_types":           types,
            "max_abs_fold_change":   round(max_fc, 4),
            "mean_abs_fold_change":  round(mean_fc, 4),
            "direction_consistency": consistency,
            "notes":                 notes,
        })

    summary = pd.DataFrame(records).sort_values("max_abs_fold_change", ascending=False)
    summary.to_csv(output_path, index=False)

    n_multi  = (summary["selected_probe_count"] > 1).sum()
    n_mixed  = (summary["direction_consistency"] == "mixed").sum()
    n_x_at   = summary["probe_types"].str.contains("_x_at").sum()
    print(f"\nGene-level summary saved: {output_path}")
    print(f"  {len(summary)} unique gene symbols from {len(probe_cols)} selected probes")
    print(f"  {n_multi} genes represented by >1 probe")
    print(f"  {n_mixed} genes with mixed FC direction across probes")
    print(f"  {n_x_at} genes have at least one _x_at (cross-hybridising) probe")

    return summary


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame, ranking: pd.Series, method: str) -> None:
    print("\n" + "=" * 60)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 60)
    print(f"Ranking method    : {'|fold change|' if method == 'fc' else 'variance'}")
    print(f"Input probes      : {len(ranking)}")
    print(f"Selected probes   : {df.shape[1] - 1}")
    print(f"Samples retained  : {df.shape[0]}")
    print(f"\nClass distribution:")
    print(df["class"].value_counts().to_string())
    print(f"\nTop 10 selected probes:")
    print(ranking.head(10).round(4).to_string())
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Load top100_features.arff into Weka Explorer")
    print("  2. Set 'class' as the class attribute")
    print("  3. Run classifiers with 10-fold CV (n=40 is large enough)")
    print("  4. Compare: NaiveBayes, J48, RandomForest, SMO (SVM), IBk (k-NN)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Select top differentially expressed probes for Weka")
    p.add_argument("--top",    type=int,   default=100,  help="Number of top probes to select (default: 100)")
    p.add_argument("--method", type=str,   default="fc", choices=["fc", "var"],
                   help="Ranking method: fc=|fold change| (default), var=variance")
    p.add_argument("--input",  type=str,   default=INPUT_CSV,    help="Path to preprocessed_matrix.csv")
    p.add_argument("--outdir", type=str,   default=OUTPUT_DIR,   help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    eda_dir = EDA_DIR
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(eda_dir, exist_ok=True)

    # 1. Load
    df = load_preprocessed(args.input)

    # 2. Compute full rankings (needed for bar chart and CSV)
    fc_ranking  = rank_by_fold_change(df).rename("abs_log_fold_change")
    var_ranking = rank_by_variance(df).rename("variance")

    # 3. Select top probes
    selected_df, ranking = select_top_probes(df, top_n=args.top, method=args.method)

    # 4. Save selected CSV
    csv_path = os.path.join(args.outdir, f"top{args.top}_features.csv")
    selected_df.to_csv(csv_path)
    print(f"\nSelected features CSV: {csv_path}")

    # 5. Save full probe rankings with gene symbols for inspection / Methods writeup
    rankings_path = os.path.join(args.outdir, "gene_rankings.csv")
    annotation    = load_probe_annotation(SOFT_GZ)
    combined = pd.DataFrame({"abs_log_fold_change": fc_ranking, "variance": var_ranking})
    if len(annotation):
        ann_df   = annotation.reset_index()
        ann_df.columns = ["probe_id", "gene_symbol"]
        combined = combined.reset_index().rename(columns={"index": "probe_id"})
        combined = combined.merge(ann_df, on="probe_id", how="left").set_index("probe_id")
        combined = combined[["gene_symbol", "abs_log_fold_change", "variance"]]
    combined.to_csv(rankings_path)
    print(f"Full probe rankings  : {rankings_path}  (includes gene_symbol column)")

    # 6. Save ARFF
    arff_path = os.path.join(args.outdir, f"top{args.top}_features.arff")
    write_arff(selected_df, relation_name="femoral_head_necrosis", path=arff_path)

    # 7. EDA — Fold change bar chart (why these probes were selected)
    bar_path = os.path.join(eda_dir, "fold_change_top20.png")
    plot_fold_change_bar(fc_ranking, bar_path, top_n=20)

    # 8. EDA — PCA plot (sample separation in top-100-probe space)
    pca_path = os.path.join(eda_dir, "pca_plot.png")
    plot_pca(selected_df, pca_path)

    # 9. EDA — Heatmap of top 20 probes × 40 samples
    heatmap_path = os.path.join(eda_dir, "heatmap_top20.png")
    plot_heatmap(selected_df, heatmap_path)

    # 10. EDA — Box plots of top 6 probes by class
    box_path = os.path.join(eda_dir, "boxplots_top6.png")
    plot_boxplots(selected_df, annotation, box_path, top_n=6)

    # 11. EDA — Sample correlation heatmap (40×40)
    corr_path = os.path.join(eda_dir, "sample_correlation.png")
    plot_sample_correlation(selected_df, corr_path)

    # 12. EDA — Volcano-style plot (FC vs variance, all probes)
    top_probe_ids = selected_df.columns[:-1].tolist()
    volcano_path  = os.path.join(eda_dir, "volcano_plot.png")
    plot_volcano(fc_ranking, var_ranking, top_probe_ids, annotation, volcano_path)

    # 13. Gene-level summary — post-selection interpretation (probe → gene grouping)
    gene_summary_path = os.path.join(args.outdir, "gene_level_summary.csv")
    build_gene_level_summary(selected_df, fc_ranking, annotation, gene_summary_path)

    # 14. Summary
    print_summary(selected_df, ranking, args.method)

    print(f"\nWeka files in  : {args.outdir}/")
    print(f"EDA plots in   : {eda_dir}/")
