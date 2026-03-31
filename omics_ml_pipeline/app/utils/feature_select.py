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
from scipy import stats
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
# WEKA STANDALONE PATHS  (used only when running this file directly)
# ---------------------------------------------------------------------------
_PROJECT_ROOT    = _APP_DIR.parent.parent   # /Omics_Capstone/
_WEKA_INPUT_CSV  = str(_PROJECT_ROOT / "data" / "femoral_head_necrosis" / "parsed" / "preprocessed_matrix.csv")
_WEKA_EDA_DIR    = str(_PROJECT_ROOT / "data" / "femoral_head_necrosis" / "EDA")
_WEKA_OUT_DIR    = str(_PROJECT_ROOT / "data" / "femoral_head_necrosis" / "feature_selection")
_WEKA_MODELS_DIR = str(_PROJECT_ROOT / "data" / "femoral_head_necrosis" / "weka_models")


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


def rank_by_hybrid_score(
    df: pd.DataFrame,
    disease_label: str = "SONFH",
    control_label: str = "control",
) -> pd.DataFrame:
    """
    Rank probes by hybrid_score = zscore(abs_fold_change) + zscore(abs_t_stat).

    Per-probe columns computed and stored:
      mean_sonfh, mean_control  — group means in log2 space
      log_fold_change           — signed mean difference (SONFH − control)
      abs_fold_change           — |log_fold_change|  (kept for downstream use)
      t_stat                    — Welch t-statistic
      abs_t_stat                — |t_stat|  (used in hybrid score)
      p_value                   — two-sided Welch p-value
      iqr                       — interquartile range across all samples
      variance                  — variance across all samples
      probe_type                — suffix type (_at, _s_at, _x_at, _a_at, other)
      hybrid_score              — zscore(abs_fold_change) + zscore(abs_t_stat)

    Returns a DataFrame sorted by hybrid_score descending. Index = probe_id.
    abs_fold_change is preserved so downstream callers (biomarker_job, plots)
    can still use fold-change values directly.
    """
    probe_cols   = df.columns[:-1]
    sonfh_df     = df[df["class"] == disease_label][probe_cols]
    control_df   = df[df["class"] == control_label][probe_cols]

    mean_sonfh   = sonfh_df.mean()
    mean_control = control_df.mean()
    log_fc       = mean_sonfh - mean_control     # signed log2 FC
    abs_fc       = log_fc.abs()

    # Welch t-test — vectorized over all probes simultaneously
    t_vals, p_vals = stats.ttest_ind(
        sonfh_df.values, control_df.values, axis=0, equal_var=False
    )
    t_series = pd.Series(t_vals, index=probe_cols)
    p_series = pd.Series(p_vals, index=probe_cols)

    probe_data = df[probe_cols]
    iqr_s = probe_data.quantile(0.75) - probe_data.quantile(0.25)
    var_s = probe_data.var()

    def _probe_type(pid: str) -> str:
        if pid.endswith("_x_at"):  return "_x_at"
        if pid.endswith("_s_at"):  return "_s_at"
        if pid.endswith("_a_at"):  return "_a_at"
        if pid.endswith("_at"):    return "_at"
        return "other"

    type_s = pd.Series([_probe_type(p) for p in probe_cols], index=probe_cols)

    ranking = pd.DataFrame({
        "mean_sonfh":      mean_sonfh.round(4),
        "mean_control":    mean_control.round(4),
        "log_fold_change": log_fc.round(4),
        "abs_fold_change": abs_fc.round(4),
        "t_stat":          t_series.round(4),
        "abs_t_stat":      t_series.abs().round(4),
        "p_value":         p_series,
        "iqr":             iqr_s.round(4),
        "variance":        var_s.round(4),
        "probe_type":      type_s,
    })

    def _zscore(s: pd.Series) -> pd.Series:
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma

    ranking["hybrid_score"] = (
        _zscore(ranking["abs_fold_change"]) + _zscore(ranking["abs_t_stat"])
    ).round(4)

    ranking.index.name = "probe_id"
    return ranking.sort_values("hybrid_score", ascending=False)


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
    gene_map: pd.Series = None,
) -> None:
    """
    Horizontal bar chart of the top N probes by |log2-fold-change|.
    Colour-codes by rank tier (top 10 vs 11–20).
    This is the key EDA plot explaining the probe selection decision.
    """
    top       = ranking.head(top_n)
    probe_ids = top.index.tolist()
    values    = top.values.tolist()

    labels = []
    for p in probe_ids:
        sym = (gene_map.get(p, "") if gene_map is not None and len(gene_map) else "")
        sym = sym if sym and sym != "---" else ""
        labels.append(f"{p} — {sym}" if sym else p)

    colours = ["#c0392b" if i < top_n // 2 else "#2980b9" for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(labels[::-1], values[::-1], color=colours[::-1],
                   edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8, color="#333333")

    ax.set_xlabel(f"|Log Fold Change| ({disease_label} mean − {control_label} mean, log2 scale)", fontsize=10)
    ax.set_ylabel("Probe ID — Gene Symbol", fontsize=10)
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
# GENE-LEVEL DEDUPLICATION — one best probe per gene for interpretation
# ---------------------------------------------------------------------------
def build_gene_level_dedup(
    hybrid_df: pd.DataFrame,
    gene_map: pd.Series,
    gene_level_path: str,
    top_genes_path: str,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Deduplicate the full probe-level hybrid ranking to one representative
    probe per gene.  Used for biomarker interpretation and literature
    comparison only — does NOT replace the probe-level Weka top-100 file.

    Probe selection priority (tie-breaking in order):
      1. Highest hybrid_score
      2. Probe specificity: _at > _s_at > _a_at > _x_at > other
      3. Higher IQR
      4. Deterministic (pandas groupby stable sort)

    Exports:
      gene_level_rankings.csv — one row per gene, all ranked genes
      top{n}_genes.csv        — top N genes by hybrid_score

    Parameters:
      hybrid_df       : DataFrame from rank_by_hybrid_score() (probe_id index)
      gene_map        : Series  probe_id → gene_symbol
      gene_level_path : output path for gene_level_rankings.csv
      top_genes_path  : output path for top{n}_genes.csv
      top_n           : number of top genes written to top_genes_path (default 100)

    Returns:
      pd.DataFrame — gene-level deduped table (all genes, sorted by hybrid_score)
    """
    _TYPE_PRIORITY = {"_at": 0, "_s_at": 1, "_a_at": 2, "_x_at": 3, "other": 4}

    work = hybrid_df.copy()
    work.index.name = "probe_id"
    work = work.reset_index()
    work["gene_symbol"]   = work["probe_id"].map(gene_map).fillna("---")
    work["type_priority"] = work["probe_type"].map(_TYPE_PRIORITY).fillna(4).astype(int)
    work["probe_rank"]    = range(1, len(work) + 1)

    # Sort so groupby.first() picks the best probe per gene
    work_sorted = work.sort_values(
        ["hybrid_score", "type_priority", "iqr"],
        ascending=[False, True, False],
    )
    best = work_sorted.groupby("gene_symbol", sort=False).first().reset_index()
    best = best.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    best["gene_rank"] = range(1, len(best) + 1)

    keep_cols = [
        "gene_rank", "gene_symbol", "probe_id", "probe_rank", "probe_type",
        "hybrid_score", "abs_fold_change", "log_fold_change",
        "mean_sonfh", "mean_control", "t_stat", "p_value", "iqr", "variance",
    ]
    out_cols = [c for c in keep_cols if c in best.columns]
    gene_df  = best[out_cols]

    gene_df.to_csv(gene_level_path, index=False)
    print(f"\nGene-level deduped rankings saved: {gene_level_path}  ({len(gene_df)} genes)")

    top_genes = gene_df.head(top_n)
    top_genes.to_csv(top_genes_path, index=False)
    print(f"Top {top_n} genes saved: {top_genes_path}")

    return gene_df


# ---------------------------------------------------------------------------
# MULTI-PANEL EDA COMPOSITE FIGURE
# ---------------------------------------------------------------------------
def plot_composite_eda(plots_dir: str, output_path: str, dataset: str = "") -> None:
    """
    Compose a 2×3 multi-panel EDA figure from the 6 individual PNG plots.
    Uses matplotlib.figure.subfigures. Panels are labelled A–F.

    Panel layout:
      A: PCA plot            B: Volcano plot        C: Heatmap top20
      D: Fold-change bar     E: Box plots top6      F: Sample correlation

    Parameters:
        plots_dir   : directory containing the individual PNG files
        output_path : where to save the composite PNG
        dataset     : optional subtitle string (e.g. "GSE123568 — top 100 probes")
    """
    import matplotlib.image as mpimg

    PANEL_FILES = [
        ("pca_plot.png",           "A"),
        ("volcano_plot.png",       "B"),
        ("heatmap_top20.png",      "C"),
        ("fold_change_top20.png",  "D"),
        ("boxplots_top6.png",      "E"),
        ("sample_correlation.png", "F"),
    ]

    panels = []
    for fname, label in PANEL_FILES:
        fpath = os.path.join(plots_dir, fname)
        if os.path.exists(fpath):
            panels.append((fpath, label))
        else:
            print(f"  Warning: {fname} not found in {plots_dir} — skipping panel {label}")

    if not panels:
        print(f"  No EDA plots found in {plots_dir} — cannot generate composite.")
        return

    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols

    fig = plt.figure(figsize=(18, 5.2 * nrows))
    if dataset:
        fig.suptitle(
            f"Exploratory Data Analysis — {dataset}",
            fontsize=13, fontweight="bold", y=1.005,
        )

    subfigs = fig.subfigures(nrows, ncols, hspace=0.04, wspace=0.02)
    flat    = subfigs.flatten() if nrows * ncols > 1 else [subfigs]

    for idx, (fpath, label) in enumerate(panels):
        sf  = flat[idx]
        ax  = sf.add_subplot(1, 1, 1)
        try:
            img = mpimg.imread(fpath)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"[Error loading {os.path.basename(fpath)}]",
                    ha="center", va="center", fontsize=9, color="red",
                    transform=ax.transAxes)
        ax.axis("off")
        sf.text(0.02, 0.97, label, fontsize=15, fontweight="bold",
                va="top", ha="left", transform=sf.transSubfigure)

    for idx in range(len(panels), nrows * ncols):
        flat[idx].set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"EDA composite figure saved: {output_path}")


# ---------------------------------------------------------------------------
# ML EVALUATION PLOTS — ROC curves, confusion matrix, feature importance
# ---------------------------------------------------------------------------

def plot_roc_curves(
    X: np.ndarray,
    y: np.ndarray,
    named_models: list,
    plots_dir: str,
    cv_splits: int = 5,
    cv_repeats: int = 10,
    random_state: int = 42,
    dataset: str = "",
) -> None:
    """
    CV-averaged ROC curves with ±1 std band for each named model.

    For each model, runs RepeatedStratifiedKFold (cv_splits × cv_repeats folds),
    computes a per-fold ROC curve, interpolates all curves to a shared FPR grid,
    then plots mean TPR ± 1 std band with per-model AUC mean ± std in the legend.

    Parameters:
        named_models : list of (display_label, unfitted_model) tuples
        cv_splits    : number of stratified folds per repeat
        cv_repeats   : number of full CV repeats
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from sklearn.base import clone

    FPR_GRID = np.linspace(0, 1, 200)
    COLOURS  = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, (display_name, model) in enumerate(named_models):
        tprs, aucs = [], []
        cv = RepeatedStratifiedKFold(
            n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state
        )
        for train_idx, test_idx in cv.split(X, y):
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])
            if hasattr(m, "predict_proba"):
                y_score = m.predict_proba(X[test_idx])[:, 1]
            elif hasattr(m, "decision_function"):
                y_score = m.decision_function(X[test_idx])
            else:
                print(f"  Skipping fold — {display_name} has no predict_proba or decision_function")
                continue
            fpr, tpr, _ = roc_curve(y[test_idx], y_score)
            aucs.append(auc(fpr, tpr))
            interp_tpr    = np.interp(FPR_GRID, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        if not tprs:
            print(f"  No folds completed for {display_name} — skipping ROC curve")
            continue

        mean_tpr     = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr      = np.std(tprs, axis=0)
        mean_auc     = float(np.mean(aucs))
        std_auc      = float(np.std(aucs))
        colour       = COLOURS[idx % len(COLOURS)]

        ax.plot(FPR_GRID, mean_tpr, colour, lw=2,
                label=f"{display_name}  (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
        ax.fill_between(FPR_GRID,
                        np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        alpha=0.15, color=colour)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    n_folds = cv_splits * cv_repeats
    ax.set_title(
        f"ROC Curves — {cv_splits}-fold CV × {cv_repeats} repeats ({n_folds} folds total)\n"
        f"Mean ± 1 std  |  {dataset}",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(plots_dir, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved: {out_path}")


def plot_confusion_matrix(
    X: np.ndarray,
    y: np.ndarray,
    model,
    model_name: str,
    class_names: list,
    plots_dir: str,
    cv_splits: int = 5,
    random_state: int = 42,
    dataset: str = "",
) -> None:
    """
    Confusion matrix from out-of-fold (OOF) predictions via StratifiedKFold.

    Uses a single StratifiedKFold pass so each sample appears in the test set
    exactly once, giving a clean OOF confusion matrix with no sample overlap.
    Plots side-by-side: raw counts (left) and row-normalised recall (right).

    Parameters:
        model       : unfitted sklearn-compatible model
        model_name  : label shown in the plot title
        class_names : string labels ordered to match LabelEncoder encoding (sorted)
        cv_splits   : number of stratified folds (default 5)
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cv     = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    cm     = confusion_matrix(y, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    disp_counts = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp_counts.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Raw OOF counts", fontsize=10)

    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=np.round(cm_norm, 2), display_labels=class_names
    )
    disp_norm.plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("Row-normalised (recall per class)", fontsize=10)

    fig.suptitle(
        f"Confusion Matrix — {model_name}\n"
        f"OOF predictions, {cv_splits}-fold stratified CV  |  {dataset}",
        fontsize=11,
    )
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {out_path}")


def plot_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    probe_cols: list,
    model,
    model_name: str,
    gene_map: pd.Series,
    plots_dir: str,
    top_n: int = 20,
    dataset: str = "",
) -> None:
    """
    Horizontal bar chart of top N features by model importance.

    Fits the model once on the full dataset for stable importance estimates,
    then labels each bar with the gene symbol from gene_map when available,
    falling back to the raw probe_id otherwise.

    Supports:
      .feature_importances_  — XGBoost, RandomForest (gain)
      .coef_                 — LogisticRegression, LinearSVC (|coef|)
      Pipeline               — inspects the final named_step for either attribute

    Note: fitting on the full dataset may yield slightly optimistic importance
    estimates; values are used for ranking only, not reported as metrics.

    Parameters:
        probe_cols : probe IDs corresponding to X columns (in order)
        gene_map   : Series probe_id → gene_symbol (from SOFT annotation)
        top_n      : number of top features to show (default 20)
    """
    from sklearn.base import clone

    m = clone(model)
    m.fit(X, y)

    def _get_importances(est):
        if hasattr(est, "feature_importances_"):
            return est.feature_importances_, "Feature importance (gain)"
        if hasattr(est, "coef_"):
            coef = est.coef_
            vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef[0])
            return vals, "|Coefficient|"
        return None, None

    if hasattr(m, "named_steps"):
        inner = list(m.named_steps.values())[-1]
        importances, importance_label = _get_importances(inner)
    else:
        importances, importance_label = _get_importances(m)

    if importances is None:
        print(f"plot_feature_importance: {model_name} has no importances or coef_ — skipping")
        return

    indices    = np.argsort(importances)[::-1][:top_n]
    top_probes = [probe_cols[i] for i in indices]
    top_vals   = importances[indices]

    top_labels = []
    for p in top_probes:
        sym = (gene_map.get(p, "") if gene_map is not None and len(gene_map) else "")
        sym = sym if sym and sym != "---" else ""
        top_labels.append(f"{p} — {sym}" if sym else p)

    colours = ["#d62728" if i == 0 else "#1f77b4" for i in range(len(top_vals))]

    def _draw_importance_bars(ax, vals, labels, title):
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals[::-1], color=colours[::-1], edgecolor="white", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel(importance_label, fontsize=11)
        ax.set_ylabel("Probe ID — Gene Symbol", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    # Full-scale plot
    fig, ax = plt.subplots(figsize=(11, 7))
    _draw_importance_bars(ax, top_vals, top_labels,
                          f"Top {top_n} Features — {model_name}\n{dataset}")
    plt.tight_layout()
    out_path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved: {out_path}")

    # Zoomed plot — clip at 1.5× the second-highest value so the outlier doesn't compress the rest
    if len(top_vals) >= 2 and top_vals[1] > 1e-8:
        zoom_xlim = top_vals[1] * 1.5
        fig, ax = plt.subplots(figsize=(11, 7))
        _draw_importance_bars(ax, top_vals, top_labels,
                              f"Top {top_n} Features — {model_name} (zoomed — top outlier truncated)\n{dataset}")
        ax.set_xlim(0, zoom_xlim)
        # Annotate the truncated outlier bar with its actual value
        # y=0 is the top bar after barh reversal
        outlier_label = f"{top_vals[0]:.3f}"
        ax.text(
            zoom_xlim * 0.98, 0,
            f"← {top_labels[0]}  ({outlier_label})",
            va="center", ha="right", fontsize=7.5, color="#d62728",
        )
        plt.tight_layout()
        zoom_path = os.path.join(plots_dir, "feature_importance_zoomed.png")
        plt.savefig(zoom_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Feature importance zoomed plot saved: {zoom_path}")


def plot_gene_importance_aggregated(
    X: np.ndarray,
    y: np.ndarray,
    probe_cols: list,
    model,
    model_name: str,
    gene_map: pd.Series,
    plots_dir: str,
    top_n: int = 20,
    dataset: str = "",
) -> None:
    """
    Horizontal bar chart of top N genes by maximum probe importance.

    Groups all probes per gene symbol and takes the max importance value —
    "strongest signal observed for this gene". Separate from probe-level
    feature_importance.png; this is the biological gene-level summary.

    Saved as gene_importance_aggregated.png.
    """
    from sklearn.base import clone

    if gene_map is None or len(gene_map) == 0:
        print("plot_gene_importance_aggregated: no gene_map — skipping")
        return

    m = clone(model)
    m.fit(X, y)

    def _get_importances(est):
        if hasattr(est, "feature_importances_"):
            return est.feature_importances_, "Aggregated feature importance (max probe importance)"
        if hasattr(est, "coef_"):
            coef = est.coef_
            vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef[0])
            return vals, "Aggregated |Coefficient| (max probe)"
        return None, None

    if hasattr(m, "named_steps"):
        inner = list(m.named_steps.values())[-1]
        importances, importance_label = _get_importances(inner)
    else:
        importances, importance_label = _get_importances(m)

    if importances is None:
        print(f"plot_gene_importance_aggregated: {model_name} has no importances — skipping")
        return

    # Build probe → importance series, map to gene symbol, group by max
    probe_imp   = pd.Series(importances, index=probe_cols)
    gene_series = probe_imp.index.map(gene_map).fillna("---")
    agg = (
        pd.DataFrame({"importance": probe_imp.values, "gene": gene_series})
        .groupby("gene")["importance"]
        .max()
        .sort_values(ascending=False)
    )
    agg = agg[agg.index != "---"].head(top_n)

    if agg.empty:
        print("plot_gene_importance_aggregated: no gene-mapped probes — skipping")
        return

    agg_vals   = agg.values
    agg_labels = list(agg.index)
    colours    = ["#d62728" if i == 0 else "#1f77b4" for i in range(len(agg_vals))]

    def _draw_gene_bars(ax, vals, labels, title):
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals[::-1], color=colours[::-1], edgecolor="white", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.set_xlabel(importance_label, fontsize=10)
        ax.set_ylabel("Gene Symbol", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    # Full-scale plot
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_gene_bars(ax, agg_vals, agg_labels,
                    f"Top {top_n} Genes by Maximum Probe Importance — {model_name}\n{dataset}")
    plt.tight_layout()
    out_path = os.path.join(plots_dir, "gene_importance_aggregated.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gene importance aggregated plot saved: {out_path}")

    # Zoomed plot — clip at 1.5× second-highest gene so the outlier doesn't compress the rest
    if len(agg_vals) >= 2 and agg_vals[1] > 1e-8:
        zoom_xlim = agg_vals[1] * 1.5
        fig, ax = plt.subplots(figsize=(9, 7))
        _draw_gene_bars(ax, agg_vals, agg_labels,
                        f"Top {top_n} Genes by Maximum Probe Importance — {model_name} (zoomed — top outlier truncated)\n{dataset}")
        ax.set_xlim(0, zoom_xlim)
        # y=0 is the top bar after barh reversal
        outlier_label = f"{agg_vals[0]:.3f}"
        ax.text(
            zoom_xlim * 0.98, 0,
            f"← {agg_labels[0]}  ({outlier_label})",
            va="center", ha="right", fontsize=8, color="#d62728",
        )
        plt.tight_layout()
        zoom_path = os.path.join(plots_dir, "gene_importance_aggregated_zoomed.png")
        plt.savefig(zoom_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gene importance aggregated zoomed plot saved: {zoom_path}")


def plot_biomarker_summary_composite(
    plots_dir:     str,
    shortlist_csv: str,
    output_path:   str,
    dataset:       str = "",
) -> None:
    """
    3-panel biomarker summary composite figure.

    Panel A — Probe-level feature importance  (loads feature_importance.png)
    Panel B — Gene-aggregated importance      (loads gene_importance_aggregated.png)
    Panel C — Shortlist scatter: abs_fold_change vs combined_score

    Saved to output_path (suggested: fig_biomarker_summary_composite.png).
    """
    import matplotlib.image as mpimg

    # --- Shortlist data for Panel C ---
    shortlist_df = None
    if os.path.exists(shortlist_csv):
        try:
            shortlist_df = pd.read_csv(shortlist_csv)
        except Exception as e:
            print(f"  plot_biomarker_summary_composite: could not read shortlist — {e}")

    # --- PNG panels A and B ---
    # TODO: replace PNG loading with direct plotting functions for full resolution control
    png_panels = [
        (os.path.join(plots_dir, "feature_importance.png"),        "A", "Probe-Level Feature Importance"),
        (os.path.join(plots_dir, "gene_importance_aggregated.png"), "B", "Gene-Aggregated Importance"),
    ]
    loaded = [(p, lbl, ttl) for p, lbl, ttl in png_panels if os.path.exists(p)]

    n_panels = len(loaded) + (1 if shortlist_df is not None else 0)
    if n_panels == 0:
        print("  plot_biomarker_summary_composite: no panels available — skipping")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    for idx, (fpath, label, title) in enumerate(loaded):
        ax = axes[idx]
        try:
            img = mpimg.imread(fpath)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"[Error: {os.path.basename(fpath)}]",
                    ha="center", va="center", fontsize=9, color="red", transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"{label}  {title}", fontsize=10, fontweight="bold", pad=6)

    # Panel C — inline scatter
    if shortlist_df is not None:
        ax = axes[len(loaded)]
        req = {"probe_id", "gene_symbol", "abs_fold_change", "combined_score"}
        if req.issubset(shortlist_df.columns):
            # Sort so top combined_score points render last (most visible)
            shortlist_df = shortlist_df.sort_values("combined_score", ascending=True)

            x    = shortlist_df["abs_fold_change"].values
            y_sc = shortlist_df["combined_score"].values
            sz   = (
                shortlist_df["rf_importance"].values * 800 + 30
                if "rf_importance" in shortlist_df.columns else 60
            )
            c = (
                shortlist_df["selection_freq"].values
                if "selection_freq" in shortlist_df.columns else "#1f77b4"
            )
            sc = ax.scatter(x, y_sc, s=sz, c=c, cmap="Blues", alpha=0.85,
                            edgecolors="#333", linewidths=0.5)
            if "selection_freq" in shortlist_df.columns:
                plt.colorbar(sc, ax=ax, label="Selection frequency", shrink=0.8)

            # Use cleaned gene name consistently — handle "GENE /// ALIAS" format
            clean_genes  = shortlist_df["gene_symbol"].apply(lambda x: str(x).split("///")[0].strip())
            gene_counts  = clean_genes.value_counts()
            for (_, row), gene in zip(shortlist_df.iterrows(), clean_genes):
                lbl = f"{gene} ({row['probe_id']})" if gene_counts.get(gene, 0) > 1 else gene
                ax.annotate(lbl, (row["abs_fold_change"], row["combined_score"]),
                            fontsize=7, xytext=(4, 2), textcoords="offset points")

            ax.set_xlabel("|Log2 Fold Change| (SONFH − control)", fontsize=10)
            ax.set_ylabel("Combined Score", fontsize=10)
            ax.set_title("C  Shortlist Probes: Fold Change vs Combined Score",
                         fontsize=10, fontweight="bold", pad=6)
            ax.grid(True, alpha=0.3, linestyle="--")
        else:
            ax.text(0.5, 0.5, "Missing columns in shortlist CSV",
                    ha="center", va="center", fontsize=9, transform=ax.transAxes)
            ax.axis("off")
            ax.set_title("C  Shortlist Probes: Fold Change vs Combined Score",
                         fontsize=10, fontweight="bold")

    if dataset:
        fig.suptitle(f"Biomarker Summary — {dataset}", fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Biomarker summary composite saved: {output_path}")


# ---------------------------------------------------------------------------
# WEKA MODEL RESULTS BAR CHART
# ---------------------------------------------------------------------------
def plot_weka_model_results(weka_models_dir: str, output_path: str) -> None:
    """
    Parse all Weka classifier result .txt files, extract Accuracy / F1 /
    ROC AUC from the weighted-average summary line, and produce a grouped
    horizontal bar chart sorted by ROC AUC.

    Expected Weka 3.8 output format (numbers may use ',' or '.' as decimal):
      Correctly Classified Instances  N  X.X %
      Weighted Avg.  TP  FP  Prec  Recall  F-Measure  MCC  ROC  PRC
    """
    import re
    import glob

    _LABELS = {
        "auto_weka":                      "Auto-Weka (PART)",
        "j48_tree":                       "J48",
        "randomforest":                   "Random Forest",
        "naive_bayes":                    "Naive Bayes",
        "multilayerpreceptron":           "MLP",
        "functions_smo":                  "SMO (SVM)",
        "lazy_ibk":                       "IBk (k=1)",
        "lazy_ibk_knn_3":                 "IBk (k=3)",
        "lazy_ibk_knn_5":                 "IBk (k=5)",
        "attribute_selection_randomforest": None,   # no classifier metrics — skip
    }

    results = []
    for fpath in sorted(glob.glob(os.path.join(weka_models_dir, "*.txt"))):
        stem  = os.path.splitext(os.path.basename(fpath))[0]
        label = _LABELS.get(stem)
        if label is None:
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()

        acc_m = re.search(
            r"Correctly Classified Instances\s+\d+\s+([\d.]+)\s+%", text
        )
        wt_m  = re.search(r"Weighted Avg\.\s+(.*)", text)

        if not (acc_m and wt_m):
            print(f"  Skipping {stem} — metrics not found in file")
            continue

        acc   = float(acc_m.group(1)) / 100.0
        parts = wt_m.group(1).split()
        nums  = []
        for p in parts:
            try:
                nums.append(float(p.replace(",", ".")))
            except ValueError:
                pass

        # Weighted Avg columns: TP, FP, Precision, Recall, F-Measure, MCC, ROC, PRC
        if len(nums) < 7:
            print(f"  Skipping {stem} — could not parse all weighted-avg columns")
            continue

        f1  = nums[4]   # F-Measure
        roc = nums[6]   # ROC Area
        results.append({"model": label, "Accuracy": acc, "F1": f1, "ROC AUC": roc})

    if not results:
        print(f"  No parseable Weka result files found in {weka_models_dir}")
        return

    df = (
        pd.DataFrame(results)
        .sort_values("ROC AUC", ascending=True)
        .reset_index(drop=True)
    )

    metrics = ["Accuracy", "F1", "ROC AUC"]
    colours = ["#2c5f8a", "#e8a838", "#c0392b"]
    bar_h   = 0.24
    y       = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.8)))

    for i, (metric, col) in enumerate(zip(metrics, colours)):
        offset = (i - 1) * bar_h
        bars = ax.barh(
            y + offset, df[metric], bar_h,
            label=metric, color=col, alpha=0.85, edgecolor="white",
        )
        for bar, val in zip(bars, df[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7.5, color="#333")

    # Highlight best model row
    best_idx = df["ROC AUC"].idxmax()
    ax.axhspan(best_idx - bar_h * 2.0, best_idx + bar_h * 2.0,
               color="#d4edda", alpha=0.45, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(
        "Weka Classifier Performance — 10-fold Cross-Validation\n"
        "GSE123568 | Top 100 Features | 40 Samples (30 SONFH / 10 Control)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1.14)
    ax.axvline(0.9, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Weka model results chart saved: {output_path}")


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame, ranking: pd.Series, method: str) -> None:
    _method_labels = {
        "fc":     "|fold change|",
        "var":    "variance",
        "hybrid": "hybrid score (|FC| + |t-stat|, z-scored)",
    }
    print("\n" + "=" * 60)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 60)
    print(f"Ranking method    : {_method_labels.get(method, method)}")
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
# Figure 2 — Model comparison bar chart
# ---------------------------------------------------------------------------
def plot_model_comparison_bar(
    models_csv: str,
    plots_dir: str,
    dataset: str = "",
    mode_suffix: str = "",
) -> None:
    """
    Horizontal bar chart of CV ROC-AUC (mean ± std) for all trained models,
    sorted best → worst (best at top). The selected model (tuned_xgboost) is
    highlighted in coral with a black edge. Saves fig_2_model_comparison.png.
    """
    _MODEL_LABELS = {
        "tuned_xgboost":                "Tuned XGBoost ★",
        "baseline_xgboost":             "XGBoost (default)",
        "tuned_random_forest":          "Tuned Random Forest",
        "baseline_random_forest":       "Random Forest (default)",
        "baseline_logistic_elasticnet": "Logistic ElasticNet",
        "baseline_linear_svc":          "Linear SVC",
        "baseline_mlp":                 "MLP",
        "baseline_gaussian_nb":         "Gaussian NB",
        "baseline_knn":                 "KNN",
    }

    try:
        df = pd.read_csv(models_csv)
    except FileNotFoundError:
        print(f"  plot_model_comparison_bar: {models_csv} not found — skipping.")
        return

    df         = df.sort_values("roc_auc_mean", ascending=True)  # ascending → best at top after invert
    labels     = [_MODEL_LABELS.get(m, m.replace("_", " ").title()) for m in df["model"]]
    auc_vals   = df["roc_auc_mean"].values
    auc_errs   = df["roc_auc_std"].values
    colors     = ["#E84B23" if m == "tuned_xgboost" else "#5B8DB8" for m in df["model"]]
    edgecolors = ["black"   if m == "tuned_xgboost" else "none"    for m in df["model"]]
    linewidths = [1.2       if m == "tuned_xgboost" else 0         for m in df["model"]]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        labels, auc_vals, xerr=auc_errs, color=colors,
        edgecolor=edgecolors, linewidth=linewidths,
        error_kw=dict(ecolor="#444", capsize=3, linewidth=1),
        height=0.6, zorder=2,
    )

    for bar, val in zip(bars, auc_vals):
        ax.text(
            val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8.5,
        )

    x_left  = max(0.80, auc_vals.min() - 0.02)
    x_right = auc_vals.max() + 0.08   # room for value annotations
    ax.set_xlim(x_left, x_right)

    ax.axvline(0.95, color="#999", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.951, 0, "0.95", color="#888", fontsize=7.5, va="bottom")

    ax.invert_yaxis()  # best model at top
    ax.set_xlabel("ROC AUC (CV mean ± std)", fontsize=10)
    title = "Model Comparison — ROC AUC"
    if dataset:
        title += f"\n{dataset}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(plots_dir, f"fig_2_model_comparison{mode_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Final model evaluation composite (ROC | CM | Feature Importance)
# ---------------------------------------------------------------------------
def plot_composite_eval(
    plots_dir: str,
    output_path: str,
    dataset: str = "",
) -> None:
    """
    Compose a 1×3 multi-panel evaluation figure from the three ML eval PNGs.
    Uses matplotlib.figure.subfigures. Panels labelled A–C with subtitles.

    Panel layout:
      A: ROC Curves    B: Confusion Matrix    C: Feature Importance
    """
    import matplotlib.image as mpimg

    PANEL_FILES = [
        ("roc_curves.png",         "A", "ROC Curves"),
        ("confusion_matrix.png",   "B", "Confusion Matrix"),
        ("feature_importance.png", "C", "Feature Importance"),
    ]

    panels = []
    for fname, label, title in PANEL_FILES:
        fpath = os.path.join(plots_dir, fname)
        if os.path.exists(fpath):
            panels.append((fpath, label, title))
        else:
            print(f"  Warning: {fname} not found in {plots_dir} — skipping panel {label}")

    if not panels:
        print(f"  No eval plots found in {plots_dir} — cannot generate composite.")
        return

    fig = plt.figure(figsize=(16, 5.5))
    if dataset:
        fig.suptitle(
            f"Final Model Evaluation — {dataset}",
            fontsize=13, fontweight="bold", y=1.005,
        )

    subfigs = fig.subfigures(1, len(panels), wspace=0.015)
    flat    = subfigs.flatten() if len(panels) > 1 else [subfigs]

    for idx, (fpath, label, title) in enumerate(panels):
        sf  = flat[idx]
        ax  = sf.add_subplot(1, 1, 1)
        try:
            img = mpimg.imread(fpath)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"[Error loading {os.path.basename(fpath)}]",
                    ha="center", va="center", fontsize=9, color="red",
                    transform=ax.transAxes)
        ax.axis("off")
        sf.suptitle(title, fontsize=10)
        sf.text(0.01, 0.98, label, fontsize=14, fontweight="bold",
                va="top", ha="left", transform=sf.transSubfigure)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model evaluation composite saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Statistical vs model importance scatter
# ---------------------------------------------------------------------------
def plot_statistical_vs_model_importance(
    X: np.ndarray,
    y: np.ndarray,
    probe_cols: list,
    model,
    gene_map: pd.Series,
    top_genes_csv: str,
    plots_dir: str,
    dataset: str = "",
    random_state: int = 42,
    label_top_n: int = 20,
    mode_suffix: str = "",
) -> None:
    """
    Scatter plot reconciling statistical signal (hybrid score) with model
    importance (XGBoost gain) for every gene in top_genes_csv.

    Axes:
      x = XGBoost feature importance (gain, full-data fit) — log scale
      y = hybrid score (statistical ranking from feature selection)

    Encoding:
      color = probe_type  (_a_at, _s_at, _at, _x_at)
      size  = abs_fold_change

    Small jitter is added to both axes to prevent point stacking.

    Quadrant lines at the median of each axis divide four regions:
      top-right    → high stat + high model  (strongest candidates)
      top-left     → high stat, low model    (biologically relevant but
                     correlated/redundant with top drivers; e.g. BPGM)
      bottom-right → low stat, high model    (model-discovered signal; e.g. TSTA3)
      bottom-left  → low both               (weaker candidates)

    Saves fig_4_stat_vs_model_importance.png to plots_dir.
    """
    import seaborn as sns
    from sklearn.base import clone

    try:
        genes_df = pd.read_csv(top_genes_csv)
    except FileNotFoundError:
        print(f"  plot_statistical_vs_model_importance: {top_genes_csv} not found — skipping.")
        return

    # Fit on full data for stable importance estimates (mirrors plot_feature_importance)
    m = clone(model)
    m.fit(X, y)
    if not hasattr(m, "feature_importances_"):
        print("  plot_statistical_vs_model_importance: model has no feature_importances_ — skipping.")
        return

    imp_series = pd.Series(m.feature_importances_, index=probe_cols, name="importance")

    # Merge: one row per gene; importance = 0 for genes not selected by the model
    df = genes_df.copy()
    df["importance"] = df["probe_id"].map(imp_series).fillna(0.0)

    # Jitter to prevent point stacking (seeded for reproducibility)
    rng = np.random.default_rng(random_state)
    df["importance_j"] = df["importance"] + rng.normal(0, 0.0015, len(df))
    df["hybrid_j"]     = df["hybrid_score"] + rng.normal(0, 0.05,   len(df))
    # Clamp jittered importance to positive values (log scale requires > 0)
    df["importance_j"] = df["importance_j"].clip(lower=1e-5)

    _PROBE_LABELS = {
        "_a_at": "antisense (_a_at)",
        "_s_at": "shared (_s_at)",
        "_at":   "standard (_at)",
        "_x_at": "cross-hyb (_x_at)",
    }
    df["probe_label"] = df["probe_type"].map(_PROBE_LABELS).fillna(df["probe_type"])

    palette = {
        "antisense (_a_at)": "#1f77b4",
        "shared (_s_at)":    "#ff7f0e",
        "standard (_at)":    "#2ca02c",
        "cross-hyb (_x_at)": "#9467bd",
    }
    present_palette = {k: v for k, v in palette.items() if k in df["probe_label"].values}

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.scatterplot(
        data=df,
        x="importance_j",
        y="hybrid_j",
        hue="probe_label",
        size="abs_fold_change",
        sizes=(40, 200),
        alpha=0.75,
        palette=present_palette,
        ax=ax,
    )

    ax.set_xscale("log")

    # Quadrant lines at medians of the original (un-jittered) values
    x_med = df["importance"].median()
    y_med = df["hybrid_score"].median()
    ax.axvline(x_med if x_med > 0 else 1e-5, linestyle="--", color="#888", linewidth=0.9, zorder=0)
    ax.axhline(y_med, linestyle="--", color="#888", linewidth=0.9, zorder=0)

    # Label top N by hybrid score + top N by model importance (N = biomarker.top_n_display)
    top_stat  = set(df.nlargest(label_top_n, "hybrid_score")["gene_symbol"])
    top_model = set(df.nlargest(label_top_n, "importance")["gene_symbol"])
    to_label  = top_stat | top_model

    for _, row in df.iterrows():
        if row["gene_symbol"] in to_label:
            ax.text(
                row["importance_j"],
                row["hybrid_j"],
                row["gene_symbol"],
                fontsize=8,
                ha="left",
                va="bottom",
            )

    ax.set_xlabel("Model Importance (XGBoost Gain, log scale)", fontsize=10)
    ax.set_ylabel("Hybrid Score (Statistical Signal)", fontsize=10)
    mode_label = mode_suffix.replace("_", "  ").strip() if mode_suffix else ""
    title_line2 = f"\nGSE123568  {mode_label}" if mode_label else "\nGSE123568"
    ax.set_title(
        f"BioStatistical vs Model Importance of Candidate Biomarkers{title_line2}",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(0.98, 0.02, f"n = {len(df)} genes",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555")
    ax.legend(
        title="Probe type  /  size = |FC|",
        fontsize=8, title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        framealpha=0.7,
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    out_path = os.path.join(plots_dir, f"fig_4_stat_vs_model_importance{mode_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Statistical vs model importance plot saved: {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Select top differentially expressed probes for Weka")
    p.add_argument("--top",    type=int,   default=100,  help="Number of top probes to select (default: 100)")
    p.add_argument("--method", type=str,   default="fc", choices=["fc", "var"],
                   help="Ranking method: fc=|fold change| (default), var=variance")
    p.add_argument("--input",  type=str,   default=_WEKA_INPUT_CSV,    help="Path to preprocessed_matrix.csv")
    p.add_argument("--outdir", type=str,   default=str(_PROJECT_ROOT / "data" / "femoral_head_necrosis"),
                   help="Root output directory. EDA plots → <outdir>/EDA/  |  CSVs/ARFF → <outdir>/feature_selection/")
    return p.parse_args()


if __name__ == "__main__":
    # Weka Graph Genration as Feature Select Util Unit test
    args = parse_args()

    eda_dir    = _WEKA_EDA_DIR
    csv_outdir = _WEKA_OUT_DIR
    os.makedirs(csv_outdir, exist_ok=True)
    os.makedirs(eda_dir,    exist_ok=True)

    # 1. Load
    df = load_preprocessed(args.input)

    # 2. Compute rankings
    #    fc_ranking / var_ranking kept as Series for EDA plots that expect them.
    #    hybrid_df is the new primary ranking for probe selection.
    fc_ranking  = rank_by_fold_change(df).rename("abs_log_fold_change")
    var_ranking = rank_by_variance(df).rename("variance")
    hybrid_df   = rank_by_hybrid_score(df)

    # 3. Select top N probes by hybrid score — Weka reproducibility branch
    top_probes  = hybrid_df.head(args.top).index.tolist()
    selected_df = df[top_probes + ["class"]].copy()
    print(f"\nTop {args.top} probes selected by hybrid score (|FC| + |t-stat|)")

    # 3b. Top 500 — Python discovery branch (always generated alongside Weka top-N)
    _n500         = max(args.top, 500)
    top500_probes = hybrid_df.head(_n500).index.tolist()
    top500_df     = df[top500_probes + ["class"]].copy()

    # 4. Save Weka top-N CSV  (top100_features.csv with default --top 100)
    csv_path = os.path.join(csv_outdir, f"top{args.top}_features.csv")
    selected_df.to_csv(csv_path)
    print(f"Selected features CSV: {csv_path}")

    # 4b. Save top 500 CSV — discovery branch
    if _n500 > args.top:
        top500_csv = os.path.join(csv_outdir, "top500_features.csv")
        top500_df.to_csv(top500_csv)
        print(f"Top 500 discovery CSV: {top500_csv}")

    # 5. Save full probe rankings (gene_rankings.csv) with all hybrid metrics
    rankings_path = os.path.join(csv_outdir, "gene_rankings.csv")
    annotation    = load_probe_annotation(SOFT_GZ)

    ranked = hybrid_df.copy()
    ranked.index.name = "probe_id"
    ranked.insert(0, "probe_rank", range(1, len(ranked) + 1))
    ranked["gene_symbol"] = ranked.index.map(annotation).fillna("---") if len(annotation) else "---"
    _col_order = ["probe_rank", "gene_symbol", "hybrid_score", "abs_fold_change",
                  "log_fold_change", "t_stat", "p_value", "iqr", "variance",
                  "mean_sonfh", "mean_control", "probe_type"]
    ranked[[c for c in _col_order if c in ranked.columns]].to_csv(rankings_path)
    print(f"Full probe rankings  : {rankings_path}  ({len(ranked)} probes, all metrics)")

    # 6. Save ARFF — Weka reproducibility branch
    arff_path = os.path.join(csv_outdir, f"top{args.top}_features.arff")
    write_arff(selected_df, relation_name="femoral_head_necrosis", path=arff_path)

    # 7. EDA — Fold change bar chart (top 20 by |FC| — keeps existing plot behaviour)
    bar_path = os.path.join(eda_dir, "fold_change_top20.png")
    plot_fold_change_bar(fc_ranking, bar_path, top_n=20, gene_map=annotation)

    # 8. EDA — PCA plot (sample separation in top-N-probe space)
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

    # 12. EDA — Volcano-style plot (FC vs variance, hybrid-selected probes highlighted)
    top_probe_ids = selected_df.columns[:-1].tolist()
    volcano_path  = os.path.join(eda_dir, "volcano_plot.png")
    plot_volcano(fc_ranking, var_ranking, top_probe_ids, annotation, volcano_path)

    # 13. Gene-level summary — post-selection probe→gene grouping (existing output)
    gene_summary_path = os.path.join(csv_outdir, "gene_level_summary.csv")
    build_gene_level_summary(selected_df, fc_ranking, annotation, gene_summary_path)

    # 13b. Gene-level deduped rankings — one best probe per gene (NEW)
    gene_level_path = os.path.join(csv_outdir, "gene_level_rankings.csv")
    top_genes_path  = os.path.join(csv_outdir, "top100_genes.csv")
    build_gene_level_dedup(hybrid_df, annotation, gene_level_path, top_genes_path, top_n=100)

    # 14. Summary
    print_summary(selected_df, hybrid_df["hybrid_score"], "hybrid")

    print(f"\nWeka files in  : {csv_outdir}/")
    print(f"EDA plots in   : {eda_dir}/")

    # 15. EDA composite multi-panel figure (2×3 grid, panels A–F)
    composite_path = os.path.join(eda_dir, "eda_composite.png")
    plot_composite_eda(eda_dir, composite_path, dataset="GSE123568 — top 100 probes (Weka pipeline)")
