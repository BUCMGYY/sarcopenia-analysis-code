from __future__ import annotations

"""Main-only Figure 2–5 rebuild for the GSE213148 mouse muscle model.

This script deliberately excludes GSE223813, GSE308816 and all other external
reference cohorts from the plotted panels.  The only plotted biological samples
are the 16-week GSE213148 libraries; downstream panels are deterministic
computations from that cohort (counterfactual sensitivity and L1000FWD
signature-reversal ranking).  External databases are not treated as cohorts.
"""

from pathlib import Path
import os
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t as student_t
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image
from sklearn.decomposition import PCA

_project_root = os.environ.get("SARCOPENIA_PROJECT_ROOT")
if not _project_root:
    raise SystemExit("Set SARCOPENIA_PROJECT_ROOT to the local project-data directory.")
ROOT = Path(_project_root).expanduser().resolve()
DATA = ROOT / "sarcopenia_disease" / "GSE213148"
RES = DATA / "results"
REAN = DATA / "reanalysis_20260928" / "results"
OUT = ROOT / "figures_main_only_20260929"
OUT.mkdir(parents=True, exist_ok=True)
PATHWAY_DIR = ROOT / "pathway_corrected_20260929"

# Publication-style defaults; SVG text remains editable.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 7.3,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.2,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

INK = "#1F2937"
MUTED = "#64748B"
GRID = "#D8DEE8"
CTRL = "#4C78A8"
CR = "#5B8DB8"
IR = "#D99A36"
MODEL = "#C85454"
TEAL = "#2C8C83"
PURPLE = "#7657A6"
GOLD = "#B77C1C"
LIGHT = "#F3F6F9"
MID = "#AAB4C2"
UP = "#C85454"
DOWN = "#4C78A8"
# Panel c uses direction-specific FDR gradients. Upregulated terms are always
# red (pale red to deep red) and downregulated terms are always blue (pale blue
# to deep blue); within each direction, colour intensity encodes -log10(FDR).
# This avoids the misleading situation in which a highly significant down term
# becomes visually red merely because it has a large -log10(FDR) value.
FDR_UP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "fdr_up", ["#FBE3E3", UP], N=256
)
FDR_DOWN_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "fdr_down", ["#E3EDF8", DOWN], N=256
)

GROUP_ORDER = ["Ctrl", "CR", "IR", "CR+IR"]
GROUP_LABELS = {"Ctrl": "Control", "CR": "CR", "IR": "IR", "CR+IR": "Sarcopenia"}
GROUP_COLORS = {"Ctrl": CTRL, "CR": CR, "IR": IR, "CR+IR": MODEL}

# Figure 2 layout-only revision. Analysis values and panel membership unchanged.
FIGURE2_LAYOUT = {
    # Slightly taller canvas keeps the three rows visually balanced after
    # opening the gap between the volcano/signature row and the heatmap.
    "figsize": (7.2, 7.3),
    "height_ratios": [1.25, 1.30, 1.65],
    "hspace": 0.40,
    # Extra horizontal gutter keeps wrapped pathway labels and the panel-b
    # side colourbar separate without shrinking the artwork into a dashboard.
    "wspace": 0.82,
    "left": 0.17,
    "right": 0.97,
    "bottom": 0.12,
    "top": 0.975,
    "box_width": 0.46,
    "whisker_iqr": 1.5,
}

# Figure 3 layout-only revision. The main figure now follows the analytical
# order used in the text: disease signature → genes → enrichment → pathway
# effects. The gene-level interaction test remains in the exported source
# table, but is not given a separate main-panel slot because it duplicated the
# same factorial question at a less interpretable level.
FIGURE3_LAYOUT = {
    # Dedicated label/colourbar axes keep the heatmap and enrichment labels
    # separate. GSVA profiles occupy six small multiples below the top row.
    "figsize": (180 / 25.4, 7.4),
    "a_bounds": [0.075, 0.575, 0.235, 0.345],
    # Give the gene heatmap more visual weight.  The previous version left
    # an unnecessarily wide visual gap before the enrichment labels.  Extend
    # the heatmap by roughly one fifth while retaining a dedicated label
    # column for panel c (the labels remain right aligned and unclipped).
    "b_bounds": [0.395, 0.575, 0.245, 0.345],
    # Match panel c's horizontal colourbar baseline rather than placing a
    # narrow vertical bar beside the heatmap.
    # Center the Row z bar under the widened heatmap (b center = 0.5175),
    # so its title and the Experimental group label share the same midpoint.
    "b_colorbar_bounds": [0.4625, 0.505, 0.110, 0.009],
    "c_label_bounds": [0.650, 0.575, 0.155, 0.345],
    "c_bounds": [0.815, 0.575, 0.150, 0.345],
    "c_colorbar_bounds": [0.835, 0.505, 0.110, 0.009],
    "d_bounds": [0.075, 0.075, 0.890, 0.390],
    "heading_y": 0.955,
    "c_heading_x": 0.715,
    "d_heading_y": 0.470,
    "profile_column_starts": [0.075, 0.4025, 0.730],
    "profile_row_starts": [0.285, 0.075],
    "profile_width": 0.235,
    "profile_height": 0.140,
    "profile_ylim": [-0.85, 0.85],
    "profile_title_size": 6.8,
    "profile_point_size": 10,
    "profile_jitter_step": 0.022,
}


def pretty_gene(value: str) -> str:
    """Keep unresolved Ensembl IDs readable in dense heatmaps."""
    value = str(value)
    if value.startswith("ENSMUSG") and len(value) > 12:
        return value[:8] + "…" + value[-6:]
    return value


def compact_pathway_name(value: str, width: int = 22) -> str:
    """Shorten database suffixes and long pathway names for panel labels."""
    text = str(value)
    for suffix in (
        " - Mus musculus (house mouse)",
        " Mus musculus (house mouse)",
        " (house mouse)",
    ):
        text = text.replace(suffix, "")
    # Pathway evidence labels may carry a database identifier before a pipe;
    # the identifier is already represented by the source line in panel d.
    if "|" in text:
        text = text.split("|", 1)[1].strip()
    replacements = {
        "Nonsense Mediated Decay (NMD) independent of the Exon Junction Complex (EJC)":
            "NMD-independent exon-junction decay",
        "Formation of a pool of free 40S subunits":
            "Free 40S subunit formation",
        "Collagen biosynthesis and modifying enzymes":
            "Collagen biosynthesis/modifying enzymes",
        "Protein digestion and absorption":
            "Protein digestion/absorption",
    }
    text = replacements.get(text, text)
    return textwrap.fill(text, width=width, break_long_words=False,
                         break_on_hyphens=False)


def compact_convergence_term(value: str) -> str:
    """Use short, reader-facing names for the six-row convergence matrix."""
    text = str(value)
    if "|" in text:
        text = text.split("|", 1)[1].strip()
    text = text.replace(" - Mus musculus (house mouse)", "")
    lower = text.lower()
    if "positive regulation of cell-cell adhesion" in lower:
        return "cell adhesion (+)"
    if "negative regulation of cell adhesion" in lower:
        return "cell adhesion (−)"
    if "rna degradation" in lower:
        return "RNA degradation"
    if "protein digestion" in lower:
        return "protein digestion"
    if "collagen biosynthesis" in lower:
        return "collagen biosynthesis"
    if "collagen formation" in lower:
        return "collagen formation"
    return textwrap.fill(text, width=20, break_long_words=False,
                         break_on_hyphens=False)


def save_pub(fig: plt.Figure, stem: str, fixed_canvas: bool = False) -> None:
    """Export editable vectors, 600 dpi PNG, and opaque RGB TIFF."""
    bbox = None if fixed_canvas else "tight"
    fig.savefig(OUT / f"{stem}.svg", bbox_inches=bbox, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches=bbox, facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches=bbox, facecolor="white")
    # The white canvas is already opaque; mode conversion removes the unused
    # alpha channel without changing pixels or upsampling the rendered image.
    with Image.open(OUT / f"{stem}.png") as raster:
        raster.convert("RGB").save(
            OUT / f"{stem}.tiff", format="TIFF", compression="tiff_lzw",
            dpi=(600, 600),
        )
    plt.close(fig)


def panel_label(ax, label: str, y: float = 1.08, x: float = -0.12) -> None:
    title = getattr(ax, "_panel_title", None)
    heading = f"{label}  {title}" if title else label
    # Nature-style panel heading: one unified 8-pt line containing the panel
    # letter and subtitle (for example, “a  Library-level QC”).
    ax.text(x, y, heading, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.0 if title else 10, fontweight="bold", color=INK)


def clean_ax(ax):
    ax.grid(axis="y", color=GRID, lw=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    return ax


def load_data():
    meta = pd.read_csv(RES / "GSE213148_sample_metadata_audited.csv")
    expr = pd.read_csv(RES / "GSE213148_log2CPM_matrix.csv")
    expr = expr.set_index("gene_id")
    de = pd.read_csv(REAN / "GSE213148_primary_DESeq2_CRIR_vs_Ctrl.csv")
    ann = pd.read_csv(RES / "GSE213148_CRIR_vs_Ctrl_DEG_annotated.csv",
                      usecols=["gene_id", "gene_name"])
    ann = ann.drop_duplicates("gene_id")
    # Fill display-only gaps with the audited official MGI Ensembl map; source
    # effect sizes and identifiers remain the DESeq2 table keys.
    mgi_map_path = ROOT / "pathway_corrected_20260929" / "gene_sets" / "MGI_MRK_ENSEMBL.rpt"
    if mgi_map_path.exists():
        mgi = pd.read_csv(mgi_map_path, sep="\t", header=None, usecols=[1, 5],
                          names=["mgi_symbol", "gene_id"], dtype=str)
        mgi = mgi.dropna(subset=["gene_id", "mgi_symbol"]).drop_duplicates("gene_id")
        ann = ann.merge(mgi, on="gene_id", how="left")
        ann["gene_name"] = ann["gene_name"].fillna(ann["mgi_symbol"])
        ann = ann[["gene_id", "gene_name"]]
    de = de.merge(ann, on="gene_id", how="left")
    interaction = pd.read_csv(REAN / "GSE213148_factorial_CRxIR_interaction_DESeq2.csv")
    cf = pd.read_csv(REAN / "GSE213148_DESeq2_bulk_counterfactual_sensitivity_top50.csv")
    enrich = pd.read_csv(RES / "GSE213148_enrichment_combined.csv")
    counts = pd.read_csv(REAN / "GSE213148_primary_counts_samples_by_genes_filtered.tsv", sep="\t")
    drug = pd.read_csv(ROOT / "figure_v2_data" / "figure6_homologene_unique_drug_rank.csv")
    return meta, expr, de, ann, interaction, cf, enrich, counts, drug


def primary_tables(meta, expr, de, ann, counts):
    # The disease model is the 16-week four-arm experiment.  The primary
    # contrast uses only 16-week Control and CR+IR libraries.
    m16 = meta[meta["source"].astype(str).str.startswith("16w_")].copy()
    cols16 = [c for c in m16["gsm"] if c in expr.columns]
    m16 = m16[m16["gsm"].isin(cols16)].copy()
    primary = m16[m16["group"].isin(["Ctrl", "CR+IR"])].copy()
    pcols = [c for c in primary["gsm"] if c in expr.columns]

    # Gene names are used only for display; IDs remain the source keys.
    gene_map = ann.drop_duplicates("gene_id").set_index("gene_id")["gene_name"].to_dict()
    expr16 = expr[cols16].copy()
    expr16.index = [
        (str(gene_map.get(g)) if pd.notna(gene_map.get(g)) else str(g))
        for g in expr16.index
    ]
    # Remove duplicate display symbols without altering the source matrix.
    expr16 = expr16[~expr16.index.duplicated(keep="first")]

    # Fixed, data-independent display rules: top 50 up and top 50 down by padj
    # among the primary DESeq2 result, then score every 16-week sample.
    de_named = de.copy()
    de_named["display_name"] = de_named["gene_name"].fillna(de_named["gene_id"])
    sig = de_named[(de_named["padj"] < 0.05) & (de_named["log2FoldChange"].abs() >= 0.5)]
    up = sig[sig["log2FoldChange"] > 0].sort_values(["padj", "gene_id"]).head(100)["display_name"].tolist()
    down = sig[sig["log2FoldChange"] < 0].sort_values(["padj", "gene_id"]).head(100)["display_name"].tolist()
    up = [g for g in up if g in expr16.index]
    down = [g for g in down if g in expr16.index]
    # Gene-wise z-score across all four 16-week groups avoids group-specific
    # baseline selection and gives one score scale for every panel.
    z = expr16.sub(expr16.mean(axis=1), axis=0).div(expr16.std(axis=1).replace(0, 1), axis=0)
    score = pd.DataFrame(index=cols16)
    score["disease_signature"] = z.loc[up, cols16].mean(axis=0) - z.loc[down, cols16].mean(axis=0)
    score["up_component"] = z.loc[up, cols16].mean(axis=0)
    score["down_component"] = z.loc[down, cols16].mean(axis=0)
    score = score.reset_index(names="gsm").merge(m16[["gsm", "group", "mouse id"]], on="gsm", how="left")

    # Primary count-based QC.
    count_cols = [c for c in counts.columns if c != "sample"]
    qc = counts.copy()
    qc["library_size"] = qc[count_cols].sum(axis=1)
    qc["detected_genes"] = (qc[count_cols] > 0).sum(axis=1)
    qc = qc[["sample", "library_size", "detected_genes"]].rename(columns={"sample": "gsm"})
    qc = qc.merge(primary[["gsm", "group", "mouse id"]], on="gsm", how="left")
    return m16, primary, expr16, de_named, score, qc, up, down


def plot_design(ax, n=12):
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis("off")
    ax.text(0.00, 1.08, "16-week experimental design", transform=ax.transAxes,
            fontsize=7.4, fontweight="bold", color=INK)
    ax.text(0.5, 0.02, "Caloric restriction", ha="center", va="bottom",
            transform=ax.transAxes, fontsize=6.0, color=INK)
    ax.text(-0.02, 0.5, "Immobilisation", ha="right", va="center", rotation=90,
            transform=ax.transAxes, fontsize=6.0, color=INK)
    labels = [["Control", "IR"], ["CR", "CR+IR"]]
    groups = [["Ctrl", "IR"], ["CR", "CR+IR"]]
    for r in range(2):
        for c in range(2):
            g = groups[r][c]
            fc = GROUP_COLORS[g]
            ax.add_patch(Rectangle((c + 0.08, 1 - r + 0.08), 0.84, 0.72,
                                   facecolor=fc, alpha=0.13, edgecolor=fc, lw=1.2))
            ax.text(c + 0.50, 1 - r + 0.51, labels[r][c], ha="center", va="center",
                    fontsize=7.2, fontweight="bold", color=INK)
            ax.text(c + 0.50, 1 - r + 0.27, f"n = {n}", ha="center", va="center",
                    fontsize=6.2, color=fc, fontweight="bold")


def figure2(meta, expr, de, ann, interaction, cf, enrich, counts, drug):
    m16, primary, expr16, de_named, score, qc, up, down = primary_tables(meta, expr, de, ann, counts)
    # PCA for the main-text disease contrast uses only the 24 primary samples
    # (Control and CR+IR, displayed as Sarcopenia).  The complete four-arm PCA
    # remains available from the source metadata but is not shown in the image.
    v = expr16.var(axis=1).sort_values(ascending=False).head(min(2000, expr16.shape[0])).index
    pca_xy = PCA(n_components=2, random_state=42).fit_transform(expr16.loc[v, primary["gsm"]].T.fillna(0))
    pca_df = primary[["gsm", "group", "mouse id"]].copy()
    pca_df["PC1"] = pca_xy[:, 0]; pca_df["PC2"] = pca_xy[:, 1]
    pca_df["display_group"] = pca_df["group"].map(GROUP_LABELS)
    pca_df.to_csv(OUT / "figure2_pca_coordinates.csv", index=False)
    qc.to_csv(OUT / "figure2_library_qc.csv", index=False)
    score_out = score.copy()
    score_out["display_group"] = score_out["group"].map(GROUP_LABELS)
    score_out.to_csv(OUT / "figure2_signature_scores.csv", index=False)

    # Heatmap: top 24 primary DESeq2 genes, shown across the 24 primary samples.
    sig = de_named[(de_named["padj"] < 0.05) & (de_named["log2FoldChange"].abs() >= 0.5)].copy()
    heat = sig.sort_values(["padj", "gene_id"]).head(24)
    heat_names = heat["display_name"].tolist()
    heat_names = [g for g in heat_names if g in expr16.index]
    hmat = expr16.loc[heat_names, primary["gsm"].tolist()]
    hz = hmat.sub(hmat.mean(axis=1), axis=0).div(hmat.std(axis=1).replace(0, 1), axis=0)
    hz.to_csv(OUT / "figure2_top24_row_z_heatmap.csv")

    # Main Figure 2 contains the biological PCA, differential-expression
    # landscape, signature contrast and top-gene heatmap.  Library-level QC is
    # written to source data but is intentionally not plotted in the main text.
    fig = plt.figure(figsize=(7.2, 5.9), dpi=300)
    gs = fig.add_gridspec(
        2, 6, height_ratios=[1.12, 1.42], hspace=0.55, wspace=0.60,
        left=0.13, right=0.975, bottom=0.12, top=0.965,
    )
    a = fig.add_subplot(gs[0, 0:3])  # PCA
    b = fig.add_subplot(gs[0, 3:6])  # volcano
    c = fig.add_subplot(gs[1, 0:2])  # signature score
    d = fig.add_subplot(gs[1, 2:6])  # heatmap

    # Uniform panel subtitles: the complete figure title stays in the
    # manuscript caption, while every panel receives the same concise,
    # bold subtitle at the same height.
    panel_titles = {
        a: "Principal-component structure",
        b: "Differential expression",
        c: "Sarcopenia signature",
        d: "Top differential-expression genes",
    }
    for ax, subtitle in panel_titles.items():
        ax._panel_title = subtitle

    for g in ["Ctrl", "CR+IR"]:
        sub = pca_df[pca_df["group"] == g]
        a.scatter(sub["PC1"], sub["PC2"], s=27, color=GROUP_COLORS[g], edgecolor="white", linewidth=0.45, label=GROUP_LABELS[g])
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.legend(loc="best", fontsize=5.5, ncol=2, handletextpad=0.2)
    a.grid(color=GRID, lw=0.45); panel_label(a, "a")

    x = de_named["log2FoldChange"].to_numpy()
    y = -np.log10(np.clip(de_named["padj"].fillna(1).to_numpy(), 1e-300, 1))
    sigmask = (de_named["padj"] < 0.05) & (de_named["log2FoldChange"].abs() >= 0.5)
    cols = np.where(sigmask & (x > 0), UP, np.where(sigmask & (x < 0), DOWN, "#CBD5E1"))
    b.scatter(x, y, s=4.4, c=cols, alpha=0.65, linewidths=0, rasterized=True)
    b.axvline(-0.5, color=MID, ls="--", lw=0.6); b.axvline(0.5, color=MID, ls="--", lw=0.6)
    b.axhline(-np.log10(0.05), color=MID, ls="--", lw=0.6)
    top = de_named.loc[sigmask].sort_values(["padj", "gene_id"]).head(6)
    # The six most significant labels cluster in the upper-right quadrant.
    # Use deterministic point offsets and very light leader lines instead of
    # allowing text to overlap; the plotted values and label selection remain
    # unchanged.
    label_offsets = {
        "Ctxn3": (0, 5, "center"),
        "Abca1": (15, 11, "left"),
        "Sh3rf2": (-23, 10, "right"),
        "Rorc": (-27, -8, "right"),
        "Egln3": (18, -3, "left"),
        "Pm20d2": (12, -14, "left"),
    }
    for _, r in top.iterrows():
        gene = str(r["display_name"])
        x0 = float(r["log2FoldChange"])
        y0 = float(-np.log10(max(r["padj"], 1e-300)))
        dx, dy, ha = label_offsets.get(gene, (0, 5, "center"))
        b.annotate(
            gene, xy=(x0, y0), xycoords="data",
            xytext=(dx, dy), textcoords="offset points",
            fontsize=5.0, ha=ha, va="center", color=INK,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": MID, "lw": 0.45,
                        "shrinkA": 1.5, "shrinkB": 1.5},
            annotation_clip=True,
        )
    b.set_xlabel("log2 fold change (Sarcopenia − control)"); b.set_ylabel("−log10 adjusted P")
    b.grid(color=GRID, lw=0.45); panel_label(b, "b")

    # Heatmap with a compact group-color strip.
    im = d.imshow(hz.values, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2, interpolation="nearest")
    d.set_yticks(np.arange(len(heat_names)), [pretty_gene(g) for g in heat_names], fontsize=5.0)
    d.set_xticks(np.arange(len(primary)), [x.replace("GSM", "") for x in primary["gsm"]], rotation=90, fontsize=5.0)
    d.tick_params(length=0)
    # Group strip beneath the heatmap.
    for j, g in enumerate(primary["group"]):
        d.add_patch(Rectangle((j - 0.5, len(heat_names) - 0.15), 1, 0.45,
                              facecolor=GROUP_COLORS[g], edgecolor="none", clip_on=False))
    cb = fig.colorbar(im, ax=d, fraction=0.02, pad=0.02)
    cb.ax.tick_params(labelsize=5.4, length=2)
    cb.set_label("row z-score", fontsize=5.8, labelpad=4)
    # Keep the same title-to-axes relationship as every other panel; the
    # additional row spacing above now prevents collision with panel c/d.
    panel_label(d, "d", x=-0.055)

    # 1.5×IQR box plots plus every observed sample. Horizontal offsets are purely
    # display coordinates; the score values, groups and sample counts are fixed.
    box_rows = []
    plotted_points = []
    for i, g in enumerate(["Ctrl", "CR+IR"]):
        group_scores = score.loc[score["group"] == g].copy()
        sub = group_scores["disease_signature"].to_numpy(dtype=float)
        assert len(sub) == 12 and np.isfinite(sub).all(), f"Invalid Figure 2c data: {g}"
        stats = mpl.cbook.boxplot_stats(sub, whis=FIGURE2_LAYOUT["whisker_iqr"])[0]
        box_rows.append({"group": g, "n": len(sub), "q1": stats["q1"],
                         "median": stats["med"], "q3": stats["q3"],
                         "whisker_low": stats["whislo"], "whisker_high": stats["whishi"],
                         "outliers": len(stats["fliers"])})
        boxes = c.bxp(
            [stats], positions=[i], widths=FIGURE2_LAYOUT["box_width"],
            patch_artist=True, showfliers=False, manage_ticks=False,
            boxprops={"edgecolor": GROUP_COLORS[g], "linewidth": 0.95},
            medianprops={"color": INK, "linewidth": 1.25},
            whiskerprops={"color": GROUP_COLORS[g], "linewidth": 0.9},
            capprops={"color": GROUP_COLORS[g], "linewidth": 0.9},
        )
        boxes["boxes"][0].set_facecolor(mpl.colors.to_rgba(GROUP_COLORS[g], 0.16))
        # One deterministic x offset per sample avoids obscuring coincident dots.
        offsets = ((np.arange(len(sub)) * 5) % len(sub) - (len(sub) - 1) / 2) * 0.023
        c.scatter(i + offsets, sub, s=21, color=GROUP_COLORS[g],
                  edgecolor="white", linewidth=0.5, zorder=3)
        for j, (_, row) in enumerate(group_scores.iterrows()):
            plotted_points.append({"gsm": row["gsm"], "group": g,
                                   "disease_signature": row["disease_signature"],
                                   "display_x": float(i + offsets[j])})
    pd.DataFrame(box_rows).to_csv(OUT / "figure2_panel_c_boxplot_statistics.csv", index=False)
    pd.DataFrame(plotted_points).to_csv(OUT / "figure2_panel_c_displayed_points.csv", index=False)
    c.set_xticks([0, 1], ["Control", "Sarcopenia"], rotation=0)
    c.set_xlim(-0.55, 1.55)
    c.margins(y=0.12)
    c.set_ylabel("Sarcopenia signature")
    ctrl_values = score.loc[score["group"] == "Ctrl", "disease_signature"].to_numpy(dtype=float)
    model_values = score.loc[score["group"] == "CR+IR", "disease_signature"].to_numpy(dtype=float)
    test = ttest_ind(model_values, ctrl_values, equal_var=False)
    p_value = float(test.pvalue)
    t_value = float(test.statistic)
    y_max = float(max(np.max(ctrl_values), np.max(model_values)))
    y_min = float(min(np.min(ctrl_values), np.min(model_values)))
    y_span = max(y_max - y_min, 1.0)
    bracket_y = y_max + 0.14 * y_span
    bracket_h = 0.035 * y_span
    c.plot([0, 0, 1, 1], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
           color=INK, lw=0.75, clip_on=False)
    p_text = "P < 0.0001" if p_value < 1e-4 else f"P = {p_value:.3g}"
    c.text(0.5, bracket_y + bracket_h + 0.025 * y_span, p_text,
           ha="center", va="bottom", fontsize=5.5, color=INK)
    c.set_ylim(y_min - 0.08 * y_span, bracket_y + 0.18 * y_span)
    pd.DataFrame([{
        "contrast": "Sarcopenia_vs_Control",
        "test": "Welch two-sample t-test",
        "n_control": int(len(ctrl_values)),
        "n_sarcopenia": int(len(model_values)),
        "t_statistic": t_value,
        "p_value": p_value,
        "annotation": p_text,
    }]).to_csv(OUT / "figure2_panel_c_statistics.csv", index=False)
    c.grid(axis="y", color=GRID, lw=0.45); panel_label(c, "c")
    c.set_axisbelow(True)
    geometry = []
    for label, ax in zip("abcd", [a, b, c, d]):
        pos = ax.get_position()
        geometry.append({"panel": label, "x0_inches": pos.x0 * fig.get_figwidth(),
                         "y0_inches": pos.y0 * fig.get_figheight(),
                         "width_inches": pos.width * fig.get_figwidth(),
                         "height_inches": pos.height * fig.get_figheight()})
    pd.DataFrame(geometry).to_csv(OUT / "figure2_layout_geometry.csv", index=False)
    save_pub(fig, "Figure2_GSE213148_primary_expression_landscape")


def figure3(meta, expr, de, ann, interaction, cf, enrich, counts, drug):
    m16, primary, expr16, de_named, score, qc, up, down = primary_tables(meta, expr, de, ann, counts)
    # Main-text Figure 3 is restricted to the prespecified disease contrast.
    # The source experiment contains additional factorial arms, but those arms
    # are not displayed in the manuscript figure.  Keeping this local display
    # order prevents CR/IR labels from leaking into the artwork while leaving
    # the complete metadata and factorial tables available for audit.
    display_groups = ["Ctrl", "CR+IR"]
    display_group_labels = ["Control", "Sarcopenia"]
    # Group-level matrices are built from the same 16-week expression matrix.
    group_means = expr16[m16["gsm"].tolist()].T.join(m16.set_index("gsm")["group"]).groupby("group").mean().reindex(GROUP_ORDER)
    sig_group = score.groupby("group")[["disease_signature", "up_component", "down_component"]].mean().reindex(GROUP_ORDER)
    sig_group.to_csv(OUT / "figure3_group_signature_means.csv")

    # Prefer the unified GO/KEGG/Reactome enrichment and GSVA outputs when
    # available. The fallback keeps this plotting script reproducible before
    # the pathway-analysis directory has been generated.
    unified_path = PATHWAY_DIR / "figure3_panel_c_unified_enrichment.csv"
    gsva_path = PATHWAY_DIR / "figure3_panel_f_selected_gsva_pathways.csv"
    candidate_path = PATHWAY_DIR / "figure3_panel_f_gene_pathway_convergence.csv"
    gsva_scores_path = PATHWAY_DIR / "gsva_sample_scores.csv"
    drug_pathway_path = PATHWAY_DIR / "drug_pathway_cross_explanation.csv"
    if unified_path.exists():
        enr_plot = pd.read_csv(unified_path)
        enr_plot["neglog10_fdr"] = -np.log10(np.clip(enr_plot["fdr"].astype(float), 1e-300, 1))
        enr_plot.to_csv(OUT / "figure3_top_enrichment_terms.csv", index=False)
    else:
        enr = enrich.copy()
        enr["neglog10_fdr"] = -np.log10(np.clip(enr["gprofiler_adjusted_p"], 1e-300, 1))
        enr = enr[enr["input_size"].notna()].copy()
        enr_up = enr[enr["query_label"].astype(str).str.lower().eq("up")].sort_values("gprofiler_adjusted_p").head(5)
        enr_down = enr[enr["query_label"].astype(str).str.lower().eq("down")].sort_values("gprofiler_adjusted_p").head(5)
        enr_plot = pd.concat([enr_up, enr_down], ignore_index=True)
        enr_plot["direction"] = enr_plot["query_label"].astype(str).str.lower()
        enr_plot["source"] = enr_plot["source"].replace({"GO:BP": "GO:BP", "REAC": "REAC"})
        enr_plot["overlap_size"] = enr_plot["intersection_size"]
        enr_plot["enrichment_ratio"] = np.nan
        enr_plot.to_csv(OUT / "figure3_top_enrichment_terms.csv", index=False)

    gsva_selected = pd.read_csv(gsva_path) if gsva_path.exists() else pd.DataFrame()
    candidate_context = pd.read_csv(candidate_path) if candidate_path.exists() else pd.DataFrame()
    gsva_scores = pd.read_csv(gsva_scores_path) if gsva_scores_path.exists() else pd.DataFrame()
    drug_pathway_context = pd.read_csv(drug_pathway_path) if drug_pathway_path.exists() else pd.DataFrame()

    # Interaction coefficient table and explicit threshold count.
    interaction["neglog10_p"] = -np.log10(np.clip(interaction["pvalue"], 1e-300, 1))
    interaction.to_csv(OUT / "figure3_interaction_coefficients.csv", index=False)
    interaction_hits = int(((interaction["padj"] < 0.05) & (interaction["log2FoldChange"].abs() >= 0.5)).sum())

    # The heatmap is deliberately ordered by direction: the ten most
    # significant upregulated genes are followed by the ten most significant
    # downregulated genes.  This makes the direction of the primary contrast
    # readable without relying on a hidden row order.
    sig_all = de_named[(de_named["padj"] < 0.05) & (de_named["log2FoldChange"].abs() >= 0.5)].copy()
    sig_up = sig_all[sig_all["log2FoldChange"] > 0].sort_values(["padj", "gene_id"]).head(10)
    sig_down = sig_all[sig_all["log2FoldChange"] < 0].sort_values(["padj", "gene_id"]).head(10)
    sig = pd.concat([sig_up, sig_down], ignore_index=True)
    heat_names = [g for g in sig["display_name"].tolist() if g in expr16.index]
    gm = group_means.loc[display_groups, heat_names].T
    gmz = gm.sub(gm.mean(axis=1), axis=0).div(gm.std(axis=1).replace(0, 1), axis=0)
    heat_order = sig[sig["display_name"].isin(heat_names)][["display_name", "log2FoldChange", "padj"]].copy()
    heat_order["direction"] = np.where(heat_order["log2FoldChange"] > 0, "Up", "Down")
    heat_order["display_order"] = np.arange(len(heat_order))
    heat_order.to_csv(OUT / "figure3_gene_heatmap_order.csv", index=False)
    gmz.to_csv(OUT / "figure3_control_sarcopenia_gene_program_heatmap.csv")

    fig = plt.figure(figsize=FIGURE3_LAYOUT["figsize"], dpi=300)
    a = fig.add_axes(FIGURE3_LAYOUT["a_bounds"])
    b = fig.add_axes(FIGURE3_LAYOUT["b_bounds"])
    c_labels = fig.add_axes(FIGURE3_LAYOUT["c_label_bounds"])
    c = fig.add_axes(FIGURE3_LAYOUT["c_bounds"])
    # Four panels are retained in the main figure. The former f pathway
    # concordance matrix is moved forward to panel d, immediately after the
    # enrichment result. The gene-level interaction test and its redundant
    # histogram remain available as source tables but are not given a second
    # main-panel slot.
    d = fig.add_axes(FIGURE3_LAYOUT["d_bounds"])
    d.axis("off")

    # Keep panel-level subtitles uniform: every panel has one concise,
    # descriptive subtitle in the same position and style. The full figure
    # title remains in the manuscript caption rather than inside the artwork.
    panel_titles = {
        a: "Directional signature components",
        b: "Directional gene expression",
        c: "Functional enrichment",
        d: "GSVA profiles across groups",
    }
    for ax, subtitle in panel_titles.items():
        ax._panel_title = subtitle
    for label, ax, xpos in zip("abc", [a, b, c],
                               [FIGURE3_LAYOUT["a_bounds"][0],
                                FIGURE3_LAYOUT["b_bounds"][0],
                                FIGURE3_LAYOUT["c_heading_x"]]):
        fig.text(xpos, FIGURE3_LAYOUT["heading_y"],
                 f"{label}  {panel_titles[ax]}", fontsize=8.0,
                 fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(FIGURE3_LAYOUT["d_bounds"][0], FIGURE3_LAYOUT["d_heading_y"],
             f"d  {panel_titles[d]}", fontsize=8.0,
             fontweight="bold", color=INK, ha="left", va="bottom")

    # Panel a: grouped violin/raincloud view of the two directional components.
    # This keeps every library visible while showing the distribution shape,
    # group mean and 95% CI without the previous connecting line.
    component_specs = [("up_component", "Up"),
                       ("down_component", "Down")]
    component_rows, component_points, component_pvals = [], [], []
    violin_positions = {"Up": [0 - 0.18, 0 + 0.18],
                        "Down": [1 - 0.18, 1 + 0.18]}
    for component_index, (column, label) in enumerate(component_specs):
        values_by_group = []
        max_component = -np.inf
        for group_index, group in enumerate(display_groups):
            sub_df = score[score["group"] == group].sort_values("gsm")
            values = sub_df[column].to_numpy(dtype=float)
            assert len(values) == 12 and np.isfinite(values).all(), f"Invalid Figure 3a data: {group}"
            values_by_group.append(values)
            max_component = max(max_component, float(values.max()))
            position = violin_positions[label][group_index]
            vp = a.violinplot([values], positions=[position], widths=0.28,
                              showmeans=False, showmedians=False,
                              showextrema=False, points=200)
            body = vp["bodies"][0]
            body.set_facecolor(GROUP_COLORS[group])
            body.set_edgecolor(GROUP_COLORS[group])
            body.set_linewidth(0.7)
            body.set_alpha(0.28)
            offsets = (((np.arange(len(values)) * 5) % len(values)
                        - (len(values) - 1) / 2) * 0.018)
            a.scatter(position + offsets, values, s=17,
                      color=GROUP_COLORS[group], alpha=0.78,
                      edgecolor="white", linewidth=0.45, zorder=3)
            mean = float(values.mean())
            ci = float(student_t.ppf(0.975, len(values) - 1)
                       * values.std(ddof=1) / np.sqrt(len(values)))
            a.errorbar(position, mean, yerr=ci, fmt="D", markersize=3.2,
                       markerfacecolor="white", markeredgecolor=INK,
                       markeredgewidth=0.55, ecolor=INK, elinewidth=0.7,
                       capsize=2.0, capthick=0.65, zorder=5)
            for j, (_, sample_row) in enumerate(sub_df.iterrows()):
                component_points.append({"component": label, "gsm": sample_row["gsm"],
                                         "group": group, "value": float(values[j]),
                                         "display_x": float(position + offsets[j])})
            component_rows.append({"component": label, "group": group, "n": len(values),
                                   "mean": mean, "sd": float(values.std(ddof=1)),
                                   "ci95_low": mean - ci, "ci95_high": mean + ci})
        component_pvals.append(float(ttest_ind(values_by_group[1], values_by_group[0],
                                               equal_var=False).pvalue))
        bracket_y = max_component + 0.13
        a.plot([violin_positions[label][0], violin_positions[label][0],
                violin_positions[label][1], violin_positions[label][1]],
               [bracket_y, bracket_y + 0.035, bracket_y + 0.035, bracket_y],
               color=INK, lw=0.65, clip_on=False, zorder=6)
        component_rows[-2]["bracket_y"] = bracket_y
        component_rows[-1]["bracket_y"] = bracket_y
    order = np.argsort(component_pvals)
    qvals = np.empty(len(component_pvals), dtype=float)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        idx = order[rank_index]
        running = min(running, component_pvals[idx] * len(order) / (rank_index + 1))
        qvals[idx] = running
    for component_index, q_value in enumerate(qvals):
        label = "q < 0.001" if q_value < 0.001 else f"q = {q_value:.3g}"
        max_y = max(float(score.loc[score["group"].isin(display_groups),
                                    component_specs[component_index][0]].max()),
                    float(score.loc[score["group"].isin(display_groups),
                                    component_specs[component_index][0]].min()))
        bracket_y = max(float(r["bracket_y"]) for r in component_rows[component_index * 2:component_index * 2 + 2])
        a.text(component_index, bracket_y + 0.07, label, ha="center", va="bottom",
               fontsize=5.2, color=INK)
        for row in component_rows[component_index * 2:component_index * 2 + 2]:
            row["welch_p"] = component_pvals[component_index]
            row["BH_q"] = q_value
    pd.DataFrame(component_rows).to_csv(OUT / "figure3_panel_a_component_statistics.csv", index=False)
    pd.DataFrame(component_points).to_csv(OUT / "figure3_panel_a_component_points.csv", index=False)
    a.axhline(0, color=MID, lw=0.65, ls="--", zorder=1)
    a.set_xticks([0, 1], ["Upregulated\ncomponent", "Downregulated\ncomponent"], fontsize=6.0)
    a.set_ylabel("Standardized component score")
    a.set_xlabel("Signature direction", labelpad=3.5)
    a.set_xlim(-0.52, 1.52)
    a.set_ylim(-1.50, 1.95)
    a.grid(axis="y", color=GRID, lw=0.45)
    a.legend(handles=[Line2D([0], [0], marker="o", color="none",
                             markerfacecolor=GROUP_COLORS["Ctrl"], markeredgecolor="white",
                             markersize=5.2, label="Control"),
                    Line2D([0], [0], marker="o", color="none",
                             markerfacecolor=GROUP_COLORS["CR+IR"], markeredgecolor="white",
                             markersize=5.2, label="Sarcopenia")],
           loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2,
           columnspacing=0.8, handletextpad=0.25, fontsize=5.6)

    # Panel b: a clean two-column heatmap. The cells use the true group-mean
    # log2 CPM values, so the printed number and the colour carry the same
    # quantitative meaning. Direction is encoded by the ordered rows and the
    # coloured gene labels.
    n_up = int((heat_order["direction"] == "Up").sum())
    heat_values = gm.loc[heat_names, display_groups].to_numpy(dtype=float)
    value_min = float(np.nanmin(heat_values))
    value_max = float(np.nanmax(heat_values))
    norm = Normalize(vmin=value_min, vmax=value_max)
    # Match panel a's Control blue and Sarcopenia red rather than using the
    # more orange-toned default diverging map.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "control_sarcopenia_diverging",
        [DOWN, "#FFFFFF", UP], N=256,
    )
    im = b.imshow(heat_values,
                  cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    b.set_yticks(np.arange(len(heat_names)), [pretty_gene(g) for g in heat_names], fontsize=5.9)
    b.set_xticks(range(2), display_group_labels, rotation=0, ha="center")
    b.set_xlabel("Group\n(cell text: mean log2 CPM)", labelpad=2.0)
    b.tick_params(length=0)
    b.set_xlim(-0.5, 1.5)
    b.set_ylim(len(heat_names) - 0.5, -0.5)
    # White minor-grid lines make every gene-by-group cell explicit at print
    # size while preserving the continuous colour scale.
    b.set_xticks(np.arange(-0.5, 2.0, 1.0), minor=True)
    b.set_yticks(np.arange(-0.5, len(heat_names), 1.0), minor=True)
    b.grid(which="minor", color="white", linewidth=0.75)
    b.tick_params(which="minor", bottom=False, left=False)
    if 0 < n_up < len(heat_order):
        b.axhline(n_up - 0.5, color="white", linewidth=1.6, zorder=4)
    for tick_index, tick in enumerate(b.get_yticklabels()):
        tick.set_color(UP if tick_index < n_up else DOWN)
    # Group annotation strip above the heatmap clarifies the two columns.
    for x_index, group in enumerate(display_groups):
        b.add_patch(Rectangle((x_index - 0.5, -1.05), 1.0, 0.18,
                              facecolor=GROUP_COLORS[group], edgecolor="none",
                              clip_on=False, zorder=5))
    # Numeric annotations are the same values used for the colour gradient.
    for row_index, gene in enumerate(heat_names):
        for group_index, value in enumerate(heat_values[row_index]):
            b.text(group_index, row_index, f"{value:.2f}", ha="center", va="center",
                   fontsize=5.0, color=INK, zorder=6)
    lfc_map = heat_order.set_index("display_name")["log2FoldChange"].to_dict()
    lfc_values = np.asarray([float(lfc_map[g]) for g in heat_names])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb_ax = fig.add_axes(FIGURE3_LAYOUT["b_colorbar_bounds"])
    cb = fig.colorbar(sm, cax=cb_ax, orientation="horizontal")
    cb.set_ticks(np.linspace(value_min, value_max, 3))
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1, colors=INK)
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.set_label("Mean log2 CPM", fontsize=5.7, labelpad=1, color=INK)
    heatmap_records = gm.loc[heat_names, display_groups].copy()
    heatmap_records.insert(0, "gene", heatmap_records.index)
    heatmap_records["Control_row_z"] = gmz.loc[heat_names, "Ctrl"].to_numpy(dtype=float)
    heatmap_records["Sarcopenia_row_z"] = gmz.loc[heat_names, "CR+IR"].to_numpy(dtype=float)
    heatmap_records["log2FoldChange"] = lfc_values
    heatmap_records["direction"] = ["Up" if value > 0 else "Down" for value in lfc_values]
    heatmap_records["color_scale_min"] = value_min
    heatmap_records["color_scale_max"] = value_max
    heatmap_records.to_csv(OUT / "figure3_panel_b_heatmap_with_effects.csv", index=False)
    heatmap_records.to_csv(OUT / "figure3_panel_b_group_mean_expression.csv", index=False)

    # Unified GO/KEGG/Reactome enrichment. A horizontal lollipop keeps the
    # enrichment-ratio scale explicit while avoiding the overlap caused by a bubble
    # plot. Stem colour encodes the tested direction, dot colour encodes
    # -log10(FDR), dot size encodes overlap size, and the source is written as
    # a compact prefix in each pathway label. No pathway membership is
    # invented and all twelve audited terms remain visible.
    if len(enr_plot):
        enr_plot = enr_plot.copy().reset_index(drop=True)
        if "gene_ratio" not in enr_plot or enr_plot["gene_ratio"].isna().all():
            enr_plot["gene_ratio"] = enr_plot["overlap_size"] / enr_plot["query_size"].replace(0, np.nan)
        enr_plot["neglog10_fdr"] = -np.log10(np.clip(enr_plot["fdr"].astype(float), 1e-300, 1)) if "fdr" in enr_plot else enr_plot["neglog10_fdr"]
        source_order = {"GO:BP": 0, "KEGG": 1, "REAC": 2}
        direction_order = {"up": 0, "down": 1}
        enr_plot["_source_order"] = enr_plot["source"].map(source_order).fillna(9)
        enr_plot["_direction_order"] = enr_plot["direction"].astype(str).str.lower().map(direction_order).fillna(9)
        enr_plot = enr_plot.sort_values(
            ["_direction_order", "_source_order", "gene_ratio"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        yidx = np.arange(len(enr_plot))[::-1]
        edge_colors = {"up": UP, "down": DOWN}
        sc_up = None
        sc_down = None
        fdr_norm = Normalize(vmin=0.0, vmax=max(1.0, float(enr_plot["neglog10_fdr"].max())))
        for i, row in enr_plot.iterrows():
            direction = str(row.get("direction", row.get("query_label", ""))).lower()
            source = str(row.get("source", ""))
            x_value = float(row["enrichment_ratio"])
            y_value = float(yidx[i])
            direction_color = edge_colors.get(direction, INK)
            direction_cmap = FDR_UP_CMAP if direction == "up" else FDR_DOWN_CMAP
            c.hlines(y_value, 0, x_value, color=direction_color, linewidth=1.45,
                     alpha=0.72, zorder=2)
            point = c.scatter(
                [x_value], [y_value],
                s=float(np.clip(row["overlap_size"] * 1.35 + 18, 26, 145)),
                c=[float(row["neglog10_fdr"])], cmap=direction_cmap, norm=fdr_norm,
                marker="o", edgecolor=direction_color, linewidth=1.0,
                alpha=0.92, zorder=3,
            )
            if direction == "up":
                sc_up = point
            elif direction == "down":
                sc_down = point
        source_short = {"GO:BP": "GO", "KEGG": "KEGG", "REAC": "Reactome"}
        short_name_replacements = {
            "drug metabolism - cytochrome p450": "Drug metabolism:\ncytochrome P450 (KEGG)",
            "external encapsulating structure organization": "External encapsulating\nstructure organisation (GO)",
            "nmd-independent exon-junction decay": "EJC-independent\nNMD (Reactome)",
            "free 40s subunit formation": "Free 40S subunit\nformation (Reactome)",
            "cytoskeleton in muscle cells": "muscle-cell cytoskeleton",
            "extracellular matrix organization": "Extracellular matrix\norganisation (Reactome)",
            "lipid metabolic process": "lipid metabolism",
        }
        labels = []
        for _, row in enr_plot.iterrows():
            # Normalize before wrapping so the short-name map also catches
            # terms that would otherwise be split across lines.
            short = compact_pathway_name(row["term_name"], width=80).replace("\n", " ")
            replacement = short_name_replacements.get(short.lower(), short)
            if "\n" in replacement:
                label = replacement
            else:
                label = textwrap.fill(
                    f"{replacement} ({source_short.get(str(row['source']), str(row['source']))})",
                    width=25, break_long_words=False, break_on_hyphens=False)
            labels.append(label)
        # Labels have a separate axes with the exact same y limits as the
        # data. They cannot drift into the gene heatmap or its colourbar.
        c.set_yticks(yidx)
        c.set_yticklabels([])
        c.tick_params(axis="y", length=0)
        c_labels.set_ylim(-0.65, len(enr_plot) - 0.25)
        c_labels.set_xlim(0, 1)
        c_labels.axis("off")
        for yy, label in zip(yidx, labels):
            c_labels.text(1.0, yy, label, ha="right", va="center", fontsize=6.0,
                          color=INK, linespacing=1.02)
        enr_plot["display_label"] = labels
        enr_plot["display_y"] = yidx
        enr_plot.to_csv(OUT / "figure3_panel_c_displayed_enrichment.csv", index=False)
        c.set_ylim(-0.65, len(enr_plot) - 0.25)
        # Explicitly set the same label padding used by panel b so that the
        # two x-axis titles share an identical vertical position.
        c.set_xlabel("Enrichment ratio", labelpad=4.3)
        c.set_xlim(0, max(6.0, float(enr_plot["enrichment_ratio"].max()) * 1.12))
        c.set_xticks([0, 2, 4, 6])
        c.grid(axis="x", color=GRID, lw=0.45)
        # Two compact, aligned colourbars preserve the directional meaning:
        # red intensity is the FDR scale for up terms and blue intensity is the
        # same scale for down terms.  They share the same numeric norm.
        cbar_x, cbar_y, cbar_w, cbar_h = FIGURE3_LAYOUT["c_colorbar_bounds"]
        cbar_gap = 0.006
        half_w = (cbar_w - cbar_gap) / 2.0
        cbar_specs = [(sc_up, FDR_UP_CMAP, "Up", cbar_x),
                      (sc_down, FDR_DOWN_CMAP, "Down", cbar_x + half_w + cbar_gap)]
        cbar_max = fdr_norm.vmax
        for sm, cmap, label, xpos in cbar_specs:
            if sm is None:
                continue
            sm_ref = mpl.cm.ScalarMappable(norm=fdr_norm, cmap=cmap)
            sm_ref.set_array([])
            cb_c = fig.colorbar(sm_ref, cax=fig.add_axes([xpos, cbar_y, half_w, cbar_h]),
                                orientation="horizontal")
            # The two bars share an identical numeric norm.  Show the numeric
            # ticks only once (under the red/Up bar) so the adjacent bars do
            # not create the cramped 0/max/0/max sequence at print size.
            if label == "Up":
                cb_c.set_ticks([0, cbar_max])
                cb_c.set_ticklabels(["0", f"{cbar_max:.0f}"])
            else:
                cb_c.set_ticks([])
            cb_c.ax.set_title(label, fontsize=5.2, color=INK, pad=1.2)
            cb_c.ax.xaxis.set_ticks_position("bottom")
            cb_c.ax.tick_params(labelsize=5.0, length=1.5, pad=0.6, colors=INK)
        fig.text(cbar_x + cbar_w / 2.0, cbar_y - 0.014, "−log10 FDR",
                 ha="center", va="top", fontsize=5.3, color=INK)
        fig.legend(handles=[Line2D([0], [0], color=UP, lw=1.5, label="Up"),
                            Line2D([0], [0], color=DOWN, lw=1.5, label="Down")],
                   loc="lower left", bbox_to_anchor=(0.715, 0.923),
                   borderaxespad=0, ncol=2, columnspacing=0.8,
                   handlelength=1.2, handletextpad=0.3, fontsize=5.8,
                   labelcolor=INK)

    # Pathway-level convergence matrix. The four effect columns are sample-
    # level GSVA contrasts; the row annotation reports how many of the
    # expression-level counterfactual candidates map to the pathway. It is
    # explicitly a computational concordance summary, not causal rescue.
    if len(gsva_selected):
        ftab = gsva_selected.copy()
        ctx = candidate_context.copy()
        rows = []
        for _, row in ftab.iterrows():
            term_id = row["term_id"]
            sub = ctx[(ctx["term_id"] == term_id) & (ctx["in_gene_set"])] if len(ctx) else pd.DataFrame()
            cf_n = int(sub["candidate_symbol"].nunique()) if len(sub) else 0
            cf_max = float(sub["mean_reflection_reversal_fraction"].max()) if len(sub) else np.nan
            drug_sub = drug_pathway_context[drug_pathway_context["term_id"] == term_id] if len(drug_pathway_context) else pd.DataFrame()
            drug_valid = pd.to_numeric(drug_sub.get("pathway_reversal_fraction", pd.Series(dtype=float)), errors="coerce").dropna()
            drug_median = float(drug_valid.median()) if len(drug_valid) else np.nan
            rows.append({
                "term_id": term_id,
                "source": row["source"],
                "term_name": row["term_name"],
                "delta_GSVA_CRIR_vs_Control": row["delta_CRIR_vs_Control"],
                "beta_CR": row["beta_CR"],
                "beta_IR": row["beta_IR"],
                "beta_interaction": row["beta_interaction"],
                "fdr_CRIR_vs_Control": row["fdr_CRIR_vs_Control"],
                "fdr_interaction": row["fdr_interaction"],
                "counterfactual_candidate_n": cf_n,
                "counterfactual_max_mean_reflection_reversal": cf_max,
                "drug_median_signature_reversal_fraction": drug_median,
                "drug_estimable_signature_n": int(len(drug_valid)),
                "interpretation": "GSVA, expression-level counterfactual membership and orthology-mapped drug-signature concordance; not causal pathway reversal",
            })
        ftab = pd.DataFrame(rows)
        # The panel is now labelled d; retain the former f-named export as a
        # compatibility alias for existing audit tables and downstream scripts.
        ftab.to_csv(OUT / "figure3_panel_d_pathway_evidence_matrix.csv", index=False)
        ftab.to_csv(OUT / "figure3_panel_f_pathway_evidence_matrix.csv", index=False)
        selected_scores = gsva_scores[gsva_scores["term_id"].isin(ftab["term_id"])].copy()
        selected_scores = selected_scores[selected_scores["group"].isin(display_groups)].copy()
        assert len(ftab) == 6, "The fixed GSVA display must contain six pathways"
        assert len(selected_scores) == 144, "Six pathways must retain the 24 primary samples"
        assert not selected_scores.duplicated(["gsm", "term_id"]).any()
        assert np.isfinite(selected_scores["gsva_score"]).all()
        # Metadata is checked against the scoring output rather than assumed
        # from row order. Each of the 48 samples has a unique mouse identifier.
        sample_meta = m16[["gsm", "group", "mouse id"]].copy()
        sample_meta = sample_meta[sample_meta["group"].isin(display_groups)].copy()
        assert sample_meta["gsm"].nunique() == 24
        assert sample_meta["mouse id"].nunique() == 24
        checked = selected_scores.merge(sample_meta, on="gsm", how="left",
                                        suffixes=("", "_metadata"), validate="many_to_one")
        assert checked["group_metadata"].notna().all()
        assert (checked["group"] == checked["group_metadata"]).all()
        assert (checked["mouse id"].astype(str) == checked["mouse id_metadata"].astype(str)).all()
        profile_titles = {
            "GO:BP__GO_0022409_positive_regulation_of_cell_cell_adhesion":
                "Positive regulation of\ncell–cell adhesion (GO)",
            "GO:BP__GO_0007162_negative_regulation_of_cell_adhesion":
                "Negative regulation of\ncell adhesion (GO)",
            "KEGG__KEGG_mmu03018_RNA_degradation_Mus_musculus_house_mouse":
                "RNA degradation\n(KEGG)",
            "KEGG__KEGG_mmu04974_Protein_digestion_and_absorption_Mus_musculus_house_mouse":
                "Protein digestion and\nabsorption (KEGG)",
            "REAC__REAC_R_MMU_1474290_Collagen_formation":
                "Collagen formation\n(Reactome)",
            "REAC__REAC_R_MMU_1650814_Collagen_biosynthesis_and_modifying_enzymes":
                "Collagen biosynthesis and\nmodifying enzymes (Reactome)",
        }
        # Boxplots display the full sample-level distributions, while the
        # bracket reports the two-group Welch comparison with BH correction
        # across the six displayed pathways.  The comparison is descriptive
        # for the 12 primary libraries per group and does not add a biological
        # replicate beyond the source experiment.
        profile_axes = []
        profile_points, profile_summaries, profile_records = [], [], []
        for k, (_, row) in enumerate(ftab.iterrows()):
            pathway_scores = selected_scores[selected_scores["term_id"] == row["term_id"]]
            group_data = {}
            for group in display_groups:
                obs = pathway_scores[pathway_scores["group"] == group].sort_values("gsm")
                values = obs["gsva_score"].to_numpy(dtype=float)
                assert len(values) == 12 and obs["mouse id"].nunique() == 12
                expected_column = {"Ctrl": "mean_Control", "CR+IR": "mean_CRIR"}[group]
                expected = float(gsva_selected.iloc[k][expected_column])
                assert np.isclose(values.mean(), expected, atol=1e-12)
                group_data[group] = (obs, values)
            p_value = float(ttest_ind(group_data["CR+IR"][1], group_data["Ctrl"][1],
                                      equal_var=False).pvalue)
            profile_records.append((k, row, group_data, p_value))

        pvals = np.asarray([record[3] for record in profile_records], dtype=float)
        order = np.argsort(pvals)
        qvals = np.empty_like(pvals)
        running = 1.0
        for rank_index in range(len(order) - 1, -1, -1):
            idx = order[rank_index]
            rank = rank_index + 1
            running = min(running, pvals[idx] * len(pvals) / rank)
            qvals[idx] = running

        for record_index, (k, row, group_data, p_value) in enumerate(profile_records):
            row_index, column_index = divmod(k, 3)
            ax = fig.add_axes([
                FIGURE3_LAYOUT["profile_column_starts"][column_index],
                FIGURE3_LAYOUT["profile_row_starts"][row_index],
                FIGURE3_LAYOUT["profile_width"],
                FIGURE3_LAYOUT["profile_height"],
            ])
            profile_axes.append(ax)
            values_by_group = [group_data[group][1] for group in display_groups]
            bp = ax.boxplot(
                values_by_group, positions=[0, 1], widths=0.43,
                patch_artist=True, showfliers=False, manage_ticks=False,
                boxprops={"linewidth": 0.8}, medianprops={"color": INK, "linewidth": 1.0},
                whiskerprops={"linewidth": 0.75}, capprops={"linewidth": 0.75},
            )
            for patch, group in zip(bp["boxes"], display_groups):
                patch.set_facecolor(mpl.colors.to_rgba(GROUP_COLORS[group], 0.20))
                patch.set_edgecolor(GROUP_COLORS[group])
            for line_index, group in enumerate(display_groups):
                for line in bp["whiskers"][line_index * 2: line_index * 2 + 2]:
                    line.set_color(GROUP_COLORS[group])
                for line in bp["caps"][line_index * 2: line_index * 2 + 2]:
                    line.set_color(GROUP_COLORS[group])
                obs, values = group_data[group]
                offsets = (((np.arange(len(values)) * 5) % len(values)
                            - (len(values) - 1) / 2)
                           * FIGURE3_LAYOUT["profile_jitter_step"])
                ax.scatter(line_index + offsets, values,
                           s=FIGURE3_LAYOUT["profile_point_size"],
                           color=GROUP_COLORS[group], alpha=0.70,
                           edgecolor="white", linewidth=0.3, zorder=3)
                mean = float(values.mean())
                ci = float(student_t.ppf(0.975, len(values) - 1)
                           * values.std(ddof=1) / np.sqrt(len(values)))
                ax.errorbar(line_index, mean, yerr=ci, marker="D", markersize=3.0,
                            markerfacecolor="white", markeredgecolor=INK,
                            markeredgewidth=0.55, color=INK, capsize=2.0,
                            elinewidth=0.65, capthick=0.65, zorder=5)
                for display_index, (_, observation) in enumerate(obs.iterrows()):
                    profile_points.append({
                        "term_id": row["term_id"], "term_name": row["term_name"],
                        "source": row["source"], "gsm": observation["gsm"],
                        "mouse_id": observation["mouse id"], "group": group,
                        "gsva_score": observation["gsva_score"],
                        "display_x": float(line_index + offsets[display_index]),
                    })
                profile_summaries.append({
                    "term_id": row["term_id"], "term_name": row["term_name"],
                    "source": row["source"], "group": group,
                    "n": len(values), "mean": mean,
                    "sd": float(values.std(ddof=1)),
                    "ci95_low": mean - ci, "ci95_high": mean + ci,
                    "ci_definition": "two-sided Student-t confidence interval for the group mean",
                    "mean_difference_sarcopenia_minus_control": float(values_by_group[1].mean() - values_by_group[0].mean()),
                    "welch_p": p_value, "BH_q": float(qvals[record_index]),
                })
            assert pathway_scores["gsva_score"].between(*FIGURE3_LAYOUT["profile_ylim"]).all()
            y_max = float(max(values.max() for values in values_by_group))
            y_min = float(min(values.min() for values in values_by_group))
            y_span = max(y_max - y_min, 0.35)
            bracket_y = min(y_max + 0.11 * y_span, 0.68)
            bracket_h = 0.035 * y_span
            ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + bracket_h,
                                   bracket_y + bracket_h, bracket_y],
                    color=INK, lw=0.65, clip_on=False)
            q_value = float(qvals[record_index])
            q_label = "q < 0.001" if q_value < 0.001 else f"q = {q_value:.3g}"
            ax.text(0.5, bracket_y + bracket_h + 0.018, q_label,
                    ha="center", va="bottom", fontsize=5.0, color=INK)
            ax.set_ylim(*FIGURE3_LAYOUT["profile_ylim"])
            ax.set_xlim(-0.35, 1.35)
            ax.set_yticks([-0.5, 0, 0.5])
            ax.set_xticks(range(2), display_group_labels, fontsize=6.0)
            ax.tick_params(axis="y", labelsize=6.0, pad=2, length=2, colors=INK)
            ax.tick_params(axis="x", length=2, pad=3, colors=INK)
            if column_index == 0:
                ax.set_ylabel("GSVA score", fontsize=7.0, labelpad=3)
            ax.set_title(profile_titles[row["term_id"]], loc="left", color=INK,
                         fontsize=FIGURE3_LAYOUT["profile_title_size"], pad=7,
                         linespacing=1.05)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_axisbelow(True)
        pd.DataFrame(profile_points).to_csv(OUT / "figure3_panel_d_gsva_displayed_points.csv", index=False)
        pd.DataFrame(profile_summaries).to_csv(OUT / "figure3_panel_d_gsva_group_summary.csv", index=False)
    else:
        # Fallback for a pre-pathway-analysis run.
        baseline = float(sig_group.loc["Ctrl", "disease_signature"])
        cr_main = float(sig_group.loc["CR", "disease_signature"] - baseline)
        ir_main = float(sig_group.loc["IR", "disease_signature"] - baseline)
        interaction_score = float(sig_group.loc["CR+IR", "disease_signature"] - sig_group.loc["CR", "disease_signature"] - sig_group.loc["IR", "disease_signature"] + baseline)
        vals = [cr_main, ir_main, interaction_score]
        bars = d.bar(["CR main", "IR main", "CR×IR"], vals, color=[CR, IR, PURPLE], width=0.65)
        d.axhline(0, color=INK, lw=0.75)
        d.set_ylabel("Score contrast")
        d.tick_params(axis="x", labelrotation=28)
        for bar, val in zip(bars, vals):
            d.text(bar.get_x() + bar.get_width() / 2, val + (0.05 if val >= 0 else -0.05), f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=5.7)
        d.grid(axis="y", color=GRID, lw=0.45)

    # The gene-level interaction test is intentionally not plotted as another
    # main-panel result. It is still written above to
    # figure3_interaction_coefficients.csv, preserving the negative result and
    # allowing a supplementary figure to be generated without recomputation.
    geometry = []
    for label, ax in zip("abcd", [a, b, c, d]):
        pos = ax.get_position()
        geometry.append({"panel": label, "x0_inches": pos.x0 * fig.get_figwidth(),
                         "y0_inches": pos.y0 * fig.get_figheight(),
                         "width_inches": pos.width * fig.get_figwidth(),
                         "height_inches": pos.height * fig.get_figheight()})
    if len(gsva_selected):
        for k, ax in enumerate(profile_axes):
            pos = ax.get_position()
            geometry.append({"panel": f"d_profile_{k + 1}",
                             "x0_inches": pos.x0 * fig.get_figwidth(),
                             "y0_inches": pos.y0 * fig.get_figheight(),
                             "width_inches": pos.width * fig.get_figwidth(),
                             "height_inches": pos.height * fig.get_figheight()})
    pd.DataFrame(geometry).to_csv(OUT / "figure3_layout_geometry.csv", index=False)
    save_pub(fig, "Figure3_GSE213148_factorial_programs", fixed_canvas=True)


def figure4(meta, expr, de, ann, interaction, cf, enrich, counts, drug):
    m16, primary, expr16, de_named, score, qc, up, down = primary_tables(meta, expr, de, ann, counts)
    cf = cf.sort_values(["mean_reflection_reversal_fraction", "DESeq2_padj"], ascending=[False, True]).head(15).copy()
    cf.to_csv(OUT / "figure4_displayed_counterfactual_candidates.csv", index=False)
    de_named["display_name"] = de_named["gene_name"].fillna(de_named["gene_id"])
    gene_to_fc = de_named.set_index("display_name")["log2FoldChange"].to_dict()

    # Five data-backed panels are reflowed into a balanced 2×6 layout.  The
    # former sixth slot had no independent result and is intentionally omitted
    # rather than filled with a schematic placeholder.
    fig = plt.figure(figsize=(7.2, 5.65), dpi=300)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.26, 1.06], hspace=0.72, wspace=0.46,
                          left=0.13, right=0.975, bottom=0.12, top=0.965)
    a = fig.add_subplot(gs[0, 0:2])
    b = fig.add_subplot(gs[0, 2:4])
    c = fig.add_subplot(gs[0, 4:6])
    d = fig.add_subplot(gs[1, 0:3])
    e = fig.add_subplot(gs[1, 3:6])

    panel_titles = {
        a: "Expression-level sensitivity",
        b: "Disease-score gap",
        c: "Candidate position in DESeq2",
        d: "Candidate feature matrix",
        e: "Candidate summary",
    }
    for ax, subtitle in panel_titles.items():
        ax._panel_title = subtitle

    # Hero panel: paired expression-level counterfactual sensitivity.
    y = np.arange(len(cf))[::-1]
    a.barh(y + 0.17, cf["clamp_reversal_fraction"], height=0.30, color=CTRL, alpha=0.86, label="clamp")
    a.barh(y - 0.17, cf["mean_reflection_reversal_fraction"], height=0.30, color=PURPLE, alpha=0.86, label="mean reflection")
    a.set_yticks(y, cf["gene_name"], fontsize=5.2)
    a.set_xlabel("Expression-level reversal fraction")
    a.legend(loc="lower right", fontsize=5.2, handlelength=1.2)
    a.grid(axis="x", color=GRID, lw=0.45); panel_label(a, "a")

    # Baseline disease-score gap versus the two counterfactual gaps.
    for _, r in cf.head(12).iterrows():
        a0 = r["baseline_50_feature_score_gap"]
        a1 = r["mean_reflection_score_gap"]
        b.plot([a0, a1], [0, 0], alpha=0)  # keep deterministic axis creation
    for i, (_, r) in enumerate(cf.head(12).sort_values("mean_reflection_reversal_fraction").iterrows()):
        x0, x1 = r["baseline_50_feature_score_gap"], r["mean_reflection_score_gap"]
        b.plot([x0, x1], [i, i], color="#B9C2CD", lw=1.3)
        b.scatter([x0, x1], [i, i], c=[INK, PURPLE], s=17, edgecolor="white", lw=0.3, zorder=3)
    sub = cf.head(12).sort_values("mean_reflection_reversal_fraction")
    b.set_yticks(range(len(sub)), sub["gene_name"], fontsize=5.2)
    b.set_xlabel("Disease-score gap")
    b.grid(axis="x", color=GRID, lw=0.45)
    b.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, label="baseline", markersize=4), Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE, label="mean reflection", markersize=4)], fontsize=5.0, loc="lower right")
    panel_label(b, "b")

    # Candidate position in the primary DESeq2 landscape.
    x = de_named["log2FoldChange"].to_numpy(); yv = -np.log10(np.clip(de_named["padj"].fillna(1), 1e-300, 1))
    c.scatter(x, yv, s=3.4, color="#D5DCE5", alpha=0.48, linewidths=0, rasterized=True)
    cand = de_named[de_named["display_name"].isin(cf["gene_name"])]
    c.scatter(cand["log2FoldChange"], -np.log10(np.clip(cand["padj"], 1e-300, 1)), s=17, color=PURPLE, edgecolor="white", lw=0.35, zorder=3)
    for _, r in cand.iterrows():
        c.text(r["log2FoldChange"], -np.log10(max(r["padj"], 1e-300)) + 0.23, r["display_name"], fontsize=5.0, ha="center")
    c.set_xlabel("log2 fold change"); c.set_ylabel("−log10 adjusted P")
    c.grid(color=GRID, lw=0.45); panel_label(c, "c")

    # Candidate feature matrix: no external cohort support is shown.
    tab = cf.head(10).copy().sort_values("mean_reflection_reversal_fraction", ascending=True)
    cols = ["DESeq2_log2FC", "DESeq2_padj", "mean_reflection_reversal_fraction", "mean_reflection_score_gap"]
    mat = tab[cols].astype(float).copy()
    mat["DESeq2_padj"] = -np.log10(np.clip(mat["DESeq2_padj"], 1e-300, 1))
    # Scale each displayed metric across the displayed candidates only; numeric table is exported unchanged.
    mm = mat.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1), axis=0)
    im = d.imshow(mm.values, cmap="vlag" if "vlag" in plt.colormaps() else "RdBu_r", aspect="auto", vmin=-2, vmax=2)
    d.set_yticks(range(len(tab)), tab["gene_name"], fontsize=5.2)
    d.set_xticks(range(len(cols)), ["log2FC", "−log10 padj", "reversal", "gap"], rotation=42, ha="right", fontsize=5.4)
    d.tick_params(length=0)
    cb = fig.colorbar(im, ax=d, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=5.0, length=2); cb.set_label("within-panel z", fontsize=5.2)
    panel_label(d, "d")

    # Compact ranked table keeps the raw values visible without adding another cohort.
    e.axis("off")
    headers = ["gene", "log2FC", "padj", "reversal"]
    xpos = [0.00, 0.40, 0.61, 0.84]
    for xx, h in zip(xpos, headers): e.text(xx, 0.64, h, fontsize=5.4, fontweight="bold")
    for i, (_, r) in enumerate(cf.head(8).iterrows()):
        yy = 0.53 - i * 0.06
        e.plot([0, 1], [yy - 0.035, yy - 0.035], color=GRID, lw=0.35)
        e.text(xpos[0], yy, pretty_gene(r["gene_name"]), fontsize=5.1)
        e.text(xpos[1], yy, f"{r['DESeq2_log2FC']:+.2f}", fontsize=5.2)
        e.text(xpos[2], yy, f"{r['DESeq2_padj']:.1e}", fontsize=5.0)
        e.text(xpos[3], yy, f"{r['mean_reflection_reversal_fraction']:.2f}", fontsize=5.2, color=PURPLE)
    # Keep the table in a fixed 0–1 canvas; otherwise the separator lines
    # autoscale the y-axis and lift the header above the panel title.
    e.set_xlim(0, 1); e.set_ylim(0, 1)
    panel_label(e, "e")

    save_pub(fig, "Figure4_GSE213148_candidate_nodes")


def figure5(meta, expr, de, ann, interaction, cf, enrich, counts, drug):
    # Keep the homologene_150x150 L1000FWD run as the only drug-ranking input.
    drug = drug[drug["run"].eq("homologene_150x150")].copy()
    drug = drug.sort_values(["candidate_rank", "qvalue"]).drop_duplicates("pert_id")
    drug["label"] = drug["official_pert_iname"].fillna(drug["pert_iname"]).fillna(drug["pert_id"])
    drug.to_csv(OUT / "figure5_l1000fwd_rankings_used.csv", index=False)
    top = drug.head(12).copy()
    allr = drug.copy()

    fig = plt.figure(figsize=(7.2, 5.35), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.18, 1.10], hspace=0.78, wspace=0.62)
    a, b, c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    d, e, f = [fig.add_subplot(gs[1, i]) for i in range(3)]

    panel_titles = {
        a: "Drug-signature ranking",
        b: "Full-run score space",
        c: "Top-ranked perturbagens",
        d: "Ranked evidence table",
        e: "Identity-field completeness",
        f: "Evidence boundary",
    }
    for ax, subtitle in panel_titles.items():
        ax._panel_title = subtitle

    # Hero ranking panel.
    # Reverse the plotting order so rank 1 is the top bar.
    dd = top.sort_values("candidate_rank", ascending=False)
    y = np.arange(len(dd))
    a.barh(y, -dd["combined_scores"], color=TEAL, height=0.58, alpha=0.92)
    a.set_yticks(y, [f"{r['pert_id']}" for _, r in dd.iterrows()], fontsize=5.0)
    a.set_xlabel("−combined signature score")
    a.grid(axis="x", color=GRID, lw=0.45); panel_label(a, "a")
    for yi, (_, r) in enumerate(dd.iterrows()):
        a.text(-r["combined_scores"] + 0.18, yi, f"q={r['qvalue']:.2g}", va="center", fontsize=5.0, color=INK)

    # Full run evidence space, with the top five directly labelled.
    b.scatter(-allr["combined_scores"], -np.log10(np.clip(allr["qvalue"], 1e-12, 1)), s=12, c="#C9D2DD", alpha=0.74, edgecolor="white", lw=0.25)
    for _, r in top.head(3).iterrows():
        b.scatter(-r["combined_scores"], -np.log10(max(r["qvalue"], 1e-12)), s=32, c=PURPLE, edgecolor="white", lw=0.35, zorder=3)
    b.set_xlabel("−combined signature score"); b.set_ylabel("−log10 q-value")
    b.grid(color=GRID, lw=0.45); panel_label(b, "b")

    # Top five cards show identity/annotation status without importing another cohort.
    c.axis("off")
    for i, (_, r) in enumerate(top.head(5).iterrows()):
        yy = 0.88 - i * 0.17
        verified = str(r.get("metadata_status", "")) == "verified_current_broad_export"
        ec = TEAL if verified else MID
        c.add_patch(FancyBboxPatch((0.02, yy - 0.055), 0.96, 0.11, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor="#FAFBFC", edgecolor=ec, lw=0.85))
        c.text(0.05, yy + 0.016, f"{int(r['candidate_rank'])}. {r['pert_id']}", fontsize=5.8, fontweight="bold")
        label_text = "" if str(r["label"]) == str(r["pert_id"]) else str(r["label"])
        c.text(0.95, yy + 0.016, label_text[:20], fontsize=5.4, ha="right", color=INK)
        target = r.get("target_annotation") if pd.notna(r.get("target_annotation")) else "target annotation unavailable"
        c.text(0.05, yy - 0.025, textwrap.shorten(str(target).replace("|", "/"), width=55, placeholder="…"), fontsize=5.0, color=INK)
    panel_label(c, "c")

    # q-value-ranked table with explicit data provenance.
    d.axis("off")
    headers = ["rank", "BRD", "q-value", "name / status"]
    xpos = [0.00, 0.15, 0.55, 0.76]
    for xx, h in zip(xpos, headers): d.text(xx, 0.64, h, fontsize=5.4, fontweight="bold")
    for i, (_, r) in enumerate(top.head(8).iterrows()):
        yy = 0.53 - i * 0.06
        d.plot([0, 1], [yy - 0.035, yy - 0.035], color=GRID, lw=0.35)
        d.text(xpos[0], yy, str(int(r["candidate_rank"])), fontsize=5.2)
        d.text(xpos[1], yy, str(r["pert_id"]).replace("BRD-", ""), fontsize=5.0)
        d.text(xpos[2], yy, f"{r['qvalue']:.2g}", fontsize=5.0)
        lab = str(r["label"]) if pd.notna(r["label"]) else "unresolved"
        d.text(xpos[3], yy, textwrap.shorten(lab, width=20, placeholder="…"), fontsize=5.0, color=INK)
    # Use a fixed table canvas so the header remains below the unified panel
    # heading and aligns with the rows in Figure 4e.
    d.set_xlim(0, 1); d.set_ylim(0, 1)
    panel_label(d, "d")

    # Identity fields present in the ranked table; this makes the follow-up gate explicit.
    status_counts = {
        "ranked signature": len(top),
        "current name": int(top["official_pert_iname"].notna().sum()),
        "target annotation": int(top["target_annotation"].notna().sum()),
        "PubChem ID": int(top["official_pubchem_cid"].notna().sum()),
    }
    e.barh(list(status_counts.keys())[::-1], list(status_counts.values())[::-1], color=[TEAL, CTRL, PURPLE, GOLD][::-1], alpha=0.88)
    e.set_xlim(0, max(status_counts.values()) + 2); e.set_xlabel("Candidates (n)")
    for yi, val in enumerate(list(status_counts.values())[::-1]): e.text(val + 0.12, yi, str(val), va="center", fontsize=5.5)
    e.grid(axis="x", color=GRID, lw=0.45); panel_label(e, "e")

    f.axis("off")
    f.add_patch(FancyBboxPatch((0.03, 0.16), 0.94, 0.70, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=LIGHT, edgecolor=GRID, lw=0.8))
    f.text(0.10, 0.56, "Signature reversal\n→ candidate prioritisation", fontsize=6.3, color=TEAL)
    f.text(0.10, 0.38, "Not drug efficacy, target validation,\nor docking / MD evidence", fontsize=5.5, color=INK)
    panel_label(f, "f")
    fig.text(0.5, 0.012, "Drug ranking is derived from the GSE213148 disease signature through the homologene_150x150 L1000FWD run. No external disease cohort is plotted; docking and molecular-dynamics results are reserved for Figure 6.", ha="center", fontsize=6.1, color=INK)
    save_pub(fig, "Figure5_GSE213148_drug_prioritisation")


def write_readme():
    (OUT / "README.md").write_text(
        "# Main-only Figure 2–5 rebuild\n\n"
        "Figures use the GSE213148 16-week mouse experiment as the only plotted biological cohort. "
        "GSE223813, GSE308816 and all other external cohort references are excluded from the images. "
        "Drug ranking in Figure 5 is a downstream computation from the GSE213148 signature; external drug databases are annotations, not cohorts.\n\n"
        "- Figure 2: primary expression landscape with a Control-versus-Sarcopenia PCA, DESeq2 volcano, signature score and top-gene heatmap; library QC remains in source tables.\n"
        "- Figure 3: directional signature distributions, gene-level dot plot, enrichment and six sample-level GSVA profiles.\n"
        "- Figure 4: expression-level counterfactual sensitivity and candidate nodes from the primary bulk model.\n"
        "- Figure 5: GSE213148-derived L1000FWD signature-reversal ranking and chemical-identity gate.\n"
        "- Figure 6 remains reserved for docking and MD after server result files are returned.\n\n"
        "All source tables written by the script are in this directory.\n", encoding="utf-8"
    )


def main():
    data = load_data()
    figure2(*data)
    figure3(*data)
    figure4(*data)
    figure5(*data)
    write_readme()
    print(f"Wrote Figure 2–5 to {OUT}")


if __name__ == "__main__":
    main()

