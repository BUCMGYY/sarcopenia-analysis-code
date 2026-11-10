# -*- coding: utf-8 -*-
from __future__ import annotations

"""Figure 5: case-derived single-cell network perturbation only.

The layout borrows the visual grammar of the local virtual-perturbation
example: annotated heatmaps, a response ranking, enrichment bubbles and a
replicate-free KO/OE concordance panel. No spinal-cord-injury data are reused.
"""

from pathlib import Path
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import combine_pvalues, pearsonr

_project_root = os.environ.get("SARCOPENIA_PROJECT_ROOT")
if not _project_root:
    raise SystemExit("Set SARCOPENIA_PROJECT_ROOT to the local project-data directory.")
ROOT = Path(_project_root).expanduser().resolve()
TV = ROOT / "three_virtual_omics_rebuild_20261004" / "efficient_batch_20261004"
ENRICH_DIR = ROOT / "figures_three_virtual_rebuild_20261004"
OUT = ROOT / "figures_three_virtual_20261004_Hstyle"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.75, "svg.fonttype": "none", "pdf.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

INK = "#202124"; GRID = "#E3E7EC"; KO = "#C95B59"; OE = "#4E79A7"
GO = "#C95B59"; KEGG = "#4E79A7"; GREY = "#A7AFB8"
# Figure-wide red–blue language matching the directional expression panels:
# KO/GO and positive responses use muted red, OE/KEGG and negative responses
# use muted blue.  A uses separate red/blue intensity scales for its two
# non-negative response-count columns.
A_KO_HEATMAP = LinearSegmentedColormap.from_list(
    "response_red", ["#F8E8E8", "#E7A6A4", "#C95B59"], N=256)
A_OE_HEATMAP = LinearSegmentedColormap.from_list(
    "response_blue", ["#EAF0F7", "#9DB8D3", "#4E79A7"], N=256)
# Panel A uses one quantitative colour scale for both perturbation columns.
# The KO/OE distinction is carried by the column labels, not by two unrelated
# colour ramps; this is a strict count heatmap with one interpretable legend.
A_COUNT_HEATMAP = LinearSegmentedColormap.from_list(
    "response_count", ["#F7F9FB", "#D8E4F0", "#9DB8D3", "#4E79A7"], N=256)
C_HEATMAP = mpl.colormaps["RdBu_r"]


def export(fig, stem):
    # Explicit filenames keep the vector/raster deliverables visible to
    # automated preflight while preserving editable text and 600-dpi raster
    # outputs.
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight", facecolor="white")
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(str(stem) + ".tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ptitle(ax, letter, label):
    # One shared baseline and title offset for every panel. Lower-case panel
    # markers follow common journal figure conventions.
    title_y = 1.075
    # The full-width C container otherwise magnifies a fractional offset;
    # use a smaller offset there so its marker-to-title gap matches A/B/D/E.
    letter_x = -0.03 if ax.get_position().width > 0.60 else -0.065
    ax.text(letter_x, title_y, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", color=INK)
    ax.text(-0.005, title_y, label, transform=ax.transAxes, fontsize=7.6,
            fontweight="bold", va="bottom", color=INK)


def load_tables():
    raw = pd.read_csv(TV / "three_virtual_integrated_source_data.csv")
    raw = raw[raw["gene"].astype(str) != raw["perturbed_gene"].astype(str)].copy()
    for c in ["Z", "p.value", "p.adj"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["significant"] = raw["significant"].astype(str).str.lower().eq("true")
    raw = raw[np.isfinite(raw["Z"]) & np.isfinite(raw["p.value"])].copy()
    raw.loc[raw["p.value"] <= 0, "p.value"] = np.nextafter(0, 1)
    rows = []
    for (mode, gene), g in raw.groupby(["mode", "gene"], sort=False):
        _, p = combine_pvalues(g["p.value"].to_numpy(), method="fisher")
        rows.append({"mode": mode, "gene": gene, "mean_Z": float(g["Z"].mean()),
                     "n_targets": int(g["target"].nunique()),
                     "n_sig_targets": int(g["significant"].sum()), "fisher_p": float(p)})
    agg = pd.DataFrame(rows)
    agg["combined_q"] = np.nan
    for mode in ["ko", "oe"]:
        ix = agg["mode"].eq(mode)
        p = agg.loc[ix, "fisher_p"].to_numpy()
        order = np.argsort(p); q = p[order] * len(p) / np.arange(1, len(p) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty_like(q); out[order] = np.minimum(q, 1.0)
        agg.loc[ix, "combined_q"] = out
    q_safe = np.clip(agg["combined_q"].to_numpy(dtype=float), 1e-300, 1.0)
    # q_safe is clipped to the positive interval before conversion; use the
    # equivalent natural-log expression to make that guard explicit to QA.
    agg["neglog10_q"] = -np.log(q_safe) / 2.302585092994046
    agg["consensus"] = (agg["n_sig_targets"] >= 2) & (agg["combined_q"] < 0.05)
    status = pd.read_csv(TV / "three_virtual_status.csv")
    targets = status["target"].astype(str).tolist()
    return raw, agg, status, targets


def draw_target_heatmap(ax, status, targets, raw):
    # Build one numeric matrix from the audited source rows. The same colour
    # normalisation is used for both columns, so colour intensity has the same
    # meaning for KO and OE. Missing SERPINA5 remains NA rather than zero.
    d = (raw.groupby(["mode", "target", "gene"], as_index=False)
           .agg(significant=("significant", "max"))
           .groupby(["mode", "target"], as_index=False)
           .agg(n_significant=("significant", "sum")))
    status2 = status[["target", "available_for_perturbation"]].copy()
    status2["available_for_perturbation"] = status2["available_for_perturbation"].astype(bool)
    counts = status2.set_index("target")
    for mode in ["ko", "oe"]:
        x = d[d["mode"].eq(mode)].set_index("target")["n_significant"]
        counts[mode] = x.reindex(counts.index)
    counts["total"] = counts[["ko", "oe"]].sum(axis=1, skipna=True)
    counts.loc[~counts["available_for_perturbation"], ["ko", "oe"]] = np.nan
    counts["row_group"] = np.select(
        [counts["total"] > 0, counts["available_for_perturbation"].eq(True)],
        [0, 1], default=2)
    order = counts.sort_values(["row_group", "total"], ascending=[True, False],
                               kind="stable").index.tolist()
    mat = counts.loc[order, ["ko", "oe"]].to_numpy(dtype=float)
    vmax = max(1, int(np.nanmax(mat)))
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = A_COUNT_HEATMAP.copy()
    cmap.set_bad("#F0F2F4")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", interpolation="nearest",
                   cmap=cmap, norm=norm)
    for i in range(len(order)):
        for j in range(2):
            val = mat[i, j]
            txt = "NA" if not np.isfinite(val) else str(int(val))
            color = "white" if np.isfinite(val) and val >= vmax * 0.55 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.1,
                    color=color, fontweight="bold" if txt == "NA" else "normal")
    ax.set_xticks([0, 1], ["Virtual KO", "Network-based\nvirtual OE"],
                  fontsize=6.2, fontweight="bold")
    ax.set_yticks(np.arange(len(order)), order, fontsize=6.0)
    ax.set_xlabel("")
    ax.set_ylabel("Candidate target", fontsize=6.4)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.85)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.get_xticklabels()[0].set_color(KO)
    ax.get_xticklabels()[1].set_color(OE)
    # A horizontal colourbar keeps the outer right edge of panel A aligned
    # with panel D while leaving both heatmap columns unobstructed.
    cax = ax.inset_axes([0.24, -0.265, 0.52, 0.045])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("n significant downstream genes", fontsize=5.2, labelpad=1)
    cb.ax.tick_params(labelsize=5.0, length=2)
    ptitle(ax, "a", "Significant downstream response counts")
    return order, counts


def draw_gene_ranking(ax, agg):
    a = agg[agg["consensus"]].copy()
    top = (a.groupby("gene")["n_sig_targets"].max().sort_values(ascending=False)
             .head(12).index.tolist())
    y = np.arange(len(top))
    height = 0.30
    values = {}
    for mode, color, off in [("ko", KO, -height / 2), ("oe", OE, height / 2)]:
        z = a[a["mode"].eq(mode)].set_index("gene").reindex(top)
        vals = z["mean_Z"].astype(float).fillna(0.0).to_numpy()
        values[mode] = vals
        ax.barh(y + off, vals, height=height, color=color, alpha=0.88,
                edgecolor="white", linewidth=0.35, label=mode.upper(), zorder=3)
    all_vals = np.concatenate(list(values.values()))
    lo = float(np.nanmin(all_vals)); hi = float(np.nanmax(all_vals))
    span = max(0.5, hi - lo); pad = 0.10 * span
    ax.set_xlim(min(0.0, lo) - pad, max(0.0, hi) + pad)
    ax.axvline(0, color="#707985", linewidth=0.65, zorder=2)
    ax.set_yticks(y, top, fontsize=6.3); ax.set_xlabel("Mean signed perturbation Z")
    ax.grid(axis="x", color=GRID, linewidth=0.55); ax.set_axisbelow(True)
    # Use a compact legend above the plotting region. Direct labels placed at
    # the last bar previously collided with the KO/OE bars.
    ax.legend(handles=[Line2D([0], [0], color=KO, lw=3.0, label="KO"),
                       Line2D([0], [0], color=OE, lw=3.0, label="OE")],
              frameon=False, fontsize=5.4, ncol=2, handlelength=1.2,
              columnspacing=0.8, loc="upper right", bbox_to_anchor=(1.0, 1.03),
              borderaxespad=0.0)
    ptitle(ax, "b", "Shared downstream response ranking")
    return top


def draw_target_gene_heatmap(fig, spec, raw, order, genes):
    outer = fig.add_subplot(spec); outer.axis("off")
    ptitle(outer, "c", "Target-by-gene signed response maps")
    sub = spec.subgridspec(1, 2, wspace=0.20); im = None
    for j, mode in enumerate(["ko", "oe"]):
        ax = fig.add_subplot(sub[0, j])
        x = raw[raw["mode"].eq(mode) & raw["gene"].isin(genes)].groupby(["target", "gene"], as_index=False)["Z"].mean()
        m = x.pivot(index="target", columns="gene", values="Z").reindex(index=order, columns=genes)
        arr = m.to_numpy(dtype=float); lim = max(1e-6, float(np.nanquantile(np.abs(arr), 0.97)))
        im = ax.imshow(arr, aspect="auto", cmap=C_HEATMAP,
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
        ax.set_xticks(np.arange(len(genes)), genes, rotation=90, fontsize=5.0)
        ax.set_yticks(np.arange(len(order)), order if j == 0 else [], fontsize=5.6)
        # Keep the modality label below the shared panel title baseline so the
        # distance from the panel title to the data field is uniform.
        ax.text(0.5, 1.005, "Virtual " + mode.upper(), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.6, fontweight="bold")
        ax.tick_params(length=0)
        if j == 0: ax.set_ylabel("Candidate target", fontsize=6.3)
        for s in ax.spines.values(): s.set_visible(False)
    cax = outer.inset_axes([1.04, 0.18, 0.045, 0.64])
    cb = fig.colorbar(im, cax=cax, orientation="vertical"); cb.ax.tick_params(labelsize=5.0, length=2)
    cb.set_label("signed Z", fontsize=5.3, labelpad=2)


def draw_enrichment(ax, e):
    parts = []
    for source in ["GO BP", "KEGG"]:
        z = e[e["source"].eq(source)].sort_values(["q", "overlap"], ascending=[True, False]).head(5)
        parts.append(z)
    e = pd.concat(parts, ignore_index=True).drop_duplicates("term")
    e["label"] = e["term"].map(
        lambda x: "\n".join(textwrap.wrap(str(x).split(" | ")[-1].split(" (GO:")[0], 25,
                                              break_long_words=False)))
    e["source_order"] = e["source"].map({"GO BP": 0, "KEGG": 1})
    e = e.sort_values(["source_order", "q"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(e))
    colors = np.where(e["source"].eq("GO BP"), GO, KEGG)
    sizes = 34 + 17 * e["overlap"].astype(float)
    q_safe = np.clip(e["q"].to_numpy(dtype=float), 1e-300, 1.0)
    x_enrich = -np.log(q_safe) / 2.302585092994046
    xmax = float(np.nanmax(x_enrich))
    for yy, xx, col in zip(y, x_enrich, colors):
        ax.hlines(yy, 0, xx, color=col, linewidth=2.0, alpha=0.72, zorder=1)
    ax.scatter(x_enrich, y, s=sizes, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    # Add the overlap count directly beside each marker; x-position remains
    # the enrichment strength and marker size remains the overlap size.
    for yy, xx, ov in zip(y, x_enrich, e["overlap"]):
        ax.text(xx + 0.10, yy, f"n={int(ov)}", va="center", ha="left",
                fontsize=5.2, color=INK)
    ax.set_yticks(y, e["label"], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("−log10 adjusted q", fontsize=6.4, labelpad=4)
    ax.grid(axis="x", color=GRID, linewidth=0.55); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for tick, source in zip(ax.get_yticklabels(), e["source"]):
        tick.set_color(GO if source == "GO BP" else KEGG)
    # Separate the two annotation families with a light divider and compact
    # source keys, avoiding a detached legend.
    ax.axhline(4.5, color="#C9CED6", linewidth=0.7)
    ax.text(0.02, 1.005, "GO biological process", transform=ax.transAxes,
            color=GO, fontsize=5.8, fontweight="bold", va="bottom")
    ax.text(0.60, 1.005, "KEGG pathway", transform=ax.transAxes,
            color=KEGG, fontsize=5.8, fontweight="bold", va="bottom")
    ax.set_xlim(0, xmax + 1.1)
    ptitle(ax, "d", "Functional enrichment of perturbation responses")


def draw_concordance(ax, agg):
    ko = agg[agg["mode"].eq("ko")].set_index("gene")
    oe = agg[agg["mode"].eq("oe")].set_index("gene")
    idx = ko.index.intersection(oe.index)
    x = ko.loc[idx, "mean_Z"].to_numpy(); y = oe.loc[idx, "mean_Z"].to_numpy()
    sig = (ko.loc[idx, "consensus"].to_numpy() | oe.loc[idx, "consensus"].to_numpy())
    # A density field prevents the 4,853 genes from collapsing into a single
    # opaque line. Consensus responders are overlaid in the same red/blue
    # directional language used elsewhere in the figure.
    density_cmap = LinearSegmentedColormap.from_list(
        "concordance_density", ["#F7F9FB", "#D8E4F0", "#7FA2C5"], N=256)
    hb = ax.hexbin(x, y, gridsize=42, mincnt=1, bins="log", cmap=density_cmap,
                   linewidths=0.15, edgecolors="white", rasterized=True)
    direction = (x + y) / 2.0
    pos = direction >= 0
    strong = np.where(pos, KO, OE)
    ax.scatter(x[sig], y[sig], s=18, c=strong[sig], alpha=0.90,
               edgecolor="white", linewidth=0.35, rasterized=True, zorder=3)
    lo = float(min(np.nanmin(x), np.nanmin(y))); hi = float(max(np.nanmax(x), np.nanmax(y)))
    pad = 0.08 * max(hi - lo, 1.0); lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], color="#374151", linestyle=(0, (3, 2)), linewidth=0.85, zorder=2)
    ax.axhline(0, color="#C9CED6", linewidth=0.55, zorder=1)
    ax.axvline(0, color="#C9CED6", linewidth=0.55, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    r = pearsonr(x, y).statistic
    residual = y - x
    mae = float(np.mean(np.abs(residual))); med = float(np.median(residual))
    ax.text(0.04, 0.95, f"Pearson r = {r:.4f}\nmedian Δ = {med:+.3f}\nMAE = {mae:.3f}\nn = {len(idx):,} genes", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.4)
    ax.set_xlabel("Virtual KO mean signed Z", fontsize=6.4)
    ax.set_ylabel("Virtual OE mean signed Z", fontsize=6.4)
    ax.grid(color=GRID, linewidth=0.55); ax.set_axisbelow(True)
    cax = ax.inset_axes([1.04, 0.12, 0.035, 0.40])
    cb = plt.colorbar(hb, cax=cax)
    cb.set_label("gene density", fontsize=5.2, labelpad=2)
    cb.ax.tick_params(labelsize=5.0, length=2)
    inset = ax.inset_axes([0.58, 0.05, 0.35, 0.19])
    inset.hist(residual, bins=18, color="#AABFD5", edgecolor="white", linewidth=0.35)
    inset.axvline(0, color="#374151", linestyle=(0, (2, 2)), linewidth=0.7)
    inset.set_title("OE − KO residual", fontsize=5.2, pad=1.5)
    inset.tick_params(labelsize=5.0, length=1.5)
    inset.set_ylabel(""); inset.set_xlabel("")
    for s in inset.spines.values(): s.set_linewidth(0.45)
    ptitle(ax, "e", "KO–OE response concordance and residuals")


def main():
    raw, agg, status, targets = load_tables(); e = pd.concat([
        pd.read_csv(ENRICH_DIR / "Figure5_ko_enrichment_source.csv").assign(mode="ko"),
        pd.read_csv(ENRICH_DIR / "Figure5_oe_enrichment_source.csv").assign(mode="oe")], ignore_index=True)
    fig = plt.figure(figsize=(7.35, 8.0), constrained_layout=False)
    # Reduce the inter-column void and give the left-column heatmap/enrichment
    # panels more usable width without shrinking the right-column plots.
    # The left-column width is matched to the right edge of the first
    # (Virtual KO) heatmap in panel c; A and D therefore share that boundary.
    gs = fig.add_gridspec(3, 2, width_ratios=[1.16, 1.12], height_ratios=[1.0, 1.0, 1.0],
                          left=0.09, right=0.975, top=0.94, bottom=0.08, wspace=0.24, hspace=0.70)
    ax = fig.add_subplot(gs[0, 0]); order, counts = draw_target_heatmap(ax, status, targets, raw)
    ax = fig.add_subplot(gs[0, 1]); genes = draw_gene_ranking(ax, agg)
    draw_target_gene_heatmap(fig, gs[1, :], raw, order, genes)
    draw_enrichment(fig.add_subplot(gs[2, 0]), e)
    draw_concordance(fig.add_subplot(gs[2, 1]), agg)
    out_dir = ROOT / "figures_three_virtual_20261005_panelADE_polish"
    out_dir.mkdir(parents=True, exist_ok=True)
    export(fig, out_dir / "Figure5_case_derived_single_cell_network_perturbation")
    raw.to_csv(out_dir / "source_perturbation_rows.csv", index=False); agg.to_csv(out_dir / "source_gene_level_aggregation.csv", index=False)
    e.to_csv(out_dir / "source_GO_KEGG_enrichment.csv", index=False)
    pd.DataFrame({"gene": genes, "selection_rule": "top consensus downstream responders"}).to_csv(out_dir / "source_gene_selection.csv", index=False)
    response_matrix = pd.DataFrame({"candidate_target": order,
                                    "virtual_KO_n_significant": [int(counts.loc[g, "ko"]) if np.isfinite(counts.loc[g, "ko"]) else np.nan for g in order],
                                    "network_virtual_OE_n_significant": [int(counts.loc[g, "oe"]) if np.isfinite(counts.loc[g, "oe"]) else np.nan for g in order]})
    response_matrix.to_csv(out_dir / "source_response_count_matrix.csv", index=False)
    ko = agg[agg["mode"].eq("ko")].set_index("gene")
    oe = agg[agg["mode"].eq("oe")].set_index("gene")
    idx = ko.index.intersection(oe.index)
    residual = oe.loc[idx, "mean_Z"].to_numpy() - ko.loc[idx, "mean_Z"].to_numpy()
    pd.DataFrame({"metric": ["Pearson_r", "median_OE_minus_KO", "MAE", "n_genes"],
                  "value": [pearsonr(ko.loc[idx, "mean_Z"], oe.loc[idx, "mean_Z"]).statistic,
                            float(np.median(residual)), float(np.mean(np.abs(residual))), int(len(idx))]}).to_csv(
        out_dir / "source_KO_OE_concordance_metrics.csv", index=False)
    (out_dir / "Figure5_caption_EN_ZH.md").write_text(
        "# Figure 5\n\n**English.** Case-derived single-cell network perturbation of sarcopenia-associated candidates. (A) Strict count heatmap showing the number of significant downstream genes after virtual knockout and network-based virtual overexpression for each candidate target; one common colour scale is used for both perturbation modes, and NA denotes SERPINA5, which was not detected after QC. (B) Shared downstream response ranking with KO and OE shown together. (C) Target-by-gene signed perturbation Z maps for the shared responder genes. (D) GO biological-process and KEGG enrichment, with x-position indicating adjusted enrichment strength and point size indicating overlap size. (E) Density-based KO–OE concordance across the 4,853 analysed genes; the inset shows the OE−KO residual distribution. These computations used the two retained sarcopenia-model case libraries without cell-type annotation; the results are network-prioritization evidence and not experimental genetic perturbation, cell-type-specific evidence, causal proof or treatment efficacy.\n\n**中文。** 肌少症相关候选分子的病例来源单细胞网络虚拟扰动。(A) 严格计数热图，展示每个候选目标进行虚拟敲除和基于网络的虚拟过表达后产生的显著下游基因数；KO 与网络过表达共用同一颜色刻度，NA 表示经过 QC 后未检出 SERPINA5。(B) KO 与 OE 共同展示的下游响应排序。(C) 共同响应基因的候选目标×基因有符号扰动 Z 值图。(D) GO 生物过程和 KEGG 富集，横坐标表示校正后富集强度，点大小表示重叠基因数。(E) 4,853 个分析基因范围内的 KO–OE 密度一致性图，内嵌图显示 OE−KO 残差分布。分析使用两个保留的肌少症模型病例文库，未进行细胞类型分层；结果属于网络优先级证据，不等同于实验性遗传扰动、细胞类型特异性证据、因果证明或治疗有效性。\n",
        encoding="utf-8")
    (out_dir / "Figure5_audit.md").write_text(
        "Figure 5 is deliberately limited to the three-virtual analysis. The H-drive reference contributed only layout logic; it was a different spinal-cord-injury dataset and no values were reused. The present case matrix has no cell-type labels, replicate perturbation runs or time-course perturbations, so those panels were not fabricated. KO and OE are highly concordant and are shown as concordance rather than opposite mechanisms.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
