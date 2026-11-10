from __future__ import annotations

"""Result-oriented H&E--Visium plate following the previous manuscript style.

Each panel carries a measured spatial program score over the complete H&E
capture area.  There are no context-only panels and no expression-selected ROI.
GSE223813 is retained as a technical injury/dystrophy reference and is never
labelled as a sarcopenia case-control cohort.
"""

from io import BytesIO
import importlib.util
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy import ndimage

_project_root = os.environ.get("SARCOPENIA_PROJECT_ROOT")
if not _project_root:
    raise SystemExit("Set SARCOPENIA_PROJECT_ROOT to the local project-data directory.")
ROOT = Path(_project_root).expanduser().resolve()
OUT = ROOT / "figures_main_only_20260929"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_SCRIPT = ROOT / "figure23_reconstruction_20260929" / "scripts" / "rebuild_figures_2_3.py"

spec = importlib.util.spec_from_file_location("audited_spatial_source", SOURCE_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(SOURCE_SCRIPT)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.linewidth": 0.55,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})

INK = "#111827"
MODULES = {
    "atrophy_proteostasis": "Atrophy/proteostasis",
    "contractile_fiber": "Contractile fiber",
}
GSM_INFO = {
    "GSM6996261": "MDX GAST · Rep 1",
    "GSM6996262": "MDX GAST · Rep 2",
}

# Physical panel geometry avoids aspect='equal' shrinking the axes inside
# overly wide GridSpec cells and silently recreating a large central gutter.
LAYOUT_MM = {
    "width": 184.15,
    "height": 176.0,
    "left": 2.0,
    "bottom": 2.0,
    "panel_side": 80.5,
    "column_gap": 1.67,
    "row_gap": 7.0,
    "colorbar_gap": 3.0,
    "colorbar_width": 2.5,
}
# Preserve spot diameter relative to the unchanged image crop when enlarging
# the panels; these are display sizes only, not inferred biological sizes.
PREVIOUS_PANEL_SIDE_IN = 5.55 * (0.972 - 0.022) / (2.0 + 0.075)
SPOT_AREA_PT2 = 6.9 * (LAYOUT_MM["panel_side"] / 25.4 / PREVIOUS_PANEL_SIDE_IN) ** 2


def genes_for(program: str) -> list[str]:
    tab = pd.read_csv(ROOT / "figure23_reconstruction_20260929" / "data" / "program_gene_sets.tsv", sep="\t")
    return tab.loc[tab["program"].eq(program), "gene"].astype(str).tolist()


def load_scores(gsm: str, program: str) -> tuple[np.ndarray, pd.DataFrame, dict]:
    spatial_dir, image, positions, qc = source.gse223_positions(gsm)
    names, barcodes, data, indices, indptr, _ = source.read_h5(spatial_dir / "filtered_feature_bc_matrix.h5")
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(barcodes), len(names)))
    name_to_idx = {name: i for i, name in enumerate(names)}
    genes = genes_for(program)
    present = [g for g in genes if g in name_to_idx]
    idx = [name_to_idx[g] for g in present]
    if not idx:
        raise ValueError(f"{gsm}: no genes present for {program}")
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    normalized = matrix[:, idx].multiply(1e4 / np.maximum(totals, 1.0)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    score = np.asarray(normalized.mean(axis=1)).ravel()
    detected = np.asarray(matrix[:, idx].getnnz(axis=1)).ravel()
    score_map = pd.DataFrame({
        "gsm_id": gsm, "barcode": barcodes, "score": score,
        "n_program_genes_detected": detected, "total_counts": totals,
    })
    score_map = positions.merge(score_map, on="barcode", how="inner")
    score_map = score_map.loc[score_map["in_tissue"].eq(1)].copy()
    score_map["x"] = score_map["pxl_col"] * float(qc["tissue_hires_scalef"])
    score_map["y"] = score_map["pxl_row"] * float(qc["tissue_hires_scalef"])
    return image, score_map, {**qc, "genes_present": present}


def save_pub(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    buf = BytesIO()
    fig.savefig(buf, format="tiff", dpi=600, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    with Image.open(buf) as im:
        im.convert("RGB").save(OUT / f"{stem}.tiff", dpi=(600, 600), compression="tiff_lzw")
    buf.close()
    plt.close(fig)


def tissue_component_mask(image: np.ndarray) -> np.ndarray:
    """Return the largest H&E tissue component, independent of expression."""
    arr = image.astype(float)
    saturation = (arr.max(axis=2) - arr.min(axis=2)) / 255.0
    mask = (saturation > 0.10) & (arr.mean(axis=2) < 245.0)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=5)
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def tissue_centered_crop(image: np.ndarray, crop_size: int = 1120) -> tuple[float, float, float, float]:
    """Return a fixed-size, morphology-derived crop around the tissue.

    The crop is determined from the H&E image only (largest saturated tissue
    component), never from expression values or score hotspots. A common
    square crop makes the four panels occupy the same visual area and removes
    uninformative chip margins.
    """
    mask = tissue_component_mask(image)
    if not mask.any():
        h, w = image.shape[:2]
        return (0.0, 0.0, float(w), float(h))
    ys, xs = np.where(mask)
    cx, cy = float((xs.min() + xs.max()) / 2.0), float((ys.min() + ys.max()) / 2.0)
    h, w = image.shape[:2]
    half = float(crop_size) / 2.0
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    if x0 < 0: x1 -= x0; x0 = 0.0
    if y0 < 0: y1 -= y0; y0 = 0.0
    if x1 > w: x0 -= (x1 - w); x1 = float(w)
    if y1 > h: y0 -= (y1 - h); y1 = float(h)
    return (max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1))


def draw_panel(ax, image, tab: pd.DataFrame, norm: Normalize, title: str, label: str, cmap: str = "magma"):
    h, w = image.shape[:2]
    ax.imshow(image, extent=(0, w, h, 0), interpolation="nearest", aspect="equal")
    visual_mask = tissue_component_mask(image)
    xi = np.clip(np.rint(tab["x"].to_numpy()).astype(int), 0, w - 1)
    yi = np.clip(np.rint(tab["y"].to_numpy()).astype(int), 0, h - 1)
    tab = tab.loc[visual_mask[yi, xi]].copy()
    # Draw only measured non-zero program spots. This keeps the H&E morphology
    # readable and avoids filling the figure with uninformative chip-edge
    # circles; the complete barcode-level score table is retained in source data.
    positive = tab.loc[tab["score"] > 0].copy()
    sc = ax.scatter(positive["x"], positive["y"], c=positive["score"], cmap=cmap,
                    norm=norm, s=SPOT_AREA_PT2, alpha=0.83, linewidths=0,
                    rasterized=True, zorder=3)
    crop = tissue_centered_crop(image)
    cx0, cy0, cx1, cy1 = crop
    ax.set_xlim(cx0, cx1); ax.set_ylim(cy1, cy0)
    crop_w, crop_h = cx1 - cx0, cy1 - cy0
    x0, y0 = cx0 + 0.06 * crop_w, cy0 + 0.93 * crop_h
    # The platform scale is derived from the audited spot diameter and hires
    # factor, not guessed from the displayed pixel dimensions.
    spot_d = float(tab.attrs.get("spot_diameter_hires", 55.0))
    scale_px = spot_d / 55.0 * 500.0
    ax.plot([x0, x0 + scale_px], [y0, y0], color="white", lw=2.6, solid_capstyle="butt", zorder=4)
    ax.text(x0 + scale_px / 2, y0 - 0.018 * h, "500 µm", color="white", ha="center",
            va="bottom", fontsize=5.7, fontweight="bold", zorder=4)
    ax.axis("off")
    # The panel letter and title form one compact publication-style heading.
    # Keeping them outside the image avoids a second title block and removes
    # the large vertical gaps present in the earlier version.
    ax.text(0.0, 1.012, f"{label}  {title}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.35, fontweight="bold",
            color=INK, clip_on=False)
    return sc


def main() -> None:
    gsms = ["GSM6996261", "GSM6996262"]
    payload = {}
    source_rows = []
    for program in MODULES:
        for gsm in gsms:
            image, tab, qc = load_scores(gsm, program)
            tab.attrs["spot_diameter_hires"] = float(qc.get("spot_diameter_fullres", 55.0)) * float(qc.get("tissue_hires_scalef", 1.0))
            payload[(program, gsm)] = (image, tab, qc)
            out = tab[["gsm_id", "barcode", "x", "y", "in_tissue", "score", "n_program_genes_detected", "total_counts"]].copy()
            out["program"] = program
            source_rows.append(out)
    pd.concat(source_rows, ignore_index=True).to_csv(OUT / "gse223813_result_plate_spot_scores.csv", index=False, encoding="utf-8-sig")

    norms = {}
    for program in MODULES:
        vals = np.concatenate([payload[(program, gsm)][1]["score"].to_numpy() for gsm in gsms])
        lo, hi = np.quantile(vals, [0.02, 0.98])
        norms[program] = Normalize(vmin=float(lo), vmax=float(hi))

    fig = plt.figure(figsize=(7.25, LAYOUT_MM["height"] / 25.4), dpi=300)

    def panel_rect(row: int, col: int) -> list[float]:
        side = LAYOUT_MM["panel_side"]
        x = LAYOUT_MM["left"] + col * (side + LAYOUT_MM["column_gap"])
        y = LAYOUT_MM["bottom"] + (1 - row) * (side + LAYOUT_MM["row_gap"])
        return [x / LAYOUT_MM["width"], y / LAYOUT_MM["height"],
                side / LAYOUT_MM["width"], side / LAYOUT_MM["height"]]

    axes = np.asarray([
        [fig.add_axes(panel_rect(0, 0)), fig.add_axes(panel_rect(0, 1))],
        [fig.add_axes(panel_rect(1, 0)), fig.add_axes(panel_rect(1, 1))],
    ])
    labels = [["a", "b"], ["c", "d"]]
    for row, program in enumerate(MODULES):
        for col, gsm in enumerate(gsms):
            image, tab, qc = payload[(program, gsm)]
            title = f"{gsm} | {MODULES[program]}"
            sc = draw_panel(axes[row, col], image, tab, norms[program], title, labels[row][col],
                            cmap="magma" if row == 0 else "viridis")
        # One shared colorbar per row, preserving direct comparison between
        # the two libraries without suggesting a cross-cohort effect.
        right_panel = panel_rect(row, 1)
        cax = fig.add_axes([
            right_panel[0] + right_panel[2] + LAYOUT_MM["colorbar_gap"] / LAYOUT_MM["width"],
            right_panel[1], LAYOUT_MM["colorbar_width"] / LAYOUT_MM["width"], right_panel[3],
        ])
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=5.8, width=0.4, length=2)
        cb.set_label("mean log1p(CP10K)", fontsize=6.0, labelpad=4)

    fig.canvas.draw()
    for row in range(2):
        left_box, right_box = axes[row, 0].get_position(), axes[row, 1].get_position()
        gap_mm = (right_box.x0 - left_box.x1) * LAYOUT_MM["width"]
        if not np.isclose(gap_mm, LAYOUT_MM["column_gap"], atol=0.01):
            raise ValueError(f"Unexpected row {row} horizontal gap: {gap_mm:.3f} mm")

    # The spatial plate is Figure 4 in the seven-figure manuscript order;
    # Figure 1 is reserved for the study-architecture flowchart.
    stem = "Figure4_GSE223813_HE_Visium_result_plate"
    save_pub(fig, stem)
    # Preserve the canonical aliases only after the result-oriented plate is
    # successfully rendered, so older files remain recoverable.
    for suffix in ("png", "svg", "pdf", "tiff"):
        (OUT / f"Figure4_GSE223813_HE_Visium.{suffix}").write_bytes((OUT / f"{stem}.{suffix}").read_bytes())
    manifest = {
        "figure": stem,
        "source_dataset": "GSE223813",
        "display_libraries": gsms,
        "layout": "2x2; rows are programs, columns are same-condition MDX GAST replicate libraries; every panel is H&E plus measured spot-level score",
        "layout_mm": LAYOUT_MM,
        "programs": MODULES,
        "color_scale": "one robust 2%-98% scale per program shared between the two libraries",
        "image_display": "common 1120 x 1120 hires-pixel crop centered on the largest H&E tissue component; no expression-based cropping",
        "claim_boundary": "spatial program scores in same-condition MDX GAST injury/dystrophy material; replicate-oriented spatial display, not a sarcopenia case-control comparison",
        "source_script": "rebuild_main_he_visium_result_plate.py",
    }
    (OUT / f"{stem}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {stem}")


if __name__ == "__main__":
    main()
