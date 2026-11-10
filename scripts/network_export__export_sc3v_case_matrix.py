from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse, io

_project_root = os.environ.get("SARCOPENIA_PROJECT_ROOT")
if not _project_root:
    raise SystemExit("Set SARCOPENIA_PROJECT_ROOT to the local project-data directory.")
ROOT = Path(_project_root).expanduser().resolve()
INFILE = ROOT / "GSE316316_snRNA" / "GSE316316_snRNA_qc.h5ad"
OUT = ROOT / "three_virtual_omics_rebuild_20261004" / "input_case_matrix"
OUT.mkdir(parents=True, exist_ok=True)

adata = ad.read_h5ad(INFILE, backed="r")
mask = np.asarray(adata.obs["condition"].astype(str) == "Frail_sarcopenia_model")
case = adata[mask].to_memory()
X = case.layers["counts"]
if not sparse.issparse(X):
    X = sparse.csr_matrix(X)
X = X.tocsr()
Xraw = X.copy()

# The original course script uses Seurat VST HVG selection. We use Scanpy's
# Seurat-compatible dispersion implementation only to materialize the fixed
# 5,000-gene input; the exact implementation is recorded in the manifest.
tmp = ad.AnnData(X=X, obs=case.obs.copy(), var=case.var.copy())
gene_symbols = pd.Index(case.var["gene_symbol"].astype(str))
keep = ~gene_symbols.duplicated(keep="first")
X = Xraw[:, keep]
gene_symbols = gene_symbols[keep]
tmp = ad.AnnData(X=X, obs=case.obs.copy(), var=pd.DataFrame(index=gene_symbols))
tmp.var_names_make_unique(join="__dup")
# Seurat's VST starts from library-size-normalized log counts. Materialize the
# same prerequisite before Scanpy's Seurat-dispersion implementation; the R
# package below still receives the original integer counts.
sc.pp.normalize_total(tmp, target_sum=1e4)
sc.pp.log1p(tmp)
sc.pp.highly_variable_genes(tmp, n_top_genes=5000, flavor="seurat", inplace=True)
hvg = list(tmp.var_names[tmp.var["highly_variable"]])
targets = pd.read_csv(ROOT / "GSE316316_snRNA" / "virtual_target_manifest_top20.csv")["gene"].astype(str).tolist()
# The matrix uses canonical mouse symbols (e.g. Pvalb), while the human bulk
# manifest is uppercase. Resolve case-insensitively and retain the real matrix
# symbol in the R input; the mapping is written to target_mapping.csv.
lookup = {str(g).upper(): str(g) for g in tmp.var_names}
target_map = pd.DataFrame({"human_manifest_gene": targets, "mouse_matrix_gene": [lookup.get(g.upper(), "") for g in targets]})
target_map.to_csv(OUT / "target_mapping.csv", index=False)
mapped_targets = [lookup[g.upper()] for g in targets if g.upper() in lookup]
selected = list(dict.fromkeys(hvg + mapped_targets))
idx = [tmp.var_names.get_loc(g) for g in selected]
Xsel = X[:, idx].tocsc().T  # course interface: genes x cells
io.mmwrite(OUT / "case_counts.mtx", Xsel)
pd.Series(selected, name="gene").to_csv(OUT / "genes.tsv", index=False)
pd.Series(case.obs_names.astype(str), name="cell").to_csv(OUT / "cells.tsv", index=False)
case.obs.to_csv(OUT / "case_cells_metadata.csv", index=True)
mapped_target_set = set(mapped_targets)
pd.DataFrame({"gene": selected, "is_hvg": [g in hvg for g in selected], "is_target": [g in mapped_target_set for g in selected]}).to_csv(OUT / "selected_genes.csv", index=False)
manifest = {
    "input": str(INFILE),
    "condition": "Frail_sarcopenia_model",
    "n_cells": int(Xsel.shape[1]),
    "n_genes": int(Xsel.shape[0]),
    "n_hvg": int(sum(g in hvg for g in selected)),
    "target_genes_present": mapped_targets,
    "target_genes_missing": [g for g in targets if g.upper() not in lookup],
    "hvg_materialization": "scanpy.pp.highly_variable_genes flavor=seurat n_top_genes=5000; exact source implementation recorded before R run",
    "orientation": "genes x cells after Matrix Market write",
}
(OUT / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
