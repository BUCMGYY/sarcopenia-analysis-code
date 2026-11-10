from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
_project_root = os.environ.get("SARCOPENIA_PROJECT_ROOT")
if not _project_root:
    raise SystemExit("Set SARCOPENIA_PROJECT_ROOT to the local project-data directory.")
PROJECT = Path(_project_root).expanduser().resolve()
COUNT_PATH = PROJECT / "GSE167186" / "GSE167186_counts.csv.gz"
META_PATH = PROJECT / "GSE167186_sample_module_scores.csv"
REPO = ROOT / "drugreflector"
sys.path.insert(0, str(REPO))
from drugreflector import DrugReflector  # noqa: E402


def load_model_input_names() -> list[str]:
    ckpt = ROOT / "checkpoints_clean" / "model_fold_0.pt"
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    return list(checkpoint["dimensions"]["input_names"])


def make_signature(group_a: str, group_b: str, counts: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.Series, dict]:
    sample_cols = [x for x in meta["sample"] if x in counts.columns]
    counts = counts.loc[:, ["Symbol", *sample_cols]].copy()
    counts["Symbol"] = counts["Symbol"].astype(str).str.upper()
    counts = counts.groupby("Symbol", sort=False).sum(numeric_only=True)
    library = counts.sum(axis=0).replace(0, np.nan)
    log_cpm = np.log1p(counts.div(library, axis=1) * 1e6)
    groups = meta.set_index("sample").loc[sample_cols, "group"]
    a = log_cpm.loc[:, groups[groups == group_a].index]
    b = log_cpm.loc[:, groups[groups == group_b].index]
    delta = a.mean(axis=1) - b.mean(axis=1)
    # Standardize only by the across-gene scale, producing a v-score-like
    # disease signature while preserving the direction of the contrast.
    scale = float(delta.std(ddof=1)) or 1.0
    v = delta / scale
    return v, {"group_a": group_a, "group_b": group_b, "n_a": a.shape[1], "n_b": b.shape[1], "scale": scale}


counts = pd.read_csv(COUNT_PATH, compression="gzip")
meta = pd.read_csv(META_PATH)
input_names = load_model_input_names()
model = DrugReflector(
    checkpoint_paths=[str(ROOT / "checkpoints_clean" / f"model_fold_{i}.pt") for i in range(3)]
)

all_rankings = []
coverage_rows = []
for name, group_a, group_b in [
    ("GSE167186_sarcopenia_vs_old_healthy", "Sarcopenia", "Old Healthy"),
    ("GSE167186_old_healthy_vs_young_healthy", "Old Healthy", "Young Healthy"),
]:
    signature, info = make_signature(group_a, group_b, counts, meta)
    covered = [g for g in input_names if g in signature.index]
    vector = signature.reindex(input_names).fillna(0.0)
    pd.DataFrame({"gene": vector.index, "v_score": vector.values}).to_csv(
        ROOT / f"{name}_signature_978.csv", index=False, encoding="utf-8-sig"
    )
    coverage_rows.append({"signature": name, **info, "model_input_genes": len(input_names), "covered_genes": len(covered), "coverage_fraction": len(covered) / len(input_names)})
    pred = model.predict(pd.Series(vector.values, index=input_names, name=name), n_top=100)
    out = pred.copy()
    out.insert(0, "compound_id", out.index.astype(str))
    out.insert(0, "signature", name)
    out.reset_index(drop=True).to_csv(ROOT / f"{name}_drugreflector_top100.csv", index=False, encoding="utf-8-sig")
    all_rankings.append(out)

pd.DataFrame(coverage_rows).to_csv(ROOT / "drugreflector_gene_coverage.csv", index=False, encoding="utf-8-sig")
pd.concat(all_rankings, ignore_index=True).to_csv(ROOT / "drugreflector_top100_all_signatures.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(coverage_rows).to_string(index=False))
