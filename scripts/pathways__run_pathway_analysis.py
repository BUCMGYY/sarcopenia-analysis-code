from __future__ import annotations

"""Unified pathway analysis for the GSE213148 16-week experiment.

This script keeps three evidence layers separate:
1. GO/KEGG/Reactome over-representation analysis of the primary DEG lists;
2. sample-level GSVA scores for the 48 16-week libraries;
3. pathway membership context for the existing expression-level
   counterfactual candidates and ranked L1000FWD perturbagens.

The last layer deliberately does not manufacture a drug-level pathway
reversal score: the available L1000FWD table contains aggregate perturbagen
scores but not the per-perturbagen gene-level signatures required to recompute
pathway reversal.
"""

from pathlib import Path
import os
import hashlib
import json
import re
from datetime import datetime, timezone

import gseapy
import numpy as np
import pandas as pd
import requests
from scipy.stats import hypergeom, ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests


_project_root = os.environ.get("SARCOPENIA_PROJECT_ROOT")
if not _project_root:
    raise SystemExit("Set SARCOPENIA_PROJECT_ROOT to the local project-data directory.")
ROOT = Path(_project_root).expanduser().resolve()
DATA = ROOT / "sarcopenia_disease" / "GSE213148"
RES = DATA / "results"
REAN = DATA / "reanalysis_20260928" / "results"
OUT = ROOT / "pathway_corrected_20260929"
GENESET_DIR = OUT / "gene_sets"
OUT.mkdir(parents=True, exist_ok=True)
GENESET_DIR.mkdir(parents=True, exist_ok=True)

LIBRARIES = {
    "GO:BP": {"name": "GO_BP_mmusculus_symbols_gmt", "version": "official GOA-MGI/gProfiler mouse symbols"},
    "KEGG": {"name": "KEGG_mmusculus_symbols_gmt", "version": "KEGG REST mmu + MGI mouse symbols"},
    "REAC": {"name": "REAC_mmusculus_symbols_gmt", "version": "gProfiler mouse R-MMU symbols"},
}
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")[:80]


def load_inputs():
    meta = pd.read_csv(RES / "GSE213148_sample_metadata_audited.csv")
    expr = pd.read_csv(RES / "GSE213148_log2CPM_matrix.csv").set_index("gene_id")
    # Use the frozen DESeq2 primary contrast as the sole DE input.  The older
    # Welch-test annotation remains available for gene-symbol mapping only and
    # is never used for effect sizes, p values or DEG thresholds.
    ann = pd.read_csv(
        RES / "GSE213148_CRIR_vs_Ctrl_DEG_annotated.csv",
        usecols=["gene_id", "gene_name"],
    ).drop_duplicates("gene_id")
    de = pd.read_csv(REAN / "GSE213148_primary_DESeq2_CRIR_vs_Ctrl.csv")
    de = de.merge(ann, on="gene_id", how="left")
    de["FDR"] = de["padj"]
    de["log2FC_case_minus_control"] = de["log2FoldChange"]
    cf = pd.read_csv(REAN / "GSE213148_DESeq2_bulk_counterfactual_sensitivity_top50.csv")
    drugs = pd.read_csv(ROOT / "figure_v2_data" / "figure6_homologene_unique_drug_rank.csv")

    m16 = meta[meta["source"].astype(str).str.startswith("16w_")].copy()
    m16 = m16[m16["gsm"].isin(expr.columns)].copy()
    m16["CR"] = m16["group"].isin(["CR", "CR+IR"]).astype(int)
    m16["IR"] = m16["group"].isin(["IR", "CR+IR"]).astype(int)
    samples = m16["gsm"].tolist()
    expr16 = expr[samples].copy()

    # Use the audited DESeq2 gene universe and stable mouse symbols. Duplicate
    # symbols are averaged after filtering to the 16-week expression matrix.
    mapping = ann[["gene_id", "gene_name"]].dropna(subset=["gene_name"]).copy()
    mapping["symbol"] = mapping["gene_name"].astype(str).str.upper()
    mapping = mapping[mapping["gene_id"].isin(expr16.index)]
    expr_sym = expr16.loc[mapping["gene_id"]].copy()
    expr_sym.index = mapping.set_index("gene_id").loc[expr_sym.index, "symbol"].values
    expr_sym = expr_sym.groupby(level=0).mean()

    tested = de.dropna(subset=["gene_name"]).copy()
    tested["symbol"] = tested["gene_name"].astype(str).str.upper()
    universe_symbols = sorted(set(tested["symbol"]).intersection(expr_sym.index))
    universe_ids = tested.loc[tested["symbol"].isin(universe_symbols), "gene_id"].astype(str).drop_duplicates().tolist()

    hit = de[(de["FDR"] < 0.05) & (de["log2FC_case_minus_control"].abs() >= 0.5)].copy()
    hit = hit.dropna(subset=["gene_name"])
    hit["symbol"] = hit["gene_name"].astype(str).str.upper()
    hit = hit[hit["symbol"].isin(universe_symbols)]
    queries = {
        "up": sorted(set(hit.loc[hit["log2FC_case_minus_control"] > 0, "symbol"])),
        "down": sorted(set(hit.loc[hit["log2FC_case_minus_control"] < 0, "symbol"])),
    }
    # g:Profiler accepts Ensembl IDs. Keep one audited ID per symbol.
    symbol_to_id = (
        tested.drop_duplicates("symbol").set_index("symbol")["gene_id"].astype(str).to_dict()
    )
    query_ids = {k: [symbol_to_id[s] for s in v if s in symbol_to_id] for k, v in queries.items()}
    return meta, m16, expr_sym, tested, universe_ids, queries, query_ids, cf, drugs, de


def run_unified_gprofiler(queries, query_ids, universe_ids):
    rows = []
    raw_meta = {}
    for direction in ["up", "down"]:
        payload = {
            "organism": "mmusculus",
            "query": query_ids[direction],
            "sources": ["GO:BP", "KEGG", "REAC"],
            "user_threshold": 0.05,
            "no_evidences": False,
            "domain_scope": "custom",
            "background": universe_ids,
        }
        response = requests.post(
            "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json()
        (OUT / f"gprofiler_{direction}_{RUN_ID}.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raw_meta[direction] = raw.get("meta", {})
        for x in raw.get("result", []):
            n = int(x.get("effective_domain_size") or len(universe_ids))
            k = int(x.get("intersection_size") or 0)
            term_size = int(x.get("term_size") or 0)
            qsize = int(x.get("query_size") or len(query_ids[direction]))
            rows.append({
                "run_id": RUN_ID,
                "dataset_id": "GSE213148",
                "contrast_id": "16w_CR+IR_vs_16w_Ctrl",
                "direction": direction,
                "source": x.get("source"),
                "term_id": x.get("native"),
                "term_name": x.get("name"),
                "overlap_size": k,
                "term_size": term_size,
                "query_size": qsize,
                "universe_size": n,
                "gene_ratio": (k / qsize) if qsize else np.nan,
                "enrichment_ratio": ((k / qsize) / (term_size / n)) if qsize and term_size and n else np.nan,
                "p_value": x.get("p_value"),
                "fdr": x.get("p_value"),
                "query_gene_count": len(query_ids[direction]),
                "background_rule": "all tested DESeq2 genes with non-missing mouse symbol",
                "gene_set_version": raw.get("meta", {}).get("version"),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        for direction in out["direction"].unique():
            idx = out["direction"].eq(direction)
            out.loc[idx, "fdr"] = multipletests(out.loc[idx, "p_value"].astype(float), method="fdr_bh")[1]
        out["neglog10_fdr"] = -np.log10(np.clip(out["fdr"].astype(float), 1e-300, 1))
        out = out.sort_values(["direction", "fdr", "overlap_size"], ascending=[True, True, False])
    out.to_csv(OUT / "pathway_enrichment_unified.csv", index=False)
    (OUT / "gprofiler_metadata.json").write_text(json.dumps(raw_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_or_fetch_gene_sets():
    combined = {}
    term_names = {}
    rows = []
    manifest = []
    unified_gmt = GENESET_DIR / "unified_mmusculus_symbols.gmt"
    unified_cache = {}
    if unified_gmt.exists():
        with unified_gmt.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                tid, name, genes = parts[0], parts[1], [g.upper() for g in parts[2:] if g.strip()]
                if tid.startswith("GO:"):
                    unified_cache.setdefault("GO:BP", {})[f"{tid} | {name}"] = genes
                elif tid.startswith("KEGG:"):
                    unified_cache.setdefault("KEGG", {})[f"{tid} | {name}"] = genes
                elif tid.startswith("REAC:R-MMU"):
                    unified_cache.setdefault("REAC", {})[f"{tid} | {name}"] = genes
    for source, cfg in LIBRARIES.items():
        json_path = GENESET_DIR / f"{slug(cfg['name'])}.json"
        if source in unified_cache:
            library = unified_cache[source]
            # Keep only sets compatible with the prespecified GSVA size rule.
            library = {k: v for k, v in library.items() if 10 <= len(set(v)) <= 1000}
            json_path.write_text(json.dumps(library, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        elif source == "REAC":
            # Enrichr's Reactome_2022 endpoint is human (R-HSA) despite the
            # organism argument.  Use the downloaded gProfiler mouse GMT and
            # retain only Reactome mouse pathways (R-MMU).
            gmt_path = GENESET_DIR / "gprofiler_full_mmusculus.name.gmt"
            if not gmt_path.exists():
                r = requests.get(
                    "https://biit.cs.ut.ee/gprofiler/static/gprofiler_full_mmusculus.name.gmt",
                    timeout=180,
                )
                r.raise_for_status()
                gmt_path.write_bytes(r.content)
            library = {}
            with gmt_path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 4 or not parts[0].startswith("REAC:R-MMU"):
                        continue
                    library[f"{parts[0]} | {parts[1]}"] = parts[2:]
            json_path.write_text(json.dumps(library, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        elif json_path.exists():
            library = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            library = gseapy.get_library(cfg["name"], organism="Mouse", min_size=10, max_size=1000)
            json_path.write_text(json.dumps(library, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        library = {str(k): sorted({str(g).upper() for g in v if str(g).strip()}) for k, v in library.items()}
        manifest.append({
            "source": source,
            "library_name": cfg["name"],
            "library_version": cfg["version"],
            "n_terms": len(library),
            "retrieval_file": str(json_path),
            "sha256": sha256_file(json_path),
        })
        for term_name, genes in library.items():
            term_id = f"{source}__{slug(term_name)}"
            combined[term_id] = genes
            term_names[term_id] = term_name
            gene_hash = hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()
            for gene in genes:
                rows.append({
                    "source": source,
                    "term_id": term_id,
                    "term_name": term_name,
                    "gene_symbol": gene,
                    "species": "Mus musculus",
                    "library_version": cfg["version"],
                    "gene_set_size": len(genes),
                    "gene_set_sha256": gene_hash,
                })
    pd.DataFrame(rows).to_csv(OUT / "pathway_gene_sets.tsv", sep="\t", index=False)
    pd.DataFrame(manifest).to_csv(OUT / "gene_set_manifest.csv", index=False)
    return combined, term_names, pd.DataFrame(manifest)


def run_gsva(expr_sym, m16, gene_sets, term_names):
    cached = OUT / "gsva_sample_scores.csv"
    if cached.exists():
        cached_df = pd.read_csv(cached)
        if {"gsm", "term_id", "gsva_score"}.issubset(cached_df.columns) and cached_df["gsm"].nunique() == len(m16):
            return cached_df
    # Gene sets and expression symbols are uppercase to make mouse-symbol
    # matching explicit and reproducible.
    gsva = gseapy.gsva(
        data=expr_sym,
        gene_sets=gene_sets,
        min_size=10,
        max_size=1000,
        kcdf="Gaussian",
        weight=1.0,
        mx_diff=True,
        abs_rnk=False,
        threads=4,
        seed=42,
        verbose=False,
    )
    res = gsva.res2d.rename(columns={"Name": "gsm", "Term": "term_id", "ES": "gsva_score"})
    term_lookup = []
    for term_id, genes in gene_sets.items():
        source = term_id.split("__", 1)[0]
        term_name = term_names.get(term_id, term_id.split("__", 1)[1].replace("_", " "))
        term_lookup.append({"term_id": term_id, "source": source, "term_name": term_name,
                            "gene_set_size": len(genes)})
    terms = pd.DataFrame(term_lookup)
    res = res.merge(terms, on="term_id", how="left")
    res = res.merge(m16[["gsm", "mouse id", "group", "CR", "IR"]], on="gsm", how="left")
    res["dataset_id"] = "GSE213148"
    res["contrast_id"] = "16w_four_arm"
    res["gsva_method"] = "gseapy.gsva; kcdf=Gaussian; mx_diff=True; seed=42"
    # Coverage is the fraction of gene-set members found in the expression matrix.
    expr_genes = set(expr_sym.index)
    res["coverage"] = res.apply(
        lambda r: len(set(gene_sets.get(r["term_id"], [])) & expr_genes) / max(int(r["gene_set_size"]), 1),
        axis=1,
    )
    res.to_csv(OUT / "gsva_sample_scores.csv", index=False)
    return res


def run_factorial_models(scores):
    cached_f = OUT / "gsva_factorial_results.csv"
    cached_p = OUT / "gsva_primary_contrast_results.csv"
    if cached_f.exists() and cached_p.exists():
        return pd.read_csv(cached_f), pd.read_csv(cached_p)
    rows = []
    for term_id, sub in scores.groupby("term_id", sort=False):
        sub = sub.dropna(subset=["gsva_score", "CR", "IR"]).copy()
        fit = ols("gsva_score ~ CR + IR + CR:IR", data=sub).fit()
        lookup = sub.iloc[0]
        means = sub.groupby("group")["gsva_score"].mean().to_dict()
        rows.append({
            "dataset_id": "GSE213148",
            "term_id": term_id,
            "source": lookup["source"],
            "term_name": lookup["term_name"],
            "n_samples": len(sub),
            "n_per_group": sub.groupby("group").size().min(),
            "beta_CR": fit.params.get("CR", np.nan),
            "p_CR": fit.pvalues.get("CR", np.nan),
            "beta_IR": fit.params.get("IR", np.nan),
            "p_IR": fit.pvalues.get("IR", np.nan),
            "beta_interaction": fit.params.get("CR:IR", np.nan),
            "p_interaction": fit.pvalues.get("CR:IR", np.nan),
            "mean_Control": means.get("Ctrl", np.nan),
            "mean_CR": means.get("CR", np.nan),
            "mean_IR": means.get("IR", np.nan),
            "mean_CRIR": means.get("CR+IR", np.nan),
            "delta_CRIR_vs_Control": means.get("CR+IR", np.nan) - means.get("Ctrl", np.nan),
        })
    out = pd.DataFrame(rows)
    for pcol, fdrcol in [("p_CR", "fdr_CR"), ("p_IR", "fdr_IR"), ("p_interaction", "fdr_interaction")]:
        out[fdrcol] = multipletests(out[pcol].fillna(1).astype(float), method="fdr_bh")[1]
    out.to_csv(OUT / "gsva_factorial_results.csv", index=False)

    primary = []
    for term_id, sub in scores.groupby("term_id", sort=False):
        ctrl = sub.loc[sub["group"] == "Ctrl", "gsva_score"].dropna()
        model = sub.loc[sub["group"] == "CR+IR", "gsva_score"].dropna()
        if len(ctrl) and len(model):
            t, p = ttest_ind(model, ctrl, equal_var=False)
        else:
            t, p = np.nan, 1.0
        row = out.loc[out["term_id"] == term_id].iloc[0].to_dict()
        row.update({"t_CRIR_vs_Control": t, "p_CRIR_vs_Control": p})
        primary.append(row)
    primary = pd.DataFrame(primary)
    primary["fdr_CRIR_vs_Control"] = multipletests(primary["p_CRIR_vs_Control"].fillna(1), method="fdr_bh")[1]
    primary.to_csv(OUT / "gsva_primary_contrast_results.csv", index=False)
    return out, primary


def select_display_pathways(enrichment, primary, gene_sets):
    # Fixed display rule: up to two enriched terms per direction/source for
    # the unified ORA, and up to two GSVA terms per source by primary FDR.
    # Exclude generic disease/infection labels from the compact display only;
    # the complete tables retain every tested term.
    e = enrichment[enrichment["fdr"] < 0.05].copy()
    exclude = r"disease|virus|infection|cancer|diabetes|parkinson|toxoplasma|hepatitis|alzheimer"
    e = e[~e["term_name"].str.contains(exclude, case=False, na=False)]
    e = e.sort_values(["direction", "source", "fdr", "overlap_size"], ascending=[True, True, True, False])
    e = e.groupby(["direction", "source"], as_index=False, group_keys=False).head(2)
    e.to_csv(OUT / "figure3_panel_c_unified_enrichment.csv", index=False)

    p = primary.sort_values(["source", "fdr_CRIR_vs_Control", "delta_CRIR_vs_Control"], ascending=[True, True, False])
    p = p[p["fdr_CRIR_vs_Control"] < 0.05].copy()
    p = p[p["term_name"].str.contains(
        r"ecm|collagen|adhesion|cytoskeleton|actin|ribosome|rna|splice|mitophagy|cytokine|translation|protein digestion",
        case=False, na=False,
    )]
    p = p.groupby("source", as_index=False, group_keys=False).head(2)
    if p.empty:
        p = primary.sort_values(["source", "fdr_CRIR_vs_Control"]).groupby("source", as_index=False, group_keys=False).head(2)
    p.to_csv(OUT / "figure3_panel_f_selected_gsva_pathways.csv", index=False)
    selected = p["term_id"].tolist()
    return e, p, selected


def candidate_pathway_context(cf, gene_sets, selected_terms, term_names):
    c = cf.copy()
    c["candidate_symbol"] = c["gene_name"].fillna(c["gene_id"]).astype(str).str.upper()
    rows = []
    for _, r in c.iterrows():
        for term_id in selected_terms:
            genes = set(gene_sets.get(term_id, []))
            source = term_id.split("__", 1)[0]
            term_name = term_names.get(term_id, term_id.split("__", 1)[1].replace("_", " "))
            rows.append({
                "candidate_gene": r["gene_name"],
                "candidate_symbol": r["candidate_symbol"],
                "cf_rank": int(r.name + 1),
                "mean_reflection_reversal_fraction": r["mean_reflection_reversal_fraction"],
                "source": source,
                "term_id": term_id,
                "term_name": term_name,
                "in_gene_set": r["candidate_symbol"] in genes,
                "context_only": True,
                "interpretation": "candidate-to-pathway membership context; not pathway-level causal reversal",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "figure3_panel_f_gene_pathway_convergence.csv", index=False)
    return out


def drug_pathway_context(drugs, gene_sets, selected_terms, term_names):
    rows = []
    # Gene-level L1000FWD signatures are now available for every ranked
    # perturbagen.  A pathway reversal score is defined prospectively as the
    # fraction of disease-direction genes in a gene set whose perturbagen
    # direction is opposite, among genes observed in both signatures.  It is a
    # signature concordance metric, not an in vivo rescue or causal effect.
    long_path = OUT / "drug_raw" / "l1000fwd_signature_genes_long.csv"
    long = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()
    de_path = REAN / "GSE213148_primary_DESeq2_CRIR_vs_Ctrl.csv"
    ann_path = RES / "GSE213148_CRIR_vs_Ctrl_DEG_annotated.csv"
    disease = pd.read_csv(de_path).merge(pd.read_csv(ann_path, usecols=["gene_id", "gene_name"]), on="gene_id", how="left")
    disease["symbol"] = disease["gene_name"].fillna("").astype(str).str.upper()
    disease = disease[disease["symbol"].ne("")].copy()
    disease_dir = disease.set_index("symbol")["log2FoldChange"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).to_dict()
    sig_meta = {}
    if not long.empty:
        for sig_id, sub_sig in long.groupby("sig_id", sort=False):
            sig_meta[sig_id] = {str(r["gene"]).upper(): (1 if r["signature_direction"] == "up" else -1) for _, r in sub_sig.iterrows()}
    # Human L1000FWD signatures are not directly comparable with mouse
    # expression.  Map human symbols to one-to-one mouse orthologues through
    # g:Orth and retain the complete audit table before any overlap score.
    orth_path = OUT / "drug_raw" / "human_mouse_ortholog_map.csv"
    if not long.empty:
        human_genes = sorted(set(long["gene"].astype(str).str.upper()))
        if orth_path.exists():
            orth = pd.read_csv(orth_path)
        else:
            orth_rows = []
            for start in range(0, len(human_genes), 400):
                batch = human_genes[start:start + 400]
                rr = requests.post(
                    "https://biit.cs.ut.ee/gprofiler/api/orth/orth/",
                    json={"organism": "hsapiens", "target": "mmusculus", "query": batch},
                    timeout=180,
                )
                rr.raise_for_status()
                orth_rows.extend(rr.json().get("result", []))
            orth = pd.DataFrame(orth_rows)
            if orth.empty:
                orth = pd.DataFrame(columns=["incoming", "name", "ortholog_ensg"])
            else:
                orth = orth.rename(columns={"incoming": "human_symbol", "name": "mouse_symbol"})
                orth["human_symbol"] = orth["human_symbol"].astype(str).str.upper()
                orth["mouse_symbol"] = orth["mouse_symbol"].astype(str).str.upper()
                orth = orth[[c for c in ["human_symbol", "mouse_symbol", "ortholog_ensg", "description", "n_result"] if c in orth.columns]]
            orth.to_csv(orth_path, index=False)
        if not orth.empty:
            orth["human_symbol"] = orth["human_symbol"].astype(str).str.upper()
            orth["mouse_symbol"] = orth["mouse_symbol"].astype(str).str.upper()
            orth = orth.dropna(subset=["mouse_symbol"]).drop_duplicates(["human_symbol", "mouse_symbol"])
            h2m = orth.groupby("human_symbol")["mouse_symbol"].apply(list).to_dict()
            sig_meta = {
                sid: {m: direction for h, direction in genes.items() for m in h2m.get(h, [])}
                for sid, genes in sig_meta.items()
            }
    drug_sig_lookup = {}
    if not long.empty:
        drug_sig_lookup = long.drop_duplicates("pert_id").set_index("pert_id")["sig_id"].to_dict()
    for _, d in drugs.iterrows():
        target_text = d.get("target_annotation")
        targets = [] if pd.isna(target_text) else [x.strip().upper() for x in str(target_text).split("|") if x.strip()]
        sig_id = drug_sig_lookup.get(d.get("pert_id"))
        sig_dir = sig_meta.get(sig_id, {})
        for term_id in selected_terms:
            source = term_id.split("__", 1)[0]
            term_name = term_names.get(term_id, term_id.split("__", 1)[1].replace("_", " "))
            overlap = sorted(set(targets).intersection(gene_sets.get(term_id, [])))
            genes = set(gene_sets.get(term_id, []))
            shared = sorted(genes.intersection(disease_dir).intersection(sig_dir))
            expected = sum(1 for g in shared if disease_dir[g] != 0)
            reversed_n = sum(1 for g in shared if disease_dir[g] == -sig_dir[g])
            concordant_n = sum(1 for g in shared if disease_dir[g] == sig_dir[g])
            reversal = reversed_n / expected if expected else np.nan
            rows.append({
                "pert_id": d.get("pert_id"),
                "candidate_rank": d.get("candidate_rank"),
                "qvalue": d.get("qvalue"),
                "term_id": term_id,
                "source": source,
                "term_name": term_name,
                "target_overlap_count": len(overlap),
                "target_overlap_symbols": "|".join(overlap),
                "sig_id": sig_id,
                "signature_shared_gene_count": expected,
                "signature_reversed_gene_count": reversed_n,
                "signature_concordant_gene_count": concordant_n,
                "pathway_reversal_fraction": reversal,
                "pathway_reversal_status": "signature-level descriptive metric" if expected else "not_estimable_for_this_pathway",
                "missing_reason": "" if expected else "no overlapping disease-direction and perturbagen genes in this pathway",
                "interpretation": "gene-signature concordance only; not a causal rescue or in vivo efficacy estimate",
                "orthology_rule": "g:Orth hsapiens→mmusculus; mouse symbols used for overlap",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "drug_pathway_cross_explanation.csv", index=False)
    return out


def main():
    meta, m16, expr_sym, tested, universe_ids, queries, query_ids, cf, drugs, de = load_inputs()
    enrichment = run_unified_gprofiler(queries, query_ids, universe_ids)
    gene_sets, term_names, manifest = load_or_fetch_gene_sets()
    scores = run_gsva(expr_sym, m16, gene_sets, term_names)
    factorial, primary = run_factorial_models(scores)
    display_enrich, display_gsva, selected = select_display_pathways(enrichment, primary, gene_sets)
    candidate_pathway_context(cf, gene_sets, selected, term_names)
    drug_pathway_context(drugs, gene_sets, selected, term_names)

    summary = {
        "run_id": RUN_ID,
        "dataset": "GSE213148",
        "n_16w_libraries": int(len(m16)),
        "groups": m16.groupby("group").size().to_dict(),
        "n_tested_gene_ids": int(len(universe_ids)),
        "de_method": "DESeq2 primary CR+IR vs Ctrl; padj<0.05 and abs(log2FoldChange)>=0.5",
        "n_unified_enrichment_rows": int(len(enrichment)),
        "enrichment_sources": sorted(enrichment["source"].dropna().unique().tolist()) if len(enrichment) else [],
        "n_gene_sets": int(len(gene_sets)),
        "n_gsva_scores": int(len(scores)),
        "selected_gsva_pathways": selected,
        "drug_pathway_reversal": "signature-level descriptive concordance computed from retrieved per-perturbagen up/down genes; not causal rescue",
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Unified pathway analysis\n\n"
        "GO:BP, KEGG and mouse Reactome over-representation use the DESeq2 GSE213148 CR+IR versus Control contrast and the all-tested-gene background. "
        "GSVA is computed per sample for the 48 16-week libraries and modelled with CR + IR + CR:IR. "
        "Drug pathway values are signature-level descriptive concordance scores based on retrieved up/down genes, not causal rescue or treatment efficacy.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
