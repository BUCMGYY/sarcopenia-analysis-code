#!/usr/bin/env Rscript

# Efficient scTenifoldKnk/OE batch. WT QC/CPM/networks/tensor are built once,
# then reused for each target. Legacy ko/oe outputs are not touched.
args <- commandArgs(trailingOnly=TRUE)
MODE <- if (length(args)) tolower(args[[1]]) else "both"
if (!MODE %in% c("ko", "oe", "both")) stop("mode must be ko, oe, or both")
MAX_TARGETS <- if (length(args) >= 2 && nzchar(args[[2]])) as.integer(args[[2]]) else NA_integer_
ROOT <- Sys.getenv("SARCOPENIA_PROJECT_ROOT")
if (!nzchar(ROOT)) stop("Set SARCOPENIA_PROJECT_ROOT to the local project-data directory.")
ROOT <- normalizePath(ROOT, winslash="/", mustWork=TRUE)
BASE <- file.path(ROOT, "three_virtual_omics_rebuild_20261004")
OUTROOT <- file.path(BASE, "efficient_batch_20261004")
WTROOT <- file.path(OUTROOT, "wt_network")
dir.create(WTROOT, recursive=TRUE, showWarnings=FALSE)
LIB <- file.path(ROOT, "R_libs_sc3v")
.libPaths(c(LIB, .libPaths()))
options(stringsAsFactors=FALSE)
suppressPackageStartupMessages({ library(Matrix); library(scTenifoldNet); library(scTenifoldKnk); library(jsonlite) })

SEED <- 12345L
QC_MIN_LIB_SIZE <- 500L
QC_MAX_MT_RATIO <- 0.1
QC_MIN_PCT <- 0
NC_NNET <- 10L
NC_NCELLS <- 500L
NC_NCOMP <- 3L
NC_Q <- 0.9
NC_CORES <- 6L
TD_K <- 3L
TD_MAX_ERROR <- 1e-5
TD_MAX_ITER <- 1000L
TD_NDECIMAL <- 3L
MA_D <- 2L
OE_FACTOR <- 10

logfile <- file.path(OUTROOT, paste0("run_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log"))
log_con <- file(logfile, open="wt")
sink(log_con, split=TRUE); sink(log_con, type="message")
on.exit({sink(type="message"); sink(); close(log_con)}, add=TRUE)
cat("efficient_batch_version=20261004\n", "mode=", MODE, "\n", sep="")
cat("start=", format(Sys.time(), tz="UTC"), "\n", "R=", R.version.string, "\n", sep="")
cat("scTenifoldNet=", as.character(packageVersion("scTenifoldNet")), " scTenifoldKnk=", as.character(packageVersion("scTenifoldKnk")), "\n", sep="")
cat("parameters: qc_minLibSize=", QC_MIN_LIB_SIZE, " qc_maxMTratio=", QC_MAX_MT_RATIO, " qc_minPCT=", QC_MIN_PCT, " nc_nNet=", NC_NNET, " nc_nCells=", NC_NCELLS, " nc_nComp=", NC_NCOMP, " seed=", SEED, " oe_factor=", OE_FACTOR, "\n", sep="")

INPUT <- file.path(BASE, "input_case_matrix")
X0 <- as(readMM(file.path(INPUT, "case_counts.mtx")), "dgCMatrix")
genes <- read.csv(file.path(INPUT, "genes.tsv"), stringsAsFactors=FALSE)$gene
cells <- read.csv(file.path(INPUT, "cells.tsv"), stringsAsFactors=FALSE)$cell
rownames(X0) <- genes; colnames(X0) <- cells
target_map <- read.csv(file.path(INPUT, "target_mapping.csv"), stringsAsFactors=FALSE)
target_map$mouse_matrix_gene <- trimws(target_map$mouse_matrix_gene)
target_map$human_manifest_gene <- trimws(target_map$human_manifest_gene)
target_map <- target_map[nzchar(target_map$mouse_matrix_gene), , drop=FALSE]
target_map <- target_map[!duplicated(target_map$mouse_matrix_gene), , drop=FALSE]
if (is.finite(MAX_TARGETS) && MAX_TARGETS > 0) target_map <- head(target_map, MAX_TARGETS)
targets <- target_map$mouse_matrix_gene
target_counts <- Matrix::rowSums(X0[intersect(targets, rownames(X0)), , drop=FALSE])
targets_available <- intersect(targets, names(target_counts)[target_counts > 0])
targets_missing <- setdiff(targets, targets_available)
cat("input_matrix=", nrow(X0), "x", ncol(X0), " targets=", length(targets), " available=", length(targets_available), " missing=", paste(targets_missing, collapse=","), "\n", sep="")

build_wt <- function(countMatrix) {
  set.seed(SEED)
  X <- scTenifoldNet::scQC(countMatrix, minLibSize=QC_MIN_LIB_SIZE,
      removeOutlierCells=TRUE, minPCT=QC_MIN_PCT, maxMTratio=QC_MAX_MT_RATIO)
  if (!all(targets_available %in% rownames(X))) stop("Available target(s) removed by QC: ", paste(setdiff(targets_available, rownames(X)), collapse=", "))
  X <- scTenifoldNet::cpmNormalization(X)
  set.seed(SEED)
  nets <- scTenifoldNet::makeNetworks(X=X, q=NC_Q, nNet=NC_NNET, nCells=NC_NCELLS,
      scaleScores=TRUE, symmetric=FALSE, nComp=NC_NCOMP, nCores=NC_CORES)
  set.seed(SEED)
  td <- scTenifoldNet::tensorDecomposition(xList=nets, K=TD_K, maxError=TD_MAX_ERROR,
      maxIter=TD_MAX_ITER, nDecimal=TD_NDECIMAL, seed=SEED)
  wt <- scTenifoldKnk:::strictDirection(td$X, lambda=0)
  wt <- as.matrix(wt); diag(wt) <- 0; wt <- t(wt)
  list(qc_cpm_matrix=X, networks=nets, tensor=td, WT=Matrix::Matrix(wt, sparse=TRUE))
}

wt_rds <- file.path(WTROOT, "WT_network_bundle.rds")
wt_meta <- file.path(WTROOT, "WT_network_metadata.json")
if (file.exists(wt_rds) && file.exists(wt_meta)) {
  cat("WT bundle found; loading without rebuilding: ", wt_rds, "\n", sep="")
  wt_bundle <- readRDS(wt_rds)
  if (!all(targets_available %in% rownames(wt_bundle$WT))) {
    cat("Existing WT bundle lacks one or more requested targets; rebuilding with current target set.\n")
    wt_bundle <- build_wt(X0)
    saveRDS(wt_bundle, wt_rds, compress=TRUE)
    wt_info <- list(created_utc=format(Sys.time(), tz="UTC"), input_dim=dim(X0), qc_cpm_dim=dim(wt_bundle$qc_cpm_matrix), parameters=list(seed=SEED, qc_minLibSize=QC_MIN_LIB_SIZE, qc_maxMTratio=QC_MAX_MT_RATIO, qc_minPCT=QC_MIN_PCT, nc_nNet=NC_NNET, nc_nCells=NC_NCELLS, nc_nComp=NC_NCOMP, nc_q=NC_Q, td_K=TD_K, td_maxError=TD_MAX_ERROR, td_maxIter=TD_MAX_ITER, td_nDecimal=TD_NDECIMAL, nCores=NC_CORES), rebuilt_for_targets=targets_available, missing_targets=targets_missing)
    write_json(wt_info, wt_meta, auto_unbox=TRUE, pretty=TRUE)
  }
} else {
  cat("Building WT network bundle once...\n")
  t_wt <- Sys.time(); wt_bundle <- build_wt(X0)
  saveRDS(wt_bundle, wt_rds, compress=TRUE)
  wt_info <- list(created_utc=format(Sys.time(), tz="UTC"), elapsed_sec=as.numeric(difftime(Sys.time(), t_wt, units="secs")), input_dim=dim(X0), qc_cpm_dim=dim(wt_bundle$qc_cpm_matrix), parameters=list(seed=SEED, qc_minLibSize=QC_MIN_LIB_SIZE, qc_maxMTratio=QC_MAX_MT_RATIO, qc_minPCT=QC_MIN_PCT, nc_nNet=NC_NNET, nc_nCells=NC_NCELLS, nc_nComp=NC_NCOMP, nc_q=NC_Q, td_K=TD_K, td_maxError=TD_MAX_ERROR, td_maxIter=TD_MAX_ITER, td_nDecimal=TD_NDECIMAL, nCores=NC_CORES), built_for_targets=targets_available, missing_targets=targets_missing)
  write_json(wt_info, wt_meta, auto_unbox=TRUE, pretty=TRUE)
  cat("WT elapsed_sec=", wt_info$elapsed_sec, "\n", sep="")
}
WT <- as.matrix(wt_bundle$WT)

run_one <- function(mode, human_gene, gene) {
  outdir <- file.path(OUTROOT, mode, human_gene)
  dir.create(outdir, recursive=TRUE, showWarnings=FALSE)
  t0 <- Sys.time(); cat("\n=== ", mode, " ", human_gene, " / ", gene, " ===\n", sep="")
  ans <- tryCatch({
    pert <- WT
    if (mode == "ko") pert[gene, ] <- 0 else pert[gene, ] <- pert[gene, ] * OE_FACTOR
    set.seed(SEED)
    ma <- scTenifoldNet::manifoldAlignment(WT, pert, d=MA_D, nCores=NC_CORES)
    dr <- scTenifoldKnk:::dRegulation(ma, gKO=gene)
    list(manifoldAlignment=ma, diffRegulation=dr)
  }, error=function(e) { cat("ERROR: ", conditionMessage(e), "\n", sep=""); NULL })
  elapsed <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  if (is.null(ans)) return(list(status="failed", elapsed_sec=elapsed, error="see run log"))
  dr <- as.data.frame(ans$diffRegulation, stringsAsFactors=FALSE); dr$gene <- as.character(dr$gene)
  dr$log_p_adj <- -log10(pmax(dr$p.adj, .Machine$double.xmin)); dr$significant <- is.finite(dr$p.adj) & dr$p.adj < 0.05
  sig <- dr[dr$significant & dr$gene != gene, , drop=FALSE]
  top <- dr[dr$gene != gene, , drop=FALSE]; top <- top[order(-abs(top$FC), top$p.adj), , drop=FALSE]; top <- head(top, 20)
  write.csv(dr, file.path(outdir, "diffRegulation.csv"), row.names=FALSE)
  write.csv(top, file.path(outdir, "top20_source.csv"), row.names=FALSE)
  write.csv(dr, file.path(outdir, "volcano_source.csv"), row.names=FALSE)
  write.csv(data.frame(category=c("Significant", "Not significant"), count=c(nrow(sig), sum(dr$gene != gene)-nrow(sig))), file.path(outdir, "significance_source.csv"), row.names=FALSE)
  saveRDS(ans, file.path(outdir, "result.rds"), compress=TRUE)
  meta <- list(human_manifest_gene=human_gene, mouse_gene=gene, mode=mode, n_genes=nrow(dr), n_significant=nrow(sig), seed=SEED, qc_minLibSize=QC_MIN_LIB_SIZE, qc_mtThreshold=QC_MAX_MT_RATIO, qc_minPCT=QC_MIN_PCT, nc_nNet=NC_NNET, nc_nCells=NC_NCELLS, nc_nComp=NC_NCOMP, oe_factor=if (mode == "oe") OE_FACTOR else NA, elapsed_sec=elapsed, WT_reused=TRUE, package_versions=c(scTenifoldNet=as.character(packageVersion("scTenifoldNet")), scTenifoldKnk=as.character(packageVersion("scTenifoldKnk"))))
  write_json(meta, file.path(outdir, "summary.json"), auto_unbox=TRUE, pretty=TRUE)
  cat("status=success elapsed_sec=", elapsed, " n_significant=", nrow(sig), "\n", sep="")
  list(status="success", elapsed_sec=elapsed, n_genes=nrow(dr), n_significant=nrow(sig), top_gene=if(nrow(top)) top$gene[[1]] else NA_character_)
}

for (mode in if (MODE == "both") c("ko", "oe") else MODE) {
  out_mode <- file.path(OUTROOT, mode); dir.create(out_mode, recursive=TRUE, showWarnings=FALSE)
  status_path <- file.path(out_mode, paste0(mode, "_summary.csv"))
  if (file.exists(status_path)) {
    status <- read.csv(status_path, stringsAsFactors=FALSE)
    if (!all(status$target == target_map$human_manifest_gene)) status <- NULL
  } else status <- NULL
  if (is.null(status)) {
    status <- data.frame(target=target_map$human_manifest_gene, mouse_gene=target_map$mouse_matrix_gene, mode=mode, status="pending", n_genes=NA_integer_, n_significant=NA_integer_, top_gene=NA_character_, elapsed_sec=NA_real_, error=NA_character_, stringsAsFactors=FALSE)
    write.csv(status, status_path, row.names=FALSE)
  }
  for (i in seq_len(nrow(target_map))) {
    if (status$status[i] %in% c("success", "failed_missing_target")) next
    if (!target_map$mouse_matrix_gene[[i]] %in% rownames(WT)) {
      z <- list(status="failed_missing_target", elapsed_sec=0, error="target absent after QC (no detected counts in case matrix)")
    } else {
      z <- run_one(mode, target_map$human_manifest_gene[[i]], target_map$mouse_matrix_gene[[i]])
    }
    status$status[i] <- z$status; status$elapsed_sec[i] <- z$elapsed_sec
    if (!is.null(z$n_genes)) status$n_genes[i] <- z$n_genes
    if (!is.null(z$n_significant)) status$n_significant[i] <- z$n_significant
    if (!is.null(z$top_gene)) status$top_gene[i] <- z$top_gene
    if (!is.null(z$error)) status$error[i] <- z$error
    write.csv(status, status_path, row.names=FALSE)
    rm(z); gc(verbose=FALSE)
  }
  cat("completed mode=", mode, "\n", sep="")
}
cat("finished=", format(Sys.time(), tz="UTC"), "\n", sep="")
