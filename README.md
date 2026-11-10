# Sarcopenia transcriptomic analysis code

This repository is intended to hold analysis and figure-generation source code for the manuscript on potential molecular programs associated with sarcopenia. It contains code only. No expression matrices, metadata tables, model checkpoints, ligand or receptor structures, molecular-dynamics trajectories, figures, manuscripts, or analysis outputs are included.

## Scope

The scripts are preserved analysis snapshots. They document the workflows used for bulk RNA-seq, pathway analysis, spatial visualization, network perturbation, compound prioritization, and docking-result visualization. They are not a turnkey package: workflows require separately obtained data, structures, trained model assets, and (for spatial reconstruction) another project script. Supply these inputs locally before running a workflow.

All scripts that use the project data root read it from the `SARCOPENIA_PROJECT_ROOT` environment variable. For example, in PowerShell set `$env:SARCOPENIA_PROJECT_ROOT = 'D:\path\to\your\local\project'` before running a script. The required subdirectory layout is encoded in each script. The docking figure script also accepts `FIGURE6_CONFIG`; `scripts/figure_config.example.json` documents its configuration schema and expected local input locations. No input data or molecular assets are included here.

| Script | Workflow |
|---|---|
| `scripts/bulk_figures__make_main_figures_2_5.py` | Bulk expression and figure generation |
| `scripts/docking_figure__draw_figure6.py` | Docking and compound-prioritization figure |
| `scripts/drug_inference__run_drugreflector_gse167186.py` | DrugReflector inference |
| `scripts/network_export__export_sc3v_case_matrix.py` | Single-nucleus case-matrix export |
| `scripts/network_figure__rebuild_figure5_panelADE_polish_20261005.py` | Virtual-perturbation figure generation |
| `scripts/network_run__run_three_virtual_efficient.R` | Virtual knockout and overexpression network analyses |
| `scripts/pathways__run_pathway_analysis.py` | Enrichment and pathway analyses |
| `scripts/spatial_figure__rebuild_main_he_visium_result_plate.py` | Spatial expression visualization |
| `scripts/figure_config.example.json` | Non-data configuration template for Figure 6 |

Some scripts depend on input files, model assets, external code, and software packages that are not redistributed with this code-only repository. Python scripts import packages including NumPy, pandas, Matplotlib, SciPy, scikit-learn, statsmodels, gseapy, requests, Pillow, PyMuPDF, AnnData, Scanpy, and PyTorch; the R workflow uses Matrix, scTenifoldNet, scTenifoldKnk, and jsonlite. Install the packages required by the workflow you intend to run. No lockfile is provided. No license is included; public availability does not grant reuse rights beyond applicable law.

