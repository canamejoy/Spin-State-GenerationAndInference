# 03 · Results

Analysis, evaluation, and interpretability notebooks that reproduce the paper's quantitative
results and figures from the trained models of [`02_models/`](../02_models/).

| Notebook | Description |
|----------|-------------|
| [`dataset_analysis.ipynb`](dataset_analysis.ipynb) | Parameter distributions, pairwise correlation matrix (documenting the mutually-exclusive $J_2$/$K_{DM}$ regimes), and 2D phase diagrams of the magnetization textures. |
| [`model_metrics.ipynb`](model_metrics.ipynb) | Per-parameter $R^2$ and MAE for every architecture; identifies which parameters are recoverable from the $s_z$ observation. |
| [`model_comparison.ipynb`](model_comparison.ipynb) | Cross-architecture comparison: accuracy-vs-model-size, per-phase performance, and UMAP structure of the learned latent spaces. |
| [`rams_interpretability.ipynb`](rams_interpretability.ipynb) | **Regression Activation Maps (RAMs)** — per-layer and per-parameter attribution maps, spatial overlays, and layer×parameter heatmaps. |
| [`interpretability_benchmark.ipynb`](interpretability_benchmark.ipynb) | Benchmark of **RAMs vs LIME / SHAP / SF-SHAP** via deletion curves, per-structure AUC, and computational cost. |

## Suggested reading order

`dataset_analysis` → `model_metrics` → `model_comparison` → `rams_interpretability` →
`interpretability_benchmark`.

## Inputs

These notebooks consume the dataset (`.npz`) and the model weights + target scalers exported by
the training notebooks in [`02_models/`](../02_models/).
