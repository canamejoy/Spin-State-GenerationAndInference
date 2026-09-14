# End-to-End Cycle Evaluation

## Overview

The complete cycle connects the **direct task** (DDPM: parameters → image) and the **inverse task** (Xception: image → parameters) into a closed loop. It provides a holistic assessment of how well the generative model preserves the physical information encoded in Hamiltonian parameters, evaluated through three complementary lenses: regression quality on recovered parameters, pixel-level image fidelity, and three physical observables of the spin configuration.

---

## Pipeline Diagram

```
Input parameters θ ∈ ℝ⁸  (physical units)
        │
        ▼
  ┌─────────────┐
  │    DDPM     │  params → normalised cond. → generates 40×40 image in [-1,1]
  └─────────────┘
        │
        ▼  top-left crop → 39×39 (exact, no interpolation)
        │
        ├──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
  ┌──────────────────┐                                   ┌──────────────────────┐
  │  Image metrics   │                                   │  Physical metrics    │
  │  MSE, Var(MSE)   │  ← compare with original →       │  M, C_nn, q_peak     │
  │  SSIM, FFT-MSE   │                                   │  (crop THEN mask,    │
  │  FFT-Corr        │                                   │   disk r_d = 18.25)   │
  │                  │                                   └──────────────────────┘
  └──────────────────┘
        │
        ▼  resize to 224×224, grayscale→RGB
  ┌─────────────┐
  │  Xception   │  image → predicted θ̂ ∈ ℝ⁸ (normalised) → inverse MinMax → physical units
  └─────────────┘
        │
        ▼
  Regression metrics: R², MAE, RMSE  (θ_input vs θ̂)
```

---

## Critical Preprocessing Rules

Correct scaler alignment is essential for the cycle to be consistent. Both models were trained with their own MinMaxScaler fitted **only on the internal training split**:

| Step | Model | Scaler action |
|---|---|---|
| `params_phys_to_ddpm_cond(y_phys)` | DDPM | Apply DDPM scaler (fitted on DDPM train split) |
| `ddpm_image_to_xception_input(img_40)` | — | Top-left crop 40→39, rescale to physical units, resize 39→224, broadcast to RGB |
| `xception_pred_to_phys(y_scaled)` | Xception | Inverse apply Xception scaler (fitted on Xception train split) |

**The two scalers are not the same** because the internal dataset version (`dataset-spines-united-v2`, 169,671 samples) used for training differs slightly in split seeds from the full dataset. Mixing scalers corrupts the cycle.

---

## Evaluation Protocols

### A. Internal Dataset

- Source: test split of `dataset-spines-united-v2` (15% holdout, same seed as training)
- DDPM generates $K_{ens}$ samples per parameter point
- Regression metrics compare **input parameters** $\boldsymbol{\theta}$ vs **re-estimated parameters** $\hat{\boldsymbol{\theta}}$
- Image metrics compare **original $s_z$ image** vs **generated image** (after crop)
- Physical metrics computed on the original (one image per parameter point) and on the generated side (mean over the $K_{ens}$ samples), so both sides use the same per-image estimator
- Additional analysis: **per-cluster R² and MAE** using magnetic phase labels from the dataset

### B. External Dataset

- Source: `dataset-spines-complete` (218,256 samples — the full dataset from Paper 1)
- **Scalers are still fitted on the internal dataset** (no re-fitting on external data)
- Evaluates out-of-distribution generalisation of the full cycle
- Currently the external cycle shows degraded performance, indicating a distribution shift between datasets or scaler mismatch effects
- Work in progress

---

## Metrics Reference

### Regression Metrics (parameter recovery)

| Metric | Formula | Interpretation |
|---|---|---|
| $R^2$ | $1 - \frac{\sum(\theta - \hat\theta)^2}{\sum(\theta - \bar\theta)^2}$ | Variance explained; 1.0 = perfect |
| MAE | $\frac{1}{N}\sum|\theta - \hat\theta|$ | Mean absolute error in physical units |
| RMSE | $\sqrt{\frac{1}{N}\sum(\theta - \hat\theta)^2}$ | Root mean squared error |

### Image Metrics

See [07_metrics.md](07_metrics.md) for full formulas.

| Metric | Range | Good value |
|---|---|---|
| MSE | $[0, \infty)$ | Low |
| Var(MSE) | $[0, \infty)$ | Low |
| SSIM | $[-1, 1]$ | High (→1) |
| FFT-MSE | $[0, \infty)$ | Low |
| FFT-Corr | $[-1, 1]$ | High (→1) |

### Physical Metrics

The canonical set is **exactly three**. See [07_metrics.md](07_metrics.md) for
formulas and for why $|M|$, $\chi$, $C_v$ and $E$ were removed.

| Observable | Symbol | Physical meaning | Range |
|---|---|---|---|
| Average magnetisation | $M$ | Net spin polarisation; $\approx \pm 1$ = ferromagnetic/saturated, $\approx 0$ = chiral | $[-1, 1]$ |
| Nearest-neighbour spin correlation | $C_{nn}$ | Local alignment of the $s_z$ projection | $[-1, 1]$ |
| Peak wave vector | $q_{peak}$ | Dominant spatial frequency of the texture | rad/site; `nan` if uniform |

> **Ordering rule.** Crop the $40\times40$ canvas to $39\times39$ **before**
> applying the disk mask. The padding sits on the right and bottom edges, so the
> top-left crop is exact; masking the canvas instead averages duplicated pixels
> on a disk displaced by half a pixel. `metrics.py` enforces this with a shape
> check.

---

## Per-Cluster Analysis

The internal cycle notebook additionally computes metrics broken down by **magnetic phase cluster** (using the `labels` array in the dataset). This allows identifying:
- Which phases the DDPM faithfully reproduces
- Which phases suffer most from degenerate generation
- How per-cluster image quality correlates with per-cluster regression accuracy

---

## Notebooks

### Evaluation cycle

| Notebook | Dataset | Status |
|---|---|---|
| [ciclo_completo_v2](../notebooks/cycle/ciclo_completo_v2.ipynb) | Internal test split | ✅ Complete |
| [cycle_complete_newmetrics](../notebooks/cycle/cycle_complete_newmetrics.ipynb) | Internal test split | ✅ Complete |
| [ciclo_texture_fidelity](../notebooks/cycle/ciclo_texture_fidelity.ipynb) | Internal, texture/saturation split | ✅ Complete |

Superseded runs live in `notebooks/replaced/` and are kept only as an archive —
they predate the crop-before-mask fix and the three-metric set, so their physical
numbers are not comparable with current results.

### Integrated cycle (training-time and sampling-time closure)

See [`integration_cycle/README.md`](../integration_cycle/README.md).

| Notebook | What it closes | Fine-tunes |
|---|---|---|
| `01_ddpm_cosine_frozen_encoder` | Latent cosine term in the DDPM loss | DDPM only |
| `02_ddpm_cosine_joint_finetune` | Same term, encoder also trains | DDPM + Xception encoder |
| `03_latent_guided_sampler` | Latent guidance at sampling time | Neither |
