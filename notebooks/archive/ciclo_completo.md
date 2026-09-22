# ciclo_completo_resultadosxclusters.ipynb

## Purpose

Evaluate the **complete end-to-end cycle** on the **internal test split**: given Hamiltonian parameters $\boldsymbol{\theta}$ from the test set, generate an image with the DDPM, pass it through Xception to re-estimate $\hat{\boldsymbol{\theta}}$, and assess quality through regression, image fidelity, physical observables, and per-cluster breakdowns.

**Status:** ✅ Complete and validated.  
**Frameworks:** TensorFlow (Xception) + PyTorch (DDPM) — both loaded simultaneously.

---

## Dataset

- **Source:** `dataset-spines-united-v2` (internal)
- **Evaluation subset:** 15% test split (same seed as training notebooks)
- **Cluster labels:** Available in `data['labels']` — used for per-phase analysis in Section 17–19

---

## Notebook Structure

| Cell / Section | What it does |
|---|---|
| **0. Setup** | Install dependencies, mount Kaggle, download both model checkpoints |
| **1. Config & paths** | Set `WORKDIR`, dataset paths, model checkpoint paths |
| **2. Imports** | TF, PyTorch, sklearn, matplotlib, UMAP |
| **3. Splits & scalers** | Reproduce exact 70/15/15 split and fit both `MinMaxScaler` objects (DDPM + Xception) |
| **4. Load Xception** | Reconstruct model architecture and load `.h5` weights |
| **5. Load DDPM** | Reconstruct `ConditionalUNet` + `DDPMScheduler`, load EMA weights from `.pt` |
| **6. Preprocessing functions** | `params_phys_to_ddpm_cond`, `ddpm_img_to_xception_input`, etc. |
| **7. Metrics** | Image metrics (MSE, Var-MSE, SSIM, FFT-MSE, FFT-Corr) and physical metrics (M, \|M\|, χ, Cv, E) |
| **8. Visual style** | Unified matplotlib style, HTML table display |
| **9. Cycle functions** | `run_inverse_cycle(idx)`, `plot_inverse_cycle(result)` |
| **10. Massive evaluation** | Loop over `N_BATCH_EVAL` test samples with ensemble DDPM generation |
| **11. Regression metrics** | R², MAE, RMSE per parameter (θ_input vs θ̂) |
| **12. True vs pred plots** | 2×4 scatter grid per parameter |
| **13. Image metrics** | MSE/SSIM/FFT-MSE/FFT-Corr table (mean, median, p95, std) |
| **14. Physical metrics** | M, \|M\|, χ, Cv, E for original vs generated (regime a vs b) |
| **15. Gallery** | 12 random examples: original \| generated \| \|diff\| |
| **16. Export CSV** | `inverse_cycle_params_metrics.csv`, `inverse_cycle_image_metrics.csv` |
| **17. Cluster stats** | Per-cluster physical metrics (regime a for originals, regime b for generated) |
| **19. R² and MAE per cluster** | Per-phase regression accuracy + representative image galleries |

---

## Key Functions

### `params_phys_to_ddpm_cond(y_phys)`
Converts a physical parameter vector (meV, K) to the normalised condition input expected by the DDPM.

```python
def params_phys_to_ddpm_cond(y_phys):
    y = np.array(y_phys, dtype=np.float32).reshape(1, -1)
    return scaler_ddpm.transform(y)   # [0,1] normalised
```

### `ddpm_img_to_xception_input(img_40)`
Converts a DDPM output (40×40, $[-1,1]$) to an Xception input (224×224, RGB, $[0,1]$).

```python
def ddpm_img_to_xception_input(img_40):
    img_39  = img_40[:39, :39]                   # centre crop 40→39
    img_01  = (img_39 + 1) / 2                   # [-1,1] → [0,1]
    img_224 = cv2.resize(img_01, (224, 224))
    img_rgb = np.stack([img_224]*3, axis=-1)     # grayscale → RGB
    return img_rgb[np.newaxis]                   # add batch dim
```

### `run_inverse_cycle(idx, n_steps=DDPM_FAST_STEPS)`
Executes the full cycle for a single test index:
1. Fetch $\boldsymbol{\theta}_{idx}$ and original image from the test set
2. Normalise $\boldsymbol{\theta}$ → DDPM condition
3. Generate image with DDPM (fast sampling with `n_steps` DDIM steps)
4. Crop/resize → Xception input
5. Predict $\hat{\boldsymbol{\theta}}$ with Xception
6. Return all intermediates + metrics

### `regression_metrics(y_true, y_pred, names)`
Computes R², MAE, RMSE for each of the 8 parameters and returns a styled DataFrame.

### Image metric functions
- `metric_mse(a, b)`: MSE over disk mask
- `metric_var_mse(a, b)`: variance of per-pixel squared errors
- `metric_ssim(a, b)`: SSIM (from `skimage`)
- `metric_fft(a, b)`: returns `(FFT-MSE, FFT-Corr)` computed on normalised amplitude spectra

### Physical metric functions (regime a — single-config estimator)
- `compute_M(img)`: average $s_z$ within disk mask
- `compute_absM(img)`: $|M|$
- `compute_chi_proxy(img, T)`: spatial correlation proxy for $\chi$
- `compute_E(img)`: nearest-neighbour exchange energy density
- `compute_Cv_proxy(img, T)`: Binder's block method for $C_v$

### Physical metric functions (regime b — ensemble estimator)
Applied over $K_{ens}$ DDPM samples for the same $\boldsymbol{\theta}$:
- `compute_chi_ensemble(M_list, T)`: $\chi = \frac{N_\mathcal{M}}{T}(\langle M^2 \rangle - \langle M \rangle^2)$
- `compute_Cv_ensemble(E_list, T)`: $C_v = \frac{N_\mathcal{M}}{T^2}(\langle E^2 \rangle - \langle E \rangle^2)$

---

## Scaler Reproducibility

Both scalers must be **identically reproduced** from the original training notebooks:

```python
# Xception scaler — fitted on Xception training split
idx_train_inv, idx_test_inv, _, _ = train_test_split(all_idx, params, test_size=0.15, random_state=42)
idx_train_inv, idx_val_inv, _, _  = train_test_split(idx_train_inv, ..., test_size=0.1765, random_state=42)
scaler_inv = MinMaxScaler().fit(params[idx_train_inv])

# DDPM scaler — fitted on DDPM training split (same indices, same seed)
scaler_ddpm = MinMaxScaler().fit(params[idx_train_ddpm])
```

Since both notebooks use `SEED=42` and the same dataset version, the train indices are identical and both scalers will be consistent.

---

## Per-Cluster Analysis (Sections 17–19)

The dataset provides magnetic phase `labels` (0–3). For each cluster, the notebook computes:
- Mean Hamiltonian parameters $\langle T^{(0)} \rangle$, $\langle \tilde{J}_2 \rangle$, $\langle \tilde{K}_{DM} \rangle$
- R² and MAE per parameter **within that cluster's test samples only**
- 5 representative images closest to the cluster centroid in UMAP space
- Comparison of physical metrics (regime a original vs regime b generated) per cluster

---

## Output Artifacts

| File | Description |
|---|---|
| `inverse_cycle_params_metrics.csv` | R², MAE, RMSE per parameter (global) |
| `inverse_cycle_image_metrics.csv` | MSE/SSIM/FFT statistics (global) |
| `ciclo_phys_summary.csv` | Physical metrics summary (global) |
| Various `.png` figures | Scatter plots, gallery, cluster analysis |

---

## Related Documentation

- Cycle theory: [docs/05_complete_cycle.md](../../docs/05_complete_cycle.md)
- Metrics: [docs/07_metrics.md](../../docs/07_metrics.md)
- Inverse model: [notebooks/inverse/xception_inverse.md](../inverse/xception_inverse.md)
- Generative model: [notebooks/generative/ddpm_train.md](../generative/ddpm_train.md)
