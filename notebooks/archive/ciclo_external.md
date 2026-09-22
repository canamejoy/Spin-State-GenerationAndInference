# ciclo_external_dataset.ipynb

## Purpose

Evaluate the **complete end-to-end cycle** on the **external dataset** (`dataset-spines-complete`, 218,256 samples). This tests out-of-distribution generalisation: can the cycle (trained on the internal 169k dataset) correctly process parameters from the larger, broader-coverage dataset?

**Status:** 🔧 Work in progress — currently shows degraded performance compared to the internal cycle.

---

## Dataset

| | Internal | External |
|---|---|---|
| Kaggle slug | `dataset-spines-united-v2` | `dataset-spines-complete` |
| Samples | 169,671 | 218,256 |
| Image shape | 39×39 | 39×39 |
| Used for | Training + internal cycle | External cycle only |
| Scalers fitted on | ✅ Internal train split | ❌ Not re-fitted (uses internal scalers) |

The **scalers are always fitted on the internal training split** and applied without modification to the external dataset. This is by design: the models were trained with those scalers, and the cycle must use the same normalisation.

---

## Notebook Structure

| Cell / Section | What it does |
|---|---|
| **0. Setup** | Install deps, Kaggle auth, download both datasets |
| **1. Paths & constants** | `WORKDIR`, dataset paths, checkpoint paths, `PARAM_NAMES` |
| **2. Imports** | TF, PyTorch, sklearn, matplotlib |
| **3. Splits & scalers** | Reproduce internal 70/15/15 split to get both scalers; load external dataset separately |
| **4. GPU config & Xception** | GPU memory growth, rebuild Xception architecture, load `.h5` weights |
| **5. DDPM** | Rebuild `ConditionalUNet` + `DDPMScheduler`, load EMA weights from `.pt` |
| **6. Preprocessing** | `dataset_img_to_ddpm`, `params_phys_to_ddpm_cond`, `ddpm_crop_to_xception` |
| **7. Metrics** | Image and physical metric functions (same as internal cycle) |
| **8. Visual style** | Unified matplotlib + HTML table style |
| **9. Massive evaluation** | Loop over `N_BATCH_EVAL` samples from the external dataset |
| **10. Global R² and MAE** | Regression metrics per parameter |
| **11. Image metrics** | MSE/SSIM/FFT-MSE/FFT-Corr summary table |
| **12. Gallery** | 12 random examples (cmap=jet): original \| generated \| \|diff\| |
| **13. Export** | CSVs of parameter and image metrics |

---

## Key Differences from the Internal Cycle Notebook

### Scaler alignment
Both notebooks reproduce the internal split to fit scalers:

```python
# Step 1: fit scalers on INTERNAL train split (not on external data)
all_idx = np.arange(N_internal)
idx_train, idx_test, _, _ = train_test_split(all_idx, params_internal, test_size=0.15, random_state=42)
scaler_ddpm  = MinMaxScaler().fit(params_internal[idx_train])
scaler_xcept = MinMaxScaler().fit(params_internal[idx_train])

# Step 2: load EXTERNAL dataset separately
data_ext = np.load(EXTERNAL_DATASET_PATH)
imgs_ext    = data_ext["img"]
params_ext  = data_ext["params"]
```

### Sample selection
Instead of using a pre-existing test split, the notebook randomly samples `N_BATCH_EVAL` indices from the **full external dataset**:

```python
eval_idx = np.random.RandomState(SEED).choice(len(imgs_ext), N_BATCH_EVAL, replace=False)
```

### Preprocessing rule for existing images
Unlike the internal cycle (which generates images via DDPM and compares with originals), here the external dataset's original images serve as the reference:

```python
def dataset_img_to_ddpm(img_39):
    """Existing 39×39 image → 40×40 zero-padded → [-1,1] for DDPM comparison."""
    img_padded = np.pad(img_39, ((0,1),(0,1)), mode='constant')  # zero-pad (not reflect)
    return img_padded * 2 - 1
```

> **Note:** Zero-padding is used here (not reflect) because the image already represents a completed spin field, and we only need to match the DDPM output size — we're not generating from scratch.

---

## Known Issues / Degradation Sources

The external cycle currently shows lower performance than the internal cycle. Likely causes under investigation:

1. **Distribution shift:** The external dataset (218k samples) covers a broader/different region of the parameter space than the internal dataset (169k). Parameters outside the internal training distribution may be poorly handled.

2. **Scaler range mismatch:** The `MinMaxScaler` fitted on the internal training split may not cover the full range of external parameters if the external dataset samples different extremes.

3. **Dataset version differences:** The internal and external datasets were generated with slightly different simulation setups or parameter sampling strategies. Structural differences in the image statistics may degrade both the DDPM generation quality and the Xception prediction accuracy.

4. **Image comparison validity:** The external cycle compares DDPM-generated images against original Monte Carlo images. The DDPM was trained on a one-to-many mapping; even with the correct parameters, the generated texture may differ in spatial phase from the original — penalising pixel-level metrics unfairly.

---

## Debugging Checklist

- [ ] Verify that `scaler_ddpm.data_min_` and `scaler_ddpm.data_max_` cover all external parameter values
- [ ] Check if external parameter ranges exceed internal training ranges for any of the 8 parameters
- [ ] Compare FFT-Corr distributions (internal vs external) to detect spectral texture differences
- [ ] Run the cycle on a subset of external samples that fall **within** the internal parameter range and compare to the full external results

---

## Output Artifacts

| File | Description |
|---|---|
| `ciclo_params_r2_mae.csv` | R², MAE per parameter (external dataset) |
| `ciclo_imagen_metrics.csv` | Image metric summary (external dataset) |

---

## Related Documentation

- Cycle theory: [docs/05_complete_cycle.md](../../docs/05_complete_cycle.md)
- Metrics: [docs/07_metrics.md](../../docs/07_metrics.md)
- Internal cycle: [notebooks/cycle/ciclo_completo.md](ciclo_completo.md)
- Datasets: [docs/06_datasets.md](../../docs/06_datasets.md)
