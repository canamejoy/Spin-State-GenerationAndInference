# XceptionFullDataBaseV3100.ipynb

## Purpose

Train an **Xception CNN regression model** to solve the inverse problem: predict 8 Hamiltonian parameters $\boldsymbol{\theta}$ from a 39×39 $s_z$ magnetisation image. This is the inverse task component of the end-to-end cycle.

**Status:** ✅ Complete and validated.

---

## Dataset

- **Source:** Kaggle `carloscanamejoy/dataset-spines-united-v2`
- **File:** `dataset_unificado_v2.npz`
- **Size:** 169,671 samples — images of shape `(39, 39)`, parameter vectors of shape `(8,)`

---

## Notebook Structure

| Cell / Section | What it does |
|---|---|
| **Libraries** | Imports TF/Keras, sklearn, UMAP, matplotlib |
| **Seed** | Fixed `SEED=42` for reproducibility |
| **Load dataset** | Downloads from Kaggle via API, loads `.npz` |
| **Preprocessing** | Adds channel dim, splits 70/15/15, fits `MinMaxScaler` on training targets only |
| **tf.data pipeline** | Batches of 512, resize 39→224, grayscale→RGB broadcast, prefetch |
| **Model definition** | Xception backbone + custom regression head |
| **Training** | Adam + MSE loss + EarlyStopping + ReduceLROnPlateau |
| **Save weights** | Saves `.h5` model to Google Drive |
| **Evaluation** | Scatter plots (true vs predicted), R²/MAE/RMSE/MAPE per parameter |
| **Dashboard** | Horizontal bar charts for R², MAE, RMSE per parameter |

---

## Model Architecture

```
Input: (39, 39, 1)
  → Resize (224, 224)
  → Grayscale → RGB (broadcast channels)
  → Xception backbone (ImageNet pretrained)
      Entry Flow:  Conv 3×3 stem + SepConv blocks with pooling
      Middle Flow: 8× repeated SepConv blocks (14×14 feature maps)
      Exit Flow:   SepConv + pooling
  → Global Average Pooling (spatial → 2048-dim vector)
  → BatchNormalization
  → Dropout(0.4)
  → Dense(256, activation='relu', kernel_regularizer=L2(1e-4))
  → BatchNormalization
  → Dropout(0.3)
  → Dense(8, activation='linear')   ← θ̂ in [0,1] (normalised)
```

**Total trainable parameters:** ~20.8 M

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | $10^{-4}$ |
| Loss | MSE |
| Batch size | 512 |
| Max epochs | 100 |
| EarlyStopping patience | 8 (on `val_loss`) |
| ReduceLROnPlateau factor | 0.3, patience 4 |
| Min LR | $10^{-6}$ |

---

## Key Functions

### `preprocess_batch(x)`
Resizes an image batch from 39×39 to 224×224 and broadcasts the single grayscale channel to 3 RGB channels. Applied inside the `tf.data` pipeline for GPU-accelerated preprocessing.

```python
def preprocess_batch(x):
    x = tf.image.resize(x, (224, 224))
    x = tf.image.grayscale_to_rgb(x)
    return x
```

### Scaler workflow
```python
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train)   # fit ONLY on train
y_val_scaled   = scaler.transform(y_val)
y_test_scaled  = scaler.transform(y_test)

# After prediction: inverse transform back to physical units
y_pred = scaler.inverse_transform(y_pred_scaled)
```

> ⚠️ **Important for the cycle:** Save this `scaler` object (or its `data_min_`, `data_range_` attributes). The cycle notebook must use the same scaler to interpret Xception outputs in physical units.

---

## Results Summary

Achieved on the 15% holdout test set:

| Parameter | $R^2$ | MAE (physical units) |
|---|---|---|
| $\tilde{K}_{DM}$ | 0.9498 | 0.049 meV |
| $\tilde{J}_2$ | 0.9146 | 0.030 meV |
| $T^{(0)}$ | 0.8430 | 0.890 K |
| $\tilde{H}_{ex}$ | 0.6107 | 0.007 meV/atom |
| $\tilde{K}_{anS}$ | 0.5686 | 0.030 meV/atom |
| $\tilde{K}_{an1}$ | 0.5012 | 0.035 meV/atom |
| $\tilde{J}_4$ | −0.012 | 0.032 meV |
| $\tilde{J}_3$ | −0.101 | 0.046 meV |

$\tilde{J}_3$ and $\tilde{J}_4$ are not identifiable from the $s_z$ projection alone (see [docs/03_inverse_problem.md](../../docs/03_inverse_problem.md)).

---

## Output Artifacts

| File | Description |
|---|---|
| `modelo_xception_fulldatabaseV3100.h5` | Saved model weights (Google Drive) |
| `vit_predictions_grid.png` | True vs predicted scatter plots per parameter |
| `vit_metrics_dashboard.png` | R²/MAE/RMSE bar chart dashboard |

---

## Related Documentation

- Theory: [docs/03_inverse_problem.md](../../docs/03_inverse_problem.md)
- Dataset: [docs/06_datasets.md](../../docs/06_datasets.md)
- Metrics: [docs/07_metrics.md](../../docs/07_metrics.md)
