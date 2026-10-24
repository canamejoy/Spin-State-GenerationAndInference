# ddpm_spines_train.ipynb

## Purpose

Train a **conditional Denoising Diffusion Probabilistic Model (DDPM)** to solve the direct problem: synthesise a physically faithful 39×39 $s_z$ magnetisation image from 8 Hamiltonian parameters $\boldsymbol{\theta}$. This is the generative component of the end-to-end cycle.

**Status:** ✅ Complete and validated.  
**Framework:** PyTorch (not TensorFlow — separate from the Xception notebook)

---

## Dataset

- **Source:** Kaggle `carloscanamejoy/dataset-spines-united-v2`
- **File:** `dataset_unificado_v2.npz`
- **Size:** 169,671 samples
- **Split:** 70/15/15 with `SEED=42` (same indices as Xception notebook)

---

## Notebook Structure

| Cell / Section | What it does |
|---|---|
| **0. Header** | Context, instructions for Colab A100 runtime |
| **1. Setup** | GPU check (`nvidia-smi`), install `pytorch-msssim`, upload `kaggle.json` |
| **2. Imports** | PyTorch, SSIM, numpy, matplotlib global config |
| **3. Dataset loading** | Download from Kaggle, load `.npz`, build 70/15/15 split |
| **4. PyTorch Dataset** | `SpinesDataset` — handles padding 39→40 and scaling |
| **5. Noise schedule** | `DDPMScheduler` — cosine schedule |
| **6. U-Net** | `ConditionalUNet` — sinusoidal time embedding + cross-attention conditioning |
| **7. Helpers** | `EMA`, loss functions, fast sampling (DDIM-style) |
| **8. Training function** | `train_one_run` — main training loop with validation |
| **9. Final training** | Launch with best Optuna hyperparameters, full dataset |
| **10. Save checkpoint** | Save model + EMA state to `.pt` file |
| **11. Training curves** | Loss, SSIM evolution plots |
| **12. Test evaluation** | Load checkpoint, generate on test set, report SSIM |
| **13. Visual comparison** | 8 real vs generated image pairs |
| **14. Export artifacts** | Zip and download checkpoint + metrics |
| **15. Drive backup** | (Optional) backup to Google Drive |
| **16. Sanity test** | R²(T) on generated images via quick inverse cycle check |

---

## Key Classes and Functions

### `SpinesDataset`
PyTorch `Dataset` that loads images and parameters, applies reflect padding from 39×39 → 40×40, and scales images to $[-1, 1]$.

```python
class SpinesDataset(Dataset):
    def __init__(self, imgs, params, img_size=40):
        # imgs: (N, 39, 39), params: (N, 8) — already MinMax scaled [0,1]
        ...
    def __getitem__(self, idx):
        img = F.pad(img_tensor, (0,1, 0,1), mode='reflect')  # 39→40 reflect
        img = img * 2 - 1                                      # [0,1]→[-1,1]
        return img, cond
```

**Why reflect padding?** Preserves boundary texture statistics (no artificial zero border). The U-Net requires spatial dimensions divisible by 2^n for its downsampling path.

### `DDPMScheduler`
Manages the noise variance schedule. Supports `'linear'` and `'cosine'` schedules. Precomputes $\bar\alpha_t$ for direct $X_0 \to X_t$ sampling.

```python
scheduler = DDPMScheduler(T=1000, beta_start=1e-4, beta_end=0.02, schedule='cosine')
# Key tensors: betas, alphas, alphas_cumprod (ᾱ_t)
```

**Cosine schedule** (chosen after Optuna search): smoother noise growth, avoids abrupt transitions that destabilise training.

### `ConditionalUNet`
The denoising backbone. Architecture:
- **Encoder path:** residual blocks with time + condition injection, downsampling
- **Bottleneck:** attention block
- **Decoder path:** residual blocks with skip connections, upsampling
- **Time encoding:** sinusoidal embeddings → projected MLP
- **Condition injection:** parameter vector $Y_n$ injected via **cross-attention** at each resolution level

```python
model = ConditionalUNet(
    img_channels=1,
    base_ch=...,
    cond_emb_dim=...,
    beta_schedule='cosine'
)
```

### `sinusoidal_embedding(t, dim)`
Encodes the scalar timestep $t$ into a $d$-dimensional vector using sinusoidal frequencies, analogous to positional encoding in transformers.

```python
def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)
```

### `EMA` (Exponential Moving Average)
Maintains a shadow copy of model weights with decay=0.999. At inference, EMA weights are loaded instead of the raw trained weights, yielding smoother and more stable generation.

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1-self.decay) * v
```

### `train_one_run(params, split_dict, epochs, ...)`
Main training loop:
1. For each batch: sample random timestep $t$, compute $X_t$ via the closed-form forward process, predict $\hat\epsilon_\theta$, compute MSE loss
2. Backpropagate, update model + EMA
3. Validate every `val_batches` steps using fast sampling (fewer DDIM steps)
4. Track best validation SSIM, save best checkpoint

---

## Training Configuration

| Hyperparameter | Value (from Optuna) |
|---|---|
| Learning rate | ≈ 2.264×10⁻⁴ |
| Timesteps $T$ | 1000 |
| Noise schedule | Cosine |
| Batch size | (from best HP) |
| EMA decay | 0.999 |
| Validation metric | SSIM on fast-sampled images |
| Max epochs | 150 |
| Optimiser | AdamW |

---

## Checkpoint Structure

The saved `.pt` file contains:

```python
{
    'model':       model.state_dict(),        # raw model weights
    'ema':         ema.shadow,                 # EMA shadow weights (use these at inference)
    'hyperparams': BEST_HPARAMS,              # full HP dict
    'history':     {'train_loss': [...], 'val_ssim': [...]},
    'best_val_ssim': float,
    'scaler_min':  scaler_ddpm.data_min_,     # MinMaxScaler internals
    'scaler_scale': scaler_ddpm.scale_,
}
```

> ⚠️ **Always load EMA weights for inference**, not the raw `model` weights.

---

## Output Artifacts

| File | Description |
|---|---|
| `ddpm_final_checkpoint.pt` | Full checkpoint with EMA weights + scaler |
| `ddpm_final_metrics.json` | Best/last val SSIM, epoch count |
| `ddpm_final_artifacts.zip` | Zip of both for download |

---

## Related Documentation

- Theory: [docs/04_direct_problem.md](../../docs/04_direct_problem.md)
- Dataset: [docs/06_datasets.md](../../docs/06_datasets.md)
- Cycle usage: [docs/05_complete_cycle.md](../../docs/05_complete_cycle.md)
