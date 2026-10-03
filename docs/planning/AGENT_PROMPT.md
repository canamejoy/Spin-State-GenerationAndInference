# End-to-End Generative Framework for Magnetic Domain Images — Full Implementation Prompt

## Context & Mission

You are tasked with completing a research-grade machine learning project whose goal is to build and evaluate an **end-to-end deep learning framework** for generating 2D magnetic domain images from Heisenberg Hamiltonian parameters. The project will produce a scientific article in Elsevier format (~40 pages), with two main sections:

- **Part 1 — Generative comparison**: three models (CVAE-Xception, CVAE-ViT, DDPM) evaluated on image and physical metrics.
- **Part 2 — Full cycle with DDPM**: DDPM-generated images fed into an inverse Xception model to recover Hamiltonian parameters; evaluated on R², MAE, and physical consistency metrics.

Everything you produce must be **fully documented in English**, with professional presentation, clean code, and publication-ready figures.

---

## Repository Structure (current state)

```
End-to-endMSGenFreamework/
├── notebooks/
│   ├── generative/
│   │   ├── cvae-vit.ipynb             ← CVAE with ViT encoder (trained on 10% only)
│   │   ├── cvae-xception.ipynb        ← CVAE with Xception encoder (trained on 10% only)
│   │   └── ddpm_spines_train.ipynb    ← DDPM training (complete, trained on 100%)
│   ├── cycle/
│   │   ├── ciclo_completo_resultadosxclusters.ipynb  ← Full cycle internal (partial)
│   │   ├── ciclo_external_dataset.ipynb              ← External cycle (to replaced/)
│   │   ├── ciclo_external_fixes.ipynb                ← External fixes (to replaced/)
│   │   └── diagnostico-ciclo-externo.ipynb           ← External diagnostics (to replaced/)
│   └── inverse/
│       └── XceptionFullDataBaseV3100.ipynb           ← Xception training (complete)
└── src/  (empty or does not exist)
```

---

## IMMEDIATE FIRST TASK — Repository Cleanup

Move the following notebooks to a new folder `notebooks/replaced/` (create it if needed). Do **not** delete them:

- `notebooks/cycle/ciclo_external_dataset.ipynb`
- `notebooks/cycle/ciclo_external_fixes.ipynb`
- `notebooks/cycle/diagnostico-ciclo-externo.ipynb`

Keep everything else in place.

---

## Environment Setup — Two Execution Environments

Every notebook you create or modify must contain **two clearly separated environment setup cells**, one for each platform. Use a markdown cell as a header to label each:

### Kaggle Cell Pattern
```python
# ============================================================
# ENVIRONMENT SETUP — KAGGLE
# Run this cell only when executing on Kaggle.
# Datasets are pre-mounted at /kaggle/input/datasets/carloscanamejoy/
# ============================================================
import os, sys

BASE = "/kaggle/input/datasets/carloscanamejoy"

DATASET_PATH    = f"{BASE}/dataset-spines-united-v2/dataset_unificado_v2.npz"
XCEPTION_WEIGHTS = f"{BASE}/weights-xception-model/modelo_xception_fulldatabaseV3100.h5"
DDPM_CHECKPOINT  = f"{BASE}/weights-models/ddpm_spines_final_39crop.pt"
CVAE_XCP_WEIGHTS = f"{BASE}/weights-cvae-xception/cvae_xception_best.h5"   # after retraining
CVAE_VIT_WEIGHTS = f"{BASE}/weights-cvae-vit/cvae_vit_best.h5"             # after retraining

OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

### Colab Cell Pattern
```python
# ============================================================
# ENVIRONMENT SETUP — GOOGLE COLAB
# Run this cell only when executing on Google Colab.
# Datasets are downloaded via the Kaggle API using kaggle.json.
# Upload your kaggle.json file to the Colab session before running.
# ============================================================
import os, sys, shutil, zipfile
from google.colab import files

# --- Kaggle credentials ---
os.makedirs("/root/.kaggle", exist_ok=True)
uploaded = files.upload()                          # upload kaggle.json when prompted
shutil.move("kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)

# --- Install dependencies ---
os.system("pip install -q kaggle torch torchvision tensorflow scikit-learn scikit-image tqdm matplotlib seaborn")

# --- Download datasets ---
os.system("kaggle datasets download -d carloscanamejoy/dataset-spines-united-v2 -p /content/data --unzip")
os.system("kaggle datasets download -d carloscanamejoy/weights-xception-model -p /content/weights --unzip")
os.system("kaggle datasets download -d carloscanamejoy/weights-models -p /content/weights --unzip")
os.system("kaggle datasets download -d carloscanamejoy/weights-cvae-xception -p /content/weights --unzip")  # after retraining
os.system("kaggle datasets download -d carloscanamejoy/weights-cvae-vit -p /content/weights --unzip")       # after retraining

BASE = "/content"
DATASET_PATH     = f"{BASE}/data/dataset_unificado_v2.npz"
XCEPTION_WEIGHTS = f"{BASE}/weights/modelo_xception_fulldatabaseV3100.h5"
DDPM_CHECKPOINT  = f"{BASE}/weights/ddpm_spines_final_39crop.pt"
CVAE_XCP_WEIGHTS = f"{BASE}/weights/cvae_xception_best.h5"
CVAE_VIT_WEIGHTS = f"{BASE}/weights/cvae_vit_best.h5"

OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

> **NOTE**: In notebooks where both environments are present, place them at the very top, clearly labeled with markdown headers `## Kaggle Environment` and `## Google Colab Environment`. After each environment block, add a shared `## Shared Configuration` cell that uses the variables defined above (DATASET_PATH, XCEPTION_WEIGHTS, etc.) so the rest of the notebook is environment-agnostic.

---

## Domain Knowledge — Critical Reference

### Dataset
- **File**: `dataset_unificado_v2.npz`
- **Keys**: `img` shape `(N, 39, 39, 1)`, `params` shape `(N, 8)`
- **Total samples**: 169,671
- **Split**: 70% train / 15% val / 15% test, `SEED=42`, stratified by magnetic structure label
- **Image range**: `[-1, 1]` (spin sz values, tanh-compatible)
- **Parameters (8)**: `[T⁰, J̃₂, K̃DM, H̃ex, K̃anS, K̃an1, J̃₃, J̃₄]` — stored as `params` array columns 0–7 in the same order
- **Structure labels**: available in dataset as a separate key (check actual key name in the npz). Four magnetic phases: Ferromagnetic, Others (paramagnetic), Labyrinthine & Conical, Helical

### Image visualization
- **Colormap**: `cmap="jet"` for sz magnetization images
- **Range**: `vmin=-1, vmax=1`
- **Figures**: save as both `.png` (300 Dpi) and `.svg` (vector) for publication

### Model architecture summary (for loading, not retraining)

**Xception inverse model** (TensorFlow/Keras):
```python
def build_xception(n_out=8):
    inp  = tf.keras.Input(shape=(224, 224, 3))
    base = tf.keras.applications.Xception(weights=None, include_top=False, input_tensor=inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization_4")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization_5")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(n_out, activation="linear", name="dense_1")(x))

# Always load on CPU to avoid contention with PyTorch DDPM on GPU
with tf.device("/cpu:0"):
    xception_model = build_xception()
    xception_model.load_weights(XCEPTION_WEIGHTS)
```

**DDPM** (PyTorch — ConditionalUNet):
- The checkpoint `.pt` file contains keys: `hyperparams` (dict), `ema` (preferred state dict), `model` (fallback state dict)
- Architecture: `ConditionalUNet(img_channels=1, base_ch=hp["base_ch"], ch_mults=(1,2,4), cond_dim=8, emb_dim=hp["cond_emb_dim"], dropout=0.0)`
- Always pin explicitly to `cuda:0` when available to avoid multi-GPU device mismatch:
```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(DDPM_CHECKPOINT, map_location=DEVICE, weights_only=False)
hp   = ckpt["hyperparams"]
ddpm = ConditionalUNet(...).to(DEVICE)
state = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
ddpm.load_state_dict(state)
ddpm.eval()
MODEL_DEVICE = next(ddpm.parameters()).device
scheduler = DDPMScheduler(T=1000, schedule=hp["beta_schedule"], device=MODEL_DEVICE)
```

**TF + PyTorch coexistence pattern** (use in every notebook that needs both):
```python
# TensorFlow: allow GPU memory growth (prevents TF from claiming all VRAM)
import tensorflow as tf
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# PyTorch: run on cuda:0
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Xception inference: always on CPU
with tf.device("/cpu:0"):
    predictions = xception_model.predict(batch, verbose=0)
```

---

## NOTEBOOKS TO CREATE OR FULLY REWRITE

### NOTEBOOK 1 — `notebooks/generative/cvae_xception_retrain.ipynb`
**Purpose**: Retrain CVAE-Xception from scratch on 100% of the dataset. The user will run this manually on Kaggle or Colab.

**Architecture**:
- Encoder backbone: Xception pretrained on ImageNet, blocks 1–10 frozen, blocks 11–14 trainable (31/132 trainable layers)
- Input images: resized to 224×224, grayscale→RGB, range [-1,1] → normalized for Xception as needed
- Encoder output: GAP → (B, 2048) feature vector
- CVAE encoder network: `Dense(512, relu) → Dense(256, relu) → [mu_q (128), log_var_q (128)]`
- Prior network (conditioned on parameter embedding): `Dense(64, relu) → Dense(128, relu) → [mu_p (64), log_var_p (64)]`, with `cond_emb_dim=32`
- z_dim = 192
- Decoder: transposed conv `5×5 → 10×10 → 20×20 → 40×40` + 2 residual blocks, **tanh output** (ensures [-1,1] range)
- Conditioning: concatenate `[z_encoder, parameter_embedding]` at decoder input

**Loss function**:
```
L = (1 - SSIM(x, x̂)) + 0.05 * L1(x, x̂) + β * KLD(q(z|x,y) || p(z|y))
```

**β-KLD schedule** (anti-posterior-collapse, single schedule):
- β ramps from `1e-6` to `β_max=0.18` linearly over 12 epochs
- β stays constant at 0.18 for remaining epochs (total 40 epochs)

**Training config**:
- Optimizer: Adam, lr=1e-4 with ReduceLROnPlateau (patience=5, factor=0.5)
- Batch size: 64
- Split: 70/15/15, SEED=42
- EarlyStopping: patience=10 on val composite loss
- Save best model checkpoint as `cvae_xception_best.h5` and best weights as `cvae_xception_best_weights.h5`
- Log per-epoch: train_loss, val_loss, kl_weight (β), kl_active (fraction of active KLD units > threshold), var_q (mean variance of posterior)

**Metrics to track per epoch** (log and plot at end):
- Reconstruction: SSIM, L1, MSE
- KLD: raw KLD, β·KLD, kl_active ratio
- var_q: mean diagonal variance of approximate posterior q

**Output figures** (save to `OUTPUT_DIR/cvae_xception_training/`):
1. Training curves: loss components per epoch (SSIM, L1, β·KLD, total)
2. β schedule visualization
3. Sample reconstructions at epoch 1, 10, 20, 40: 5 random images → original | reconstruction | difference map

---

### NOTEBOOK 2 — `notebooks/generative/cvae_vit_retrain.ipynb`
**Purpose**: Retrain CVAE-ViT from scratch on 100% of the dataset. Ready to run on Kaggle or Colab.

**Architecture**:
- Encoder backbone: ViT (Vision Transformer), first 8 layers frozen, layers 9–12 trainable
- CVAE encoder: GAP on ViT output → `Dense(512, relu) → Dense(256, relu) → [mu_q (128), log_var_q (128)]`, z_dim=128
- Prior network: same structure as Xception CVAE, `cond_emb_dim=64`
- Decoder: transposed conv `5×5 → 10×10 → 20×20 → 40×40` + 2 residual blocks, **tanh output**
- Conditioning: concatenate `[z_encoder, parameter_embedding]`

**Loss function**:
```
L = (1 - SSIM(x, x̂)) + 0.0147 * L1(x, x̂) + 9.023 * MSE_patch_var(x, x̂) + β * KLD
```
where `MSE_patch_var` is the MSE between the local patch variance maps of original and reconstructed images (encourages texture reproduction).

**β-KLD schedule** — **F_plateau** (this is the chosen schedule, do NOT implement F_linear):
```python
def beta_for_epoch(epoch, total_epochs, beta_start=1e-6, beta_max=0.0657, warmup_epochs=5):
    """
    Warmup: linearly ramp from beta_start to beta_max over warmup_epochs.
    Plateau: hold at beta_max for remaining epochs.
    """
    if epoch <= warmup_epochs:
        p = (epoch - 1) / max(1, warmup_epochs - 1)
        return beta_start + (beta_max - beta_start) * p
    else:
        return beta_max
```

**Training config**:
- Optimizer: Adam, lr=1e-4 with ReduceLROnPlateau (patience=5, factor=0.5)
- Batch size: 32 (ViT is heavier)
- Epochs: 30
- Split: 70/15/15, SEED=42
- EarlyStopping: patience=8 on val composite loss
- Save best model checkpoint and weights (`.h5` format)

**Same per-epoch metrics and output figures as NOTEBOOK 1**, adapted for ViT architecture. Save figures to `OUTPUT_DIR/cvae_vit_training/`.

---

### NOTEBOOK 3 — `notebooks/evaluation/image_metrics_robustness.ipynb`
**Purpose**: Demonstrate why pixel-based metrics (MSE, BCE, SSIM) are insufficient for evaluating magnetic domain images, and why cosine similarity in Xception latent space is a better metric. This is a standalone experiment using **only real images** — no generative model needed.

**Requires**: Xception weights only (TF/CPU). No DDPM, no CVAE.

#### Section 1 — Data Loading & Preprocessing
- Load `dataset_unificado_v2.npz`
- Apply 70/15/15 split (SEED=42) — use only the **test split**
- Select N_ref = 500 images, stratified by structure label (125 per phase)
- Images already in [-1, 1] range

#### Section 2 — Circular Disk Mask
```python
def circular_mask(h=39, w=39):
    """Binary mask: True inside the inscribed circle of the 39x39 image."""
    cy, cx = h // 2, w // 2
    r = min(cy, cx)
    Y, X = np.ogrid[:h, :w]
    return (X - cx)**2 + (Y - cy)**2 <= r**2

MASK = circular_mask()  # shape (39, 39), dtype bool
```
Apply this mask to all metric computations (set pixels outside mask to 0 before computing metrics).

#### Section 3 — Four Robustness Conditions
```python
def shift_image(img, px, axis=1):
    """Shift image by px pixels along axis using np.roll."""
    return np.roll(img, px, axis=axis)

def reflect_image(img):
    """Horizontal flip (left-right reflection)."""
    return img[:, ::-1]
```

Generate comparison pairs for each of the N_ref reference images:
- **E0** — `(img, img)` — reference, should yield perfect scores
- **E1** — `(img, shift_image(img, 1))` — 1-pixel horizontal shift
- **E2** — `(img, shift_image(img, 5))` — 5-pixel horizontal shift
- **E3** — `(img, reflect_image(img))` — horizontal reflection

#### Section 4 — Metric Computation
For each condition, compute over all N_ref pairs:

```python
# MSE (within mask)
def masked_mse(a, b, mask=MASK):
    diff = (a - b) ** 2
    return diff[mask].mean()

# BCE (images in [-1,1] → shift to [0,1] for BCE)
def masked_bce(a, b, mask=MASK, eps=1e-7):
    a_ = (a[mask] + 1) / 2
    b_ = (b[mask] + 1) / 2
    return -np.mean(a_ * np.log(b_ + eps) + (1 - a_) * np.log(1 - b_ + eps))

# SSIM (use skimage, full image then masked region or directly on masked)
from skimage.metrics import structural_similarity as ssim
def masked_ssim(a, b, mask=MASK):
    return ssim(a, b, data_range=2.0)  # data_range=2 for [-1,1]

# Cosine similarity in Xception latent space
def extract_xception_features(imgs_batch, xception_model, layer_name="dense"):
    """
    Extract 256-dim features from the Dense(256) layer (GAP→BN→Drop→Dense256).
    imgs_batch: (B, 39, 39) numpy array, values in [-1, 1]
    Returns: (B, 256) numpy array
    """
    # Resize and convert grayscale to RGB
    imgs_4d = imgs_batch[..., np.newaxis]           # (B, 39, 39, 1)
    import tensorflow as tf
    imgs_resized = tf.image.resize(imgs_4d, (224, 224)).numpy()
    imgs_rgb = np.repeat(imgs_resized, 3, axis=-1)  # (B, 224, 224, 3)
    # Xception preprocessing: scale [-1,1] → [0,1] → then tf.keras.applications.xception.preprocess_input
    imgs_rgb = (imgs_rgb + 1) / 2.0
    imgs_rgb = tf.keras.applications.xception.preprocess_input(imgs_rgb * 255.0)
    # Build feature extractor
    feature_extractor = tf.keras.Model(
        inputs=xception_model.input,
        outputs=xception_model.get_layer(layer_name).output
    )
    with tf.device("/cpu:0"):
        features = feature_extractor.predict(imgs_rgb, batch_size=32, verbose=0)
    return features  # (B, 256)

def cosine_similarity(z1, z2):
    """Cosine similarity between paired feature vectors."""
    norm1 = np.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8
    norm2 = np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8
    return np.sum((z1 / norm1) * (z2 / norm2), axis=-1)  # (B,)
```

Compute for each condition E0–E3:
- `mse_scores`: array of shape (N_ref,)
- `bce_scores`: array of shape (N_ref,)
- `ssim_scores`: array of shape (N_ref,)
- `cosine_scores`: array of shape (N_ref,)

Also compute **per-structure breakdown**: split scores by the 4 magnetic phases.

#### Section 5 — Figures (save to `OUTPUT_DIR/robustness/`)

**Figure 1** — `robustness_metrics_summary.png/.svg`
4-panel figure (2×2 grid), one panel per metric (MSE, BCE, SSIM, Cosine).
Each panel: grouped bar chart, x-axis = conditions E0–E3, bars = mean ± std.
Color palette: use a consistent 4-color palette (e.g., `["#4C72B0","#DD8452","#55A868","#C44E52"]`).
Add horizontal dashed line at E0 value (reference).
Title each panel with the metric name and a short interpretation note.

**Figure 2** — `robustness_metrics_per_structure.png/.svg`
4-panel figure, one panel per metric. Within each panel: grouped bars where groups = conditions E0–E3, bar color = structure (4 colors, consistent legend). 
Ferromagnetic: `#1f77b4`, Others: `#ff7f0e`, Labyrinthine & Conical: `#2ca02c`, Helical: `#d62728`.

**Figure 3** — `robustness_qualitative.png/.svg`
Grid of 8 rows × 5 columns. Each row = one example image (2 per structure phase).
Columns: Original | E0 (ref) | E1 (+1px) | E2 (+5px) | E3 (reflected).
Use `cmap="jet"`, `vmin=-1, vmax=1`. Show metric values (MSE, SSIM, Cosine) as text below each generated image.
One title row above with column names.

**Figure 4** — `robustness_cosine_vs_ssim_scatter.png/.svg`
Scatter plot: x-axis = SSIM score, y-axis = Cosine score, colored by condition (E0–E3), markers shaped by structure.
This directly illustrates that cosine similarity is stable where SSIM degrades.

---

### NOTEBOOK 4 — `notebooks/evaluation/generative_comparison.ipynb`
**Purpose**: Compare the three generative models (CVAE-Xception, CVAE-ViT, DDPM) on image metrics, physical metrics, and qualitative inspection. This is the **core Part 1 notebook** for the article.

**Requires**: All 5 weight files (Xception, DDPM, CVAE-Xception, CVAE-ViT). Both TF and PyTorch.

> **Note**: CVAE weights will not exist until the user re-trains them (Notebooks 1 and 2). Add a clear warning cell at the top:
> ```python
> # ⚠ WARNING: CVAE weights (CVAE_XCP_WEIGHTS, CVAE_VIT_WEIGHTS) will not be available
> # until Notebooks 1 and 2 (cvae_xception_retrain.ipynb, cvae_vit_retrain.ipynb) have
> # been run and the resulting checkpoints uploaded to Kaggle datasets.
> # All sections up to and including "Section 2 — DDPM Evaluation" can run immediately.
> ```

#### Section 0 — Shared Setup
- Load dataset (test split only, SEED=42, 70/15/15)
- Build common test set: take N_eval = 1000 samples, stratified by structure (250 per phase)
- Load all models (Xception on CPU, DDPM on cuda:0)
- TF memory growth + PyTorch cuda:0 pinning (see coexistence pattern above)

**Ensemble size**: `K = 32` (configurable constant at top of notebook)

#### Section 1 — Image Generation
For each of the N_eval parameter vectors θ:
- **DDPM**: generate K=32 images by running `ddpm.sample(theta, K)` → shape `(K, 39, 39)`
- **CVAE-Xception**: sample K=32 images from prior p(z|θ) → decode → shape `(K, 39, 39)`
- **CVAE-ViT**: same as above
- Store: `ddpm_samples[i]` = `(K, 39, 39)`, similarly for cvae_xcp and cvae_vit
- For each sample set, pick `best_sample[i]` = the image with highest SSIM vs reference (used for Regime A metrics)

#### Section 2 — Image Metrics (Regime A — single best image per θ)
For each model, for each of the N_eval samples, compute:
- `masked_mse(ref, best_sample)` using circular disk mask
- `masked_bce(ref, best_sample)` using circular disk mask  
- `masked_ssim(ref, best_sample)`
- `cosine_sim(z_ref, z_best_sample)` where z is 256-dim Xception feature

Aggregate:
- Mean ± std over N_eval — overall summary table
- Mean ± std per structure (4 phases) — per-structure table
- Save all per-sample scores to CSV: `image_metrics_per_sample.csv`

#### Section 3 — Physical Metrics
Implement all four physical metrics:

##### 3a — Magnetization M
```python
def magnetization(img, mask=MASK):
    """Mean sz over the disk mask. Range [-1, 1]."""
    return img[mask].mean()

def abs_magnetization(img, mask=MASK):
    return np.abs(img[mask].mean())
```

**Regime A**: compute per best_sample image.
**Regime B**: compute per-ensemble mean: `M_ens[i] = mean([magnetization(img) for img in ensemble[i]])`

##### 3b — Nearest-Neighbor Correlation C_nn
```python
def cnn(img, mask=MASK):
    """
    Mean of sz_i * sz_j for all horizontally or vertically adjacent pairs
    where both pixels i and j are inside the circular mask.
    """
    h, w = img.shape
    total, count = 0.0, 0
    for dy, dx in [(0, 1), (1, 0)]:
        for y in range(h - dy):
            for x in range(w - dx):
                if mask[y, x] and mask[y + dy, x + dx]:
                    total += img[y, x] * img[y + dy, x + dx]
                    count += 1
    return total / count if count > 0 else 0.0
# Vectorized version preferred for speed — implement with np.roll or scipy.ndimage
```

Compute in Regime A and Regime B (ensemble mean).

##### 3c — Ornstein-Zernike proxy for χ_zz and ξ
The OZ fit gives: `1/S(q) = a + b·q²` where `S(q)` is the azimuthally averaged structure factor.

```python
import numpy as np
from numpy.fft import fft2, fftshift

def structure_factor(img):
    """Compute 2D structure factor S(q) = |FFT(img)|^2 / N."""
    N = img.size
    ft = fftshift(fft2(img))
    return np.abs(ft) ** 2 / N

def azimuthal_average(sq_2d, n_bins=15):
    """
    Azimuthally average S(q) into radial bins.
    Returns (q_bins, sq_avg) arrays.
    """
    h, w = sq_2d.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    q_bins, sq_avg = [], []
    for r in range(1, min(cy, cx) + 1):
        ring = sq_2d[R == r]
        if len(ring) > 0:
            q_bins.append(r)
            sq_avg.append(ring.mean())
    return np.array(q_bins, dtype=float), np.array(sq_avg)

def oz_fit(img, q_max_fraction=0.4):
    """
    Fit Ornstein-Zernike: 1/S(q) = a + b*q^2
    Returns: chi_proxy = 1/a (susceptibility proxy),
             xi = sqrt(b/a) (correlation length),
             r2 (R^2 of the fit — report this to flag validity).
    Returns (nan, nan, nan) if fit fails.
    """
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score

    sq_2d = structure_factor(img)
    q, sq = azimuthal_average(sq_2d)

    # Restrict to low-q regime
    q_max = q_max_fraction * q.max()
    mask_q = (q > 0) & (q <= q_max) & (sq > 0)
    q_fit, sq_fit = q[mask_q], sq[mask_q]
    if len(q_fit) < 4:
        return np.nan, np.nan, np.nan

    inv_sq = 1.0 / sq_fit

    def oz_model(q, a, b):
        return a + b * q ** 2

    try:
        popt, _ = curve_fit(oz_model, q_fit, inv_sq, p0=[1.0, 1.0], maxfev=5000)
        a, b = popt
        if a <= 0 or b <= 0:
            return np.nan, np.nan, np.nan
        chi_proxy = 1.0 / a
        xi = np.sqrt(b / a)
        r2 = r2_score(inv_sq, oz_model(q_fit, a, b))
        return chi_proxy, xi, r2
    except Exception:
        return np.nan, np.nan, np.nan
```

**IMPORTANT NOTE about OZ fit validity**: The OZ fit is physically meaningful only in disordered or paramagnetic regimes (phase "Others") where long-range fluctuations dominate. In ordered phases (Ferromagnetic, Helical), the fit R² will be low — **always report R² alongside χ and ξ**, and flag NaN/low-R² results in figures with a different marker or hatching.

**Regime A**: compute `chi_proxy`, `xi`, `oz_r2` per best_sample image.  
**Regime B (ensemble)**: compute `chi_ens` = `(1/T) * Var(m)` where m is ensemble magnetization:
```python
def chi_ensemble(ensemble_imgs, mask=MASK, temperature=1.0):
    """
    Fluctuation-dissipation estimate of susceptibility from K-sample ensemble.
    chi = N/T * Var(m), where m_k = mean sz over mask for sample k.
    """
    ms = np.array([img[mask].mean() for img in ensemble_imgs])
    N = mask.sum()
    return N * ms.var() / temperature
```

**Three chi comparisons** (compute for all 3 models):
1. `chi_orig_proxy` vs `chi_gen_proxy` (Regime A proxy: how well does best single image reproduce susceptibility?)
2. `chi_gen_proxy` vs `chi_gen_ens` (internal consistency: does single-image proxy agree with ensemble FDT estimate?)
3. `chi_orig_proxy` vs `chi_gen_ens` (direct: does ensemble generation reproduce original susceptibility?)

##### 3d — Summary: compute all metrics and save
Save per-sample CSV with columns:
```
idx, structure, theta_T, theta_J2, theta_KDM, theta_Hex, theta_KanS, theta_Kan1, theta_J3, theta_J4,
[model]_mse, [model]_bce, [model]_ssim, [model]_cosine,
[model]_M_A, [model]_M_B, [model]_Cnn_A, [model]_Cnn_B,
[model]_chi_proxy_A, [model]_xi_A, [model]_oz_r2_A, [model]_chi_ens_B,
[model]_chi_orig_proxy, [model]_chi_gen_proxy, [model]_chi_gen_ens
```
for model in {ddpm, cvae_xcp, cvae_vit}.

#### Section 4 — Figures

All figures use this color scheme for models:
- DDPM: `"#2563EB"` (blue)
- CVAE-Xception: `"#16A34A"` (green)
- CVAE-ViT: `"#DC2626"` (red)

Structure colors (consistent across all figures):
- Ferromagnetic: `"#1f77b4"`
- Others: `"#ff7f0e"`
- Labyrinthine & Conical: `"#2ca02c"`
- Helical: `"#d62728"`

**Figure 1** — `comparison_image_metrics_overall.png/.svg`
Grouped bar chart: x-axis = metric (MSE, BCE, SSIM, Cosine), groups = 3 models. Mean ± std bars.

**Figure 2** — `comparison_image_metrics_per_structure.png/.svg`
4-panel figure (2×2), one panel per metric. In each panel: x-axis = 4 structures, groups = 3 models.

**Figure 3** — `comparison_physical_metrics_overall.png/.svg`
3-panel figure (M, C_nn, χ_ens), grouped bar per model, Regime B values shown, Regime A as lighter overlay or separate bar.

**Figure 4** — `comparison_physical_metrics_per_structure.png/.svg`
Same as Figure 3 but split by structure (4×3 subpanels or 4-panel with grouped structure bars).

**Figure 5** — `comparison_chi_three_comparisons.png/.svg`
3-panel figure (one per chi comparison ①②③): scatter plot original vs generated, colored by structure. Add diagonal line (ideal agreement). Show Pearson r² in legend. Separate panel for each model or overlay with markers.

**Figure 6** — `comparison_xi_per_structure.png/.svg`
Box plots of ξ per structure per model. Mark invalid fits (R² < 0.7) with different symbol/hatch.

**Figure 7** — `comparison_qualitative_grid.png/.svg`
Grid: 4 structures × 5 columns. Columns: Reference | DDPM | CVAE-Xception | CVAE-ViT | Difference (|ref - gen|, cmap="hot").
Select 1 representative example per structure (median SSIM sample).
cmap="jet", vmin=-1, vmax=1 for sz images. vmin=0, vmax=1 for difference maps.
Print SSIM and Cosine values below each generated image column.

---

### NOTEBOOK 5 — `notebooks/cycle/ciclo_completo_v2.ipynb`
**Purpose**: Full end-to-end cycle evaluation with DDPM on the internal test set. This is the **core Part 2 notebook**. It is a clean rewrite of `ciclo_completo_resultadosxclusters.ipynb` (the old notebook goes to `replaced/`).

**Requires**: DDPM weights (PyTorch, cuda:0) + Xception weights (TF, CPU).

#### Section 0 — Setup
Same TF+PyTorch coexistence pattern. Load dataset test split (same SEED=42 split, N=~25,450 test samples).

**Ensemble size**: `K = 32`

#### Section 1 — Generate images for test set
For each θ in test set:
- Generate K=32 images with DDPM
- Pick `best_sample` = highest SSIM vs reference
- Store ensemble for physical metrics

#### Section 2 — Inverse model: parameter recovery
For each best_sample:
- Run Xception (on CPU): predict θ̂ from generated image
- Compare θ̂ vs θ (ground truth)

Compute per parameter (8 total):
- R² score
- MAE
- RMSE

**Per-structure breakdown**: same 4 phases.

Save CSV: `cycle_predictions.csv` with columns:
```
idx, structure, theta_T, theta_J2, ..., theta_J4 (ground truth),
pred_T, pred_J2, ..., pred_J4 (from generated image)
```

#### Section 3 — Cosine similarity in cycle
For each test sample:
- Extract Xception 256-dim features from reference image: `z_ref`
- Extract from generated best_sample: `z_gen`
- Compute cosine similarity

Report: mean ± std overall and per structure.

#### Section 4 — Physical metrics in cycle
Compute for each test sample's generated ensemble:
- M (Regime A: best_sample; Regime B: ensemble mean)
- C_nn (Regime A and B)
- χ_proxy from best_sample (Regime A)
- χ_ens from ensemble (Regime B)
- ξ from best_sample OZ fit (with R² reported)
- All three chi comparisons ①②③

Save to `cycle_physical_metrics.csv`.

#### Section 5 — Figures (save to `OUTPUT_DIR/cycle/`)

**Figure 1** — `cycle_r2_mae_per_parameter.png/.svg`
Two side-by-side bar charts (R² and MAE), x-axis = 8 parameters, bars colored by parameter group.
Style matches the reference article (SpinesIA_Inverso_param.pdf, Figures 11–12).

**Figure 2** — `cycle_r2_per_structure.png/.svg`
Grouped bar chart: x-axis = 8 parameters, groups = 4 structures. 4 bars per parameter.
Show R² for each (structure, parameter) pair.

**Figure 3** — `cycle_scatter_predictions.png/.svg`
2×4 grid of scatter plots (one per parameter). x-axis = θ_true, y-axis = θ_pred.
Color by structure. Add diagonal line. Print R² and MAE in each panel.

**Figure 4** — `cycle_cosine_per_structure.png/.svg`
Bar chart: cosine similarity distribution (mean ± std) per structure. Compare with direct (non-cycle) cosine from Section 3.

**Figure 5** — `cycle_chi_comparisons.png/.svg`
Same three-comparison figure as in generative_comparison.ipynb but for cycle evaluation (DDPM only).
Scatter: chi_orig vs chi_gen, per comparison ①②③, per structure.

**Figure 6** — `cycle_physical_per_structure.png/.svg`
4-panel: M, C_nn, χ_ens, ξ — each shows distribution per structure (original vs cycle-generated) as overlapping histograms or violin plots.

---

## SHARED UTILITIES FILE — `notebooks/utils/metrics.py`

Create this utility module that all evaluation notebooks import:

```python
"""
metrics.py — Shared metric functions for generative evaluation.
All functions assume images in [-1, 1] range, shape (H, W).
"""
import numpy as np
from numpy.fft import fft2, fftshift
from skimage.metrics import structural_similarity as skssim
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


STRUCTURE_NAMES = ["Ferromagnetic", "Others", "Labyrinthine & Conical", "Helical"]
STRUCTURE_COLORS = {
    "Ferromagnetic": "#1f77b4",
    "Others": "#ff7f0e",
    "Labyrinthine & Conical": "#2ca02c",
    "Helical": "#d62728",
}
MODEL_COLORS = {
    "DDPM": "#2563EB",
    "CVAE-Xception": "#16A34A",
    "CVAE-ViT": "#DC2626",
}
PARAM_NAMES = ["T⁰", "J̃₂", "K̃DM", "H̃ex", "K̃anS", "K̃an1", "J̃₃", "J̃₄"]


def circular_mask(h=39, w=39):
    cy, cx = h // 2, w // 2
    r = min(cy, cx)
    Y, X = np.ogrid[:h, :w]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


MASK = circular_mask()


def masked_mse(a, b, mask=MASK):
    return ((a - b) ** 2)[mask].mean()


def masked_bce(a, b, mask=MASK, eps=1e-7):
    a_ = (a[mask] + 1) / 2
    b_ = (b[mask] + 1) / 2
    return -np.mean(a_ * np.log(b_ + eps) + (1 - a_) * np.log(1 - b_ + eps))


def masked_ssim(a, b):
    return skssim(a, b, data_range=2.0)


def cosine_similarity_pair(z1, z2):
    n1 = np.linalg.norm(z1) + 1e-8
    n2 = np.linalg.norm(z2) + 1e-8
    return float(np.dot(z1 / n1, z2 / n2))


def magnetization(img, mask=MASK):
    return img[mask].mean()


def cnn_correlation(img, mask=MASK):
    """Vectorized nearest-neighbor spin correlation."""
    shifts = [(0, 1), (1, 0)]
    total, count = 0.0, 0
    for dy, dx in shifts:
        a = img[:-dy or None, :-dx or None] if (dy > 0 or dx > 0) else img
        b = img[dy:, dx:]
        ma = mask[:-dy or None, :-dx or None] if (dy > 0 or dx > 0) else mask
        mb = mask[dy:, dx:]
        valid = ma & mb
        total += (a * b)[valid].sum()
        count += valid.sum()
    return total / count if count > 0 else 0.0


def structure_factor(img):
    N = img.size
    ft = fftshift(fft2(img))
    return np.abs(ft) ** 2 / N


def azimuthal_average(sq_2d, n_bins=None):
    h, w = sq_2d.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    q_bins, sq_avg = [], []
    for r in range(1, max_r + 1):
        ring = sq_2d[R == r]
        if len(ring) > 0:
            q_bins.append(float(r))
            sq_avg.append(ring.mean())
    return np.array(q_bins), np.array(sq_avg)


def oz_fit(img, q_max_fraction=0.4):
    sq_2d = structure_factor(img)
    q, sq = azimuthal_average(sq_2d)
    q_max = q_max_fraction * q.max()
    mask_q = (q > 0) & (q <= q_max) & (sq > 0)
    q_fit, sq_fit = q[mask_q], sq[mask_q]
    if len(q_fit) < 4:
        return np.nan, np.nan, np.nan
    inv_sq = 1.0 / sq_fit
    try:
        popt, _ = curve_fit(lambda q, a, b: a + b * q ** 2, q_fit, inv_sq,
                             p0=[1.0, 1.0], maxfev=5000)
        a, b = popt
        if a <= 0 or b <= 0:
            return np.nan, np.nan, np.nan
        r2 = r2_score(inv_sq, a + b * q_fit ** 2)
        return 1.0 / a, np.sqrt(b / a), r2
    except Exception:
        return np.nan, np.nan, np.nan


def chi_ensemble(ensemble_imgs, mask=MASK, temperature=1.0):
    ms = np.array([img[mask].mean() for img in ensemble_imgs])
    N = int(mask.sum())
    return N * float(ms.var()) / temperature


def save_figure(fig, path_no_ext, dpi=300):
    """Save figure in both PNG (300 dpi) and SVG formats."""
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.svg", bbox_inches="tight")
```

---

## FIGURE STYLE GUIDE (apply everywhere)

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# Global style
plt.rcParams.update({
    "figure.dpi": 150,                  # screen display
    "savefig.dpi": 300,                 # saved PNG
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})
```

Every figure must have:
- Descriptive title (what the figure shows, not a variable name)
- Axis labels with units or value range where applicable
- Legend when more than one series
- Error bars (std) wherever means are reported
- Caption-ready: figure is self-explanatory without reading the notebook

---

## DOCUMENTATION STANDARD

Every notebook must begin with a markdown cell:
```markdown
# [Notebook Title]

**Purpose**: One-sentence description of what this notebook does.  
**Inputs**: List of required files/models.  
**Outputs**: List of produced files/figures.  
**Execution environment**: Kaggle / Google Colab (run only one setup cell).  
**Dependencies**: tensorflow, torch, scikit-learn, scikit-image, matplotlib, seaborn, scipy, tqdm, numpy
```

Every major section must have a markdown header and a brief explanation of what the section computes and why.

Code cells must be clean with:
- No debugging `print` statements left in (use `tqdm` progress bars instead)
- Meaningful variable names
- Short inline comments only where logic is non-obvious (no paragraph comments)

---

## SUMMARY — FILES TO CREATE OR MODIFY

| Action | File |
|--------|------|
| **Move to `replaced/`** | `ciclo_external_dataset.ipynb`, `ciclo_external_fixes.ipynb`, `diagnostico-ciclo-externo.ipynb` |
| **Move to `replaced/`** | `ciclo_completo_resultadosxclusters.ipynb` (superseded by Notebook 5) |
| **Create** | `notebooks/generative/cvae_xception_retrain.ipynb` |
| **Create** | `notebooks/generative/cvae_vit_retrain.ipynb` |
| **Create** | `notebooks/evaluation/image_metrics_robustness.ipynb` |
| **Create** | `notebooks/evaluation/generative_comparison.ipynb` |
| **Create** | `notebooks/cycle/ciclo_completo_v2.ipynb` |
| **Create** | `notebooks/utils/metrics.py` |

**Total new notebooks: 5. Shared utility module: 1. Notebooks retired to `replaced/`: 4.**

---

## FINAL CHECKLIST (verify before declaring complete)

- [ ] `notebooks/replaced/` contains exactly 4 notebooks
- [ ] All 5 new notebooks open without import errors in a clean kernel
- [ ] Each notebook has both Kaggle and Colab environment cells at the top
- [ ] `notebooks/utils/metrics.py` imports cleanly and all functions pass a quick smoke test
- [ ] All path variables (DATASET_PATH, weights) are defined in environment cells, not hardcoded elsewhere
- [ ] All figures saved as both `.png` and `.svg`
- [ ] CVAE training notebooks contain warning cell about missing weights
- [ ] OZ fit R² is always reported alongside χ and ξ
- [ ] Circular disk mask applied to all pixel-level metric computations
- [ ] K=32 defined as a named constant at the top of evaluation notebooks
- [ ] All documentation, comments, figure titles, axis labels, and variable names are in English
