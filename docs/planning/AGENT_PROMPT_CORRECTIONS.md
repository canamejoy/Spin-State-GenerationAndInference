# Correction Prompt — End-to-End Generative Framework

Apply all corrections in order on top of the work already done from `AGENT_PROMPT.md`.
Edit existing notebooks — do not regenerate from scratch.

---

## CORRECTION 1 — Replace `metrics.py` with Kaggle Dataset import

### Problem
`notebooks/utils/metrics.py` exists locally but Kaggle and Google Colab cannot access local repository files.
The file has already been uploaded to a public Kaggle dataset at:
```
https://www.kaggle.com/datasets/carloscanamejoy/physicalmetrics
```
The file in that dataset is named `metrics.py`.

### What to do

**Step 1 — Update `metrics.py` itself** with all changes specified in Correction 2 below before proceeding.

**Step 2 — In every notebook** (`cvae_xception_retrain.ipynb`, `cvae_vit_retrain.ipynb`, `image_metrics_robustness.ipynb`, `generative_comparison.ipynb`, `ciclo_completo_v2.ipynb`), replace any `from utils.metrics import *` or `import sys; sys.path.insert(...)` pattern with the following environment-aware import block. Insert it as the **first code cell after the environment setup cells**, under a markdown header `## Load Shared Metrics`.

```python
# ── Load shared metrics module from Kaggle dataset ───────────────────────────
# On Kaggle: metrics.py is pre-mounted as part of the physicalmetrics dataset.
# On Colab:  metrics.py is downloaded via the Kaggle API along with the other datasets.

import importlib.util, sys, os

_METRICS_KAGGLE = "/kaggle/input/datasets/carloscanamejoy/physicalmetrics/metrics.py"
_METRICS_COLAB  = "/content/weights/metrics.py"
_metrics_path   = _METRICS_KAGGLE if os.path.exists(_METRICS_KAGGLE) else _METRICS_COLAB

spec    = importlib.util.spec_from_file_location("metrics", _metrics_path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
sys.modules["metrics"] = metrics

from metrics import (
    STRUCTURE_MAP, STRUCTURE_NAMES, STRUCTURE_COLORS, MODEL_COLORS, PARAM_NAMES,
    circular_mask, MASK,
    masked_mse, masked_bce, masked_ssim,
    cosine_similarity_pair, cosine_similarity_batch,
    magnetization, abs_magnetization, cnn_correlation,
    structure_factor, azimuthal_average, oz_fit, chi_ensemble,
    shift_image, reflect_image,
    normalize_metrics,
    save_figure, apply_figure_style,
    center_crop, get_structure_label,
)
```

**Step 3 — Update the Colab environment setup cell** in every notebook to also download `physicalmetrics`:
```python
os.system("kaggle datasets download -d carloscanamejoy/physicalmetrics -p /content/weights --unzip")
```
Add this line immediately after the other `kaggle datasets download` commands in the Colab cell.

---

## CORRECTION 2 — Update structure taxonomy (6 phases, not 4)

### Problem
The original prompt assumed 4 magnetic phases. The actual dataset has cluster labels that map to **6 distinct structures**.

### Cluster-to-structure mapping
```python
STRUCTURE_MAP = {
    4:  "Helical",
    5:  "Helical",
    13: "Helical",
    6:  "Labyrinthine & Conical",
    14: "Labyrinthine & Conical",
    8:  "Bimeron",
    10: "Ferromagnetic",
    11: "Ferromagnetic",
    12: "Ferromagnetic",
    15: "Skyrmions",
    16: "Skyrmions",
    17: "Field-Saturated",
}

STRUCTURE_NAMES = [
    "Ferromagnetic",
    "Helical",
    "Labyrinthine & Conical",
    "Bimeron",
    "Skyrmions",
    "Field-Saturated",
]
```

### What to do

**Step 1 — Update `notebooks/utils/metrics.py`** with all of the following changes:

- Replace `STRUCTURE_NAMES` (currently 4 entries) with the 6-entry list above.
- Replace `STRUCTURE_COLORS` with:
```python
STRUCTURE_COLORS = {
    "Ferromagnetic":          "#1f77b4",   # blue
    "Helical":                "#d62728",   # red
    "Labyrinthine & Conical": "#2ca02c",   # green
    "Bimeron":                "#9467bd",   # purple
    "Skyrmions":              "#8c564b",   # brown
    "Field-Saturated":        "#e377c2",   # pink
}
```
- Add `STRUCTURE_MAP` dict (the mapping above) to the module.
- Add helper function:
```python
def get_structure_label(cluster_id):
    """Map a numeric cluster ID to its structure name string."""
    return STRUCTURE_MAP.get(int(cluster_id), f"Unknown({cluster_id})")
```
- Add crop function:
```python
def center_crop(img, size=39):
    """
    Crop a (40, 40) DDPM output to (size, size) using top-left crop.
    No interpolation — preserves original pixel values.
    img: numpy array of shape (H, W) or (H, W, C), H >= size, W >= size.
    """
    return img[:size, :size]
```
- Add normalization function:
```python
def normalize_metrics(scores_dict, reference_key="E0", worst_key="E2"):
    """
    Normalize raw metric scores for interpretability.

    Similarity metrics (ssim, cosine): divided by mean of reference condition (E0).
        Normalized E0 = 1.0. Degraded conditions < 1.0.

    Error metrics (mse, bce): divided by mean of worst condition (E2).
        Normalized E2 = 1.0. Reference E0 ≈ 0.0.

    Args:
        scores_dict: dict mapping condition keys (e.g. "E0","E1","E2","E3") to
                     sub-dicts {"mse": array, "bce": array, "ssim": array, "cosine": array}
        reference_key: condition key used as denominator for similarity metrics
        worst_key: condition key used as denominator for error metrics

    Returns:
        norm_dict: same structure as scores_dict with normalized values
        denominators: dict of the actual denominator values used
    """
    sim_metrics   = ["ssim", "cosine"]
    error_metrics = ["mse", "bce"]

    denom_sim   = {m: float(np.mean(scores_dict[reference_key][m])) for m in sim_metrics}
    denom_error = {m: float(np.mean(scores_dict[worst_key][m]))     for m in error_metrics}
    denominators = {**denom_sim, **denom_error}

    norm_dict = {}
    for cond, metric_scores in scores_dict.items():
        norm_dict[cond] = {}
        for m, vals in metric_scores.items():
            vals = np.asarray(vals)
            denom = denom_sim[m] if m in sim_metrics else denom_error[m]
            norm_dict[cond][m] = vals / denom if denom > 1e-9 else vals

    return norm_dict, denominators
```

**Step 2 — In every notebook**, wherever samples are grouped by structure:
- Replace any hardcoded list of 4 structure names with `STRUCTURE_NAMES` (imported from metrics).
- Replace any direct use of cluster label as a display name with `get_structure_label(cluster_id)`.
- Standard grouping pattern to use throughout all notebooks:
```python
structure_labels = np.array([get_structure_label(c) for c in clusters_test])
```

**Step 3 — Update all figure generation code** that loops over structures to loop over `STRUCTURE_NAMES` (6 items) and use `STRUCTURE_COLORS[name]` for consistent coloring. Figures that previously used 4 bars or 4 panels must now use 6.

---

## CORRECTION 3 — 40→39 crop after DDPM generation

### Problem
The DDPM generates images of shape `(40, 40)` (it internally pads 39→40 with reflect padding before the U-Net). All downstream metrics, Xception inference, and comparisons with dataset images require shape `(39, 39)`. The correct crop is a **top-left crop** (no interpolation):
```python
img_39 = img_40[:39, :39]
```
Confirmed by existing production code: `chunk = chunk[:, :39, :39]  # crop 40->39`.

### What to do

In every notebook that calls the DDPM sampler, immediately after the generation loop apply the crop before any metric computation or Xception inference:

```python
# DDPM generates (B, 40, 40) — crop to (B, 39, 39) before any downstream use
generated = generated[:, :39, :39]   # top-left crop, no interpolation
```

For per-image loops use `center_crop(img_40, size=39)` (imported from metrics).

This applies to:
- `notebooks/evaluation/generative_comparison.ipynb` — Section 1 (image generation loop)
- `notebooks/cycle/ciclo_completo_v2.ipynb` — Section 1 (image generation loop)
- `notebooks/generative/cvae_xception_retrain.ipynb` — decoder output is 40×40 via transposed conv; crop before loss and metric computation
- `notebooks/generative/cvae_vit_retrain.ipynb` — same

Verify the circular mask in `metrics.py` is `circular_mask(h=39, w=39)` — confirm no accidental change to 40.

---

## CORRECTION 4 — Update all Kaggle paths to exact values

### Problem
The original prompt used placeholder or slightly wrong Kaggle paths.

### What to do

Use these **exact paths** everywhere — replace any other form.

**Kaggle paths (pre-mounted, read-only)**:
```python
DATASET_PATH  = "/kaggle/input/datasets/carloscanamejoy/dataset-spines-united-v2/dataset_unificado_v2.npz"
XCEPTION_PATH = "/kaggle/input/datasets/carloscanamejoy/weights-xception-model/modelo_xception_fulldatabaseV3100.h5"
DDPM_PATH     = "/kaggle/input/datasets/carloscanamejoy/weights-models/ddpm_spines_final_39crop.pt"
CVAEXPN_PATH  = "/kaggle/input/datasets/carloscanamejoy/weights-cvae-models/cvae_xception.h5"
CVAEVIT_PATH  = "/kaggle/input/datasets/carloscanamejoy/weights-cvae-models/cvae_vit.h5"
METRICS_PATH  = "/kaggle/input/datasets/carloscanamejoy/physicalmetrics/metrics.py"
```

**Colab download commands** (inside the Colab setup cell):
```python
os.system("kaggle datasets download -d carloscanamejoy/dataset-spines-united-v2   -p /content/data    --unzip")
os.system("kaggle datasets download -d carloscanamejoy/weights-xception-model      -p /content/weights --unzip")
os.system("kaggle datasets download -d carloscanamejoy/weights-models              -p /content/weights --unzip")
os.system("kaggle datasets download -d carloscanamejoy/weights-cvae-models         -p /content/weights --unzip")
os.system("kaggle datasets download -d carloscanamejoy/physicalmetrics             -p /content/weights --unzip")
```

**Shared configuration cell** — place this after both setup cells; it auto-detects the environment and defines all paths. The rest of each notebook uses only these variables, never hardcoded paths:
```python
import os
_ON_KAGGLE = os.path.exists("/kaggle/input")

DATASET_PATH  = ("/kaggle/input/datasets/carloscanamejoy/dataset-spines-united-v2/dataset_unificado_v2.npz"
                 if _ON_KAGGLE else "/content/data/dataset_unificado_v2.npz")
XCEPTION_PATH = ("/kaggle/input/datasets/carloscanamejoy/weights-xception-model/modelo_xception_fulldatabaseV3100.h5"
                 if _ON_KAGGLE else "/content/weights/modelo_xception_fulldatabaseV3100.h5")
DDPM_PATH     = ("/kaggle/input/datasets/carloscanamejoy/weights-models/ddpm_spines_final_39crop.pt"
                 if _ON_KAGGLE else "/content/weights/ddpm_spines_final_39crop.pt")
CVAEXPN_PATH  = ("/kaggle/input/datasets/carloscanamejoy/weights-cvae-models/cvae_xception.h5"
                 if _ON_KAGGLE else "/content/weights/cvae_xception.h5")
CVAEVIT_PATH  = ("/kaggle/input/datasets/carloscanamejoy/weights-cvae-models/cvae_vit.h5"
                 if _ON_KAGGLE else "/content/weights/cvae_vit.h5")
METRICS_PATH  = ("/kaggle/input/datasets/carloscanamejoy/physicalmetrics/metrics.py"
                 if _ON_KAGGLE else "/content/weights/metrics.py")
OUTPUT_DIR    = "/kaggle/working/outputs" if _ON_KAGGLE else "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

Replace ALL manually separated Kaggle/Colab path blocks in every notebook with this single shared block. Keep the Kaggle and Colab **download/install cells** separate (user runs only one), but merge path definitions here.

---

## CORRECTION 5 — CVAE retraining notebooks: checkpoint save path

### Problem
CVAE retraining notebooks save weights locally. On Kaggle `/kaggle/working/` is writable; on Colab `/content/` is writable. Save paths must reflect this.

### What to do

In `cvae_xception_retrain.ipynb` and `cvae_vit_retrain.ipynb`, use:
```python
CKPT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
BEST_CKPT = os.path.join(CKPT_DIR, "cvae_xception_best.h5")   # or cvae_vit_best.h5
```

Add a markdown cell at the **end** of each retraining notebook:
```markdown
## After Training — Upload Weights to Kaggle

After training completes, the best checkpoint is saved at:
- Kaggle: `/kaggle/working/outputs/checkpoints/cvae_xception_best.h5`
- Colab:  `/content/outputs/checkpoints/cvae_xception_best.h5`

Upload this file to the Kaggle dataset `carloscanamejoy/weights-cvae-models` before
running `generative_comparison.ipynb` or `ciclo_completo_v2.ipynb`.
```

---

## CORRECTION 6 — Always use the full test split, never a subsample

### Problem
The original prompt specified subsampled evaluation sets (N_eval = 1000, N_PER_STRUCTURE = 166).
All evaluations must use the **complete test split** without further subsampling.

### Rule
Dataset: 169,671 samples. Split 70/15/15 SEED=42 → test split ≈ **25,451 samples**. Use all of them.

### What to do

In every evaluation notebook (`image_metrics_robustness.ipynb`, `generative_comparison.ipynb`, `ciclo_completo_v2.ipynb`), load the full test split as follows:

```python
from sklearn.model_selection import train_test_split

SEED = 42

data      = np.load(DATASET_PATH, mmap_mode="r")
imgs      = data["img"]       # (N, 39, 39, 1)
params    = data["params"]    # (N, 8)
# Inspect available keys with list(data.keys()) if unsure of the cluster key name.
# Common candidates: "labels", "clusters", "cluster", "structure"
clusters  = data["labels"]    # adjust key name as needed
all_idx   = np.arange(len(imgs))

idx_train, idx_temp = train_test_split(all_idx, test_size=0.30, random_state=SEED,
                                        stratify=clusters)
idx_val,   idx_test = train_test_split(idx_temp, test_size=0.50, random_state=SEED,
                                        stratify=clusters[idx_temp])

# COMPLETE test split — no further subsampling
imgs_test     = imgs[idx_test]       # (N_test, 39, 39, 1)
params_test   = params[idx_test]     # (N_test, 8)
clusters_test = clusters[idx_test]   # (N_test,)

print(f"Test split size: {len(idx_test)} samples")
```

Remove all lines that further subsample `idx_test`:
- `rng.choice(idx_test, size=1000, ...)`
- `idx_test[:1000]`
- `N_eval = 1000` used to slice the test set
- Any `N_PER_STRUCTURE`-based stratified subsampling of the test set

Use `tqdm` for progress bars on long loops but still iterate over all samples.

---

## CORRECTION 7 — Normalized image metrics in the robustness experiment

### Problem
`image_metrics_robustness.ipynb` reports raw MSE, BCE, SSIM, and cosine similarity.
These must be normalized relative to known reference conditions for interpretability.

### Normalization convention

| Metric | Denominator | E0 result | Direction |
|--------|-------------|-----------|-----------|
| SSIM   | `mean(SSIM_E0)` | 1.0 | 1.0 = best |
| Cosine | `mean(Cosine_E0)` | 1.0 | 1.0 = best |
| MSE    | `mean(MSE_E2)` | ≈ 0.0 | 0.0 = best |
| BCE    | `mean(BCE_E2)` | ≈ 0.0 | 0.0 = best |

`normalize_metrics()` (added to `metrics.py` in Correction 2) implements this convention.

### What to do

**Step 1 — In `image_metrics_robustness.ipynb`, after computing raw scores**, add:

```python
raw_scores = {
    "E0": {"mse": mse_E0, "bce": bce_E0, "ssim": ssim_E0, "cosine": cosine_E0},
    "E1": {"mse": mse_E1, "bce": bce_E1, "ssim": ssim_E1, "cosine": cosine_E1},
    "E2": {"mse": mse_E2, "bce": bce_E2, "ssim": ssim_E2, "cosine": cosine_E2},
    "E3": {"mse": mse_E3, "bce": bce_E3, "ssim": ssim_E3, "cosine": cosine_E3},
}

norm_scores, denominators = normalize_metrics(raw_scores, reference_key="E0", worst_key="E2")

print("Normalization denominators:")
for m, v in denominators.items():
    print(f"  {m}: {v:.6f}")

# Sanity check: E0 SSIM/Cosine should be 1.0; E0 MSE/BCE should be ~0.0
print("\nNormalized condition means:")
for cond in ["E0", "E1", "E2", "E3"]:
    means = {m: float(np.mean(norm_scores[cond][m])) for m in ["mse","bce","ssim","cosine"]}
    print(f"  {cond}: {means}")
```

**Step 2 — Use `norm_scores` (not `raw_scores`) for ALL figures.**

Update axis labels:
- `"Normalized SSIM (relative to E0)"`, range [0, 1.1]
- `"Normalized Cosine Similarity (relative to E0)"`, range [0, 1.1]
- `"Normalized MSE (relative to E2)"`, range [0, 1.1]
- `"Normalized BCE (relative to E2)"`, range [0, 1.1]

Add annotation to each figure:
```python
fig.text(0.01, 0.01,
         "SSIM & Cosine normalized to E0 (identical images) = 1.0  |  "
         "MSE & BCE normalized to E2 (5-px shift) = 1.0",
         fontsize=7, color="gray", ha="left", va="bottom")
```

**Step 3 — Save both raw and normalized scores to CSV:**
```python
import pandas as pd

rows = []
for cond in ["E0", "E1", "E2", "E3"]:
    for i in range(len(idx_test)):
        rows.append({
            "condition":    cond,
            "structure":    structure_labels[i],
            "mse_raw":      raw_scores[cond]["mse"][i],
            "bce_raw":      raw_scores[cond]["bce"][i],
            "ssim_raw":     raw_scores[cond]["ssim"][i],
            "cosine_raw":   raw_scores[cond]["cosine"][i],
            "mse_norm":     norm_scores[cond]["mse"][i],
            "bce_norm":     norm_scores[cond]["bce"][i],
            "ssim_norm":    norm_scores[cond]["ssim"][i],
            "cosine_norm":  norm_scores[cond]["cosine"][i],
        })

pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/robustness/robustness_scores.csv", index=False)
```

---

## SUMMARY OF FILES TO EDIT

| File | Changes |
|------|---------|
| `notebooks/utils/metrics.py` | Update `STRUCTURE_NAMES` (6 phases), `STRUCTURE_COLORS` (6 colors); add `STRUCTURE_MAP`, `get_structure_label()`, `center_crop()`, `normalize_metrics()` |
| `notebooks/generative/cvae_xception_retrain.ipynb` | `importlib` metrics import; auto-detect paths; crop 40→39 after decoder; 6-structure figures; checkpoint save to `OUTPUT_DIR/checkpoints/`; upload instructions cell |
| `notebooks/generative/cvae_vit_retrain.ipynb` | Same as above |
| `notebooks/evaluation/image_metrics_robustness.ipynb` | `importlib` metrics import; auto-detect paths; full test split; normalization step; `norm_scores` in all figures; raw+normalized CSV |
| `notebooks/evaluation/generative_comparison.ipynb` | `importlib` metrics import; auto-detect paths; full test split; crop 40→39 in generation loop; 6-structure figures |
| `notebooks/cycle/ciclo_completo_v2.ipynb` | `importlib` metrics import; auto-detect paths; full test split; crop 40→39 in generation loop; 6-structure figures |

---

## FINAL VERIFICATION CHECKLIST

- [ ] `metrics.py` exports: `STRUCTURE_MAP`, `STRUCTURE_NAMES` (6), `STRUCTURE_COLORS` (6), `get_structure_label()`, `center_crop()`, `normalize_metrics()`
- [ ] Every notebook imports metrics via `importlib.util.spec_from_file_location` — no `sys.path` hacks
- [ ] Every notebook has a single `_ON_KAGGLE` auto-detect path block; no duplicated path strings
- [ ] Colab setup cell in every notebook includes `kaggle datasets download -d carloscanamejoy/physicalmetrics`
- [ ] Every DDPM generation call is followed by `[:, :39, :39]` or `center_crop()` before metrics or Xception
- [ ] `circular_mask` in `metrics.py` is `h=39, w=39`
- [ ] All figures loop over `STRUCTURE_NAMES` (6 structures)
- [ ] CVAE retraining notebooks save to `OUTPUT_DIR/checkpoints/` and end with upload instructions
- [ ] All three evaluation notebooks print test split size at load time (should be ~25,451)
- [ ] No notebook uses `rng.choice`, `[:N]` slicing, or `N_PER_STRUCTURE` to subsample the test split
- [ ] `image_metrics_robustness.ipynb` prints normalization denominators and uses `norm_scores` in all figures
- [ ] Robustness CSV contains both `_raw` and `_norm` columns
