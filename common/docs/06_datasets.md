# Datasets

Two Kaggle datasets are used in this project. Both contain 39×39 grayscale images of the out-of-plane magnetisation field $s_z$ and the corresponding 8 Hamiltonian parameter vectors.

---

## Dataset 1 — Internal Dataset (v2)

**Kaggle slug:** `carloscanamejoy/dataset-spines-united-v2`  
**File:** `dataset_unificado_v2.npz`  
**Used in:** Xception training, DDPM training, complete cycle (internal)

| Property | Value |
|---|---|
| Total samples | 169,671 |
| Image shape | 39 × 39 × 1 (grayscale) |
| Parameter vector | 8 scalars |
| File format | NumPy `.npz` |

### NPZ Keys

| Key | Shape | Description |
|---|---|---|
| `img` | (169671, 39, 39) | $s_z$ magnetisation field, values in $[0, 1]$ |
| `params` | (169671, 8) | Hamiltonian parameters in physical units |
| `labels` | (169671,) | Integer cluster label per sample |
| `label_keys` | — | Cluster index → integer mapping |
| `label_names` | — | Cluster names (Ferromagnetic, Others, Labyrinthine & Conical, Helical) |
| `column_names` | (8,) | Parameter names: `[T, Jex2, Jex3, Jex4, Kan1, KanS, Hex, KDM]` |

### Download (Colab)

```python
import os, shutil, zipfile
os.makedirs("/root/.kaggle", exist_ok=True)
shutil.move("kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)

os.system("kaggle datasets download -d carloscanamejoy/dataset-spines-united-v2")

with zipfile.ZipFile("dataset-spines-united-v2.zip", 'r') as z:
    z.extractall("dataset")

import numpy as np
data = np.load("dataset/dataset_unificado_v2.npz")
imgs   = data["img"]     # (169671, 39, 39)
params = data["params"]  # (169671, 8)
labels = data["labels"]  # (169671,)
```

---

## Dataset 2 — External Dataset (new-hez)

**Kaggle slug:** `carloscanamejoy/dataset-new-hez`  
**File:** `dataset-hez-profdario.npz`  
**Used in:** Complete cycle (external), generalisation test

| Property | Value |
|---|---|
| Image shape | 39 × 39 × 1 (grayscale) |
| Parameter vector | 8 scalars |
| Image content | $\langle s_z \rangle$ averaged over **all 5 layers** of the nanodot |

> ⚠️ **Critical difference from Dataset 1:** The internal dataset uses the $s_z$ component of the **central layer only** (`z = ⌊L̃/2⌋`). This dataset computes $\langle s_z \rangle$ as an **average over all 5 layers**. This creates a fundamental domain shift at the image level: averaged images are smoother, have lower contrast, and different spatial frequency content — even for the same Hamiltonian parameters. This is the primary suspected cause of poor image metrics in the external cycle.

### Download (Colab)

```python
os.system("kaggle datasets download -d carloscanamejoy/dataset-new-hez -p /content/kaggle_data/dataset-new-hez --unzip")
```

---

## Train/Validation/Test Split

Both models use a **70% / 15% / 15%** split with a **fixed random seed** (`SEED=42`) to ensure strict reproducibility and prevent information leakage:

```python
from sklearn.model_selection import train_test_split

# Step 1 — hold out test set (15% of total)
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Step 2 — hold out validation set (15% of total ≈ 17.65% of remaining 85%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool, y_train_pool, test_size=0.1765, random_state=42
)
```

All partitions are executed with the same seed. This strategy ensures that topological samples remain strictly independent across subsets while preserving the underlying thermodynamic distribution.

---

## Preprocessing

### Xception (TensorFlow / Keras)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train)   # fit ONLY on training set
y_val_scaled   = scaler.transform(y_val)
y_test_scaled  = scaler.transform(y_test)

# Image preprocessing (per batch, inside tf.data pipeline)
def preprocess_batch(x):
    x = tf.image.resize(x, (224, 224))       # 39x39 → 224x224
    x = tf.image.grayscale_to_rgb(x)         # (H,W,1) → (H,W,3)
    return x
```

The **grayscale→RGB broadcast** is necessary to leverage ImageNet-pretrained weights (which expect 3-channel input). No further normalisation is applied to images since $s_z$ is already in $[0, 1]$.

### DDPM (PyTorch)

```python
# Images: pad 39×39 → 40×40 with reflect padding, then scale to [-1, 1]
# Reflect padding preserves boundary texture statistics
imgs_padded = F.pad(imgs_tensor, (0, 1, 0, 1), mode='reflect')
imgs_scaled = imgs_padded * 2 - 1   # [0,1] → [-1,1]

# Parameters: same MinMaxScaler fitted on DDPM training split
scaler_ddpm = MinMaxScaler()
p_train_scaled = scaler_ddpm.fit_transform(p_train)
```

**Important:** The Xception scaler and DDPM scaler are **different objects** fitted on the same split indices but potentially different data versions. They must not be interchanged in the cycle pipeline.

### Cycle Preprocessing

```python
# DDPM → Xception: generated image (40×40, [-1,1]) → Xception input (224×224, [0,1], RGB)
img_gen_40 = ...                             # DDPM output
img_crop_39 = img_gen_40[:39, :39]          # centre crop 40→39
img_01      = (img_crop_39 + 1) / 2         # [-1,1] → [0,1]
img_224     = cv2.resize(img_01, (224,224))  # 39→224
img_rgb_224 = np.stack([img_224]*3, axis=-1) # grayscale → RGB
```

---

## Parameter Names and Physical Units

| Index | Name | Physical unit | Role |
|---|---|---|---|
| 0 | `T` | K | Annealing temperature |
| 1 | `Jex2` | meV | 2nd-shell exchange constant |
| 2 | `Jex3` | meV | 3rd-shell exchange constant |
| 3 | `Jex4` | meV | 4th-shell exchange constant |
| 4 | `Kan1` | meV/atom | Bulk magnetocrystalline anisotropy |
| 5 | `KanS` | meV/atom | Surface magnetocrystalline anisotropy |
| 6 | `Hex` | meV/atom | External Zeeman field |
| 7 | `KDM` | meV | DMI strength |

$\tilde{J}_1 = 1.0$ meV is the fixed energy reference and is **not** included in the parameter vector.
