# Inverse Problem: Image → Hamiltonian Parameters

> **Paper:** *Regression-Based Explainable Deep Learning for Estimating Hamiltonian Parameters from Magnetic Nanodot Images*
> Méndez-Rondón et al., submitted to Elsevier.

## Problem Formulation

The inverse problem asks: given a 2D $s_z$ magnetisation image $\mathbf{X}_n$, recover the 8 Hamiltonian parameters $\boldsymbol{\theta}_n$ that generated it.

This is formulated as a **continuous multi-output regression** task. We learn a parameterised mapping:

$$
f_\Theta : \mathbb{R}^{\breve{H} \times \breve{W}} \to \mathbb{R}^{\breve{P}}
$$

The optimal weights $\hat{\Theta}$ minimise the MSE loss over the training distribution:

$$
\hat{\Theta} = \arg\min_\Theta \frac{1}{\breve{N}} \sum_{n=1}^{\breve{N}} \mathcal{L}\!\left(\boldsymbol{\theta}_n,\, f_\Theta(\mathbf{X}_n)\right)
$$

The inverse problem is **inherently ill-posed**: distinct parameter combinations can yield nearly indistinguishable spin configurations (degenerate states), and the $s_z$ projection discards information about in-plane components $s_x, s_y$.

---

## Architecture: Xception Regression Head

The Xception (Extreme Inception) backbone decouples spatial correlations from cross-channel correlations via **depthwise separable convolutions**:

$$
\text{Conv}_{sep}(\mathbf{X}) = \text{Conv}_{pointwise}\!\left(\text{Conv}_{depthwise}(\mathbf{X})\right)
$$

For input $\mathbf{X} \in \mathbb{R}^{\breve{H} \times \breve{W} \times \breve{C}}$:
- **Depthwise convolution** — applies a single spatial filter per channel independently (captures local spin-texture frequencies)
- **Pointwise convolution** (1×1) — projects across channels to capture cross-channel correlations (maps textures to the parameter space)

**Full regression pipeline:**

```
Input (39×39×1)
  → Resize to 224×224
  → Grayscale → RGB (broadcast)
  → Xception backbone (ImageNet pretrained, frozen/fine-tuned)
  → Global Average Pooling (spatial)
  → BatchNorm
  → Dropout(0.4)
  → Dense(256, ReLU, L2 λ=1e-4)
  → BatchNorm
  → Dropout(0.3)
  → Dense(8, linear)          ← predicted θ̂
```

**Optimiser:** Adam, lr = $10^{-4}$  
**Loss:** MSE  
**Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau (factor=0.3, patience=4)  
**Batch size:** 512  
**Max epochs:** 100

---

## Identifiability Hierarchy

Not all 8 parameters are equally recoverable from the $s_z$ projection:

| Identifiability | Parameters | Reason |
|---|---|---|
| **High** ($R^2 \geq 0.85$) | $T^{(0)}$, $\tilde{J}_2$, $\tilde{K}_{DM}$ | Leave strong, distinctive spatial signatures in $s_z$ |
| **Partial** ($0.50 \leq R^2 < 0.85$) | $\tilde{K}_{an1}$, $\tilde{K}_{anS}$, $\tilde{H}_{ex}$ | Couple primarily to $s_x, s_y$; recoverable only indirectly |
| **Not identifiable** ($R^2 \approx 0$) | $\tilde{J}_3$, $\tilde{J}_4$ | Energetically dominated by $\tilde{J}_1, \tilde{J}_2$; signatures masked |

The high MAE observed for $T^{(0)}$, $\tilde{J}_2$, $\tilde{K}_{DM}$ despite high $R^2$ reflects the **high-temperature paramagnetic regime**: visually identical disordered configurations correspond to widely different parameter values, making the inverse mapping fundamentally ill-posed in that subset.

---

## RAMs: Regression Activation Maps

Standard Grad-CAM targets a discrete class score $y^c$. For continuous regression, this fails because there is no natural "class" to differentiate. RAMs address this with a **scale-invariant, error-based activation objective**.

### Error Objective

For target parameter $\theta_p$ and prediction $\hat{\theta}_p$:

$$
\mathcal{O}_p = \exp\!\left(-\gamma_p\,(\theta_p - \hat{\theta}_p)^2\right)
$$

- $\gamma_p > 0$: precision factor controlling gradient selectivity (set to $\gamma_p = 10$ after sensitivity analysis)
- $\mathcal{O}_p \in (0, 1]$: smooth, bounded — equals 1 at perfect prediction, decays to 0 with increasing error
- Eliminates the mathematical discontinuities of absolute error formulations

### Spatial Importance Weights (CNN)

For feature map $\mathbf{A}^k \in \mathbb{R}^{H' \times W'}$ at the terminal convolutional layer:

$$
\alpha_k^{(p)} = \frac{1}{H'W'} \sum_{i=1}^{H'}\sum_{j=1}^{W'} \frac{\partial \mathcal{O}_p}{\partial \mathbf{A}^k_{ij}}
$$

### RAM Construction

$$
\mathcal{R}^{(p)}_{\text{RAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^{(p)}\,\mathbf{A}^k\right)
$$

ReLU retains only spatially positive contributions to accurate prediction. The map is then upsampled to $\breve{H} \times \breve{W}$ and normalised to $[0,1]$.

### Extension to ViT

The ViT processes images as $N_r = 196$ non-overlapping 16×16 patches. The spatial token sequence $\mathbf{Z}_s \in \mathbb{R}^{N_r \times D}$ is reshaped into a pseudo-feature map:

$$
\mathbf{A}_{\text{ViT}} = \text{Reshape}(\mathbf{Z}_s) \in \mathbb{R}^{\sqrt{N_r} \times \sqrt{N_r} \times D}
$$

The same gradient-based RAM formula then applies, providing **architecture-agnostic** attribution.

### Normalised RAMs for Layer-Wise Analysis

To compare across layers $l \in \{1, \dots, L\}$ and parameters $p$, all RAMs are globally normalised:

$$
\tilde{\mathcal{R}}^{(l,p)}_{\text{RAM}} = \frac{\mathcal{R}^{(l,p)}_{\text{RAM}}}{\max_{l', p'}\!\left(\max_{i,j}\left[\mathcal{R}^{(l',p')}_{\text{RAM}}\right]_{i,j}\right)}
$$

This reveals the **hierarchical decoupling** across network depth: early layers encode short-range spin-winding and domain wall chirality ($\tilde{J}$, $\tilde{D}$), while deeper layers aggregate global thermodynamic order ($T^{(0)}$, $\tilde{K}$).

---

## Latent Space Analysis

The ViT's final encoder block features are projected to 2D via **UMAP**, revealing four physically meaningful magnetic phase clusters identified by density-based clustering (HDBSCAN + KMeans):

| Cluster | $\langle T^{(0)} \rangle$ | $\langle \tilde{K}_{DM} \rangle$ | Description |
|---|---|---|---|
| Ferromagnetic | ~5.0 K | ~0.0 meV | Uniform $s_z$, minimal spatial modulation |
| Others | ~5.5 K | ~0.1 meV | Thermally disordered / transitional |
| Labyrinthine & Conical | ~5.0 K | ~0.7 meV | Stripe and labyrinthine domains |
| Helical | ~4.0 K | ~0.8 meV | Regular helical stripes, most ordered |

Per-phase $R^2$ confirms that accuracy rises substantially in ordered low-$T$ phases:
- Labyrinthine & Conical: $R^2 = 0.978$ ($T^{(0)}$), $R^2 = 0.970$ ($\tilde{K}_{DM}$)
- Helical: $R^2 = 0.968$ ($T^{(0)}$), $R^2 = 0.866$ ($\tilde{K}_{DM}$)

---

## Notebook

See [section2-inverse/cuadernillos/xception_inverse.md](../section2-inverse/cuadernillos/xception_inverse.md) for implementation details.
