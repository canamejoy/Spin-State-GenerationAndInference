# Direct Problem: Hamiltonian Parameters → Magnetic Domain Images

> **Paper:** *Image-Driven Estimation of Magnetic Configurations in Nanodots*
> Méndez-Rondón, Canamejoy-Alegría et al., in preparation.

## Problem Formulation

The direct (generative) problem asks: given a vector of 8 Hamiltonian parameters $\boldsymbol{\theta}$, synthesise a physically faithful 2D $s_z$ magnetisation image $\mathbf{X}$.

This is an inherently **one-to-many stochastic mapping**: the same $\boldsymbol{\theta}$ can produce multiple distinct thermodynamic configurations (degenerate states, metastable phases). The goal is to learn a conditional generative model $g_\psi$ that samples from this distribution:

$$
\hat{\mathbf{X}} = g_\psi(\boldsymbol{\theta}) \sim p(\mathbf{X} \mid \boldsymbol{\theta})
$$

Two architectures are evaluated: a **Conditional Variational Autoencoder (CVAE)** and a **Denoising Diffusion Probabilistic Model (DDPM)**.

---

## General Conditional Generation Framework

Given dataset pairs $\{Y_n \in \mathbb{R}^P,\, X_n \in \mathbb{R}^{H \times W \times C}\}_{n=1}^N$, the objective is to learn $f_\theta : \mathbb{R}^P \to \mathbb{R}^{H \times W \times C}$ minimising:

$$
\theta^* = \arg\min_\theta \frac{1}{N}\sum_{n=1}^{N} \mathcal{L}(X_n,\, f_\theta(Y_n))
$$

---

## Architecture 1: Conditional Variational Autoencoder (CVAE)

The CVAE extends the standard VAE by conditioning both encoder and decoder on the physical parameter vector $Y_n$.

### Training Phase

1. **Encoder** maps $(X_n, Y_n)$ to a posterior distribution $q(z \mid X_n, Y_n)$ with parameters $(\mu_q, \log\sigma^2_q)$
2. **Prior network** maps $Y_n$ alone to a conditioned prior $p(z \mid Y_n)$ with parameters $(\mu_p, \log\sigma^2_p)$
3. **Reparameterisation trick:** $z = \mu_q + \sigma_q \odot \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$
4. **Decoder** reconstructs $\hat{X}$ from $(z, Y_n)$

### Inference Phase

1. Sample $z \sim \mathcal{N}(\mu_p, \sigma^2_p)$ using the **prior network** (encoder not used)
2. Decode $(z, Y_n) \to \hat{X}$

### Loss Function

The CVAE is trained using the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}_{total} = \mathcal{L}_{recon} + \beta(\tilde{p}) \cdot \mathcal{L}_{reg}
$$

**Reconstruction term** (binary cross-entropy):

$$
\mathcal{L}_{BCE}(X, \hat{X}) = -\frac{1}{B}\sum_{b=1}^{B}\sum_{h,w,c}\left[x_{b,h,w,c}\log\hat{x}_{b,h,w,c} + (1 - x_{b,h,w,c})\log(1 - \hat{x}_{b,h,w,c})\right]
$$

**Reconstruction term** (structural similarity):

$$
\mathcal{L}_{SSIM}(X, \hat{X}) = \frac{1}{B}\sum_{b=1}^{B}\left(1 - \text{SSIM}(x_b, \hat{x}_b)\right)
$$

where:

$$
\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}
$$

**Regularisation term** (KL divergence between posterior and conditioned prior):

$$
\mathcal{L}_{KLD} = \frac{1}{B}\sum_{b=1}^{B}\frac{1}{2}\sum_{k=1}^{d_z}\left(\log\sigma^2_{p,b,k} - \log\sigma^2_{q,b,k} + \frac{\sigma^2_{q,b,k} + (\mu_{q,b,k} - \mu_{p,b,k})^2}{\sigma^2_{p,b,k}} - 1\right)
$$

**Dynamic $\beta$ scheduling** (avoids posterior collapse at early training):

$$
\beta(\tilde{p}) = \frac{1 - e^{-\delta \tilde{p}}}{1 + e^{-\delta \tilde{p}}}, \quad \tilde{p} = \frac{\text{epoch}}{\text{total\_epochs}}
$$

where $\delta > 0$ controls the growth rate.

---

## Architecture 2: Denoising Diffusion Probabilistic Model (DDPM)

The DDPM learns to reverse a gradual Gaussian noising process applied directly in the image space, conditioned on $Y_n$.

### Forward Process

A clean image $X_0$ is progressively corrupted over $T = 1000$ timesteps. Any noisy state $X_t$ can be sampled directly from $X_0$ without simulating each step:

$$
X_t = \sqrt{\bar{\alpha}_t}\,X_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

where $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$, and $\{\beta_t\}_{t=1}^T$ is the **cosine noise schedule** (chosen over linear after hyperparameter search with Optuna).

### Reverse Process (Denoising)

A conditional U-Net $\epsilon_\theta(X_t, t, Y_n)$ is trained to predict the noise $\epsilon$ added at timestep $t$. The physical parameters $Y_n$ are injected via **cross-attention layers**. Timestep $t$ is encoded via **sinusoidal embeddings**.

**Training objective:**

$$
\mathcal{L}_{diff} = \mathbb{E}_{t, X_0, \epsilon}\left[\left\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\,X_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\; t,\; Y_n\right)\right\|^2\right]
$$

**Inference:** starting from pure Gaussian noise $X_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, iteratively apply the learned denoising step $T$ times to recover $\hat{X}_0$.

### Key Implementation Details

| Aspect | Value |
|---|---|
| Image size input | 39×39 → padded to 40×40 (reflect padding) |
| Timesteps $T$ | 1000 |
| Noise schedule | Cosine |
| Backbone | Conditional U-Net |
| Conditioning mechanism | Cross-attention on $Y_n$ |
| Time encoding | Sinusoidal embedding |
| EMA | Exponential Moving Average (decay=0.999) on model weights |
| Optimiser | AdamW, lr ≈ 2.26×10⁻⁴ (from Optuna) |
| Loss | MSE on predicted noise |
| Evaluation metric | SSIM on generated vs real images |

**Why 40×40 (reflect padding) instead of 39×39?** The U-Net requires spatial dimensions divisible by powers of 2 for its downsampling/upsampling path. Reflect padding preserves boundary texture statistics better than zero-padding. The generated 40×40 images are **centre-cropped back to 39×39** before passing to the Xception inverse model.

---

## Complete Cycle Pipeline

Both models participate in the end-to-end evaluation:

```
θ (8 params)
  → [DDPM] (40×40 generation)
  → centre crop (39×39)
  → [Xception] (224×224 resized)
  → θ̂ (8 params estimated)
```

See [05_complete_cycle.md](05_complete_cycle.md) for the full cycle evaluation methodology.

---

## Notebooks

- DDPM training: [notebooks/generative/ddpm_train.md](../notebooks/generative/ddpm_train.md)
- Cycle evaluation (internal): [notebooks/archive/ciclo_completo.md](../notebooks/archive/ciclo_completo.md)
- Cycle evaluation (external): [notebooks/archive/ciclo_external.md](../notebooks/archive/ciclo_external.md)
