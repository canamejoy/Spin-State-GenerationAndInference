# Evaluation Metrics

Three families of metrics are used to evaluate the complete cycle: **regression metrics** (parameter recovery quality), **image metrics** (pixel/spectral fidelity), and **physical metrics** (three observables of the spin configuration on the nanodot disk).

---

## 1. Regression Metrics (Parameter Recovery)

Applied to compare input parameters $\boldsymbol{\theta}$ vs re-estimated parameters $\hat{\boldsymbol{\theta}}$ after the full cycle.

### Coefficient of Determination ($R^2$)

$$
R^2 = 1 - \frac{\sum_{n=1}^{N}(\theta_{n,p} - \hat\theta_{n,p})^2}{\sum_{n=1}^{N}(\theta_{n,p} - \bar\theta_p)^2}
$$

- $\bar\theta_p$: mean of the ground-truth values for parameter $p$
- Range: $(-\infty, 1]$; $R^2 = 1$ is perfect, $R^2 = 0$ equals predicting the mean, $R^2 < 0$ is worse than the mean
- **Note:** $R^2$ is statistically undefined when the target has near-zero variance (e.g., $\tilde{K}_{DM} \approx 0$ in the ferromagnetic cluster). Reported as N/A in those cases.

### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{N}\sum_{n=1}^{N}|\theta_{n,p} - \hat\theta_{n,p}|
$$

- Reported in the **original physical units** of each parameter (K for $T^{(0)}$, meV for exchange/DMI, meV/atom for anisotropy/field)
- When computed on Min-Max normalised targets $[0,1]$, MAE is dimensionless and directly comparable across parameters

### Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{n=1}^{N}(\theta_{n,p} - \hat\theta_{n,p})^2}
$$

---

## 2. Image Metrics

Applied to compare original $s_z$ images vs DDPM-generated images (both cropped to 39×39). All metrics are restricted to the **circular disk mask** $\mathcal{M}$ (radius $r_d = 18.25$ MUC).

### MSE Variance (Var-MSE)

Measures spatial heterogeneity of per-pixel squared errors:

$$
\text{Var}_{\text{MSE}}(a, b) = \frac{1}{N}\sum_{i=1}^{N}\left[e_i^2 - \overline{e^2}\right]^2, \quad e_i = (a_i - b_i)^2
$$

- $e_i$: squared error at pixel $i$
- $\overline{e^2} = \frac{1}{N}\sum_i e_i$: mean squared error (MSE)
- High Var-MSE indicates spatially concentrated errors (e.g., on domain walls only)

### SSIM (Structural Similarity Index)

$$
\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}
$$

- Computed over local windows (Gaussian kernel)
- $\text{SSIM} \in [-1, 1]$; higher is better
- **Limitation for periodic textures:** phase-shifted configurations (same frequency, different phase) are energetically equivalent but penalised by SSIM. FFT metrics address this.

### Spectral MSE (FFT-MSE)

$$
\text{FFT-MSE}(a, b) = \frac{1}{N}\sum_{i=1}^{N}\left(\tilde{S}_a^{(i)} - \tilde{S}_b^{(i)}\right)^2, \quad \tilde{S}_x = \frac{|\mathcal{F}\{x\}|}{\max|\mathcal{F}\{x\}|}
$$

- $\mathcal{F}\{x\}$: 2D DFT of image $x$, zero-frequency shifted to centre
- $\tilde{S}_x \in [0,1]^{H \times W}$: normalised amplitude spectrum
- Captures differences in **spatial frequency content** regardless of phase shifts
- Low FFT-MSE: the generated image has the same periodicity and domain-wall spacing as the original

### Spectral Pearson Correlation (FFT-Corr)

$$
\text{FFT-Corr}(a, b) = \frac{\sum_{i=1}^{N}(\tilde{S}_a^{(i)} - \bar{S}_a)(\tilde{S}_b^{(i)} - \bar{S}_b)}{\sqrt{\sum_{i=1}^{N}(\tilde{S}_a^{(i)} - \bar{S}_a)^2}\;\sqrt{\sum_{i=1}^{N}(\tilde{S}_b^{(i)} - \bar{S}_b)^2}}
$$

- $\text{FFT-Corr} \in [-1, 1]$; higher is better
- Measures the linear correlation between the amplitude spectra of the two images
- Robust to differences in absolute spectral amplitude; captures spectral shape similarity

---

## 3. Physical Metrics

The canonical set is **exactly three** observables. All are computed on the
$39\times39$ physical image, restricted to the circular nanodot disk

$$
\mathcal{M} = \left\{(y, x) \in \mathbb{Z}^2 : (y - c_y)^2 + (x - c_x)^2 \leq r_d^2\right\},
\qquad (c_y, c_x) = \left(\tfrac{H-1}{2}, \tfrac{W-1}{2}\right),
\quad r_d = 18.25\ \text{MUC},
$$

which gives $N_\mathcal{M} = |\mathcal{M}| = 1049$ pixels.

> **Crop precedes mask.** The DDPM works on a $40\times40$ canvas obtained by
> reflect-padding the physical image on its *right and bottom* edges. Metrics
> must be computed after `topleft_crop`, never on the canvas: the extra row and
> column are duplicated pixels, and a $40\times40$ disk is centred at 19.5,
> half a pixel off the physical centre. `metrics.py` raises `ValueError` if a
> $40\times40$ array reaches a masked helper.

> **Radius.** $r_d = 18.25$ magnetic unit cells is the nanodot radius used in
> the predecessor paper. On the $39\times39$ grid it discretises to the same
> 1049-pixel mask as 18.3. The inscribed
> circle of the pixel grid ($r = 19$) admits 80 additional background pixels
> that dilute every disk average; earlier revisions of `metrics.py` used it by
> mistake.

### Average Magnetisation $M$

$$M = \frac{1}{N_\mathcal{M}}\sum_{i \in \mathcal{M}} s_z(i)$$

- Range $[-1, 1]$. $M \approx \pm 1$ is ferromagnetic or field-saturated;
  $M \approx 0$ is chiral, helical or skyrmionic.
- Net spin polarisation of the dot — the order parameter conjugate to $\tilde{H}_{ex}$.

### Nearest-Neighbour Spin Correlation $C_{nn}$

$$C_{nn} = \frac{1}{|\mathcal{B}|}\sum_{\langle i,j \rangle \in \mathcal{B}} s_z(i)\,s_z(j)$$

- $\mathcal{B}$: the set of row- and column-adjacent pairs with **both** ends
  inside $\mathcal{M}$; each pair counted once.
- Range $[-1, 1]$. $C_{nn} \to 1$ is aligned, $\approx 0$ disordered,
  $< 0$ antiferromagnetic or short-period modulated.
- Captures only the $s_z^i s_z^j$ projection of the exchange; transverse
  components are not recoverable from a scalar $s_z$ image.

### Peak Wave Vector $q_{\text{peak}}$

The structure factor is computed on the **fluctuation field** — disk mean
removed, background zeroed — so that it describes the texture rather than the
disk aperture:

$$
\tilde{s}_z = \left(s_z - \bar{s}_z^{\,\mathcal{M}}\right)\cdot\mathbb{1}_\mathcal{M},
\qquad
S(\mathbf{q}) = \frac{\left|\mathcal{F}\{\tilde{s}_z\}\right|^2}{N}
$$

$S(\mathbf{q})$ is azimuthally averaged into integer radial bins, and

$$
q_{\text{peak}} = \frac{2\pi}{L}\,r^{*},
\qquad
r^{*} = \arg\max_{r > 0}\ \big\langle S(\mathbf{q})\big\rangle_{|\mathbf{q}| = r},
\qquad L = 39.
$$

- Units: rad per lattice site. A texture of wavelength $\ell$ sites peaks at
  $q = 2\pi/\ell$.
- The $r = 0$ bin is excluded, so a uniform (saturated) configuration has no
  spectral power and returns `nan` — this is meaningful, not a failure.
- Radial bins are integer, so $q_{\text{peak}}$ is quantised in steps of
  $2\pi/39 \approx 0.161$ rad/site; a wavelength of 6 sites ($q = 1.047$) is
  reported at the $r = 6$ bin, $q = 0.967$.

### What was removed, and why

| Removed | Reason |
|---|---|
| $\lvert M \rvert$ | Redundant with $M$ for comparison purposes; $\langle\lvert M\rvert\rangle \neq \lvert\langle M\rangle\rangle$ made the ensemble and single-configuration regimes non-comparable. |
| $\chi$ (susceptibility) | Requires either a $K$-sample ensemble or an Ornstein–Zernike proxy whose fit is valid only in disordered regimes. Not a uniform axis across the six magnetic phases. |
| $C_v$ (specific heat) | Same objection as $\chi$, compounded by the block-subsystem estimator's sensitivity to block size near the disk boundary. |
| $E$ (exchange energy density) | Monotonically tied to $C_{nn}$ under the $s_z$ projection, so it adds no independent information. |

The three retained observables are well defined on **every single
configuration in every phase**, which is what makes them usable as a comparison
axis between simulated and generated images.

---

## 4. Implementation Reference

`notebooks/utils/metrics.py` is the single source of truth.

| Symbol | Function |
|---|---|
| $\mathcal{M}$ | `MASK` (39×39, $r_d$ = 18.25), `N_MASK` = 1049 |
| — | `topleft_crop(img)` — 40→39, exact; alias `center_crop` kept for old notebooks |
| $M$ | `magnetization(img)` |
| $C_{nn}$ | `spin_correlation(img)` — alias `cnn_correlation` kept |
| $q_{\text{peak}}$ | `peak_wave_vector(img)` |
| all three | `physical_metrics(img)` → dict; `physical_metrics_batch(imgs)` → dict of arrays |
| names / labels | `PHYSICAL_METRIC_NAMES`, `PHYSICAL_METRIC_LABELS` |

Drive per-metric loops from `PHYSICAL_METRIC_NAMES` rather than hardcoding, so
a future change to the set propagates to every figure automatically.
