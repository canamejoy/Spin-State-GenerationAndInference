# Evaluation Metrics

Four families of metrics are used to evaluate the complete cycle: **regression metrics** (parameter recovery quality), **image metrics** (pixel/spectral fidelity), **physical metrics** (three continuous observables of the spin configuration on the nanodot disk), and **topological descriptors** (discrete counts on the level sets of $s_z$).

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

## 4. Topological Descriptors of the $s_z$ Level Sets

The three canonical observables are **disk averages**, and a disk average is
exactly what a small perturbation can move. The perturbation that latent
guidance writes is about 1% of the image range and lives near the Nyquist
frequency: it shifts $M$ and $C_{nn}$ in the third decimal while taking the
encoder's reading of $\tilde{J}_3$ from 0.001 to 0.985. A descriptor that
**counts** things cannot be moved that way, which is why a discrete family is
carried alongside the scalars.

### 4.1 What this is *not*: the skyrmion number

The topological charge of a spin texture,

$$
Q = \frac{1}{4\pi}\int \mathbf{n}\cdot\left(\partial_x\mathbf{n}\times\partial_y\mathbf{n}\right)\,\mathrm{d}^2r,
\qquad \mathbf{n} = \mathbf{S}/|\mathbf{S}|,
$$

requires the **full three-component** field $\mathbf{n} = (n_x, n_y, n_z)$; on a
lattice it is evaluated through the Berg–Lüscher spherical-triangle
construction over the three spins of each elementary triangle
(Berg & Lüscher 1981, <https://doi.org/10.1016/0550-3213(81)90568-X>).

This dataset stores **only the $s_z$ projection**. The in-plane components were
discarded at image-export time, so $Q$ is not recoverable from these images by
any method, and **no quantity defined in this section is a skyrmion number**.
Recovering genuine topology from a projection needs additional experimental
channels — see e.g. the three-dimensional reconstruction of
[Raftrey *et al.* (2024)](https://doi.org/10.1126/sciadv.adp8615), which
requires vector tomography rather than a single scalar map.

What the $s_z$ projection *does* determine is the topology of its **level
sets**, and that is enough to separate a bubble lattice from a helical stripe —
which the scalar observables alone do not do.

### 4.2 Formal definition: Minkowski functionals of the excursion sets

Fix a threshold $u > 0$ and define the two **excursion sets** (also called
level sets or superlevel sets) inside the disk:

$$
\mathcal{E}^{+}_{u} = \{\,i \in \mathcal{M} \;:\; s_z(i) > u\,\},
\qquad
\mathcal{E}^{-}_{u} = \{\,i \in \mathcal{M} \;:\; s_z(i) < -u\,\}.
$$

By **Hadwiger's theorem**, every functional on subsets of $\mathbb{R}^2$ that is
additive, motion-invariant and conditionally continuous is a linear combination
of exactly **three** *Minkowski functionals*: area $V_0$, perimeter $V_1$, and
Euler characteristic $V_2 = \chi$. Three numbers are therefore not an arbitrary
choice — they are a complete morphological basis in two dimensions
(Mecke 2000, <https://doi.org/10.1007/3-540-45043-2_6>). The family was
introduced into physics as a morphological descriptor by
[Mecke, Buchert & Wagner (1994)](https://arxiv.org/abs/astro-ph/9412061)
(*A&A* **288**, 697), and the Euler characteristic of excursion sets had already
entered physics as the **genus statistic** of
[Gott, Dickinson & Melott (1986)](https://doi.org/10.1086/164347)
(*ApJ* **306**, 341). The distributional theory for excursion sets of random
fields is Adler & Taylor, *Random Fields and Geometry* (Springer, 2007).

In two dimensions the Euler characteristic decomposes into Betti numbers,

$$
\chi = b_0 - b_1 = (\text{connected components}) - (\text{holes}),
$$

which is what makes its **sign** phase-discriminating: a set of $k$ bubbles each
enclosing a hole gives $\chi < 0$, while unbranched stripes or a saturated
domain give $\chi > 0$.

### 4.3 The five numbers computed

All five are evaluated on $\mathcal{E}^{\pm}_{u}$ with $u = 0.25$, under
**8-connectivity** (two domains touching at a corner are one domain, which is
what a physical texture does).

| Quantity | Definition | Formal name |
|---|---|---|
| $n_{+}$, $n_{-}$ | number of connected components of $\mathcal{E}^{+}_{u}$, $\mathcal{E}^{-}_{u}$ | $b_0$ — zeroth Betti number |
| $\chi_{+}$ | Euler characteristic of $\mathcal{E}^{+}_{u}$ | $V_2$ — genus statistic |
| $w$ | fraction of pairs in $\mathcal{B}$ that straddle zero | discrete proxy for $V_1$ (perimeter) — domain-wall density |
| $a_{\max}$ | area of the largest component of $\mathcal{E}^{+}_{u}$, over $N_\mathcal{M}$ | $V_0$ — largest-cluster fraction (percolation order parameter) |

The wall density reuses **the same pair set $\mathcal{B}$ as $C_{nn}$** —
row- and column-adjacent pairs with both ends inside $\mathcal{M}$, each counted
once — so it is the sign-only companion of the correlation:

$$
w = \frac{1}{|\mathcal{B}|}\sum_{\langle i,j\rangle \in \mathcal{B}}
\mathbb{1}\!\left[\,s_z(i)\,s_z(j) < 0\,\right],
\qquad
C_{nn} = \frac{1}{|\mathcal{B}|}\sum_{\langle i,j\rangle \in \mathcal{B}} s_z(i)\,s_z(j).
$$

$C_{nn}$ weights each pair by its amplitude; $w$ counts sign flips and ignores
amplitude entirely. That is precisely why $w$ resists a low-amplitude
perturbation that drags $C_{nn}$.

### 4.4 Choices that are ours, not the literature's

The functionals above are standard. These four decisions are **ours** and must
be reported as such:

| Choice | Value | Justification |
|---|---|---|
| Threshold $u$ | 0.25 | At $u = 0$ a disordered configuration fragments into hundreds of single-pixel specks and $n_\pm$ measures noise, not structure. $u = 0.25$ keeps only domains with real amplitude. **Not a value taken from any paper.** |
| Connectivity | 8 (corner-touching merges) | A physical domain is not cut by a diagonal contact. |
| Support | the disk $\mathcal{M}$, not the square | Same rationale as every other physical metric; boundary pixels outside the dot are not part of the texture. |
| Normalisation of $w$ | pairs, not arc length | $w$ is a discrete sign-flip fraction, **not** a calibrated perimeter $V_1$ in lattice units. It is comparable across images of this dataset; it is not comparable to a continuum $V_1$. |

Because $u$ is a free parameter, the honest presentation of $\chi_{+}(u)$ is as a
curve over $u$ — the *Minkowski functional profile*, which is how the cosmology
and statistical-physics literature reports it. A single $u$ is a convenience for
tabulating one number per image, not a claim that $u = 0.25$ is special. This
curve is now **implemented**, not merely recommended: see §4.7 for the full
threshold sweep and the threshold-free summaries derived from it.

### 4.5 Validation

Verified on synthetic fields at introduction (commit `fe04e40`):

- **Counting.** 1, 4 and 7 planted islands are counted as $n_{+} = 1, 4, 7$.
- **Phase separation by sign of $\chi_{+}$ alone.** $+1$ ferromagnetic; $-6$ for
  seven bubbles; $+7$ for stripes of period 6; $-46$ disordered.
- **Robustness to the guidance perturbation.** Adding near-Nyquist noise at 1%,
  5% and 10% leaves $n_{+} = 7$ throughout, while $M$ drifts $0.6130 \to 0.6060$
  and $C_{nn}$ drifts $0.8063 \to 0.7880$.

The third point is the entire reason the family exists.

### 4.6 Related work on topological descriptors of skyrmion textures

The closest published analogues, for positioning this section in a thesis:

- **Minkowski functionals on simulated magnetic domain patterns** —
  S. Katletz and R. L. Stamps, *Simulations of magnetic domain pattern formation
  and analysis using geometric measures*, **IEEE Trans. Magn. 40**(4), 2155–2157
  (2004), [doi:10.1109/TMAG.2004.830409](https://doi.org/10.1109/TMAG.2004.830409).
  Uses the same functional family to discriminate simulated domain morphologies.
- **Persistent homology as a phase descriptor for skyrmion lattices** —
  M. Taniwaki, T. B. Winkler, J. Rothörl, R. Gruber, C. Mitsumata, M. Kotsugi and
  M. Kläui, *Persistent homology as efficient phase descriptor for 2D skyrmion
  lattices*, **APL Mach. Learn. 4**(2), 026111 (2026),
  [doi:10.1063/5.0331214](https://doi.org/10.1063/5.0331214);
  preprint [arXiv:2504.14688](https://arxiv.org/abs/2504.14688). Treats skyrmions
  as quasi-particles and filters over their positions — a **particle-based**
  filtration, whereas this section uses a **field-based** single-threshold level
  set. Persistent homology is the natural generalisation of §4.4's threshold
  problem: instead of fixing $u$, track $b_0$ and $b_1$ across all $u$.
- **Tensorial extension for anisotropy** — G. E. Schröder-Turk, S. Kapfer,
  B. Breidenbach, C. Beisbart and K. Mecke, *Tensorial Minkowski functionals and
  anisotropy measures for planar patterns*, **J. Microsc. 238**(1), 57–74 (2010),
  [doi:10.1111/j.1365-2818.2009.03331.x](https://doi.org/10.1111/j.1365-2818.2009.03331.x).
  Relevant if stripe **orientation** ever needs to be quantified; the scalar
  functionals used here are orientation-blind by construction.

### 4.7 The Euler characteristic curve (threshold-free)

Instead of fixing $u = 0.25$, sweep it. Define the superlevel filtration

$$
\mathcal{S}_u = \{\,i \in \mathcal{M} \;:\; s_z(i) > u\,\}, \qquad u \in \{-0.9, -0.8, \dots, 0.9\},
$$

and, at every $u$ on that grid, the three curves

$$
\beta_0(u) = \text{connected components of } \mathcal{S}_u, \qquad
\beta_1(u) = \text{holes of } \mathcal{S}_u, \qquad
\chi(u) = \beta_0(u) - \beta_1(u).
$$

Sweeping $u$ removes the arbitrary $0.25$ of §4.4: no single threshold has to
be defended, and the summaries below are functionals of the **whole curve**
rather than a reading at one point.

| Summary | Meaning |
|---|---|
| `chi_min` | minimum of $\chi(u)$ over the grid — most negative when the majority phase is riddled with holes (bubble-like) |
| `chi_max` | maximum of $\chi(u)$ over the grid — most positive for unbranched stripes or a saturated domain |
| `u_chi_min` | the threshold $u$ at which $\chi$ attains `chi_min` |
| `b0_max` | maximum of $\beta_0(u)$ over the grid — the threshold-free replacement for $n_{+}$: the peak domain count over **all** thresholds, not one fixed one |
| `ecc_l1` | $\int \lvert\chi(u)\rvert\,\mathrm{d}u$ — total topological activity of the curve |
| `b1_total` | $\int \beta_1(u)\,\mathrm{d}u$ — total loop content, a "total persistence"-flavoured summary |

This object is documented in the literature as the **Euler Characteristic
Curve (ECC)**. It is provably stable and far cheaper to compute than full
persistent homology:

- A. Smith and V. M. Zavala, *The Euler characteristic: A general topological
  descriptor for complex data*, **Comput. Chem. Eng. 154**, 107463 (2021),
  [doi:10.1016/j.compchemeng.2021.107463](https://doi.org/10.1016/j.compchemeng.2021.107463).
- O. Hacquard and V. Lebovici, *Euler characteristic tools for topological
  data analysis*, **JMLR 25** (2024),
  <https://jmlr.org/papers/volume25/23-0353/23-0353.pdf>.

The ECC is the Euler-characteristic shadow of the cubical sublevel-set
persistent homology of the same field, so it captures the threshold sweep
without needing a persistence library.

> **When the sweep buys nothing.** On a field that is effectively two-valued
> ($s_z \approx \pm 1$ everywhere, i.e. a saturated low-temperature texture),
> $\mathcal{S}_u$ does not change anywhere in the interior of the grid, so
> $\chi$ is **constant**: `chi_min` $=$ `chi_max`, `ecc_l1` collapses to
> $\lvert\chi\rvert \times 1.8$ and `b1_total` to $\beta_1 \times 1.8$,
> both redundant. Worse, `u_chi_min` degenerates to the first grid point
> ($-0.9$) purely because `argmin` breaks ties on the left, which is an
> artefact and not a physical threshold. All five synthetic phases of §4.5 are
> two-valued and show exactly this. **Treat `chi_min`, `chi_max` and `b0_max`
> as the primary summaries, and read `u_chi_min` only when
> `chi_min < chi_max`.** The curve carries genuine extra information only
> where $s_z$ takes intermediate values — finite-temperature textures,
> resolved domain walls, and DDPM output, which is continuous by
> construction. That is the regime this family is for.

### 4.8 Stripe defects: junctions and terminals

The ECC counts components and holes, but says nothing about whether the
stripes **branch** — a question this descriptor answers instead. Define the
domain set $\mathcal{D} = \{\,i \in \mathcal{M} : s_z(i) > 0\,\}$, its
morphological skeleton (a 1-pixel-wide medial curve), and classify each
skeleton pixel by its 8-neighbour count within the skeleton:

| Neighbours | Class |
|---|---|
| 0 | isolated point (a degenerate bubble) |
| 1 | terminal (a stripe end) |
| $\geq 3$ | junction (a branch point) |

| Quantity | Meaning |
|---|---|
| `n_terminals` | skeleton pixels with exactly 1 neighbour |
| `n_junctions` | skeleton pixels with $\geq 3$ neighbours |
| `n_isolated` | skeleton pixels with 0 neighbours |
| `skeleton_len` | number of skeleton pixels counted |
| `defect_density` | $(n_{\text{terminals}} + n_{\text{junctions}}) / \text{skeleton\_len}$, or $0$ when the skeleton is empty |

**Margin exclusion is a correctness requirement, not a tuning knob.** A
stripe that is merely cut by the edge of the dot produces a spurious
terminal at the cut, which has nothing to do with the stripe morphology
itself. Counts are restricted to the skeleton eroded away from $\mathcal{M}$
by 2 pixels, so a domain that exits the disk is not misread as ending inside
it.

**Phase reading, measured** on the synthetic fields of §4.5 (not asserted —
these are the values the implementation actually returns):

| Field | `n_terminals` | `n_junctions` | `n_isolated` | `defect_density` |
|---|---|---|---|---|
| Saturated ferromagnet | 0 | 0 | 1 | 0.000 |
| Helical stripes, period 6 | 0 | 0 | 0 | 0.000 |
| Seven $+1$ bubbles on a $-1$ background | 14 | 0 | 0 | 0.500 |
| Seven $-1$ bubbles in a $+1$ matrix | 0 | 42 | 0 | 0.290 |
| Disordered | 7 | 254 | 0 | 0.808 |

Two readings matter. First, **`defect_density` orders disorder monotonically**
(0.000 → 0.290 → 0.500 → 0.808), which is what makes it usable as a
comparison axis. Second, **the terminal/junction split separates bubbles from
a labyrinth**: a bubble lattice is terminal-rich and junction-free, because
each compact bubble skeletonises to a short segment with two ends; a
labyrinth is junction-dominated, because its stripes branch. The ECC cannot
make that distinction — it is the reason this descriptor exists.

`n_isolated` fires only in the degenerate case where a domain skeletonises to
a single pixel (the saturated disk). It is **not** the bubble signature; an
earlier draft of this section claimed it was, and the measurement above
refutes that.

This is grounded in V. Y. Okubo, K. Shimizu, B. S. Shivaram and H. Y. Kim,
*Characterization of Magnetic Labyrinthine Structures Through Junctions and
Terminals Detection Using Template Matching and CNN*, **IEEE Access 12**,
92419-92430 (2024),
[doi:10.1109/ACCESS.2024.3422259](https://doi.org/10.1109/ACCESS.2024.3422259).
The honest difference: they need TM-CNN because their microscopy images are
noisy; our simulated fields are clean enough for an exact morphological
skeleton.

**Ours, not the literature's:** `threshold = 0.0` and `margin = 2` are our
choices.

> **Input range is a silent footgun.** Every function in §4 assumes
> $s_z \in [-1, 1]$, the convention `metrics.py` declares at module level.
> The dataset NPZ stores `img` in $[0, 1]$ (see docs/06), and the DDPM
> pipeline rescales with `img * 2 - 1`. Feeding a raw $[0, 1]$ image to
> `stripe_defects` marks the **entire disk** as domain and to
> `euler_characteristic_curve` wastes the negative half of the grid — in both
> cases silently, because `_check_physical_shape` validates the shape and not
> the range. Rescale before measuring.

---

## 5. Implementation Reference

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
| $n_{+}, n_{-}, \chi_{+}, w, a_{\max}$ | `topological_descriptors(img, threshold=0.25)` → dict with keys `n_pos`, `n_neg`, `euler_pos`, `wall_frac`, `largest_frac` |
| same, batched | `topological_batch(imgs)` → dict of arrays; names in `TOPOLOGY_NAMES` |
| $u$ grid | `THRESHOLD_GRID` — 19 values, $-0.9 \dots 0.9$ in steps of $0.1$ |
| $\beta_0(u), \beta_1(u), \chi(u)$ | `euler_characteristic_curve(img, thresholds=THRESHOLD_GRID)` → dict with keys `u`, `chi`, `b0`, `b1` |
| `chi_min`, `chi_max`, `u_chi_min`, `b0_max`, `ecc_l1`, `b1_total` | `ecc_summary(img)` → dict; batched: `ecc_summary_batch(imgs)` → dict of arrays; names in `ECC_SUMMARY_NAMES` |
| $n_{\text{terminals}}, n_{\text{junctions}}, n_{\text{isolated}}, \text{skeleton\_len}, \text{defect\_density}$ | `stripe_defects(img, threshold=0.0, margin=2)` → dict; batched: `stripe_defects_batch(imgs)` → dict of arrays; names in `DEFECT_NAMES` |

Drive per-metric loops from `PHYSICAL_METRIC_NAMES` rather than hardcoding, so
a future change to the set propagates to every figure automatically.

The Euler characteristic curve and stripe-defect descriptors (§4.7, §4.8) are
covered by `tests/test_topology.py`. Run it with:

```
pytest tests/ -q
```
