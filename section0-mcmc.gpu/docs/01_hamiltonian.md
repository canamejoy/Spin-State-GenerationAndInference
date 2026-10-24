# Extended Heisenberg Hamiltonian Framework

## Overview

The atomistic foundation of this project is the **extended Heisenberg Hamiltonian**, which models the discrete spin interactions in a magnetic nanodot. It generalises the classic Ising and Heisenberg models to include symmetry-breaking relativistic effects — primarily the Dzyaloshinskii–Moriya interaction (DMI) and magnetocrystalline anisotropy — that are essential for stabilising chiral topological textures such as skyrmions and helical domains in confined nanoscale geometries.

---

## General Quadratic Form

Let $\mathcal{V} = \{1, \dots, N\}$ be the set of discrete lattice sites. Each site $i$ carries a normalised spin vector $\mathbf{s}_i \in \mathbb{R}^3$ with $\|\mathbf{s}_i\|_2 = 1$. The total effective energy of the system $\mathcal{H} \in \mathbb{R}$ [meV] is:

$$
\mathcal{H}\!\left(\mathbf{S}\,|\,\breve{\theta}\right)
= -\sum_{i=1}^{N}\sum_{j \in \mathcal{N}(i)} \langle \mathbf{s}_i,\, \tilde{\mathbf{M}}_{ij}\,\mathbf{s}_j \rangle
  - \sum_{i=1}^{N} \langle \tilde{\mathbf{h}}_i,\, \mathbf{s}_i \rangle
  - \sum_{i=1}^{N} \langle \mathbf{s}_i\,\tilde{\mathbf{K}}_i,\, \mathbf{s}_i \rangle
$$

where:
- $\mathcal{N}(i)$: set of interacting neighbours of site $i$
- $\mathbf{S} \in \mathbb{R}^{N \times 3}$: full spin matrix
- $\tilde{\mathbf{M}}_{ij} \in \mathbb{R}^{3 \times 3}$: bilinear interaction matrix
- $\tilde{\mathbf{h}}_i \in \mathbb{R}^3$: external field vector
- $\tilde{\mathbf{K}}_i \in \mathbb{R}^{3 \times 3}$: anisotropy tensor

---

## Parameter Decomposition

The free parameter vector is:
$$
\boldsymbol{\theta} = \left[\tilde{J}_1,\, \tilde{J}_2,\, \tilde{J}_3,\, \tilde{J}_4,\, \tilde{D},\, \tilde{H},\, \tilde{K}_{an1},\, \tilde{K}_{anS}\right]
$$

### 1. Bilinear Coupling — Exchange + DMI

Any real square matrix splits into symmetric and skew-symmetric parts:

$$
\tilde{\mathbf{M}}_{ij}(\tilde{J}_r, \tilde{D}) = \tilde{J}_r\,\mathbf{I} - \tilde{D}\,[\hat{\mathbf{d}}_{ij}]_\times
$$

- $\tilde{J}_r$ [meV]: isotropic exchange constant for the $r$-th neighbour shell ($r \in \{1,2,3,4\}$), ordered by increasing interatomic distance $d_1 < d_2 < d_3 < d_4$.
  - $\tilde{J}_r > 0$: ferromagnetic (parallel) alignment
  - $\tilde{J}_r < 0$: antiferromagnetic (antiparallel) alignment
- $\tilde{D}$ [meV]: DMI magnitude, acting only between first-shell neighbours
- $\hat{\mathbf{d}}_{ij} \in \mathbb{R}^3$: unit DM vector ($\|\hat{\mathbf{d}}_{ij}\|_2 = 1$)
- $[\hat{\mathbf{d}}_{ij}]_\times$: skew-symmetric cross-product matrix

**Why $\tilde{J}_1$ is fixed:** The Boltzmann distribution depends only on $\mathcal{H}/k_B T$, so any global rescaling $\lambda \boldsymbol{\theta}$ yields an identical spin configuration. Fixing $\tilde{J}_1 = 1.0$ meV breaks this scale symmetry and ensures a well-posed inverse problem.

### 2. Zeeman Interaction

$$
\tilde{\mathbf{h}}_i = \tilde{H}\,\hat{\mathbf{e}}_x \in \mathbb{R}^3
$$

- $\tilde{H}$ [meV/atom]: external field magnitude along the fixed crystallographic axis $\hat{\mathbf{e}}_x$
- Range: $[0.0,\; 2.0]$ meV, corresponding to $0$–$10$ T in laboratory units
- Fixing the direction eliminates the orientational degree of freedom, leaving $\tilde{H}$ as the single Zeeman scalar

### 3. Magnetocrystalline Anisotropy

The lattice is partitioned into two disjoint subsets: bulk sites $\mathcal{V}_{bulk}$ (full coordination shells) and surface sites $\mathcal{V}_{surf}$ (broken inversion symmetry). The site-dependent anisotropy constant is:

$$
\tilde{K}^{eff}_i = \begin{cases} \tilde{K}_{an1} & \text{if } i \in \mathcal{V}_{bulk} \\ \tilde{K}_{anS} & \text{if } i \in \mathcal{V}_{surf} \end{cases}
$$

- $\tilde{K}_{an1}$ [meV/atom]: bulk cubic anisotropy; range $[0.001,\; 1.0]$ meV in hard nanomaterials
- $\tilde{K}_{anS}$ [meV/atom]: surface anisotropy; $|\tilde{K}_{anS}| > |\tilde{K}_{an1}|$ due to enhanced symmetry-breaking at the boundary

---

## Full Explicit Hamiltonian

Substituting the decomposed terms:

$$
\mathcal{H}\!\left(\mathbf{S}\,|\,\hat{\mathbf{d}}_{ij}, \hat{\mathbf{e}}_x;\,\boldsymbol{\theta}\right)
= -\sum_{r=1}^{4} \tilde{J}_r \sum_{i=1}^{N}\sum_{j \in \mathcal{N}_r(i)} \langle \mathbf{s}_i, \mathbf{s}_j \rangle
  + \tilde{D} \sum_{i=1}^{N}\sum_{j \in \mathcal{N}_1(i)} \langle \hat{\mathbf{d}}_{ij},\, (\mathbf{s}_i \times \mathbf{s}_j) \rangle
  - \tilde{H}\sum_{i=1}^{N} \langle \hat{\mathbf{e}}_x, \mathbf{s}_i \rangle
  + \frac{1}{S_o^4}\sum_{i=1}^{N} \tilde{K}^{eff}_i \left( s^2_{i\hat{x}}s^2_{i\hat{y}} + s^2_{i\hat{x}}s^2_{i\hat{z}} + s^2_{i\hat{y}}s^2_{i\hat{z}} \right)
$$

The cubic anisotropy form $s^2_{i\hat{x}}s^2_{i\hat{y}} + s^2_{i\hat{y}}s^2_{i\hat{z}} + s^2_{i\hat{z}}s^2_{i\hat{x}}$ follows from expanding $\frac{1}{2}(1 - \|\mathbf{s}_i \mathbf{E}\|^4_4)$ under the $L_2$ normalisation constraint, normalised by $S_o^4$ for dimensional consistency.

---

## Free Parameter Sampling Ranges

| Parameter | Physical Role | Min | Max | Median |
|---|---|---|---|---|
| $T^{(0)}$ [K] | Annealing temperature | 0.000 | 20.000 | 3.400 |
| $\tilde{J}_2$ [meV] | 2nd-shell exchange | -0.286 | 0.659 | 0.000 |
| $\tilde{J}_3$ [meV] | 3rd-shell exchange | -0.290 | 0.290 | 0.000 |
| $\tilde{J}_4$ [meV] | 4th-shell exchange | -0.234 | 0.235 | 0.000 |
| $\tilde{K}_{an1}$ [meV/atom] | Bulk anisotropy | 0.000 | 0.600 | 0.082 |
| $\tilde{K}_{anS}$ [meV/atom] | Surface anisotropy | 0.000 | 0.200 | 0.070 |
| $\tilde{H}_{ex}$ [meV/atom] | External Zeeman field | 0.000 | 1.199 | 0.022 |
| $\tilde{K}_{DM}$ [meV] | DMI strength | 0.000 | 1.200 | 0.665 |

> **Measured, not specified.** These are the actual minimum, maximum and median
> of each column of `params` in `dataset_unificado_v2.npz` (169,671 samples),
> read off the released dataset rather than copied from the generation script.
> An earlier version of this table quoted a nominal sampling design in which
> $\tilde{H}_{ex}$ reached only 0.05 and $\tilde{K}_{an1}$ only 0.2. Both are
> wrong by a wide margin -- the field reaches 1.199 and the bulk anisotropy
> 0.600 -- and the error is not cosmetic: a parameter search bounded by the old
> table cannot reach the skyrmion phase at all, because the field never becomes
> strong enough to collapse stripe domains into isolated skyrmions. Note also
> that $\tilde{J}_2$ is sampled asymmetrically ($-0.286$ to $+0.659$), not over
> the symmetric interval the old table implied.

$\tilde{J}_1 = 1.0$ meV is **fixed** as the energy reference unit.

---

## Emergent Magnetic Phases

The competition among $\tilde{J}$, $\tilde{D}$, $\tilde{K}$, and $\tilde{H}$ produces the following dominant topological regimes identifiable in the $s_z$ projection:

| Phase | Key driver | Visual signature |
|---|---|---|
| Ferromagnetic | $\tilde{J}_1 \gg \tilde{D}$, low $T$ | Uniform out-of-plane polarisation |
| Helical | $\tilde{D}/\tilde{J}_1 \approx 1$, low $T$ | Regular stripe pattern with chiral winding |
| Labyrinthine & Conical | Intermediate $\tilde{D}$, moderate $T$ | Stripe/labyrinthine domains with clear periodicity |
| Paramagnetic / Others | High $T$ | Spatially uncorrelated noise |

---

## References

- Eriksson et al., *Atomistic Spin Dynamics: Foundations and Applications*, OUP, 2017
- Fert, Reyren & Cros, *Nature Reviews Materials* 2, 17031, 2017
- Cullity & Graham, *Introduction to Magnetic Materials*, Wiley, 2011
