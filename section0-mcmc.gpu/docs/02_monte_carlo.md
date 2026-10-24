# Monte Carlo Simulation & Dataset Generation

## Overview

Generating thermodynamically faithful magnetic domain configurations requires formal statistical mechanics: the atomistic Hamiltonian $\mathcal{H}$ defines energies, but thermal fluctuations govern which spin configurations are actually realised. This is handled through the **Metropolis–Hastings Monte Carlo algorithm** combined with **simulated annealing**, implemented in PyTorch with CUDA parallelisation.

---

## Thermodynamic Coupling via the Boltzmann Distribution

Temperature $T$ [K] couples to the spin system through the canonical ensemble. A magnetic microstate $\mu$ is a specific realisation of the spin matrix $\mathbf{S}_\mu$. Its thermodynamic energy is:

$$
\mathcal{H}(\mu) \equiv \mathcal{H}\!\left(\mathbf{S}_\mu \,|\, \hat{\mathbf{d}}_{ij}, \hat{\mathbf{h}}, \mathbf{E};\, \boldsymbol{\theta}\right)
$$

At thermal equilibrium, the probability of occupying microstate $\mu$ is:

$$
P(\mu) = \frac{1}{\mathcal{Z}}\exp\!\left(-\frac{\mathcal{H}(\mu)}{k_B T}\right)
$$

The canonical partition function enforces normalisation over all valid spin configurations:

$$
\mathcal{Z} = \sum_{\mathbf{S}_\nu} \exp\!\left(-\frac{\mathcal{H}\!\left(\mathbf{S}_\nu\,|\,\hat{\mathbf{d}}_{ij}, \hat{\mathbf{h}}, \mathbf{E};\,\boldsymbol{\theta}\right)}{k_B T}\right)
$$

---

## Metropolis–Hastings Algorithm

The distribution $P(\mu)$ is sampled stochastically. When a localised spin perturbation proposes a transition from microstate $\mu^{(t)}$ to $\mu^{(t+1)}$, the energy difference $\Delta\mathcal{H} = \mathcal{H}(\mu^{(t+1)}) - \mathcal{H}(\mu^{(t)})$ is evaluated **locally** at the perturbed site for computational efficiency. The proposed state is accepted with probability:

$$
P\!\left(\mu^{(t)} \to \mu^{(t+1)}\right) = \begin{cases} 1 & \text{if } \Delta\mathcal{H} < 0 \\ \exp\!\left(-\dfrac{\Delta\mathcal{H}}{k_B T^{(t)}}\right) & \text{if } \Delta\mathcal{H} \geq 0 \end{cases}
$$

- **Downhill moves** ($\Delta\mathcal{H} < 0$): always accepted — the system moves to a lower-energy state
- **Uphill moves** ($\Delta\mathcal{H} \geq 0$): accepted with Boltzmann probability — allows escape from local minima

---

## Simulated Annealing Protocol

The instantaneous temperature $T^{(t)}$ is systematically reduced during the simulation:

- **High $T^{(t)}$**: $k_B T^{(t)} \gg \Delta\mathcal{H}$ — thermal noise dominates, the system explores freely and escapes metastable minima (misaligned ferromagnetic domains, etc.)
- **Low $T^{(t)}$**: $k_B T^{(t)} \ll \Delta\mathcal{H}$ — the spin field settles into the global energy minimum, stabilising complex chiral textures driven by the DMI

**Exchange-normalised initial temperature:** To ensure every simulation starts from a fully randomised paramagnetic phase regardless of $\boldsymbol{\theta}$, the initial temperature $T^{(0)}$ is dynamically rescaled in proportion to the dominant exchange energy $\tilde{J}_1$. This erases initial-state bias and prevents premature collapse into metastable local minima.

**Cooling schedule (as used in experiments):**
- Final temperature: $T^{(f)} = 0.1$ K
- Linear decrement: $\Delta T = -0.1$ K per step
- At each temperature step: **8,000 Monte Carlo Sweeps (MCS)** for thermalisation
- Followed by **100 measurement MCS** for thermodynamic averages (specific heat, susceptibility) and convergence verification

---

## Nanodot Geometry

The magnetic nanodot is modelled on a three-dimensional discrete lattice with cylindrical confinement:

- **Radius:** $R_d = 18.25$ MUC (Magnetic Unit Cells)
- **Thickness:** $\tilde{L} = 5$ MUC
- **Active spins:** ~5,200 localised spins per simulated dot
- **Geometry mask:** binary circular mask defining valid lattice sites

---

## Image Generation Pipeline

After the annealing converges to the near-ground state at $T^{(f)} = 0.1$ K, the final topological state is extracted and converted to a 2D image:

1. **Geometry parsing** — establish binary circular lattice mask with fixed radius and thickness
2. **Spin state mapping** — extract the out-of-plane component $s_z \in [-1, 1]$ from the central layer ($z = \lfloor \tilde{L}/2 \rfloor$) of the 3D spin field
3. **Image rendering** — rasterise onto a fixed $\breve{H} \times \breve{W}$ pixel grid using a divergent red–blue colormap encoding opposite spin polarities
4. **Preprocessing** — normalise pixel intensities to $[0, 1]$; **no spatial augmentations** (rotations, flips) are applied to preserve intrinsic DMI-dictated chirality

**Why only $s_z$?** Experimental techniques such as magnetic force microscopy and Lorentz TEM observe the out-of-plane component. Using $s_z$ mimics this projection and defines the observational modality. This introduces a fundamental partial observability: parameters that couple primarily to $s_x, s_y$ (e.g., $\tilde{K}_{anS}$, $\tilde{H}_{ex}$) cannot be fully recovered from $s_z$ alone.

---

## Resulting Dataset Structure

The simulation pipeline produces a labelled dataset:

$$
\mathcal{D} = \left\{\mathbf{X}_n, \boldsymbol{\theta}_n\right\}_{n=1}^{\breve{N}}
$$

- $\mathbf{X}_n \in [0,1]^{\breve{H} \times \breve{W}}$: normalised 2D $s_z$ magnetisation field (39 × 39 pixels)
- $\boldsymbol{\theta}_n \in \mathbb{R}^8$: vector of 8 Hamiltonian + thermodynamic parameters

Each sample corresponds to a unique combination of Hamiltonian parameters, sampled independently and uniformly within the bounds in [01_hamiltonian.md](01_hamiltonian.md).

For dataset details and splits, see [06_datasets.md](06_datasets.md).

---

## Implementation Notes

- Simulation optimised using **PyTorch tensor operations on CUDA-enabled GPUs** to overcome the computational bottleneck of sequential MC methods
- Approximate wall-clock time: minutes per sample on CPU; parallelised GPU batches dramatically reduce total time
- Each simulation is **statistically independent** — no shared state between parameter configurations
