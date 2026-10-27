# 01 · Data Generation

GPU-accelerated Monte Carlo simulation that generates the labelled synthetic dataset used to
train the regression models.

| Notebook | Description |
|----------|-------------|
| [`monte_carlo_simulation.ipynb`](monte_carlo_simulation.ipynb) | Atomistic **Metropolis–Hastings Monte Carlo** with **simulated annealing** for the extended Heisenberg Hamiltonian, implemented in **JAX**. |

## What it does

- Relaxes magnetic nanodots governed by symmetric exchange ($J_1\!-\!J_4$), the Dzyaloshinskii–Moriya
  interaction ($K_{DM}$), magnetocrystalline anisotropy, and the Zeeman field, into thermodynamically
  converged spin textures.
- Evaluates the energy variation $\Delta\mathcal{H}$ for the entire lattice at once (tensor-shift
  neighbor gathering), with three levels of parallelization:
  - **(a) spins** — checkerboard sublattice decomposition $(x+y+z)\bmod 2$;
  - **(b) replicas** — independent stochastic chains batched on a GPU;
  - **(c) experiments** — independent parameter sets distributed across GPUs (`pmap`, SPMD).
- The annealing temperature and the Monte Carlo sweeps stay **strictly sequential**, so the
  parallelization introduces no approximation to the physics.

## Requirements

- A **multi-GPU** environment: the notebook asserts `jax.device_count() >= 2`.
- JAX with GPU support (FP32).

## Output

Labelled pairs $\{\mathbf{X}_n, \boldsymbol{\theta}_n\}$ — 2D out-of-plane magnetization ($s_z$)
images and their generating Hamiltonian parameters — saved per batch (`LOTE_ID`). The full
assembled dataset is published on
[Kaggle](https://www.kaggle.com/datasets/carloscanamejoy/dataset-spines-united-v2).
