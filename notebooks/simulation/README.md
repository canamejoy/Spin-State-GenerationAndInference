# Simulation notebooks

## `00_benchmark_mc_observables.ipynb` — the reference, do not edit

Written by the project's PhD lead and copied here **verbatim**. Its md5 is
`d704a6ec4e8bdc65392a771ef22372c2`; check it before and after any session that touches this directory.

Every other notebook here derives from it. If something in it needs to change,
that is a conversation with its author, not an edit.

### What it establishes

It is already a three-mode benchmark, which is the structure the CPU-vs-GPU
comparison needs:

| mode | engine | device |
|---|---|---|
| 1 | site-by-site over the lattice, **vectorised over the replica batch** (NumPy) | CPU |
| 2 | checkerboard + tensors + `jit` | 1 GPU |
| 3 | checkerboard + tensors + `pmap` | 2x T4 |

and it measures E, M, C_v and chi against T for two experiments (helical and
skyrmionic), plus the middle-layer spin visualisation.

Mode 1 is labelled *"CPU clásico (site-by-site)"*, and the label undersells it:
`get_neighbors` indexes `sp[:, z, y, (x+1) % Nx]`, so every numpy operation is
spread across the whole replica batch. It already carries a matrix decomposition
-- on the replica axis rather than the lattice axis. That matters for reading the
ablation: at batch 1 the same loop is 18x *slower* than scalar arithmetic,
because a numpy call on a `(1,3)` array is nearly all overhead, and the honest
baseline for "what the group's CPU path costs" is 4.01 ms per replica-sweep, not
a scalar loop's 12.77. See `docs/10_ablation_and_cpu_rate.md`.

### The conventions it fixes

These are the project's physics and every derived notebook must match them:

- **DMI on the in-plane bonds only** (x and y), with sign `-kDM`. This is
  *interfacial* DMI, not an omission: in a thin film the DM vector comes from
  the interface and acts on in-plane bonds. Including the z bond would model
  bulk DMI, a different material class.
- **Proposal: a uniform draw on the sphere**, with no cone move and no adaptive
  width.
- **Checkerboard `(x+y+z) % 2`**, cubic anisotropy with a bulk/surface split,
  Zeeman through the `gamma` angle.

Measured consequence of getting the DMI wrong: on the same spin configuration at
`K_DM = 0.9382`, using x/y/z with a `+` sign changes the total energy by
**50%**. It is a different Hamiltonian, not a detail.

### Its configuration is deliberately reduced

`Rd = 15.0`, 20 temperature points, 50 + 50 sweeps, 128 replicas — sized so
mode 1 finishes inside a Kaggle session. The production configuration
(`Rd = 18.30`, 200 temperature points, 10,000 + 5,000 sweeps, 100 replicas)
is roughly four orders of magnitude more work, and the derived notebooks are
where that scaling is handled.


---

## Derived notebooks

| Notebook | Purpose |
|---|---|
| `01_validation_gpu_vs_prof_dario.ipynb` | Agreement with the tutor's released states at matched parameters. |
| `02_cpu_rate.ipynb` | The CPU per-sweep cost and the extrapolated ladder. |
| `03_ablation_dp_bp.ipynb` | What each acceleration contributes, and what dropping the checkerboard costs in physics. |
| `05_parallel_axes.ipynb` | Replicas, states or both, on CPU and 1/2/4 H100. Six full ladders measured. |
| `06_physics_across_runs.ipynb` | The observables those runs produced, per state, six panels side by side. |
| `04_timing_summary.ipynb` | Every arm, every comparable time unit, and the bar charts. Mirrored to Kaggle as `carloscanamejoy/04-timing-summary`. |

Compute lives in `modal/`; the notebooks load JSON from `../simulation/results/`
and present it. Keeping the two apart is what lets a 50-hour CPU extrapolation
and a 15-minute H100 run be reported in the same table without either being
re-executed to redraw a figure.
