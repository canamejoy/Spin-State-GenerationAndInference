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
| 1 | site-by-site sequential (NumPy) | CPU |
| 2 | checkerboard + tensors + `jit` | 1 GPU |
| 3 | checkerboard + tensors + `pmap` | 2x T4 |

and it measures E, M, C_v and chi against T for two experiments (helical and
skyrmionic), plus the middle-layer spin visualisation.

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
