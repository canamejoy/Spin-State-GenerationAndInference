# Test plan (from `planning/pruebas-v1.xlsx`)

Transcribed so it is reviewable in version control, cross-checked against what
has already been run, with the problems found while checking it.

---

## Part 1 — Simulation: 16 runs

CPU / 1 GPU / 2 GPU / 3 GPU, crossed with `MD` (matrix decomposition) and
`BP` (bipartite / checkerboard update), on and off.

### Measured: see `10_ablation_and_cpu_rate.md`

**This section's estimates were wrong and are superseded.** They are kept with
the correction because the error is instructive.

The estimate put a site-by-site attempt at 10.0 us and therefore a 65x penalty
for dropping `MD`, which made eight of the sixteen cells look unaffordable and
argued for running them at a reduced configuration. Both figures were inflated by
per-sweep setup inside the timed loop -- the neighbour table and the per-site
anisotropy constant were being rebuilt every sweep, which a real implementation
builds once. Hoisted out, the per-site cost is **2.44 us** and the `MD` penalty
is **8.6x**, so every cell runs complete lattice sweeps at the production
geometry with no reduced configuration and no pro rata scaling.

The `no BP` question below was resolved in the direction this section
anticipated, and more sharply: the colouring costs **0.98x** -- nothing. It is
not an accelerator at all. Its contribution is correctness, which is now measured
rather than argued: dropping it while keeping the parallel update shifts the
Berg-Lüscher charge of the middle layer by up to a full topological unit.

### `no BP` needs its meaning fixed before it is run

Switching the checkerboard off is not a speed setting. Without it, updating
sites in parallel is simply **wrong**: two neighbours would each decide against
the other's stale value, the joint move's energy is not the sum of the
individual ones, and the chain stops sampling the Boltzmann distribution.

So `no BP` has to mean *strictly sequential single-site updates* -- correct,
and slow. If that is the intent, then `no MD, no BP` and `MD, no BP`
are nearly the same run, because a sequential update cannot use a vectorised
energy for anything but one site at a time.

**Measured, and the last sentence is wrong.** They are not nearly the same run:
`MD, no BP` is 62x *slower* than `no MD, no BP`, because building a
whole-lattice neighbour field per accepted site costs far more than the scalar
arithmetic it replaces. Vectorising a sequential chain is a pessimisation.

One more correction, and it changes what the `CPU` column means. The reference
notebook's own CPU mode is labelled site-by-site, but it is **vectorised over the
replica batch**, so it already carries a matrix decomposition -- on the replica
axis instead of the lattice axis. The `no MD` baseline for the group's actual
code is therefore 4.01 ms per replica-sweep, not the scalar loop's 12.77.

### The checkerboard is exact only for shells 1 and 3

Under the parity colouring `(x+y+z) mod 2`:

| shell | neighbours | example | colour change |
|---|---|---|---|
| 1 | 6 | (1,0,0) | 1 — bipartite, exact |
| 2 | 12 | (1,1,0) | **0 — same colour, NOT exact** |
| 3 | 8 | (1,1,1) | 1 — bipartite, exact |
| 4 | 6 | (2,0,0) | **0 — same colour, NOT exact** |

With `J2` or `J4` non-zero there are bonds *inside* a sublattice, and the
simultaneous update is no longer an exact reordering of sequential Metropolis.
The planned parameters have `J2=J3=J4=0`, so the planned runs are safe -- but
**18.9% of the released dataset is not**: clusters 8, 10, 11 and 12 (Bimeron and
Ferromagnetic, 32,074 of 169,671 samples) were generated with `J2` between
-0.077 and +0.346.

How much that biases those samples is unmeasured. The package's own test is
named `test_no_shell_1_neighbour_shares_a_sublattice` — scoped to the one shell
where the property holds.

---

## Part 2 — Generative models: 4 runs

| # | model |
|---|---|
| 1 | CVAE |
| 2 | DDPM FiLM — T only |
| 3 | DDPM FiLM — 8 parameters |
| 4 | DDPM FiLM — 8 parameters + spin correlation |

This is a **conditioning ablation**, not the architecture comparison that
`notebooks/evaluation/physical-metrics-comparison-4models.ipynb` already ran
(DDPM-mine / DDPM-paper / NCSN / CVAE). Both are worth having; they answer
different questions and should not be presented as one table.

Note that 2 → 3 → 4 is a clean ladder (how much conditioning, then physics in
the loss), which is what makes the hypothesis tests below meaningful.

---

## Part 3 — Full cycle: 8 rows, 6 distinct tests

| # | test | status |
|---|---|---|
| 1 | cosine, **sampler**, freeze all | **done** — `integration/training/03`, evaluated in `evaluation/04` |
| 2 | cosine, sampler, retrain DDPM | not run |
| 3 | cosine, sampler, retrain DDPM + Xception | not run |
| 4 | cosine, sampler, freeze all | **duplicate of 1** |
| 5 | cosine, **cost function**, freeze all | **done** — `integration/training/01`, evaluated in `evaluation/02` |
| 6 | cosine, cost function, retrain DDPM | not run |
| 7 | cosine, cost function, retrain DDPM + Xception | **done** — `integration/training/02`, evaluated in `evaluation/03` |
| 8 | cosine, cost function, freeze all | **duplicate of 5** |

Rows 4 and 8 repeat rows 1 and 5 verbatim. Six distinct tests, three already
run.

A result that bears on the three remaining ones: row 7 (joint retrain) collapses
without a regression anchor — the cosine reaches 0.0000 by epoch 3 and the
inverse model's R^2 goes to -1.1e5. Rows 2, 3 and 6 all retrain something, so
each needs that anchor from the start.

And the measured outcome of rows 1, 5 and 7 should be read before committing to
the rest: the latent guidance raises cycle R^2 from 0.6387 to 0.9161 while
leaving every physical metric unchanged, but the gain is not physical. Feeding
the head the parameters with no image at all scores 0.9596, and a 32-point sweep
showed the guidance has exactly one degree of freedom — it aligns the image's
latent with `predictor(theta)` and can do nothing else.

---

## Hypothesis tests

| # | test | note |
|---|---|---|
| 1 | vary the seed, compare models 3 vs 4 | the right unit is theta (n=1000), not the K samples per theta |
| 2 | k-folds, compare models 3 vs 4 | paired over theta; Wilcoxon rather than a t-test, since per-theta errors are skewed |

Both compare the same pair, so they share a null hypothesis and the pair of them
needs one multiple-comparison correction, not two independent readings.

## Open questions in the sheet

| question | status |
|---|---|
| how latent vectors are compared | **answered**: UMAP over four populations on the same theta, with distances reported in the original 256-d space as well as the projection. Guidance displaces the latent by 1.357, a message by 0.144 |
| behaviour at high T | the notebook referred to as "06 high T guidance" — not in either repository; needs a path |
| noise schedule: linear vs exp vs cosine | not run. The DDPM uses a cosine schedule (`BEST_HPARAMS`) |

---

## Simulation parameters as given

| parameter | value | note |
|---|---|---|
| thermalisation sweeps | 10,000 | |
| measurement sweeps | 5,000 | `docs/02_monte_carlo.md` says 8,000 + 100; the two disagree |
| replicas | 100 | |
| `Rd` | 18.25 | the benchmark uses 18.30; both give `N=39`, so no numerical difference |
| `z` | 5 | |
| `Jex` | 1.0 | fixed energy reference |
| `Jex2`, `Jex3`, `Jex4` | 0 | keeps the checkerboard exact — see Part 1 |
| `Kan1` | 0.3107 | |
| `KanS` | 0.0 | |
| `Hex` | 0.3823 | |
| `KDM` | 0.9382 | |
| `dt` | −0.1 K | |
| `Ti` → `Tf` | 20.0 → **0.0** | **`beta = 1/(k_B T)` diverges at T=0, and `C_v` divides by `T^2`**. Use 0.1 K, or treat the last point as a greedy quench |

`Kan1`, `KanS`, `Hex` and `KDM` are cluster 16's values — a verified skyrmion
point (topological charge quantised per replica at −7, −8, −9, with `M_z` = +0.52).
`Kan2` and `kd` are blank in the sheet and do not exist in `HamiltonianParams`.
