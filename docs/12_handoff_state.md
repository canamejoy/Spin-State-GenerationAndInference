# Handoff: state of the simulation study

Written before a migration. Everything needed to pick this up elsewhere.

---

## 1. What is NOT in git

**Nothing from this study has been committed.** The whole simulation arm lives in
the working tree only:

```
 M docs/08_notebook_index.md         references to the new notebooks
 M docs/09_test_plan.md              superseded estimates marked
 M docs/planning/pruebas-v1.xlsx     Part 1 RUN column, 16 measured cells
 M notebooks/simulation/README.md    notebook 00's CPU mode corrected
 M tests/conftest.py                 makes the engine importable from tests
?? docs/10_ablation_and_cpu_rate.md
?? docs/11_parallel_axes.md
?? docs/12_handoff_state.md          this file
?? notebooks/simulation/0{1..6}_*.ipynb
?? notebooks/simulation/modal/       the engine and the Modal app
?? notebooks/simulation/results/     every measurement, as JSON
?? tests/test_ablation_equivalence.py
?? tests/test_ladder_checkpoint.py
```

`*.npz` is gitignored, so `results/npz/` (169 MB of spins and raw accumulators)
will **not** survive a `git clone`. Those files are reproducible from the Modal
volumes, and the derived quantities are already in the committed JSON.

**Commit before migrating.** Suggested split: one commit for
`notebooks/simulation/modal/` plus the two test files, one for the notebooks and
`results/*.json`, one for the docs and the spreadsheet.

---

## 2. Where the code lives

```
notebooks/simulation/
  00_benchmark_mc_observables.ipynb   REFERENCE, never edit (md5 d704a6ec…)
  01_validation_gpu_vs_prof_dario.ipynb
  02_cpu_rate.ipynb
  03_ablation_dp_bp.ipynb
  04_timing_summary.ipynb
  05_parallel_axes.ipynb
  06_physics_across_runs.ipynb
  results/*.json                      every measurement
  results/npz/                        heavy artefacts, gitignored
  modal/
    sim_core.py        the engine: geometry, substep, energy, anneal, reduce_obs
    ablation.py        the four MD x BP cells, pure numpy, no JAX
    device_ablation.py the same cells across devices
    parallel_axes.py   the replica/state groupings
    ladder_cpu.py      checkpointed CPU ladder, addressed randomness
    run_ablation.py    local runner for the ablation
    app.py             every Modal function and entrypoint
    theta4.json        the four parameter sets
```

`app.py` entrypoints: `main`, `rate`, `ablation_run`, `gpu_nodp`, `cpu_ladder`,
`cpu_full`, `hardware`, `overhead`, `axes`, `axis_ladder`, `batch_scan`.

---

## 3. Tests

```
.venv/bin/python -m pytest tests/ -q      # 42 passed
```

`tests/test_ablation_equivalence.py` pins the scalar single-site `dE` against the
reference engine's total-energy difference, proves the checkerboard update equals
sequential colour-order exactly, proves the uncoloured update does **not**, and
proves the replica-batched loop is the scalar chain.

`tests/test_ladder_checkpoint.py` proves a resumed ladder is bit-identical to an
uninterrupted one. That is load-bearing: the CPU ladder runs ~50 h in blocks.

Both need `jax[cpu]` in the venv. `tests/conftest.py` puts
`notebooks/simulation/modal` on the path.

---

## 4. Credentials and accounts

**Before every launch, run the spend guard** — `scripts/modal_budget_guard.sh
check <account> <estimated-usd>`. It reads the live balance and refuses when the
account is short. Three accounts went past their credit in one day because the
balance was tracked by subtracting predicted costs instead of reading it, and
the predictions counted GPU-seconds only: CPU, memory and container startup add
about 8% on a GPU job and dominate a long CPU job. Estimate the run, let the
guard read the truth, and re-read the balance afterwards.

Modal profiles live in `~/.modal-accounts/<name>/.modal.toml`, used via
`MODAL_CONFIG_PATH`. **Never paste their contents.** `modal token new` writes to
`~/.modal.toml`; move it into the per-account folder.

| account | status |
|---|---|
| ccanamejoy | exhausted, retired by the user |
| <account-b> | ~$10 left |
| <account-c> | ~$8 left |
| <account-a> | ~$18 left, running the CPU ladder |

Kaggle uses the `kg <account> <args>` wrapper over
`~/.kaggle-accounts/<account>/kaggle.json`. The Kaggle account `ccanamejoy`
(username `carloscanamejoy`) owns the published notebooks and the dataset — it is
a different service from the exhausted Modal account of the same name and is
still in use.

Published: `carloscanamejoy/0{2,4,5,6}-*` notebooks and the dataset
`carloscanamejoy/nanodisk-timing-results`.

---

## 5. Work in flight

**The 200-point CPU ladder at 100 chains**, tag `cpu_ladder_100`, on
`<account-a>`. At the time of writing it had resumed at block 6 of 20 with
16.61 h already spent; it needs roughly 36 h more and about $15.

Driver: `scratchpad/cpu_driver2.sh`, a loop of `modal run` invocations capped at
6 h each. The cap matters — an earlier 20 h invocation died with
`AuthError: Jwt is expired`, because a long detached session outlives its token.

Resuming it elsewhere means moving `cpu_ladder_100_ckpt.npz` between volumes:
`modal volume get` from the old workspace, `modal volume put` to the new. The
checkpoint carries `spins` (float64), `done`, `elapsed_s`, `obs` and `snaps`.

It is the last extrapolated figure in the study. Everything else is measured.

---

## 6. Settled results

- **Interfacial DMI**: in-plane bonds only, minus sign. All-axes with `+` changes
  the total energy by 50% on the same configuration.
- **Cubic anisotropy**: `K1` and `K2` on direction cosines, confirmed by the user
  and verified numerically to machine precision against `sim_core._an`.
- **Validation**: |M| 0.8318 against Prof. Darío's 0.8302 (0.19%), same skyrmion
  count. Three CPU implementations and the H100 agree on E/N to 0.28% over 200
  temperature points; the sequential colour-order arm agrees to 0.000%.
- **BP costs 0.98x in a scalar implementation and 2.02x in the vectorised one**
  that ships. Two passes over the lattice, each computing the whole array to use
  half. It is the price of correctness: the single-pass version is twice as fast
  and moves the topological charge by a whole unit.
- **MD gives 5.87x with BP in place and 0.0029x without** — applying the
  decomposition to a sequential chain makes it 348x slower. The two are not
  separable.
- **An H100 without MD is 438x slower than a CPU**, and more GPUs make it worse:
  each site becomes one dispatch per device.
- **Device count is free** at fixed per-device batch (0.9% over 1, 2, 4 GPUs).
  **Per-device batch has an optimum near 50 chains**, 27% penalty at 25 or 100.
- **Parallelisation changes speed by up to 8,700x and physics not at all.**

## 7. Open questions

- **Shells 2 and 4 break the colouring.** `(x+y+z) mod 2` is exact for shells 1
  and 3 only; with `J2` or `J4` non-zero there are bonds inside a sublattice.
  18.9% of the released dataset has `J2 != 0` and that bias is unmeasured. For
  the tutor.
- **The third factor in the batch scan** (14% gap between scan and ladder at the
  same B) is unisolated. Candidates: per-chain parameter arrays versus scalars,
  and JIT compilation inside the ladder's first timed call.
- **`app.py` converts parameters to scalars** in `_simulate`, so that path runs
  one theta per batch even though the engine supports per-chain parameters —
  verified. Generating a dataset needs that wrapper changed.
- **Parts 2 and 3 of the test plan are untouched**: the generative-model
  comparison and the remaining cycle-integration cells.

## 8. Methodological rules earned the hard way

- Extrapolating **along** the saturated regime is exact; interpolating **into**
  the unsaturated regime was wrong by 48%.
- Per-sweep rates separate orders of magnitude, not 10–20% differences.
- A tight error bar on the wrong quantity reads exactly like a result.
- Cold containers over-report by ~1.6x; the first repeat of any JIT-compiled cell
  carries compilation.
- In JAX, warm up with the **same** static argument values that will be timed.
- When a figure mixes arms measured under different configurations, the varying
  parameter belongs in each bar's label, not the title.
