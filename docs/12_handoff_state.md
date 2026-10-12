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
- **The batch penalty is XLA's vector width, not the hardware.** XLA launches a
  fixed 128-thread block and gives each thread `min(4, 2^v2(elements))` elements,
  where `elements = B * L * N^2 * 3`. `N` is always odd, so
  `v2(elements) = v2(B) + v2(L)` and only the 4-wide path is expensive. The
  penalty condition is therefore `v2(B) + v2(L) >= 2`: odd `L` suffers at
  multiples of four, `L = 2 mod 4` at **every even** `B`, and `L` divisible by
  four is **immune**. Predicted 89 grid values across three `--xla_dump_to`
  dumps with zero misses. Measured +14.7% (no overlap) against +10.2% (with
  overlap for the `B % 4` rule) at `L = 6`. Ruled out first, each at the same
  shape, stride, topology and parity: raw CUDA without XLA (-0.1%), sibling
  buffer alignment (-0.1% to +0.4%), the RNG (-0.2%), XLA padding (0.000%), and
  the host-to-device path (arguments are already device arrays, and traced GPU
  time rises +15.3% in step with wall clock). Bandwidth was refuted by an H200
  control: same 132 SMs, 1.43x the bandwidth, penalty did not fall.

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
- **Width 4 costs because XLA gives each thread CONSECUTIVE elements.** A raw
  CUDA elementwise kernel at XLA's own 128-thread block, identical total work:
  blocked VEC=4 is +14.9% against VEC=1 (1699 GB/s) while strided VEC=4 is
  -45.3% (3571 GB/s). The JAX kernel pays +14.25%. One warp-wide load blocked
  spans VEC*128 bytes and needs VEC transactions for 128 useful bytes; widening
  also cuts blocks and raises ILP, which wins at 2 and loses at 4. So
  "compiler, not hardware" was a false dichotomy: XLA picks the width and the
  layout, the memory system prices it. Confirmed on a second architecture — the
  crossover moves with bandwidth, width 2 costing +6.8% on a T4 against +0.09%
  on an H100. Caveat: the microbenchmark matches the width-4 magnitude but not
  the width-2 cell (-27.2% there).
- **Immunity is insurance with a device-dependent premium, and the blanket
  advice was wrong.** `L` divisible by four means paying width 4 in every cell,
  and width 4 hurts most where bandwidth is scarce. Footprints matched to 0.1%:
  on an H100 it costs +5.6% against hitting a good odd-`L` cell and saves 7.5%
  against missing; on a T4 it costs **+30.7%** and saves 1.6%. Revised: if you
  can choose `B`, take an odd `L` and keep `v2(B) + v2(L) < 2`; take `L`
  divisible by four only when `B` is out of your hands. Also, "immune" is not
  "flat" — the eight `L = 4` cells span 16% on an H100 and 9% on a T4.
- **The geometry half of the rule now holds on two architectures, with a
  residue.** On a T4, going from `L = 5` to `L = 4` cuts the mod-4 penalty from
  +25.7% to +4.0%, an 84% reduction. But that +4.0% separates cleanly and the
  width cap of 4 cannot explain it, since all eight cells are width 4. The gap
  is marginal (0.052 us, ~0.25%, about 3 sd) and survives dropping B=64.
  Hypothesis: on `sm_75` XLA takes a path wider than 4 at higher `v2`, which
  would also connect to the unexplained mod-8 structure. Needs
  `--xla_dump_to` on a T4.
- **The ~2 MB footprint threshold is NOT an L2-capacity effect.** The five-device
  campaign (T4, L4, A100, H100, H200) only established that the penalty exists
  at multiples of four with `L = 5`, which is odd — the special case where the
  `B % 4` rule and `v2(B) + v2(L) >= 2` coincide. Every measurement of the layer
  dependence (the eight-lattice `v2` sweeps, the compiler dumps, the `L = 6`
  parity test) ran on an H100 only. XLA's codegen consults the target compute
  capability, so a different architecture may pick different widths. **The cheap
  test: the dense window at `L = 4` against `L = 5` on a T4**, which has the
  largest penalty and so the clearest signal. Untested.
- **The broad f(B) sawtooth still has no mechanism.** The vector width explains
  the step between *adjacent* batch sizes and the geometric immunity, and
  nothing more: at the reference lattice (`L = 5`, odd) all fourteen
  batches divisible by four are 4-wide, including `B = 512`, the global minimum,
  and `B = 768`, among the worst. Spread inside that group is 1.76x, larger than
  the penalty, and 4-wide against the rest is +2.7% in one measurement and
  -0.7% in the other. Why 512 wins is unexplained.
- **Why 4-wide costs and 2-wide does not** is untested. Four floats is 16 B per
  thread, so transaction size is the obvious suspect. Measured on `L = 3`, which
  spans all three widths in one lattice: width 2 is +0.09% against width 1,
  width 4 is +14.25%.
  XLA allocates 0.676 MB per replica (7.8x the spin array), so at B=60 arm A is
  16.4 MB and arm B 40.6 MB — both inside the H100's 50 MB L2, yet only arm B
  pays. The vector-width microbenchmark touches 15.7 MB, comparable to arm A,
  and pays anyway. Still unexplained, but better posed.

## 8. Methodological rules earned the hard way

- **Fusion names are not stable across batch sizes.** In the XLA dumps
  `loop_select_fusion_2` walks the full `(B, L, N, N, 3)` array at `B = 58` and
  the per-site `(B, L, N, N)` arrays elsewhere, and
  `loop_maximum_negate_fusion_1` exists only at `B = 60`. Scoring launch
  geometry per kernel name reported two false misses; score it per entry,
  against every array shape the module walks.
- Extrapolating **along** the saturated regime is exact; interpolating **into**
  the unsaturated regime was wrong by 48%.
- Per-sweep rates separate orders of magnitude, not 10–20% differences.
- A tight error bar on the wrong quantity reads exactly like a result.
- Cold containers over-report by ~1.6x; the first repeat of any JIT-compiled cell
  carries compilation.
- In JAX, warm up with the **same** static argument values that will be timed.
- When a figure mixes arms measured under different configurations, the varying
  parameter belongs in each bar's label, not the title.
