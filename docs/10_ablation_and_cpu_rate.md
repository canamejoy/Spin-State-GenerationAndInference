# Ablation `MD x BP`, and the CPU rate

Part 1 of the test plan asked what each of the two accelerations contributes, and
what the CPU arm costs so the unaffordable cells can be extrapolated instead of
run. Both are now measured.

## Reference and hardware, stated once

**Every speedup in this document is quoted against reference B.**

| | definition | ms per replica-sweep |
|---|---|---|
| **B — the reference** | no MD, no BP, numpy, **100 replicas**, CPU | **3.341** |
| A — declared separately | no MD, no BP, pure Python, **1 replica**, no JAX | 12.771 |

B is the baseline because it is the same 100-replica configuration as every other
arm, so the comparison is like for like; a ratio against a differently sized run
is not a speedup, it is two experiments. A is 3.8x slower *per replica* than B
for a reason unrelated to either acceleration under study — B's replica axis is
already an array dimension — so quoting against A would inflate every factor by
3.8x against a program nobody ran. Headline against B: **1,203x on one H100,
4,960x on four**; against A those become 4,599x and 18,959x, and must be labelled
as such wherever they appear.

Note also the notation: **MD** is the matrix decomposition and **BP** the
bipartite (checkerboard) update. Earlier drafts wrote `DP` for the matrix decomposition; that spelling is retired.

**Hardware.** Modal, container `cpu=8.0`, x86-64 **GenuineIntel**, 24 visible
vCPUs, 144 GB, gVisor sandbox (Linux 4.19). GPU arms on **NVIDIA H100 80GB
HBM3**, driver 580.95.05, compute capability 9.0, PCIe gen 5.

Two things about that hardware have to be said rather than glossed:

- **The CPU model name is not recoverable.** `lscpu` and `/proc/cpuinfo` both
  report `unknown`, because Modal runs containers under gVisor, which masks it.
  This is the environment's answer, not a parsing failure, and the honest course
  is to report the vendor and the sandbox and leave the model blank.
- **`cpu=8.0` limits CPU *time*, not *visibility*.** The container sees 24 vCPUs,
  so numpy and JAX size their thread pools to 24 while being throttled to eight
  cores' worth of time. That is a likely contributor to the ~20% scatter between
  containers documented below.

---

Everything else in this document was measured on **one Modal container**
(numpy 2.4.6, JAX 0.10.2), at the production geometry (`Rd=18.30`, `L=5`,
`N=39`, 5,245 active sites) and the validated skyrmion parameters (`Jex=1.0`, `Kan1=0.177`, `Kan2=0.034`, `KanS=0.139`,
`Hex=0.483`, `KDM=0.880`, `gamma=90`). The GPU rows come from the H100 runs of
the scaling study. Nothing was measured on a laptop: a ratio between a local
machine under WSL2 and a cloud H100 is not a quantity anybody can reproduce.

Reproduce with `notebooks/simulation/modal/`:

    modal run app.py::ablation_run --tag ablation_8core_v2 --repeats 3
    modal run app.py::rate --config cfg_A_200T.json --tag cpu_rate_8core \
        --n-temps 4 --n-therm 60 --n-meas 40 --reps 1,25,50,100,200

Both write JSON to the `nanodisk-out` volume. `run_ablation.py` runs the same
measurement locally for development.

---

## What is being ablated

- **MD**, matrix decomposition: the trial energy of many sites as array
  operations instead of arithmetic on one site at a time.
- **BP**, the checkerboard: colouring sites by `(x+y+z) mod 2` so that no two
  sites updated together are neighbours.

Every variant evaluates the **same Hamiltonian** as `sim_core` -- same
interfacial DMI, both cubic-anisotropy constants, the surface constant.
`tests/test_ablation_equivalence.py` pins the un-vectorised arm's single-site
`dE` against the reference engine's total-energy difference at interior and
surface sites before any timing is believed, because an ablation across two
different Hamiltonians measures nothing.

---

## Result 1 -- what the bipartite update costs depends on the implementation

| cell | loop | ms/sweep (1 replica) | us/site |
|---|---|---|---|
| `MD + BP` | one array update per colour | **1.509 ± 0.003** | 0.288 |
| `no MD, BP` | scalar, colour by colour | 13.031 ± 0.158 | 2.484 |
| `no MD, no BP` | scalar, lattice order | 12.771 ± 0.079 | 2.435 |
| `MD, no BP` | array expressions, one site at a time | 796.98 ± 2.25 | 151.95 |

**This section originally read "the checkerboard contributes no speed at all",
on the strength of `no MD, no BP / no MD, BP = 0.98x`. That ratio is correct and
the conclusion drawn from it was too broad.**

Both arms of that comparison are *scalar*: they visit one site at a time, so the
ordering of the sites is all that changes and the cost cannot differ. The
comparison is blind to what the colouring costs a *vectorised* kernel, which is
the kernel that ships.

Measured on the production kernel at 100 chains, with the whole-disk single-pass
update as the comparator:

| implementation | with BP | without BP | cost of BP |
|---|---|---|---|
| scalar, site by site | 12.77 | 13.03 ms/sweep | **0.98x** |
| numpy, vectorised | 156.69 | 66.68 ms/sweep | **2.35x** |
| **JAX, the shipped kernel** | **65.16** | **32.23 ms/sweep** | **2.02x** |

The reason is in the loop: a bipartite sweep makes **two** passes over the
lattice, one per colour, and each pass computes `dE` for the whole array and then
discards the half it does not own. Same 5,245 attempts, twice the arithmetic.

So the correct statement is: **the colouring is free in a sequential
implementation and costs 2x in a vectorised one, and that 2x is the price of
correctness.** The single-pass version is twice as fast and samples the wrong
distribution -- it is the one that moves the topological charge by a whole unit.
It is still worth paying, because matrix decomposition returns far more than 2x
and is only legal because the colouring is there.

The original ratio, for the scalar arms, stands: `0.98x`. Within the run-to-run
scatter, switching the colouring on in a scalar loop costs nothing and saves
nothing: the same sites are visited, in a
different order. **In that implementation the checkerboard is not an
accelerator**; in the vectorised one it is a 2x tax paid to make the simultaneous
update legal. Either way its contribution is correctness (Result 3).

`md_given_bp = 8.64x`. With the colouring in place, matrix decomposition is the
entire speedup on this axis.

`MD, no BP` is the literal fourth cell -- keep the decomposition, drop the
colouring, stay sequential -- and it is **528x slower than the production
kernel and 62x slower than the plain scalar loop**. A sequential chain can only
accept one site at a time, so it builds a whole-lattice neighbour field and
discards all but one element of it. This is the cell that shows the two
accelerations are not separable: MD is a speedup *because* BP makes a
simultaneous update legal, and vectorising a sequential chain is a
pessimisation.

### The test plan's 65x was wrong, and so was its 10 us/site

`docs/09_test_plan.md` estimated a site-by-site attempt at 10.0 us and therefore
a 65x penalty for dropping MD. Both figures were inflated by per-sweep setup --
the neighbour table and the per-site anisotropy constant were being rebuilt
inside the timed loop, which a real implementation builds once. With that
hoisted out, the per-site cost is **2.44 us** and the penalty is **8.6x**.

The practical consequence is better than the estimate: every cell runs complete
lattice sweeps at the production geometry. No reduced configuration and no pro
rata scaling were needed anywhere in this ablation.

---

## Result 2 -- notebook 00's CPU mode already carries a decomposition

The reference notebook's `run_mode_cpu` is labelled *"CPU clásico
(site-by-site)"*, but its `get_neighbors` indexes `sp[:, z, y, (x+1) % Nx]`:
every numpy operation is spread across the whole replica batch. The
decomposition is already there, **on the replica axis rather than the lattice
axis**.

The two axes are not equivalent, and the difference is exactly the point of this
ablation. Replicas are independent chains, so batching over them needs no
colouring and raises no detailed-balance question whatsoever. The lattice axis is
the only one the checkerboard exists for.

| batch | ms/sweep | ms per replica-sweep |
|---|---|---|
| 1 | 277.5 | 277.5 |
| 25 | 322.0 | 12.88 |
| 100 | 401.0 | 4.010 |

At batch 1 this is 18x slower than the scalar loop, because a numpy call on a
`(1,3)` array is nearly all overhead. The batch axis is what makes it viable.

So the honest baseline for "what the group's CPU path costs" is **4.01 ms per
replica-sweep**, not the scalar loop's 12.77 -- and quoting the scalar number
would have compared the production kernel against something nobody ever ran.

---

## Result 3 -- dropping the colouring is wrong, and measurably so

Keeping the parallel update while dropping the colouring is the other reading of
`no BP`, and it is not a slower chain -- it is a different one. Two neighbours
each decide against the other's stale spin, so the joint move's energy is not
the sum of the individual ones.

That this is a different chain is proved exactly, not statistically:
`test_naive_parallel_update_is_not_the_sequential_chain` drives both loops with
identical proposals and identical uniforms and asserts they diverge.

How much it matters, at fixed temperature, 2,000 sweeps, 1,000 discarded:

| T | E/N exact | E/N uncoloured | ΔE | σ | abs M exact | abs M uncoloured | Q exact | Q uncoloured |
|---|---|---|---|---|---|---|---|---|
| 1.0 | −3.12592 | −3.12965 | −0.12% | 1.1 | 0.6867 | 0.7189 | −12.06 | −11.98 |
| 3.0 | −2.95843 | −2.96128 | −0.10% | 0.9 | 0.6819 | 0.6690 | −12.92 | −11.92 |
| 8.0 | −2.49552 | −2.47133 | +0.97% | 2.6 | 0.6246 | 0.6388 | −9.11 | −8.11 |

The energy bias is under 1% and only marginally significant. The informative
columns are the other two: `|M|` moves by 2-5%, and the Berg-Lüscher charge of
the middle layer shifts by **up to a full topological unit**. That last one is
the quantity the released dataset's state labels are built from, so an
"acceleration" that moved it would be changing the thing the paper reports.

**Caveat on magnitude.** These are fixed-temperature runs from a random start,
not the production annealing protocol, and `Q ≈ −12` is a high-charge disordered
state rather than the few-skyrmion states the dataset holds. The table
establishes that the deviation is real and gives its order of magnitude; it is
not a correction factor for annealed results.

---

## Result 4 -- the CPU rate, and the ladder

A 200-point ladder on CPU is days, so what is measured is the per-sweep cost and
the ladder is extrapolated from it. Cost is exactly linear in sweeps, so that is
arithmetic and not a fitted trend. The full production protocol (10,000 therm +
5,000 meas per temperature, with every observable) at 100 replicas:

| replicas | ms/sweep | us per replica-sweep |
|---|---|---|
| 1 | 7.43 | 7428 |
| 25 | 32.27 | 1291 |
| 50 | 66.60 | 1332 |
| 100 | 97.44 | 974 |
| 200 | 167.30 | 837 |

### The rate depends on how warm the container is, by 1.8x

Nine measurements at 100 replicas, all the same code and the same container
spec:

| | ms/sweep |
|---|---|
| four repeats inside one container, in order | 67.5, 63.6, 61.8, **56.5** |
| first measurement in four fresh containers | 101.6, 94.4, 79.0 |
| first measurement, earlier container | 97.4 |

Within a container the figure improves **monotonically** -- 67.5 down to 56.5,
a 16% gain over four repeats -- as frequency, page cache and allocator settle.
Every single-shot container measurement is therefore biased high, and the 2x
spread across containers is mostly that bias, not hardware variation.

Which number belongs in the extrapolation follows from what is being
extrapolated. The production ladder is **one call lasting tens of hours**, so it
runs warm essentially throughout. The right figure is the warm steady state,
**62.4 ± 4.0 ms/sweep**, and the cold figures are what a short benchmark
measures rather than what a long run costs.

This also retires an earlier number: 97.4 ms/sweep, quoted in the first pass of
this study, was a cold first measurement.

**The CPU does not saturate the way the H100 does.** On one H100 the per-replica
cost was flat at 2.92 ± 0.25 us from 25 replicas all the way to 400 -- the device
is already full at 25. On CPU the per-replica cost keeps falling, 7428 → 837 us
from 1 to 200 replicas. On a GPU a bigger batch buys nothing; on CPU it still
pays.

### The device table, at 200 T points x 15,000 sweeps = 3,000,000 sweeps

All rows at 100 replicas.

All rows at 100 replicas. The CPU row uses the warm rate; `± ` on the ladder
follows the ±4.0 ms/sweep scatter.

| path | ms/sweep | full ladder | vs CPU |
|---|---|---|---|
| **1x H100, no decomposition** (`gpu_sitewise`) | 121,151 | **11.5 years** | 0.003x |
| notebook 00 CPU mode (site-by-site, batched) | 401.0 | 334 h = 13.9 d | 0.16x |
| production kernel, JAX on CPU (warm) | 62.4 ± 4.0 | **52 ± 3 h** | 1.0x |
| 1x H100 | 0.303 | **15.2 min** | 206x |
| 2x H100 | 0.133 | 6.7 min | 469x |
| 4x H100 | 0.084 | 4.2 min | 743x |

Three CPU-ish rows because they answer three different questions. **52 h** is
the device comparison -- the same code on a different processor, and the figure
to quote for "how long would this take on CPU". **334 h** is what the group's own
CPU path would cost. The first row is Result 5.

---

## Result 5 -- a GPU without matrix decomposition is far worse than a CPU

The test plan's `no MD` column has GPU rows, and they are not slow versions of
the GPU runs. Measured on an H100 with the identical site-by-site loop and no
`jit` -- because fusing the sweep into one kernel *is* the decomposition:

| path | ms per replica-sweep |
|---|---|
| CPU, site-by-site, batched over replicas | 4.01 |
| **H100, site-by-site, batched over replicas** | **1,211.5** |

The H100 is **302x slower than the CPU** at the same algorithm. A sequential
chain hands the device a few dozen floats at a time and pays a dispatch and a
round trip for each of 5,245 sites; the CPU just does the arithmetic. One sweep
takes 121 seconds, and the 200-point ladder would take **11.5 years**.

So the accelerator and the decomposition are not independent choices either.
Without `MD` a GPU is not a slower GPU -- it is much worse than having no GPU at
all. That is why six of the test plan's sixteen cells are not worth running at
production scale: they are all this cell with more devices, and adding devices to
a sequential chain only adds latency.

---

## Declared limits

- **Warm-up, not hardware, drives the CPU scatter.** Within a container the rate
  improves 16% over four repeats; across containers the first measurement spans
  79-102 ms/sweep against a 62.4 ± 4.0 warm steady state. Quote the warm figure
  for long runs and the cold one for short benchmarks, and say which. The GPU
  arm showed ~15% scatter and was measured once per configuration, so the
  CPU/GPU ratios carry roughly ±20% and no difference smaller than that should
  be read as real.
- **The scalar arms are CPython.** 8.6x is the gain over a straightforward
  *Python* site-by-site loop, which is what the group actually has. A compiled
  sequential implementation (C, Fortran, numba) would narrow that gap
  substantially, and the paper should say "over a straightforward Python
  implementation" rather than "over a sequential implementation".
- **`MD + BP` here is numpy float64, not the shipped kernel.** The production
  kernel is JAX float32 and, on CPU, is *slower* than numpy at small batch
  (11.7 ms/sweep at 1 replica against 1.5). JAX's advantage is the GPU, not
  vectorisation per se. The ablation is reported in numpy so that the MD axis is
  the only thing changing across the four cells.
- **The checkerboard is exact only for shells 1 and 3.** Unchanged from
  `09_test_plan.md`: shells 2 and 4 connect same-parity sites, so with `J2` or
  `J4` non-zero the simultaneous update is not an exact reordering of sequential
  Metropolis. These runs have `J2=J3=J4=0`; 18.9% of the released dataset does
  not, and that bias is still unmeasured. Open question for the tutor.
