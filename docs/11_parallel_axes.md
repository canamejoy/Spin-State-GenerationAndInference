# Where to put the parallelism

Part 1 of the test plan asked what each acceleration contributes. This asks the
next question: with the algorithm fixed and correct, **where should the parallel
hardware go** — over replicas (seeds), over states (parameter sets), or both.

The bipartite update is mandatory and is on in every configuration here.

**Fixed configuration:** 4 states x 100 replicas = 400 independent chains,
annealed over 200 temperature points x (10,000 + 5,000) sweeps. All groupings
move that same total work; only the arrangement changes.

Notebooks: `05_parallel_axes.ipynb` (timings), `06_physics_across_runs.ipynb`
(the observables those runs produced). Raw results in
`notebooks/simulation/results/`.

---

## The four groupings

| mode | grouping | axis used |
|---|---|---|
| BP only | 400 launches of 1 chain | the bipartite update alone |
| BP + replicas | 4 launches of 100 chains | batch over seeds |
| BP + states | 100 launches of 4 chains | batch over parameter sets |
| BP + replicas + states | 1 launch of 400 chains | both |

The naming follows the three levels in the shared notebooks
(`migarciaq/mc-gpu-spines`, `mc-gpu-replicas`, `mc-gpu-experiments`): replicas
batch with `vmap` inside a device, states distribute with `pmap` across devices.

The four states are the medians of four phase clusters in the released dataset —
`uniform`, `helical`, `points` (skyrmions), `lines` (labyrinth). All carry
`Jex2 = Jex3 = Jex4 = 0`, so the bipartite colouring is exact for each;
`parallel_axes.load_thetas` asserts it rather than trusting it.

---

## Total execution time

Six cells measured end to end (`measured_full_ladder: True` in their result
files); the other ten extrapolated from a measured per-sweep rate.

| mode | CPU | 1x H100 | 2x H100 | 4x H100 |
|---|---|---|---|---|
| BP only | 79.4 d *ext* | 5.8 h *ext* | 5.8 h *ext* | 5.8 h *ext* |
| **BP + replicas** | 10.3 d *ext* | **51.7 min** | **27.6 min** | **16.9 min** |
| BP + states | 45.4 d *ext* | 1.8 h *ext* | 1.8 h *ext* | 1.7 h *ext* |
| **BP + replicas + states** | 7.6 d *ext* | **1.0 h** | **30.1 min** | **16.4 min** |

Scaling 1 -> 4 devices: BP only **0.99x**, BP + states **1.06x**,
BP + replicas **3.06x**, BP + replicas + states **3.80x** (ideal 4.00x).

**Without a batch axis, extra GPUs do nothing.** `BP only` cannot shard a single
chain, so three of four devices idle and are billed; its penalty against the best
mode grows from 6.7x on one GPU to 21x on four. `BP + states` is barely better:
four chains do not fill an H100, and on four devices that is one chain each.

**The grouping that looks natural for dataset generation is the second worst.** A
dataset varies the state, not the seed, so `BP + states` is the instinct; batch
many states at once instead.

---

## A claim this study published and retracted

An earlier version reported that `BP + replicas` beat `BP + replicas + states` on
2 and 4 GPUs by 18.6% and 13.8%, and called it real at **154 and 301 sigma** on
three repeats inside one container.

Measuring the full ladders overturned it: on four GPUs the second is now faster
(16.4 against 16.9 min), and the superlinear `4.69x` scaling that prompted a
cache hypothesis does not exist — measured, 3.06x.

**The sigma figure was not wrong; it measured the wrong quantity.** Three repeats
of the same 2,000-sweep loop agree to 0.0001 ms/sweep, so the *rate* was pinned.
But that loop is the bare spin update, while a production ladder also computes
every observable on every measurement sweep. A tight error bar on the wrong
quantity reads exactly like a result.

### How far off the extrapolations were

| cell | extrapolated | measured | error |
|---|---|---|---|
| replicas 1x H100 | 61.4 min | 51.7 min | **+18.7%** |
| replicas 2x H100 | 22.1 min | 27.6 min | −19.9% |
| replicas 4x H100 | 13.1 min | 16.9 min | −22.5% |
| both 1x H100 | 54.1 min | 62.1 min | −13.0% |
| both 2x H100 | 27.1 min | 30.1 min | −9.9% |
| both 4x H100 | 15.2 min | 16.4 min | −7.3% |

Mean |error| 15.2%, range −22.5% to +18.7%. Two effects pull opposite ways, so
no correction factor exists: missing observables push the estimate **low**
(+8.2%, +15.3%, +19.1% overhead on 1, 2, 4 GPUs, measured separately), while
fixed per-launch cost pushes it **high** — the rate study ran 2,000-sweep
batches, a real ladder 750,000 per launch. `replicas 1x` is where the second wins
outright, and it is the one cell that came out high.

**Standing rule:** per-sweep rates extrapolate well enough to separate orders of
magnitude and not well enough to rank configurations differing by 10–20%.

---

## Isolating batch size from device count

The six ladders showed the same per-device batch (B=100) costing 2.586
us/chain-sweep on one GPU and 3.272 on four, which looked like a multi-device
cost. Two scans, each inside one container, three repeats per cell, full
200-point ladders. The first repeat of every cell is discarded: it carries JIT
compilation.

**Device count, at fixed per-device batch B=25:**

| GPUs | chains | device-us per chain-sweep |
|---|---|---|
| 1 | 25 | 2.826 |
| 2 | 50 | 2.800 |
| 4 | 100 | 2.807 |

Spread **0.9%**. Coordinating four devices is free. Replicas are independent
chains with no collective, and `pmap` adds nothing measurable. **The 27%
attributed to device count was not device count.**

**Batch size, at fixed device count of 1:**

| B | device-us per chain-sweep | sd | vs best |
|---|---|---|---|
| 25 | 2.826 | 0.0010 | 1.22x |
| **50** | **2.323** | 0.0020 | **1.00x** |
| 100 | 2.960 | 0.0020 | 1.27x |

A clear minimum near **B = 50**, with **27% penalty at the extremes**. The
standard deviations are 0.002, so this is signal.

**Practical recommendation: aim for about 50 chains per GPU.** It explains why
`BP + replicas` on four GPUs (B=25) underperformed its own scaling.

### Testing the recommendation in production

The optimum above came from a scan of the bare update loop, so it was a
prediction, not a measurement of a real run. The `pairs` configuration tests it:
the four states are split into **two launches of two states x 100 replicas**, so
each launch carries 200 chains, which on four devices is exactly **50 per
device**. Everything else is the standard ladder -- 200 temperatures, 15,000
sweeps, 100 replicas, 400 states total.

| config (4x H100) | chains/device | wall clock | ladder us per chain-sweep |
|---|---|---|---|
| `replicas` | 25 | 16.87 min | 3.373 |
| **`pairs`** | **50** | **13.96 min** | **2.792** |
| `both` | 100 | 16.36 min | 3.272 |

`pairs` is the fastest four-GPU configuration measured: **15% under `both` and
17% under `replicas`.** Nothing changed but how 400 chains were grouped.

The three ladder points reproduce the scan's shape. Minimum at B=50 in both,
with the extremes 1.21x and 1.17x above it in the ladder against 1.22x and 1.27x
in the scan. The scan's *ranking* transfers; its absolute values do not. Every
ladder point sits 11-20% above its scan counterpart, which is the same offset
already recorded below -- the ladder also computes observables, the scan timed
only the update. The prediction of 11.6 min was 20% optimistic for that reason,
and the direction was right.

This is the one place in the study where a cheap measurement predicted a
production result before it was run, and it is worth being precise about what it
predicted: the ordering, not the number.

### The 26.5% at B=100, settled

`replicas 1x` read 2.586 us/chain-sweep and `both 4x` read 3.272 at the same
per-device batch -- 26.5% apart, larger than the effect being optimised. Seven
candidates, each measured inside one container rather than argued:

| candidate | effect | verdict |
|---|---|---|
| pmap over 4 devices vs jit over 1 | -0.2% | not it |
| 4 sequential launches vs 1 | +0.9% | not it |
| different container AND different account | +0.3% | not it |
| scalar params vs per-chain columns | **+12.9%** | real, widens the gap |
| schedule length, n_temps 40 vs 200 | -4.2% | not it |
| parameter values, ours vs theta4.json's | -0.2% | not it |
| replaying `_axis_ladder` itself | +27 s per launch | the harness, below |

A direct replication of the ladder's exact configuration -- per-chain columns,
the real thetas, one GPU, jit, B=100, 200 temperatures -- costs **3.174 us**.
Replaying `_axis_ladder`'s own code at 40 temperatures costs 3.63 us per launch,
which is that 3.17 plus about 27 s of compilation. The code is self-consistent
and reproduces the direct measurement.

**The 2.586 does not reproduce.** Scaled to 200 temperatures the ladder's own
code needs 65.2 min; the original run reported 51.72, which is 21% faster than
the code can run. No design candidate accounts for it, so it is recorded as an
irreproducible measurement rather than explained.

**What this settles:** there is no multi-device cost. pmap over four devices
costs -0.2% against jit over one at B=100, which agrees with the 0.9% spread
found at B=25. The 27% that looked like device count was never device count,
and the remainder was one bad number.

**Two findings that outlive the question:**

1. **Parameter form costs 12.9%.** Per-chain columns against scalars, same
   container. `anneal` builds `K1_eff` from whatever it is handed, so columns
   make that tensor (B,L,N,N,1) instead of (L,N,N,1). Any `f(B)` curve meant to
   predict a ladder must be measured in the production column form.
2. **pmap on a single device costs +3.6%**, slower than jit on one device and
   than pmap on two or four. Production never takes that path, since `run_one`
   picks jit when the batch does not span devices.

### The harness timed its own compilation

`run_one` started `t0` before its first call, so each launch charged XLA
compilation -- about 27 s -- to the kernel. This is not a constant offset: it
scales with the launch count, so it taxed `replicas` (4 launches) four times as
hard as `both` (1). Now fixed with `.lower(...).compile()`, which compiles the
exact shapes that will be timed and does no work. A warm-up call on a shorter
schedule would NOT have worked: `betas` sets the length of the temperature axis,
so a shorter one is a different shape and compiles a second executable, leaving
the timed call cold.

Correcting the seven existing ladders for the bias, at 27.4 s per launch:

| config (4x H100) | launches | raw | less compilation |
|---|---|---|---|
| `replicas` | 4 | 16.87 min | 15.04 min |
| **`pairs`** | 2 | 13.96 min | **13.05 min** |
| `both` | 1 | 16.36 min | 15.90 min |

The bias favoured `both` and penalised `replicas`, so removing it **widens**
`pairs`' margin over `both` from 15% to 17.9%. The ranking is unchanged and the
batch-optimum result stands. The absolute wall clocks in the cost table above
are the raw figures and each carries roughly 27 s per launch of compilation.

So the batch *shape* (optimum near 50, ±27%) and the device-count result (free)
stand; the absolute scan values should not be quoted as the ladder cost.

---

## Cost

H100 at **$0.001097 per second** (modal.com/pricing, verified).

| run | GPUs | wall clock | GPU-seconds | USD |
|---|---|---|---|---|
| replicas 1x | 1 | 51.7 min | 3,103 | $3.40 |
| replicas 2x | 2 | 27.6 min | 3,307 | $3.63 |
| replicas 4x | 4 | 16.9 min | 4,050 | $4.44 |
| both 1x | 1 | 62.1 min | 3,727 | $4.09 |
| both 2x | 2 | 30.1 min | 3,612 | $3.96 |
| both 4x | 4 | 16.4 min | 3,926 | $4.31 |
| pairs 4x | 4 | 14.0 min | 3,350 | $3.68 |
| **total** | | | **25,076** | **$27.51** |

**Wall clock falls 3x while cost rises 30%.** From `replicas 1x` to `replicas 4x`:
51.7 to 16.9 minutes, $3.40 to $4.44. The 24% short of ideal scaling is paid in
money. If the deadline does not press, one GPU is cheapest.

The seven runs produced **2,800 annealed states** with full observables for
$27.51 — **$0.0098 per state**. `pairs 4x` is both the fastest four-GPU run and
the cheapest of the three, at $3.68: the batch optimum buys wall clock and money
at the same time, because the GPUs are rented by the second either way.

The batch/device scans cost a further ~$6.50.

---

## Physics

`06_physics_across_runs.ipynb` reads back what these runs produced. The six
groupings differ only in how the chains were handed to the hardware, so their
curves should coincide. Worst-case spread across the six, per state:

| state | E | M | Cv | chi | Q |
|---|---|---|---|---|---|
| uniform | 0.02% | 0.02% | 6.1% | 3.3% | 5.1% |
| helical | 0.07% | 1.8% | 8.2% | 13.2% | 15.8% |
| points | 0.04% | 0.29% | 7.2% | 18.7% | 2.5% |
| lines | 0.05% | 9.5% | 8.0% | 25.9% | 34.7% |

**Energy agrees to 0.07% in all four states.** `Cv` and `chi` scatter more
because they are variances estimated from 100 replicas — Monte Carlo noise, not
disagreement between methods. `lines` is the most scattered because `Hex = 0.022`
leaves it nearly degenerate, so independent chains fall into different
labyrinths: the same Hamiltonian, different local minima.

**Parallelisation changes the speed by up to 8,700x and the physics not at all.**
That separation is what makes the speed result defensible.

### One correction those curves needed

The reduction saved with each run averages over all 400 chains, and in the
`repl+states` grouping those carry four different parameter sets — an average
across states is not a physical quantity. The published curves are recomputed
from the raw per-chain accumulators, sliced into four 100-chain blocks. The chain
ordering is identical in both groupings, so one slice expression serves all six.
