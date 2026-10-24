# Results and Discussion

<!--
papersmith contract: sections/03-results-and-discussion.md
  extent          1170–4150 words of prose · two or three content blocks,
                  plus limitations last
  carries         5–14 figures and 0–4 tables
  does NOT carry  a preamble, a closing artefact, or a single numbered equation
  chain           problem i -> contribution i -> property i -> instrument i ->
                  evidence i, the property keeping the SAME NAME through all five
  per artefact    point at it -> read it in numbers -> give the mechanism

  SCOPE NOTE. This section reports the simulator of Section 3 and nothing else.
  The reference implementation appears only where correctness requires a
  reference to be correct against; it is never a performance baseline, and no
  intermediate version, abandoned attempt or development sequence appears here.
  Those belong to the repository's documentation.
-->

## The simulator samples the intended chain

Table 1 collects the four exactness tests and the three statistical ones, with
the number each returns. Fed identical randomness, the fused kernel of
Section 3.2 and the reference implementation of the chain agree on every spin of
the configuration (0 of 182520 differing, Table 1 row 1), with the single-site
energy difference matching to 1.4 × 10⁻⁶ in absolute value, which is the
resolution of single precision on that quantity. The observable kernels of
Section 3.4 agree with the reference observables on a common configuration to
1.2 × 10⁻⁷ relative for the energy and 8.4 × 10⁻⁸ absolute for the topological
charge (rows 3 and 4). These rows carry no statistics: with the randomness held
fixed the only admissible difference is arithmetic, so a non-zero spin count
would identify a term of equation (1) wrongly transcribed rather than a sampling
discrepancy.

**Table 1.** Agreement with the reference implementation of the chain. Rows 1–4
hold the randomness fixed and are exactness tests; rows 5–7 let each engine draw
its own and are statistical.

| # | test | quantity | result |
|---|---|---|---|
| 1 | one pass, even class, identical randomness | differing spins | 0 / 182520 |
| 2 | one pass, odd class, identical randomness | differing spins | 0 / 182520 |
| 3 | observable kernels, energy | relative | 1.2 × 10⁻⁷ |
| 4 | observable kernels, topological charge | absolute | 8.4 × 10⁻⁸ |
| 5 | both chains to equilibrium, T = 20, 5, 1 | σ | 0.09 – 0.23 |
| 6 | trial direction norm | deviation from S₀ | 2.4 × 10⁻⁷ |
| 7 | reduced curves, against the reference run against itself | mean σ / max σ | 0.99 / 1.90 |

Row 7 is the instrument that settles the question, and reading it requires the
control the row names. Running the reference implementation against *itself*
with a second seed returns 0.87 and 2.53 for the same two statistics, which is
the noise floor of the comparison rather than a property of either engine; the
fused kernel's 0.99 and 1.90 therefore sit inside that floor, its maximum
deviation falling below the control's own. Without that control a figure in this
row is not interpretable in either direction, because the two engines draw from
different generators and are therefore distinct Markov chains that can agree
only in distribution.

The admissibility condition of equation (7) is checked computationally for the
colouring in use rather than argued from the table in Section 2. Over the
displacement sets of the active shells, the two-class colouring is admissible
for shells {1} and {1,3}, the eight-class colouring for any set containing
shell 2, and the twenty-seven-class colouring for any set containing shell 4;
in every case the colour classes partition the sites inside the disc exactly.
Because the no-barrier guarantee of Section 3.2 follows from that condition and
not from the class count, it holds unchanged at eight and twenty-seven classes.

## Each component delivers the property it was defined under

### Contiguous neighbour access and the synchronisation-free halo

Table 3 reports the per-site update time the two components jointly achieve:
0.129 ns on one datacentre GPU and 0.033 ns aggregated over four (Table 3, rows
1 and 2). At the geometry of Section 1 that is 7.74 site updates per nanosecond
per device. The mechanism is the one the layout was defined for: the weighted
neighbour sum of equation (5) is formed from contiguous segments displaced by
one element, so each translate costs one coalesced transaction per segment
rather than a gather, and no translate is ever written to memory.

### Counter-based proposals

Table 2 presents the acceptance rate of the in-kernel generator against the
reference generator at five temperatures spanning the ladder. The ratio scatters
on both sides of unity (0.996 to 1.029, Table 2 column 4), each value within one
standard error of it, where the standard error of a ratio of two independently
measured acceptance rates is √2·√((1−p)/(N p)) and runs from 0.47 % at the
hottest temperature to 3.4 % at the coldest, the acceptance rate p falling by
two orders of magnitude across that range. The absence of a systematic sign
**confirms** the independence that Section 3.3 requires between the acceptance
variate and the trial direction: a shared input between them would bias the
test in one direction at every temperature, which is the signature this
instrument is built to detect.

**Table 2.** Acceptance rate of the in-kernel generator against the reference
generator, from a common configuration. The expected ratio is unity; the
standard error grows as the acceptance rate falls.

| T (K) | reference | in-kernel | ratio | std. error of the ratio |
|---|---|---|---|---|
| 20.0 | 0.4233 | 0.4218 | 0.996 | 0.47 % |
| 10.0 | 0.1614 | 0.1611 | 0.998 | 0.97 % |
| 5.0 | 0.0723 | 0.0718 | 0.994 | 1.5 % |
| 1.0 | 0.0132 | 0.0136 | 1.029 | 3.4 % |
| 0.1 | 0.0029 | 0.0029 | 0.997 | 7.3 % |

### Deterministic observables

Rows 3 and 4 of Table 1 are also this component's evidence, and they are
available *as* exactness tests only because the reduction is order-independent:
a reduction whose result depended on the order in which blocks retired could be
compared only within a tolerance, and the figures of 1.2 × 10⁻⁷ and 8.4 × 10⁻⁸
would carry no information about the formulation. That the comparison is
admissible at all is **attributed to** the partial-sum reduction of Section 3.4
rather than inferred from the agreement.

## The simulator at production scale

Figure 2 is the principal result. It presents the wall clock of the complete
production ladder — 200 temperatures by 20 000 sweeps, so 4 × 10⁶ sweeps per
replica — for one hundred states at five thousand replicas, across the CPU
baseline, one and two consumer GPUs and one, two and four datacentre GPUs.

Read at four datacentre GPUs (Figure 2, rightmost bar), the simulator completes
the hundred-state ladder in 57.6 minutes; at one device (Figure 2, fourth bar)
in 3.8 hours; and on two consumer GPUs (Figure 2, third bar) in 25.8 hours.
Against the CPU baseline on the identical workload, 130.7 days, the separation
is 834-fold per site update (Table 3, rows 1 and 3). Figure 3 shows the same
comparison without the CPU bar and on a linear axis, since at 130.7 days the
baseline compresses everything else onto the axis.

The hundred-state configuration is the one that exhibits the scaling, which is
why it is the principal figure rather than the single-state case. Over one to
four devices the simulator gains a factor of 3.91 with a hundred states and only
1.10 with one, on the same hardware and the same kernel. This is **attributed
to** the replicas available per device rather than inferred from the two
figures: at one state and four devices each device receives twelve replicas, and
Figure 4 shows the per-site cost against replicas per device, where the curve
rises steeply below roughly twenty-five. The practical consequence is that
additional devices are paid for by additional states — one state per device
requires no sharding and preserves the per-state cost — and not by dividing one
state across them.

Figure 5 presents the per-site cost against the replica count at fixed geometry
and shows it very nearly insensitive to the arithmetic form of the array extent.
Let v₂ be the 2-adic valuation, B the replicas per device and L the layer count.
The condition v₂(B) + v₂(L) ≥ 2 can be flipped by varying B only when v₂(L) < 2,
which among the layer counts measured holds for L = 5 and L = 6 and not for
L = 4 or L = 8, where the condition is already satisfied for every B. Across the
two informative layer counts the condition changes the cost by 0.30 % and
1.70 % on the datacentre device, and by 0.56 % and 0.97 % on the consumer
device — at most 1.7 %, with no consistent sign. This is **attributed to** the formulation: the block shape is fixed by the
kernel of Section 3.2 and the layer count only sets how many blocks exist along
the out-of-plane axis, so no code-generation decision depends on the extent.
The practical consequence is that the replica count may be chosen for the
physics or for the scheduling.

The margin is not uniform across devices, and Figure 4 shows the sign. On the
datacentre device the per-site cost improves as the batch widens and then
plateaus at 0.677 µs per replica-sweep; on the consumer device it degrades
monotonically, from 7.282 to 9.201 µs over the same range. This **suggests** a
memory-hierarchy origin — the consumer device has an order of magnitude less
last-level cache and a third of the bandwidth, against a working set of 456 MB
at five thousand replicas — but that mechanism was not measured here, and only
the sign is established.

### Against published implementations

Table 3 places the simulator beside published GPU Monte Carlo spin codes on a
common unit, the time per single-site update.

**Table 3.** Time per single-site update, per device. This work's rows are all
taken at five thousand replicas per device, so they share a basis. Every
published figure is read from the paper itself, not from a secondary source,
and the rightmost column states the per-site arithmetic each model carries —
which is what decides whether a comparison in this unit means anything.

| implementation | model | device | ns / update | per-site work |
|---|---|---|---|---|
| **this work** | Heisenberg + iDMI + cubic + surface + field | H100 | **0.129** | heavy |
| this work | same | T4 | 1.75 | heavy |
| this work, CPU baseline | same | 24 threads | 107.7 | heavy |
| Charilaou [1] | 3-D anisotropic Heisenberg | Tesla V100 | 1.55 | moderate |
| Block, Virnau & Preis [2] | 2-D Ising, multi-spin coding | Tesla C1060 / GTX 480 | 0.127 | minimal |
| Weigel & Yavors'kii [3] | lattice spin models, $k=100$ systems | GTX 480 | 0.03 | minimal |
| Bisson et al. [4] | 2-D Ising | GB200 | 5.5 × 10⁻⁴ | minimal |

The aggregate figures those works report are larger again and are not in the
table, because they measure a different thing: 206 updates per nanosecond over
sixty-four devices [2] and 1.15 × 10⁵ over a rack [4], against this work's
0.033 ns per update aggregated over four devices. A per-device row is the only
one that compares.

Against the published implementation whose Hamiltonian is closest — an
anisotropic Heisenberg model, the only entry carrying continuous spins and an
anisotropy — the raw ratio is twelvefold (0.129 against 1.55 ns, Table 3 rows 1
and 4). That ratio should not be quoted on its own, because the two rows are
measured on hardware two generations apart: the published figure is from a
Tesla V100 and this work's from an H100.

The comparison that controls for the device is between row 4 and row 2. This
work's consumer-class device returns 1.75 ns against the V100's 1.55, and a
V100 carries twice the multiprocessors and nearly three times the memory
bandwidth of that device. Normalising on that basis, the formulation is ahead
by something between a factor of two and three rather than twelve — smaller than
the raw ratio and attributable to the method rather than to the hardware.

The Ising entries are faster per update, by up to two orders of magnitude on a
per-device basis, and that is not evidence about this work in either
direction. The per-site
arithmetic differs by roughly two orders of magnitude: a bit-packed Ising flip
is a handful of integer operations with no transcendental and one random number,
while one update here performs on the order of 290 arithmetic operations, one
exponential, one reciprocal square root and four random draws, as the kernel of
Section 3.2 makes explicit. Reading Table 3 as a ranking would be reading a unit
the models do not share.

What the table supports, read across rows 1 and 5, is narrower and more useful:
the simulator reaches the per-device rate of a multi-spin coded Ising
implementation on 2010 hardware — 0.129 against 0.127 ns — while carrying a
Hamiltonian some two orders of magnitude heavier per site. The comparison that
is strictly controlled remains the one against the CPU baseline, where the code,
the physics and the workload are identical, and there the figure is 834-fold.

**References for Table 3.** [1] M. Charilaou, *GPU-accelerated Monte Carlo
simulations of anisotropic Heisenberg ferromagnets*, arXiv:2106.11286. [2] B.
Block, P. Virnau and T. Preis, *Multi-GPU accelerated multi-spin Monte Carlo
simulations of the 2D Ising model*, arXiv:1007.3726. [3] M. Weigel and T.
Yavors'kii, *GPU accelerated Monte Carlo simulations of lattice spin models*,
arXiv:1107.5463. [4] M. Bisson, M. Bernaschi, M. Fatica et al.,
*Massive-scale simulations of 2D Ising and Blume-Capel models on rack-scale
multi-GPU systems*, arXiv:2502.18624.

## Limitations

<!-- Developed under sections/04-limitations.md; pointed at from here. -->

Four limitations bound the claims above. The per-device replica floor, below
which the per-site cost rises steeply (Figure 4). The architecture dependence of
the margin, whose sign is measured and whose mechanism is not. The widening of
the colouring that shells 2 and 4 require, which divides the parallelism
available per pass by eight or twenty-seven while leaving the work per sweep
unchanged, and which the fused kernel of Section 3.2 does not yet implement
beyond the first shell. And the single geometry in which the cost behaviour of
Figure 5 was characterised. They are developed in Section 4.
