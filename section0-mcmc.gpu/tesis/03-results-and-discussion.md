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
-->

## The simulator reproduces the reference chain exactly

Table 1 collects the two exactness tests and the two statistical ones, with the
number each returns. Fed identical randomness, the fused kernel of Section 3.2
and the reference implementation agree on every spin of the configuration
(0 of 182520 differing, Table 1 row 1), with the single-site energy difference
matching to 1.4 × 10⁻⁶ in absolute value — the resolution of single precision
on that quantity. The observable kernels of Section 3.4 agree with the
reference observables on a common configuration to 1.2 × 10⁻⁷ in relative terms
for the energy and 8.4 × 10⁻⁸ in absolute terms for the topological charge
(Table 1 rows 3 and 4). These two tests carry no statistics at all: with the
randomness held fixed the only admissible difference is arithmetic, so a
non-zero spin count would indicate a term of the Hamiltonian wrongly
transcribed rather than a sampling discrepancy.

The statistical tests compare the two engines running each from its own
generator, which makes them distinct Markov chains that can agree only in
distribution. The energy per site agrees to between 0.09 and 0.23 standard
deviations at three temperatures spanning the ladder (Table 1 row 5), and the
trial directions remain on the sphere to 2.4 × 10⁻⁷ (row 6).

**Table 1.** Agreement between the fused kernel and the reference
implementation. Rows 1–4 hold the randomness fixed and are therefore exactness
tests; rows 5–7 let each engine use its own generator and are statistical.

| # | test | quantity | result |
|---|---|---|---|
| 1 | stencil vs reference, even sublattice | differing spins | 0 / 182520 |
| 2 | stencil vs reference, odd sublattice | differing spins | 0 / 182520 |
| 3 | observable kernels, energy | relative | 1.2 × 10⁻⁷ |
| 4 | observable kernels, topological charge | absolute | 8.4 × 10⁻⁸ |
| 5 | both chains to equilibrium, T = 20, 5, 1 | σ | 0.09 – 0.23 |
| 6 | trial direction norm | abs. deviation from S₀ | 2.4 × 10⁻⁷ |
| 7 | reduced curves vs reference-against-itself | mean σ / max σ | 0.99 / 1.90 |

Row 7 is the instrument that settles the question, and it requires a control the
other rows do not. Running the *reference* implementation against *itself* with a
second seed yields 0.87 and 2.53 for the same two statistics, which is the noise
floor of the comparison rather than a property of either engine; the fused
kernel's 0.99 and 1.90 therefore sit inside that floor, its maximum deviation
falling below the control's own. Without that control the comparison is not
interpretable: an early version of the measurement, lacking it, flagged one
temperature at 4.94 σ and that figure was an artefact of a measurement window
too short to resolve the quantity, not of the engine.

The control also identified a defect that no other row could. Measuring the
acceptance rate directly revealed the kernel accepting about 1.1 % fewer
proposals than the reference, with all five temperatures falling on the same
side. Rows 1 and 2 could not see it, because in those tests the kernel's own
generator is bypassed. The cause is the one Section 3.3 names: drawing the
direction and the acceptance variate from adjacent *keys* at a shared counter
gives the variate an input in common with the direction. After the separation
of counters prescribed there, the acceptance ratios scatter on both sides of
unity (0.996, 0.998, 0.994, 1.029, 0.997), each within one standard error.
That the defect reached the sampling distribution while leaving the energy
arithmetic exact **demonstrates** that the two levels of test are not redundant.

## The components deliver the properties they were defined under

### Contiguous neighbour access and the synchronisation-free halo

Figure 2 presents the per-replica cost of one sweep against the replica count,
for the reference engine and for the fused kernel. Decomposing the change
isolates the two components: with the trial spins still supplied by the host, so
that only the stencil of Section 3.2 and the layout of Section 3.1 differ, the
kernel runs 51.3 % faster than the reference on average over six replica counts
(Figure 2, middle curve). The mechanism is the one the layout was defined for:
instruction-level inspection of the reference engine attributes 31.5 % of the
sweep to the six materialised translates, and the stencil removes that
round-trip through memory while leaving the access pattern a shifted contiguous
segment.

Three earlier realisations of the same stencil, built on a tiling abstraction
that computes one address per lane, were between 15 % and 31 % *slower* than the
reference despite being exact to single-precision round-off. That the same
stencil loses when its addresses are per-lane and wins when they are
vector-shifted **confirms** that contiguity, and not the number of loads, is
the property at issue — the fused kernel issues 21 loads per site, no fewer than
before.

### Counter-based proposals

Adding the in-kernel generator of Section 3.3 on top of the stencil takes a
further 31.4 % off what remains (Figure 2, lower curve). Instruction-level
accounting attributes 17.1 % of the reference sweep to its host-side generator,
so the measured share is consistent with removing that work and additionally
not writing the proposals to memory.

### Deterministic observables

Moving the observables into the kernel recovers 4.5 percentage points on the
full ladder. The figure is smaller than the composition of Sections 3.1–3.3
would suggest, and the reason is instructive: a first measurement over four
temperatures put the recovery at 0.8 points, which would have been grounds to
discard the component. The measurement window was too short for the quantity.
This **indicates** that the remaining gap between the bare sweep and the full
ladder lies not in the observables but in the eight accumulators and the
temperature scan, neither of which the four components address.

## The optimised simulator at the scale the study runs

Figure 3 is the flagship result. It presents the wall clock of the complete
production ladder — 200 temperatures by 20 000 sweeps — for one hundred states
at five thousand replicas, across the CPU baseline, one and two consumer GPUs,
and one, two and four datacentre GPUs, with both engines shown side by side.

Read at four datacentre GPUs (Figure 3, rightmost pair), the optimised engine
completes the hundred-state ladder in 57.6 minutes against 2.7 hours for the
reference engine. Read at one GPU (Figure 3, fourth pair), the same comparison
is 3.8 hours against 17.2. Against the CPU baseline on the same workload the
separation is of a different order: 130.7 days against 57.6 minutes.

The scaling behaviour is what makes this the flagship rather than the
single-state figure. Over one to four GPUs the optimised engine gains a factor
of 3.91 with a hundred states and only 1.10 with one, on the same hardware and
the same kernel. The mechanism is the per-device batch: at one state and four
GPUs each card receives twelve replicas, and at twelve replicas per card the
kernel's grid falls to roughly six hundred blocks and **loses** to the reference
engine by 47.7 %. The crossover sits between thirteen and twenty-five replicas
per card. The practical consequence is that additional cards are paid for by
additional states, not by dividing one state across them — one state per card is
embarrassingly parallel, requires no sharding, and preserves the per-state cost.

Figure 4 shows the same comparison without the CPU bar and on a linear axis,
since at 130.7 days the baseline compresses everything else onto the axis.

### The cost model holds where it was not designed to

Table 2 reports the penalty the reference engine pays when the element count of
its arrays acquires two factors of two, measured on two architectures, beside
the same penalty for the fused kernel.

**Table 2.** Cost penalty for the condition v₂(B) + v₂(L) ≥ 2, where v₂ is the
2-adic valuation, B the replicas per device and L the layer count. Measured over
layer counts 4, 5, 6 and 8.

| architecture | reference engine | fused kernel |
|---|---|---|
| datacentre (H100) | +17.8 % | +0.3 % |
| consumer (T4) | +49.9 % | +0.8 % |

The penalty is substantial for the reference engine on both architectures and
absent for the fused kernel (Table 2, right column). This is **attributed to**
the formulation rather than inferred from the pattern: the penalty arises from a
code-generation decision the array compiler takes about the loop it is handed,
and the fused kernel hands it no such loop — the block shape is fixed by the
kernel and the layer count only sets how many blocks exist along the
out-of-plane axis.

The advantage does not, however, transfer uniformly across architectures. On the
datacentre card the kernel's per-site cost improves as the batch widens and then
plateaus, and its margin over the reference engine grows from 63.9 % to 78.1 %;
on the consumer card the per-site cost degrades monotonically and the margin
*contracts* from 57.0 % to 49.6 %. This **suggests** a memory-hierarchy origin —
the consumer card has an order of magnitude less last-level cache and a third of
the bandwidth, against a working set of 456 MB at five thousand replicas — but
that mechanism was not measured, and only the sign is established here.

### Against published implementations

Table 3 places the simulator beside published GPU Monte Carlo spin codes on a
common unit, the time per single-site update.

**Table 3.** Time per single-site update. The unit is common; the models are not.
The rightmost column states the per-site arithmetic each model carries, which is
what makes or breaks a comparison in this unit.

| implementation | model | hardware | ns / update | per-site work |
|---|---|---|---|---|
| this work, reference engine | Heisenberg + iDMI + cubic + surface + field | 1 × H100 | 0.590 | heavy |
| **this work, fused kernel** | same | 1 × H100 | **0.129** | heavy |
| this work, fused kernel | same | 4 × H100 (aggregate) | 0.033 | heavy |
| this work, CPU baseline | same | 24 threads | 107.7 | heavy |
| Charilaou (2021) | 3-D anisotropic Heisenberg | unstated GPU | 1.55 | moderate |
| Preis et al. (2010) | 2-D Ising, multi-spin coding | 1 GPU (2010) | 0.127 | minimal |
| Weigel & Yavors'kii (2011) | lattice spin models | GTX 480 | 0.03 | minimal |
| Romero et al. | 2-D Ising, 4-bit packing | V100 | 0.0022 | minimal |
| rack-scale (2025) | 2-D Ising | multi-GPU GB200 | 8.7 × 10⁻⁶ | minimal |

Against the published implementation with the closest Hamiltonian — an
anisotropic Heisenberg model, the only entry in Table 3 carrying continuous
spins and an anisotropy — the fused kernel is twelve times faster per update
(0.129 against 1.55 ns, Table 3 rows 2 and 5), on newer hardware whose
generation the source does not state.

The Ising entries are faster per update, by up to four orders of magnitude, and
that comparison is **not** evidence about this work in either direction. The
per-site arithmetic differs by roughly two orders of magnitude: a bit-packed
Ising flip is a handful of integer operations with no transcendental and one
random number, while one update here performs on the order of 290 arithmetic
operations, one exponential, one reciprocal square root and four random draws.
Reading Table 3 as a ranking would be reading a unit that the models do not
share.

What the table does support, read across rows 2 and 7, is a narrower and more
useful statement: the fused kernel reaches the per-site rate of a multi-spin
coded Ising implementation on 2010 hardware while carrying a Hamiltonian some two
orders of magnitude heavier per site. The comparison that is strictly controlled
remains the internal one — identical code, identical physics, identical
workload — and there the optimised simulator is 834 times faster than the CPU
baseline (Table 3, rows 2 and 4).

## Limitations

<!-- Per the contract, limitations close the section under their own
     contract (sections/04-limitations.md). Drafted there, pointed at here. -->

Four limitations bound the claims above, and each is stated where the evidence
for it was gathered: the per-device batch floor below which the fused kernel
loses to the reference engine; the architecture dependence of the margin, whose
sign is measured and whose mechanism is not; the restriction of the chain's
exactness to a bipartite interaction graph, which holds the second and fourth
exchange shells at zero; and the single geometry in which the cost model was
calibrated. They are developed in Section 4.
