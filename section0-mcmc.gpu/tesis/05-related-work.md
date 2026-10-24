# Related Work

<!--
papersmith contract: sections/05-related-work.md
  extent          820–1700 words
  carries         one block per stated problem, and one synthesis figure or table
  does NOT carry  equations
  structure       preamble -> general panorama -> one block per specific problem
                  -> closing with open fronts and the gap -> synthesis artefact
  rule            thematic, not catalogue
-->

Accelerating lattice spin Monte Carlo on graphics hardware has two largely
separate literatures, and the simulator of Section 3 sits between them. One
optimises the cheapest possible site update to its hardware limit; the other
carries a physically rich Hamiltonian and accepts whatever rate follows. This
section reviews each, then states what neither supplies.

## The panorama

Graphics hardware suits lattice Monte Carlo for a structural reason that
predates any particular implementation: a bipartite colouring of the
interaction graph makes one colour's updates mutually independent, so a sweep
becomes two data-parallel passes. Every implementation reviewed here rests on
that observation, and the differences between them lie in what each does with
the freedom it buys. Two strategies have been pursued, and they diverge almost
immediately.

The first compresses the state. If a spin takes two values it fits in one bit,
and a machine word then holds dozens of independent lattice sites or dozens of
independent replicas, updated by bitwise operations in a single instruction.
The second keeps the state as it is — three floating-point components per site —
and parallelises only over sites and replicas.

The reported rates differ by four orders of magnitude, and the gap is almost
entirely the choice above rather than engineering quality. That makes the
literature difficult to read as a ranking, and reading it as one is a recurring
error: the unit shared across these works, time per single-site update, is not
commensurable across models whose per-site arithmetic differs by two orders of
magnitude.

## Bit-compressed Ising implementations

The line begins with multi-spin coded implementations of the two-dimensional
Ising model, which established both the method and the benchmark unit still in
use. Preis and co-workers reported a single-GPU rate on 2010 hardware and
demonstrated that the approach scales across a cluster of devices, reaching
aggregate rates two orders of magnitude above the single device by partitioning
a lattice of 800 000 sites on a side across sixty-four GPUs.

Subsequent work sharpened the packing rather than the parallelism. Representing
each spin in four bits instead of one raised the single-device rate on a V100 by
more than a factor of three over the earlier generation's figure, and a
sixteen-device configuration multiplied that again. The most recent entries move
to rack-scale systems and report aggregate lattice-update rates another two
orders of magnitude higher.

Two properties of this line matter for the present work. The rates are genuinely
remarkable and they are also **model-specific**: the bit packing that produces
them requires a spin with two states, and a continuous spin admits no analogue.
And the quantities these implementations compute are correspondingly narrow —
magnetisation and energy of a two-state model, with no anisotropy, no
chiral interaction and no vector observable.

A parallel strand pursues the same compression for spin glasses, where
asynchronous multi-spin coding across replicas serves parallel tempering rather
than a single long chain. The structural lesson there is relevant and
transferable: when the physics requires many weakly coupled chains, the replica
axis is a better place to find parallelism than the lattice, because it needs no
halo at all.

## Continuous-spin implementations

The second literature keeps the vector spin. Weigel's surveys of lattice spin
models on graphics hardware cover both the Metropolis update and cluster
algorithms, and report per-update rates for continuous models on hardware of
their period, obtained in part by running many independent systems
concurrently — the same replica-axis strategy, arrived at independently.

The closest published work to the simulator of Section 3 is Charilaou's
Monte Carlo implementation of the three-dimensional anisotropic Heisenberg
model, which carries continuous spins and an anisotropy and reports a per-update
time together with a demonstration at one hundred million spins on a single
device. It is the only entry in this review whose Hamiltonian is of the same
kind as the one in Section 1, and it is therefore the only one against which a
per-update comparison carries meaning.

Beyond Monte Carlo, the atomistic spin dynamics community maintains several
frameworks that integrate the Landau–Lifshitz–Gilbert equation on the same kind
of lattice and with a comparable interaction set, some with Monte Carlo modes
alongside. Their published performance figures are reported in iterations per
second rather than per-update times, and in at least one case the documented
bottleneck is a correlation-function evaluation rather than the spin update
itself. A recent kernel-fusion study in that family reports a speedup over
multi-core processors for the dynamic structure factor specifically. These
frameworks are the natural comparison for a *production* workflow and the wrong
comparison for a *sweep rate*, because the quantity they optimise is not the
one at issue here.

## Open fronts, and the gap

Three fronts remain open in this literature, and the third is the one this work
addresses.

**Compression without a two-state spin.** Nothing in the bit-packing line
transfers to a continuous spin, and no published alternative achieves comparable
compression for one. The question of whether a reduced-precision vector
representation could recover part of that advantage appears unexamined.

**Portability of the optimisation across devices.** The reported rates are
tightly coupled to the device they were measured on, and the works reviewed here
report a single architecture each. Whether an optimisation that wins on a
datacentre device also wins on a consumer one is therefore not addressed
anywhere in this literature, although the question determines whether a result
is usable by a group without datacentre access.

**A physically rich Hamiltonian at a competitive sweep rate.** This is the gap.
The implementations that report the highest rates carry the least physics, and
the implementation whose Hamiltonian most resembles the one this work requires
reports a rate an order of magnitude slower than the bit-packed codes of a
decade earlier. No reviewed work carries an interfacial chiral interaction, a
site-dependent surface anisotropy and a cubic anisotropy series together, and
none reports a per-update rate for such a model. The simulator of Section 3 is
built for exactly that combination, and Table 3 of Section 4 reports its rate in
the unit this literature established, with the arithmetic weight of each model
stated alongside so that the comparison can be read for what it is.

A fourth front is methodological rather than computational, and it recurs across
both literatures: performance claims are generally reported without a stated
control against the implementation being replaced. Where a reimplementation is
compared against a reference, the comparison is usually of final observables
rather than of the chain itself, which cannot distinguish an engine that samples
the right distribution from one whose sampling is subtly biased but whose
equilibrium averages happen to agree. The validation protocol of Section 4
addresses that directly.

**Table 4.** Synthesis of the reviewed implementations against the three
properties this work requires. A dash marks a property the source does not
report rather than one it lacks.

| line of work | continuous spin | chiral + surface anisotropy | per-update rate reported | multiple architectures |
|---|---|---|---|---|
| bit-compressed Ising | no | no | yes | — |
| Ising spin glass, multi-spin + tempering | no | no | yes | — |
| continuous-spin surveys | yes | no | yes | — |
| anisotropic Heisenberg MC | yes | partial (anisotropy only) | yes | — |
| atomistic spin dynamics frameworks | yes | yes | no (iterations/s) | partial |
| **this work** | **yes** | **yes** | **yes** | **yes (2)** |

Table 4 makes the gap legible as a row rather than as an argument: no reviewed
line fills every column, and the properties missing from each are the ones
Section 3 was built to supply.
