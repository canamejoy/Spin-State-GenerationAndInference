# Fixing the cost without constraining the layer count

Written 2026-10-13, after the dummy-layer attempt failed. The question: how do
we get a cost that does not depend on the batch size, for ANY number of layers?

## 1. First, what "the 80%" actually is

It is not one effect. From the trio on a T4 (36 measurements):

| | us/replica-sweep | vs B=65 |
|---|---|---|
| B=65, width 1 | 13.798 | — |
| B=63, **also width 1** | 16.926 | +22.7% |
| B=64, width 4 | 24.968 | **+81.0%** |

The width penalty measured on its own, same lattice, is **+28.6%**. If width
were the whole story B=64 would cost 17.742; it measured 24.968, so there is a
further **1.41x** from the broad sawtooth. The 80% is

    width (29%)  x  sawtooth (41%)

and it is the **worst pairing you can construct**, not a single mechanism. Any
strategy aimed at "the 80%" is aimed at a number that does not exist as one
thing. The two halves have to be attacked separately, and they are very
different problems: the width is understood, the sawtooth is not.

## 2. Why no reshaping can fix the width half

    v2(elements) = v2(B) + v2(L) + v2(N^2 * 3),  and v2(N^2 * 3) = 0 since N is odd

Every form of padding **multiplies** the element count, so it can only ADD to
`v2`, never subtract. **There is no reshaping that drives `v2(elements)` below
`v2(L)` while keeping the physical sites as a sub-block.** This is arithmetic,
not an empirical limitation, so it closes the whole family of ideas at once:
padding N, padding the component axis, chunking the batch, folding axes.

The single exception is making the ARRAY's layer count odd while the PHYSICAL
count stays — the masked dummy layer. That was measured and it loses 6 of 6
cells (mean -20.6%), because one extra layer costs `1/L` (25% at L=4, 12.5% at
L=8) against a width penalty of 14.3% on an H100: same order, no margin. See
`results/curves/dummy_layer_refuted.json`.

So within XLA, for `L` divisible by four, **there is no fix**. That is settled.

## 3. The reframing: XLA leaves 1.8x on the table

Experiment 5, raw CUDA on an H100, identical total work, only the index mapping
changing (`results/curves/width4_is_uncoalesced.json`):

| what | configuration | us per pass |
|---|---|---|
| what XLA picks at B divisible by 4 | width 4, **blocked** | 9.67 |
| what XLA picks otherwise | width 1 | 8.41 |
| **the best available** | width 4, **strided** | **4.60** |

- strided width 4 against width 1: **1.83x faster**
- strided width 4 against XLA's bad case: **2.10x faster**

**The prize is not avoiding a 14-29% penalty. It is capturing a ~1.8x that XLA
leaves unclaimed in EVERY batch, good ones included.** XLA's own documentation
says the vectoriser fires when the access is "contiguously w.r.t. the
`%vector_index`" — the blocked layout — and its coalescing cost model, which
knows about exactly this waste, is not consulted by that decision.

That changes the economics of the only remaining route. It is no longer a
defensive measure worth 20%; it is the difference between 8.41 and 4.60.

## 4. The strategy: port `substep` to Pallas

Pallas is JAX's kernel language. It makes the tiling and the indexing ours while
the rest of the program stays in JAX. It is the only option that is
**independent of `L`** (we choose the width, so `v2(L)` stops mattering) and the
only one that can take the strided layout.

### Phase 0 — feasibility spike. One day, about $1.
Write a Pallas kernel for the **neighbour sum and renormalise only** — no RNG,
no Hamiltonian. Compare against the JAX path for equivalence and speed at
B = 63, 64, 65 on the reference lattice.
**Gate: if it does not beat the JAX path, stop here.** Everything after this
depends on Pallas being able to express the stencil without the materialisation
that `jnp.roll` forces, and that is cheap to find out.

### Phase 1 — the full Metropolis substep. Two to three days.
Port exchange, anisotropy (uniaxial, cubic, surface), DMI, Zeeman, the trial
spin and the acceptance test. There is a head start: `_METRO` in `app.py`
already has counter-based RNG, the acceptance branch and the uniaxial term in
CUDA, measured flat across B.

**Verification is the real work, because the trajectory will NOT match bit for
bit** — a different RNG stream draws different numbers. So:
- the **energy** of a fixed configuration must match to float tolerance
  (the pattern is in `_dummy_layer`, which caught 1.18e-07 and 0.0)
- the **acceptance rate** must match statistically at several temperatures
- **equilibrium observables** must agree within error bars on the existing
  ladder

### Phase 2 — integrate behind a flag. One to two days.
Swap `substep` inside `build_engine` so both paths coexist and can be compared
at any time. `anneal`, the temperature scan, the observables and the ladder stay
in JAX untouched. Run the existing test suite against both.

### Phase 3 — verify the claim.
Re-measure the dense window and the broad curve. The prediction to check: the
cost becomes **flat in B**, since we choose the tiling, and lands near the
strided figure rather than the width-1 one.

### Risks, named
- **Pallas on GPU is less mature than on TPU.** Phase 0 exists to find this out
  for $1 instead of after a week.
- **The DMI term reads across components of neighbouring sites**, which is the
  most awkward thing to express in a tile. If anything forces a fallback, it is
  this.
- **We own the kernel's correctness from then on.** The JAX version is readable
  and tested; a hand-written tile kernel is neither for free.
- **It does not touch the sawtooth.** The broad curve has no mechanism and
  Pallas is not aimed at it. Expect a flat-in-B cost for the substep and
  whatever the rest of the program still does.

### If Phase 0 fails
Fall back to what already works: measure the curve for each geometry and pick
`B`. That costs about **$0.60 and half an hour per lattice**, and for `L` odd
the rule makes even that unnecessary.

## 5. What this does and does not promise

**Does:** a substep cost that does not depend on `B` or on `v2(L)`, at the
strided rate rather than the blocked one. That is the "fixed cost regardless of
the layer count" the question asked for.

**Does not:** a flat total. The broad sawtooth lives in the rest of the
program — `input_concatenate_fusion` from the rolls, the unnamed fusions — and
every cheap route to it is closed: wave quantisation, buffer-offset conflicts,
the reduction emitter and pad-for-roll, all refuted and recorded.
