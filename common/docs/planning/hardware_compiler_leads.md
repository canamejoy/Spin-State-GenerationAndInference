# Hardware and compiler specifications, and where they point

Gathered 2026-10-12 to attack the two questions the batch-penalty work left
open: the ~2 MB footprint threshold, and the broad f(B) sawtooth where B=512 is
the global minimum and B=768 near the worst while both are vector width 4.

No GPU time was spent on this: every test below is arithmetic against data
already in `section0-mcmc.gpu/resultados/curves/`.

## 1. Specifications, with sources

### Memory transaction granularity
- A cache line is **128 B** and maps to a 128-B aligned segment of device
  memory. The minimum granularity is **one 32-B sector**; one tag covers the
  four sectors of a line, and each sector has its own valid/dirty bits, so
  sectors are individually replaceable.
- Accesses cached in **L1 and L2 are serviced in 128-B transactions**; accesses
  cached in **L2 only are serviced in 32-B transactions**. On A100 the
  granularity is selectable at 32/64/128 B through the CUDA runtime API.
- Source: NVIDIA developer forums on L1/L2 granularity, and the Ampere
  optimisation deck (GTC 2020, s21819).

**Why it matters here.** It confirms rather than changes the width-4 arithmetic.
Blocked vectorisation puts 4 consecutive floats (16 B) in each thread, so one
warp-wide load touches 32 addresses 16 B apart, spanning 512 B. At 32-B sector
granularity that is 16 sectors fetched (512 B) to deliver 128 B of useful
payload: the same 4x waste, now expressed in the unit the hardware actually
moves.

### The H100's L2 is split in two
- GH100's L2 is **two independent partitions**, each serving a subset of GPCs:
  four GPCs connect to one partition and four to the other. The 50 MB is
  **2 x 25 MB**.
- Any thread can reach all 50 MB but not at the same speed: the **far partition
  costs nearly 2x the latency** -- chipsandcheese measured it as "about as much
  latency as VRAM on the RX 6900 XT".
- Bandwidth: **over 5.5 TB/s on the near partition**, **3.8 TB/s across the
  whole L2** -- a regression from the A100's 4.5 TB/s.
- Hopper adds an **L2 Request Coalescer (LRC)** that aggregates L1 misses from
  different SMs hitting the same L2 sector.
- Sources: NVIDIA H100 architecture whitepaper; chipsandcheese, "Nvidia's H100:
  Funny L2, and Tons of Bandwidth".

### DRAM interleaving
- Hardware interleaving by the memory controllers is on **256 B granularity**.
- HBM3 uses **pseudo-channel** mode: each 128-bit channel splits into two
  64-bit pseudo-channels sharing the command bus, which doubles the bank count.
  Pseudo-channel-level interleaving is standard in GPU-HBM systems.
- The H100's 5120-bit bus is 80 channels of 64 bit (160 pseudo-channels).

### Address translation
- GPU large pages are **2 MB**. With 4 KB pages an L2 TLB of 1024 entries
  reaches about **4 MB**; with 2 MB pages the same 1024 entries reach **256 MB**.
- Source: Mosaic (arXiv 1804.11265) and the dead-entry TLB paper
  (arXiv 2606.00486).

### XLA:GPU, from its own documentation
- The loop emitter's thread-ID map uses a **128-thread block**
  (`bl_x * 128 + th_x`). That matches every dump we took, on two architectures.
- The vectorisation pass, verbatim: *"if they are produced by
  `xla_gpu.apply_indexing` that accesses the elements **contiguously** w.r.t. to
  the `%vector_index` and the access is **aligned**, then `tensor.extract` is
  converted to `vector.transfer_read`"* and hoisted out of the loop. How the
  width is chosen is **not documented**.
- There are **seven emitter types**, one per fusion "hero". The **reduction
  emitter** codegens with warp shuffles and shared memory.
- Read coalescing is decided **heuristically**, by looking for HLO op
  combinations known to be problematic, in
  `xla/service/gpu/model/coalescing_analysis.cc`; the docs call the heuristics
  "necessarily imperfect and incomplete". The cost model carries a **coalescing
  waste factor that assumes fully strided access** -- one element per cache line
  used, the rest of the bandwidth wasted -- and assumes writes are coalesced.
- Sources: openxla.org/xla/emitters; openxla/xla discussion 10065 (Cost Models
  in XLA GPU); the MLIR open meeting deck on XLA indexing analysis.

**Two things worth noting.** First, "contiguously w.r.t. the vector_index" is
exactly the blocked scheme measured in experiment 5 -- XLA's documented
condition names the layout that costs 14.9% in raw CUDA. Second, XLA has a cost
model that knows about coalescing waste, yet its vectoriser picks the layout
that creates that waste. If the source confirms the vectoriser does not consult
the coalescing model, that is an upstream issue worth reporting.

## 2. Hypotheses killed by arithmetic, at no cost

### Wave quantisation -- REFUTED
If a grid is not a multiple of SMs x blocks-per-SM, the last wave runs partly
empty. Computed for all 19 batches at K = 1, 2, 4, 8, 16 blocks/SM on 132 SMs:
efficiencies are 0.675 to 0.999 and mostly above 0.9, and `r` against measured
cost peaks at **+0.65**. Decisive counterexample: **B=512 (0.982) and B=768
(0.997) are indistinguishable in wave efficiency and differ 57% in cost.**

### Buffer-offset conflicts -- REFUTED
Simultaneously-touched buffers sit `B x 91,260` bytes apart, so a conflict
effect would make that separation land on bad residues for the slow batches.
Tested against seven candidate periods -- 128 B line, 256 B interleave,
80 x 256 B, 160 x 256 B, the 25 MB/16-way set stride, 4 KB and 2 MB pages.
**Every correlation is |r| <= 0.21.** And the same counterexample kills it
directly: `91,260 = 2^2 x 22,815` with 22,815 odd, so B divisible by 64 puts the
separation on a 256-B multiple -- which includes 64, 128, 256, 320, 512, 768 and
1024 alike. **B=512 and B=768 have identical residue zero and opposite cost.**

### A per-sweep fixed cost -- PARTIAL, and it is not the sawtooth
Fitting `us(B) = a/B + b` gives **a = 10.9 us per sweep, b = 2.549 us**, with
**R^2 = 0.364**. The `a/B` term is real and is 29% of the cost at B=10 but only
0.4% at B=1024, so it explains the steep rise at small B and nothing else. The
residual sawtooth spans **-21.7% (B=512) to +23.2% (B=768)**, a 45 pp range that
survives the fit untouched.

## 3. Where to point next

### Lead A -- the footprint threshold is the 25 MB partition, not the 50 MB L2
The L2-capacity explanation was refuted earlier using the 50 MB figure. With the
**correct 25 MB partition** it fits, using XLA's own measured allocation of
0.676 MB per replica (probe 3a):

| arm | sites | working set at B=60 | vs 25 MB | measured |
|---|---|---|---|---|
| A | 2125 | 16.4 MB | fits | null at every v2 |
| B | 5245 | 40.6 MB | exceeds | +15.2%, clean split |

**Falsifiable prediction with a number:** at B=60, sweeping `n_active` at fixed
`v2` with everything else held, the penalty should switch on near
**n_active = 3233 sites**, where the working set crosses 25 MB. Not at 2125, and
not far above 3233. A handful of lattices between 2500 and 4500 sites settles
it, and the far-partition latency being ~2x is the right magnitude to produce a
15% average penalty once a fraction of traffic goes far.

This also gives the device-dependent amplitude a mechanism to be tested against,
since the partition structure differs per device (T4 has a small unified L2,
A100 and H100 are partitioned, Ada differs again).

### Lead B -- the broad sawtooth is XLA choosing a different PROGRAM per B
Every address-arithmetic hypothesis is dead, and the one thing that provably
changes with B and is not arithmetic is the compiled program. Evidence already
in hand:

- Fusion counts move with B even inside a 14-wide window: **71 to 78** fusions
  at B = 51..64 (probe 3a).
- Kernel counts from the thunk dumps: **43 / 45 / 43 / 40** at B = 59/60/63/64
  on an H100; **37 / 41 / 43 / 37** at B = 57/58/60/64 on a T4.
- The fusion *set* changes, not just its size:
  `loop_maximum_negate_fusion_1` exists only at some B, and
  `loop_select_fusion_2` walks the full array at one B and the per-site arrays
  at others.
- **The reduction kernel is the strongest signal.** `loop_reduce_fusion_1` on a
  T4 goes **87 -> 1724 -> 892 -> 98** blocks at B = 57/58/60/64: a 20x
  non-monotonic swing that follows no width rule. That is the signature of a
  shape-driven emitter choice, and XLA's reduction emitter is a separate
  emitter with its own tiling.
- The `input_concatenate_fusion_*` kernels are heavy (10,247 to 11,115 blocks),
  scale smoothly with B, and **never drop** -- a width-1 cost floor that
  explains why the measured penalty is ~12-15% rather than the larger number
  the loop fusions alone would imply.

**The experiment:** run `launch_dims` over the 19 broad batches, compile only,
and correlate the reduce-kernel geometry and the kernel/fusion counts against
the residual sawtooth from section 2. If the fast batches (512, 150, 75, 32)
have structurally cheaper programs than the slow ones (768, 100, 400, 256), the
broad curve is a compiler-choice artefact, same as the mod-4 penalty turned out
to be. Compile only, so pennies.

### Lead C -- check XLA's vectoriser against its own coalescing model
XLA ships a coalescing analysis with a waste factor, and a vectoriser that
chooses contiguous-per-thread access. Experiment 5 measured that choice costing
14.9% in raw CUDA while the coalesced alternative at the same width is 45%
faster. Reading `coalescing_analysis.cc` and the vectorisation pass to see
whether one consults the other is free, and if they do not, this is a reportable
upstream issue with a reproduction already in hand.

## 4. What the CPU side contributes
The host is not in the measured region -- every argument is a device array
before the timer, and traced GPU time tracks wall clock to within the effect
size. What the host does contribute is the per-sweep launch cost, which is the
`a = 10.9 us` term of section 2: real, dominant below B ~ 50, negligible above
B ~ 200, and flat in B so it cannot produce a sawtooth.
