# Interactive pages

Five self-contained pages built from the measurements in `docs/10`–`docs/12`.
Open the `.html` files directly in a browser; everything they need is here.

| page | what it answers |
|---|---|
| `reparto.html` | How many replicas belong on each GPU, how the cost curve behaves, how the optimum moves with the lattice geometry, and why batch sizes divisible by four are penalised — XLA's vector width, with the layer count deciding whether a lattice is immune. |
| `cronograma.html` | 100 replicas x 15k sweeps against 50 x 20k: cost, accuracy, and the autocorrelation time that decides between them. |
| `resumen.html` | **Start here.** The batch-size result synthesised for a reader who wants the conclusion: what we found, the T4 64-vs-65 reproducibility test in total seconds, what the width effect and the sawtooth each are worth, and the scope of what is verified. Reports wall-clock seconds throughout, not the normalised metric. |
| `replicas.html` | How many replicas belong on each GPU, for the fixed setup the tutor asked for: H100, Rd 18.30, five layers. Fifty batch sizes measured, reported as seconds per replica for a full production run. Also carries why grouping by experiments does not parallelise (1.11x against 3.86x) and why the launch count does not matter (2.7%). |
| `lotes.html` | The whole GPU-parallelisation investigation as one narrative: the optimal-batch hypothesis, every mechanism that turned out to be false (hardware, cache, bandwidth, buffer alignment, the RNG, the host transfer, the replica stride), and the rule that held — `v2(B) + v2(L) >= 2`. Six interactive widgets: a `v2` explainer, a geometry explorer, the 19-point curve, the dense per-device windows, a vector-width calculator, and a planner over the five degrees of freedom that refuses to cost a lattice whose curve has not been measured. States what is verified on five GPUs and what is verified only on an H100. |

`cronograma.html` reads `sched.json` and the `sz/` images with relative paths,
so keep the three together. Those images are the middle-layer `s_z` map of
replica 0 at ten temperatures along the descent, rendered with jet pinned to
[-1, 1] -- the physical range of a direction cosine, so the same colour means
the same value in every frame. Sites outside the disk stay at `s_z = 0` and take
jet's green, matching the convention of the released reference dataset.

All three pages are also published as private artifacts on claude.ai; these copies
are the shareable ones. `lotes.html` carries no external data files — its numbers
are embedded, and every one of them is checked against `results/curves/*.json`.
