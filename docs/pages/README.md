# Interactive pages

Two self-contained pages built from the measurements in `docs/10`–`docs/12`.
Open the `.html` files directly in a browser; everything they need is here.

| page | what it answers |
|---|---|
| `reparto.html` | How many replicas belong on each GPU, how the cost curve behaves, and how the optimum moves with the lattice geometry. |
| `cronograma.html` | 100 replicas x 15k sweeps against 50 x 20k: cost, accuracy, and the autocorrelation time that decides between them. |

`cronograma.html` reads `sched.json` and the `sz/` images with relative paths,
so keep the three together. Those images are the middle-layer `s_z` map of
replica 0 at ten temperatures along the descent, rendered with jet pinned to
[-1, 1] -- the physical range of a direction cosine, so the same colour means
the same value in every frame. Sites outside the disk stay at `s_z = 0` and take
jet's green, matching the convention of the released reference dataset.

Both pages are also published as private artifacts on claude.ai; these copies are
the shareable ones.
