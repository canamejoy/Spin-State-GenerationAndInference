# Cycle — internal evaluation notebooks

Three notebooks running the full `theta -> DDPM -> image -> Xception -> theta_hat` pipeline
on the internal dataset split, evaluated with the plain (non-integrated) checkpoints — the
published DDPM and the published Xception encoder, used independently, without any latent
cosine loss or guidance.

| Notebook | Dataset | Status |
|---|---|---|
| `ciclo_completo_v2.ipynb` | Internal test split | Complete |
| `cycle_complete_newmetrics.ipynb` | Internal test split | Complete |
| `ciclo_texture_fidelity.ipynb` | Internal, texture/saturation split | Complete |

See [`section3-cycle/docs/05_complete_cycle.md`](../../section3-cycle/docs/05_complete_cycle.md) for the theory and
evaluation protocol.

## Relationship to `section3-cycle/integracion/evaluation/`

The `section3-cycle/integracion/evaluation/` notebooks evaluate the **integrated** architectures
(cosine-loss fine-tunes and latent-guided sampling, see
[`section3-cycle/integracion/README.md`](../integration/README.md)) using the same physical-metric
and cycle-evaluation protocol as this directory. `section3-cycle/ciclo/` is the baseline: it
establishes the plain-checkpoint cycle numbers that the integration notebooks compare
against (`DDPM+cos`, joint fine-tune, and guided sampling are all reported relative to this
baseline). Superseded, pre-crop-before-mask runs are archived under
[`section3-cycle/archivo/`](../archive/), not here.
