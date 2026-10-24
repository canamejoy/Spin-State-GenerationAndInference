# Raw results from the Kaggle runs

Every number in `main/report/results.tex` comes from a file here. They were rescued from a
session-scoped temporary directory; without this copy they would have been lost when the
session ended.

## The comparison, one protocol

`04a`-`04d` are the four architectures, all at K=32, N=1000 stratified by magnetic
structure, 100 sampling steps, seed 42. Only the checkpoint and the guidance scale differ,
so these four rows are comparable by construction.

| File | Variant | cycle R2 |
|---|---|---|
| `04a_ddpm_base.json` | baseline DDPM | 0.6387 |
| `04b_cos_frozen.json` | cosine term, encoder frozen | 0.6488 |
| `04c_cos_joint.json` | cosine term, both trained, **with the regression anchor** | 0.7997 |
| `04d_latent_guidance.json` | sampling-time latent guidance, s=8 | 0.9161 |

## The control

`05_cross_encoder.json` — the same images read by two independently trained inverse models,
|Pearson r| per parameter. The Xception reads J3 at 0.985 in guided images; the ViT reads
0.058, while reading T, J2, KanS and KDM on those same images at 0.88-0.93. The pattern the
guidance writes is specific to one encoder.

## Training runs

| File | Note |
|---|---|
| `01_train_frozen.json` | 15 epochs, frozen encoder |
| `02_train_joint_anchored.json` + `02_history_anchored.json` | 15 epochs **with** the anchor: the inverse model stays positive (0.60 to 0.67) |
| `02_train_joint_COLLAPSED.json` + `02_history_COLLAPSED.json` | the same run **without** the anchor, kept as the documented failure mode: cosine term reaches 0.0000 by epoch 3, inverse R2 -114,134, cycle R2 -146,409 |

## Guidance sweeps

`03_guidance_sweep_k32.json` (N=512 unstratified, 50 steps) and
`03_guidance_sweep_matched.json` (N=1000 stratified, 100 steps). Prefer the matched one;
the other predates the protocol being fixed and its q_peak in particular is not comparable.

## A warning that must travel with these files

The `peak_wave_vector` here is the **masked, mean-subtracted** physical variant from
`metrics.py`. The Kaggle notebooks `physical-metrics-comparison-4models` and
`physical-metrics-3ddpm-comparison` call `structure_factor` with its defaults --- raw image,
no disk mask, no mean removed --- and report the raw radial bin. Their q_peak column (0.911)
and the one here (~0.60) are different quantities and must never be placed side by side.
`magnetization` and `spin_correlation` are computed identically in both and do compare.
