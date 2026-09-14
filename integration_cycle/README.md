# Integration cycle

Three Kaggle notebooks that integrate the Xception inverse-model's latent space into the
conditional DDPM, replacing the earlier physical-loss experiment. **No notebook computes a
physical loss term** — the only new loss ingredient is a cosine-similarity term between
encoder latents.

## Kaggle datasets (attach all four to every notebook)

| Dataset | File used |
|---|---|
| `carloscanamejoy/dataset-spines-united-v2` | `dataset_unificado_v2.npz` |
| `carloscanamejoy/weights-xception-model` | `xception_regressor_torch.pt` |
| `carloscanamejoy/weights-models` | `ddpm_spines_final_39crop.pt` |
| `carloscanamejoy/physicalmetrics` | `metrics.py` |

Paths are never hardcoded — each notebook resolves them at runtime with
`find_file(name)`, which globs `/kaggle/input/**/<name>`.

## The loss (notebooks 01 and 02)

```
L = L_denoise + lambda_cos * (1 - cos(z(x0_pred_39), z(x0_39)))
```

- `L_denoise`: the existing min-SNR-weighted epsilon MSE (unchanged).
- `x0_pred = ((x_t - sqrt_1a * eps_pred) / sqrt_a).clamp(-1, 1)`, then top-left cropped to
  39x39 (`metrics.topleft_crop`) before it ever reaches the encoder.
- `z(.)`: the 256-d Xception-encoder latent — the activation **after** the ReLU of
  `Dense(256)` and **before** the final `Dense(8)`. The target branch `z(x0_39)` is always
  `.detach()`ed.
- **t-window mask**: the cosine term only contributes for `t < COSINE_T_MAX` (default 400).
  At high `t`, `x0_pred` is a clamped, near-pure-noise reconstruction whose latent carries no
  usable signal — this is the known failure mode of the earlier physical-loss run, and the
  reason the mask is mandatory, not optional.
- `lambda_cos` ramps `0 -> LAMBDA_COS_FINAL` (default 0.1) with a cosine schedule over
  `LAMBDA_WARMUP_EPOCHS` (default 5) epochs.
- `COSINE_EVERY_N_STEPS` (default 1) can throttle how often the cosine term (and its encoder
  forward pass, the dominant per-step cost) is evaluated.

Both notebooks **warm-start from the published DDPM checkpoint** (`ddpm_spines_final_39crop.pt`)
and continue training with this loss added — they do not train the diffusion model from
scratch.

## Notebook 01 — `01_ddpm_cosine_frozen_encoder.ipynb`

DDPM fine-tunes with the cosine loss above; the encoder is **frozen**
(`requires_grad_(False)` + `.eval()`) and asserted to have zero trainable parameters.
Gradients flow through the encoder into `x0_pred` but never update its weights — it is a
fixed target space the DDPM is pulled towards.

Key constants: `BEST_HPARAMS` (DDPM), `FINETUNE_EPOCHS=15`, `COSINE_T_MAX=400`,
`LAMBDA_COS_FINAL=0.1`, `LAMBDA_WARMUP_EPOCHS=5`, `COSINE_EVERY_N_STEPS=1`.

**Reading the results**: training curves show total/denoise/cosine loss, validation SSIM,
and the lambda/window-fraction schedule. The final evaluation block (shared protocol, see
below) reports physical-metric R², SSIM, masked MSE, and full-cycle R²/MAE/RMSE for the
`DDPM+cos` model — compare these against the plain DDPM baseline and against the joint
fine-tune in notebook 02.

## Notebook 02 — `02_ddpm_cosine_joint_finetune.ipynb`

Same loss, but the encoder's weights **also** train, with a separate optimizer param group
(`lr_encoder = 1e-5` vs `lr = 2.26e-4` for the DDPM), the same linear warmup (scaled to its own
LR), and its own EMA object. The encoder module stays in `.eval()` mode throughout (so
BatchNorm/Dropout remain deterministic) even though its weights are being updated.

Two guardrails are logged every epoch and plotted at the end:
1. **Latent collapse** — mean per-dimension std of `z` over a fixed probe batch. A warning
   prints if it drops below 50% of the epoch-0 value (a collapsing encoder makes every
   cosine ~1, making the loss term vacuous).
2. **Inverse-task regression** — per-parameter R² of the full encoder+head, recomputed each
   epoch on the held-out inverse-model test split. This is the real cost of joint
   fine-tuning: if R² degrades, generative fidelity was bought by damaging the inverse
   model.

Key constants: same as notebook 01, plus `LR_ENCODER=1e-5`.

**Reading the results**: in addition to notebook 01's plots, check the two guardrail
subplots — the latent-std curve against its 50%-of-baseline threshold line, and the
inverse-task R² curve. If R² falls noticeably vs. notebook 01/the published 0.9498 / 0.9146 /
0.8430 reference, joint fine-tuning is trading inverse-model accuracy for generative
fidelity, and that trade should be reported explicitly, not hidden.

## Notebook 03 — `03_latent_guided_sampler.ipynb`

**No fine-tuning of anything.** DDPM and encoder are both frozen and loaded from their
published checkpoints, unchanged.

- **Step A** trains a small `LatentPredictor` (`Linear(8,256) -> SiLU -> Linear(256,512) ->
  SiLU -> Linear(512,256)`) mapping the 8 physical conditioning parameters to a target
  latent `z*`, on latents precomputed by the frozen encoder. Loss:
  `(1 - cos(pred, z_target)) + 0.1 * MSE`. This is a lightweight auxiliary network, not a
  fine-tune of either big model — the reported **held-out cosine similarity is the ceiling
  on how good the guidance can possibly be**.
- **Step B** adds classifier-guidance-style gradient steps inside `fast_sample`: while
  `t < GUIDANCE_T_MAX` (default 400), `eps_pred` is nudged towards higher cosine similarity
  between the predicted `x0`'s latent and `z*`, with the gradient normalized per sample so
  `GUIDANCE_SCALE` stays scale-free.
- **Step C** sweeps `GUIDANCE_SCALE in [0, 0.5, 1, 2, 4, 8]` (`0` is the unguided baseline
  and is asserted to reproduce the plain DDPM). For each scale it reports physical-metric
  R², SSIM, masked MSE, latent cosine, and cycle R²/MAE/RMSE.

Key constants: `PREDICTOR_EPOCHS=30`, `PREDICTOR_LR=1e-3`, `PREDICTOR_MSE_WEIGHT=0.1`,
`GUIDANCE_T_MAX=400`, `GUIDANCE_SCALES=[0, 0.5, 1, 2, 4, 8]`, `DEFAULT_EVAL_SCALE=1`.

**Reading the results**: the three-panel sweep plot shows physical-metric R² and SSIM
degrading, and latent cosine rising, as `GUIDANCE_SCALE` increases. The expected failure
mode — texture artefacts with a *rising* latent cosine at high scale (the image looks less
physical even as it moves closer to `z*`) — is the headline result to look for, not an
error. The canonical evaluation block at the end runs the full shared protocol at
`DEFAULT_EVAL_SCALE` for a single representative parity plot and per-phase breakdown.

## Shared evaluation protocol (every notebook)

Using `metrics.py` only, on the internal test split (`TEST_FRACTION` controls how much of it
— lower it for a cheap smoke run):
1. Crop to 39x39 first (`topleft_crop`), then compute every metric — never on the 40x40
   canvas.
2. The three canonical physical metrics (`magnetization`, `spin_correlation`,
   `peak_wave_vector`), original vs. generated, with R² and a parity plot.
3. SSIM and masked MSE.
4. Full cycle `theta -> DDPM -> encoder+head -> theta_hat`: per-parameter R², MAE, RMSE.
5. Per-magnetic-phase breakdown via `get_structure_label` (skipped if the dataset has no
   `labels` array).
6. `apply_figure_style()` runs once near the top of each notebook; every figure is saved as
   both PNG and SVG via `save_figure()` into `/kaggle/working/<notebook>/`.

## Non-negotiables honoured

- No physical loss term anywhere (the removed loss used `magnetization`,
  `abs_magnetization`, `spin_correlation`, and a soft peak-wavevector proxy — none of it
  survives here).
- No notebook imports or computes `abs_magnetization`, `oz_fit`, or `chi_ensemble` (all
  three were removed from `metrics.py` upstream).
- No physical metric is ever computed on a 40x40 array — every call site is preceded by a
  crop to 39.
- `SEED = 42`, and `random`, `numpy`, `torch` are all seeded.
- Each notebook resolves its inputs only from the four required Kaggle datasets via
  `find_file()` — no hardcoded paths.

## What could not be executed

These notebooks were built and statically verified (JSON validity, cell schema, `compile()`
over the concatenated source, and the `grep`/crop-ordering checks above) but **not run** —
there is no GPU, no Kaggle environment, and no `numpy`/`torch` in the tool sandbox available
to this build. The mandatory encoder parity check (Section 9.2 of the spec) and the
guidance-scale-0 sanity assertion (notebook 03) are real runtime assertions that will only
be exercised the first time each notebook runs on Kaggle.
