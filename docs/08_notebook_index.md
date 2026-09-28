# Notebook and test index

What exists, where it lives, and what it produced. Two repositories are
involved and they are easy to confuse:

| Repository | Holds |
|---|---|
| `Spin-State-GenerationAndInference` | generative models, the inverse model, the cycle, and all image-space metrics |
| `Diffusion-Accelerated-MCMC` | the `damcmc` Monte Carlo simulator and the device-scaling benchmark |

`01_baseline_simulation.ipynb` at the root of this repository is a copy
downloaded from Windows. It imports `src/damcmc`, which exists only in the other
repository, so it cannot run here; the maintained version is
`notebooks/01_baseline_benchmark.ipynb` over there.

---

## 1. Generative models — `notebooks/generative/`

| Notebook | What it did |
|---|---|
| `ddpm_spines_train.ipynb` | Final conditional DDPM retrain with the 39x39 crop (Colab + A100). Produced `ddpm_spines_final_39crop.pt`, the base checkpoint every later arm is measured against. 18.06 M parameters, `base_ch=80`, `ch_mults=(1,2,4)`. |
| `ddpm-spines-lee-style.ipynb` | DDPM following the reference paper's architecture literally, for comparison against the adapted one. |
| `ddpm-spines-newlossfunction.ipynb` | DDPM with physical metrics added to the loss. Scored best on every physical metric in the three-DDPM comparison. |
| `ncsn_v2_adagn.ipynb` | Conditional NCSN with AdaGN and a Karras-style lambda(sigma). |
| `cvae-xception.ipynb` | CVAE with an Xception encoder plus SSIM; fixed beta-schedule with reduced beta_max. |
| `cvae-vit.ipynb` | CVAE with a ViT encoder, dual-GPU, linear vs plateau schedule using the Optuna v2 winners. |
| `cvae-8params.ipynb` | CVAE conditioned on all 8 Hamiltonian parameters. |
| `cvae_xception_retrain.ipynb` / `cvae_vit_retrain.ipynb` | Both CVAEs retrained on 100% of the dataset. |

## 2. Inverse model — `notebooks/inverse/`

| Notebook | What it did |
|---|---|
| `XceptionFullDataBaseV3100.ipynb` | Original Keras training of the Xception regressor (image -> 8 parameters). |
| `xception-keras-to-torch.ipynb` | Faithful hand-port of the Keras weights to PyTorch with parity verification. 288 state-dict tensors, 21.3 M parameters. `KERAS_BN_EPS = 1e-3` is set explicitly because eps is not stored in the state dict. |

## 3. Full cycle — `notebooks/cycle/`

| Notebook | What it did |
|---|---|
| `ciclo_completo_v2.ipynb` | End-to-end evaluation theta -> DDPM -> image -> Xception -> theta. |
| `cycle_complete_newmetrics.ipynb` | The same cycle scored with the three physical metrics (magnetisation, spin correlation, peak wavevector). |
| `ciclo_texture_fidelity.ipynb` | Texture fidelity restricted to the non-saturated regime (low H_ex), where the texture actually carries information. |

## 4. Evaluation — `notebooks/evaluation/`

| Notebook | What it did |
|---|---|
| `generative_comparison.ipynb` | CVAE-Xception vs CVAE-ViT vs DDPM. |
| `physical_metrics_comparison.ipynb` | Generative models compared on physical metrics (Colab). |
| `physical-metrics-comparison-4models.ipynb` | Final four-model comparison: physical metrics plus the visual panel. |
| `ddpm-physical-metrics-check.ipynb` | Physical-metric sanity check on DDPM output. |
| `ddpm_steps_sweep.ipynb` | Sampling-step sweep, 25 vs 50 vs 100. 100 steps is what the comparison protocol uses. |
| `image-metrics-robustness.ipynb` | Robustness of the image metrics to noise and perturbation. |

## 5. Cycle integration — `notebooks/integration/`

Three ways of closing the loop between the DDPM and the inverse model, each
trained separately, then all four arms evaluated under one protocol.

### Training — `integration/training/`

| Notebook | What it did |
|---|---|
| `01_ddpm_cosine_frozen_encoder.ipynb` | Latent cosine loss added to DDPM training with the Xception encoder frozen. `COSINE_T_MAX = 400` restricts the term to low-noise timesteps; in-window subsetting before the encoder is numerically identical and 3.6x faster. |
| `02_ddpm_cosine_joint_finetune.ipynb` | The same term with the encoder fine-tuned jointly. Without an anchor this collapses (cosine 0.0000 by epoch 3, inverse R^2 = -1.1e5); the regression anchor `lambda_inv * \|\|f_phi(X) - theta\|\|^2` keeps the inverse model positive (0.60 -> 0.67). |
| `03_latent_guided_sampler.ipynb` | Classifier-style latent guidance at sampling time: `eps_hat <- eps_hat + s * sqrt(1 - alpha_bar) * grad_x(1 - cos) / \|\|grad\|\|`. Trains the small `theta -> z*` head; fine-tunes nothing. |

### Evaluation — `integration/evaluation/`

Same protocol in all four arms: K = 32 samples per theta, N = 1000 stratified
thetas, 100 sampling steps. Split one notebook per architecture because four do
not fit in a single Kaggle session.

| Notebook | What it did |
|---|---|
| `01_ddpm_base.ipynb` | Reference arm: base checkpoint, guidance scale 0. |
| `02_cos_frozen.ipynb` | `ddpm_cosine_frozen_encoder.pt`, guidance scale 0. |
| `03_cos_joint.ipynb` | `ddpm_cosine_joint_finetune.pt` (anchored), guidance scale 0. |
| `04_latent_guidance.ipynb` | Base checkpoint with guidance on at scale 8.0. Writes a partial `.npz` every 25 theta chunks; it is the slowest arm. |
| `05_cross_encoder_test.ipynb` | Not an architecture arm. Asks whether the information guidance adds is readable by a *different* encoder. It is not: on guided images Xception recovers J3 at R^2 = 0.985 while a separately trained ViT recovers it at 0.058, on the same images from which it reads T, J2, KanS and KDM at 0.88-0.93. |
| `_nbcheck.py` | Order-aware AST checker: catches names used before the cell that defines them, and orphaned decorators. `compile()` catches neither. |

### Results

| Model | magnet. | C_nn | q_peak | SSIM | cycle R^2 | V_pix | V_lat |
|---|---|---|---|---|---|---|---|
| DDPM base | 0.7954 | 0.8521 | 0.5924 | 0.1753 | 0.6387 | 13.29 | 1.51 |
| + cos frozen | 0.7886 | 0.8292 | 0.5046 | 0.1591 | 0.6488 | 14.22 | 1.43 |
| + cos joint (anchored) | 0.7948 | 0.8398 | 0.5168 | 0.1746 | 0.7997 | 13.10 | 1.45 |
| + guidance | 0.7953 | 0.8509 | 0.5996 | 0.1731 | **0.9161** | 13.31 | 1.09 |

The cycle-R^2 gain from guidance is **not** a physical improvement. Feeding the
head the parameters with no image at all scores 0.9596, above guidance's 0.9161;
the cross-encoder test above shows the channel is private to Xception; and a
low-pass filter at r <= 14 collapses the guided/unguided gap from +0.823 to
+0.049, so the message lives at wavelengths below 2.8 lattice sites.

Raw outputs are in `results/kaggle/*.json`.

## 6. Archive — `notebooks/archive/`

Superseded cycle notebooks kept for provenance: `ciclo_completo_resultadosxclusters`,
`ciclo_external_dataset`, `ciclo_external_fixes`, `diagnostico-ciclo-externo`.

## 7. Monte Carlo benchmark — the other repository

`Diffusion-Accelerated-MCMC/notebooks/`

| Notebook | What it does |
|---|---|
| `01_baseline_benchmark.ipynb` | Original baseline notebook. Its `cpu` / `single_gpu` / `multi_gpu` modes all executed the same code -- there was no multi-device path in the package. |
| `02_ddpm_fewshot_finetune.ipynb` | Few-shot DDPM fine-tune and exact-sampler study. |
| `bench/00_pick_state.ipynb` | Picks the skyrmion parameter point with evidence: anneals a (K_DM, H_ex) grid at production geometry and ranks by topological charge, its stability across replicas, and M_z. |
| `bench/01_bench_cpu.ipynb` | CPU arm. |
| `bench/02_bench_gpu1.ipynb` | 1 GPU. |
| `bench/03_bench_gpu2.ipynb` | 2 GPUs, replicas split 50/50. |
| `bench/04_bench_gpu3_runpod.ipynb` | 3 GPUs on RunPod, replicas split 34/33/33. |
| `bench/05_compare.ipynb` | Timing table, speedup, physics-agreement check, side-by-side final textures, LaTeX table. |
| `bench/_generate.py` | Generates arms 01-04 from one shared body so they cannot drift apart. |
| `scripts/kaggle_chain.py` | Chains the CPU arm across ~11 Kaggle sessions via a checkpoint dataset round-trip. |

Shared configuration for all four arms: `Rd = 18.30`, 5 layers (39x39), 200
temperature points from 20.0 K to 0.1 K in 0.1 K steps, 100 replicas,
10,000 + 5,000 sweeps. Measured cost is 143 ms per sweep, so the CPU arm is
about 119 h.

---

## 8. Tests

### `Spin-State-GenerationAndInference/tests/` — 31 tests

| File | Covers |
|---|---|
| `test_topology.py` (14) | Level-set topological descriptors and Euler characteristic curves in `notebooks/utils/metrics.py`. |
| `test_orientation.py` (17) | Orientational order (psi2, psi6, anisotropy) and correlation range. |

### `Diffusion-Accelerated-MCMC/tests/` — 126 tests

| File | Covers |
|---|---|
| `test_hamiltonian.py` | Both engines produce identical total energies; the two documented convention switches. |
| `test_geometry.py` | Disk mask, checkerboard split, bulk/surface classification, neighbour tables. |
| `test_moves.py` | Cone and uniform proposals, sigma adaptation and its bounds. |
| `test_mcmc_boltzmann.py` | A single-spin chain reproduces the Boltzmann distribution. |
| `test_mcmc_sigma_adaptation.py` | Sigma is frozen during measurement -- adapting there would break detailed balance silently. |
| `test_checkpoint.py` | Save and resume round-trip. |
| `test_symmetry.py` | Exact D4 x time-reversal augmentation. |
| `test_path_mh.py` | Exact path-space Metropolis-Hastings with diffusion proposals. |
| `test_topological_charge.py` (7) | Berg-Luscher charge: quantisation (Q -> -1 as the texture closes), antiskyrmion sign flip, exact zero on a ferromagnet, and that inert out-of-disk sites are excluded rather than evaluated. |
| `test_device_invariance.py` (11) | The load-bearing property of the benchmark: splitting the replicas across devices leaves the accept/reject sequence exactly unchanged and the observables equal to within float32 reassociation. Also the uneven 34/33/33 split, and that a resumed run reproduces an uninterrupted one while the clock keeps accumulating. |

---

## 9. Known gaps

- **Two metric modules.** `notebooks/utils/metrics.py` (image space) and
  `src/damcmc/observables.py` (Monte Carlo) both compute "physical metrics" and
  have diverged. Their disk masks are, however, identical pixel for pixel
  (1049 px on 39x39), so a common module is reachable without regenerating data.
- **Topological charge is not comparable across the two.** Berg-Luscher Q needs
  the full spin vector; the generative models emit s_z only. Q can validate the
  Monte Carlo reference state but cannot be an axis of model comparison. See
  `07_metrics.md` section 4.1.
- `01_baseline_simulation.ipynb` at the repository root is a dead duplicate.
