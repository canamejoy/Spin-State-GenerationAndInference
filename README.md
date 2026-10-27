# Spin State Generation and Inference

An end-to-end deep learning framework for **generating and inverting 2D magnetic domain images** of nanodots governed by the extended Heisenberg Hamiltonian.

This repository contains the code for two complementary papers:

| | Paper | Task | Status |
|---|---|---|---|
| 📄 | *Regression-Based Explainable DL for Estimating Hamiltonian Parameters from Magnetic Nanodot Images* | **Inverse:** image → parameters | Published: https://doi.org/10.1016/j.matdes.2026.116816 |
| 📝 | *Image-Driven Estimation of Magnetic Configurations in Nanodots* | **Direct:** parameters → image | In preparation |

---

## Overview

Magnetic nanodots exhibit rich topological spin textures (skyrmions, helical domains, labyrinthine patterns) governed by a competition of exchange, Dzyaloshinskii–Moriya (DMI), anisotropy, and Zeeman interactions. Both predicting these textures from parameters and recovering parameters from images are challenging inverse/direct problems due to the **degeneracy** of the magnetic energy landscape.

This project addresses both directions with a closed **cycle**:

```
θ (8 Hamiltonian parameters)
        │
        ▼
   ┌─────────┐
   │  DDPM   │  Direct task: conditional image generation
   └─────────┘
        │  generated image (39×39)
        ▼
   ┌──────────┐
   │ Xception │  Inverse task: parameter regression
   └──────────┘
        │
        ▼
      θ̂ (recovered parameters)
```

---

## Repository Structure

```
.
├── section0-mcmc.gpu/                 # Monte Carlo on the GPU: why the replica
│   │                                  #   count changes the cost, and the kernel
│   ├── informe/                       #   START HERE — 32-page report, HTML + PDF
│   ├── paginas/                       #   the six interactive pages
│   ├── codigo/                        #   sim_core, triton_halo, triton_bench, app
│   ├── cuadernillos/                  #   notebooks, including the Kaggle ones
│   ├── resultados/                    #   ~130 measurement files
│   └── docs/                          #   01_hamiltonian, 02_monte_carlo, 10, 11
│
├── section1-generative/               # Direct problem: params → image
│   ├── modelos/                       #   CVAE and DDPM training
│   ├── evaluacion/                    #   cross-model comparison, robustness
│   └── docs/04_direct_problem.md
│
├── section2-inverse/                  # Inverse problem: image → params (PaperInverseProblemEstimation)
│   ├── 01_data_generation/            #   Monte Carlo simulation (JAX GPU)
│   ├── 02_models/                     #   CNN baseline, ResNet-50, DenseNet-121, Xception, ViT
│   ├── 03_results/                    #   Metrics, comparison, RAMs interpretability
│   ├── cuadernillos/                  #   Historical Xception notebooks & PyTorch port
│   └── docs/03_inverse_problem.md
│
├── section3-cycle/                    # End-to-end cycle: θ → DDPM → Xception → θ̂
│   ├── ciclo/                         #   the cycle notebooks
│   ├── integracion/                   #   training/ + evaluation/ (see its README)
│   ├── archivo/                       #   superseded runs, kept as a record
│   └── docs/05_complete_cycle.md
│
├── main/                              # The article
│   ├── main.tex, Makefile, figures/   #   LaTeX source
│   └── papers/                        #   bibliography PDFs      (gitignored)
│
├── common/                            # Shared across sections
│   ├── utils/metrics.py               #   metric module — single source of truth
│   ├── tests/                         #   59 tests
│   ├── scripts/                       #   tooling (Modal budget guard)
│   └── docs/                          #   06_datasets, 07_metrics, 08, 09, 12,
│                                      #   latex/, planning/
│
├── openspec/                          # Change records (paths left historical)
├── requirements.txt
└── README.md
```

> **Sections are the unit of organisation.** Each one holds its own notebooks,
> results and documentation, so a section can be read without the others.
> `section0-mcmc.gpu/informe/informe.pdf` is the model the other sections'
> reports should follow.

---

## Datasets

Both datasets are publicly available on Kaggle:

| Dataset | Slug | Samples | Used for |
|---|---|---|---|
| Internal | `carloscanamejoy/dataset-spines-united-v2` | 169,671 | Model training + internal cycle |
| External | `carloscanamejoy/dataset-spines-complete` | 218,256 | External cycle (generalisation test) |

Each sample: a 39×39 grayscale $s_z$ magnetisation image + 8 Hamiltonian parameters.  
See [common/docs/06_datasets.md](common/docs/06_datasets.md) for full details and preprocessing rules.

---

## Notebooks at a Glance

### 1. Xception Inverse Model — `section2-inverse/cuadernillos/`

Trains an Xception CNN to predict 8 Hamiltonian parameters from a magnetisation image.

**Key results on test set:**

| Parameter | $R^2$ | MAE |
|---|---|---|
| $\tilde{K}_{DM}$ | 0.9498 | 0.049 meV |
| $\tilde{J}_2$ | 0.9146 | 0.030 meV |
| $T^{(0)}$ | 0.8430 | 0.890 K |

See [section2-inverse/cuadernillos/xception_inverse.md](section2-inverse/cuadernillos/xception_inverse.md).

### 2. DDPM Generative Model — `section1-generative/modelos/`

Trains a conditional DDPM (U-Net backbone, cosine schedule, EMA) to generate 39×39 magnetisation images conditioned on 8 Hamiltonian parameters. Uses PyTorch.

See [section1-generative/modelos/ddpm_train.md](section1-generative/modelos/ddpm_train.md).

### 3. Complete Cycle — Internal Dataset — `section3-cycle/ciclo/`

Runs the full θ → DDPM → image → Xception → θ̂ pipeline on the internal test split. Evaluates with regression metrics (R², MAE), image metrics (MSE, SSIM, FFT-Corr), three physical observables (M, C_nn, q_peak), and per-magnetic-phase breakdowns.

See [section3-cycle/archivo/ciclo_completo.md](section3-cycle/archivo/ciclo_completo.md).

### 4. Complete Cycle — External Dataset — `section3-cycle/ciclo/` 🔧

Same pipeline applied to the larger external dataset (218k). Currently under investigation for distribution shift issues.

See [section3-cycle/archivo/ciclo_external.md](section3-cycle/archivo/ciclo_external.md).

### 5. Integrated Cycle — `section3-cycle/integracion/`

Three ways of closing the loop between the generator and the inverse model, using
the 256-d Xception latent (the regression head removed) as the meeting point.
The training loss is `denoise_MSE + λ·(1 − cos(z(x̂₀), z(x₀)))` — no physical term.

| Notebook | Closure | Fine-tunes |
|---|---|---|
| `01_ddpm_cosine_frozen_encoder` | Latent cosine in the DDPM loss | DDPM only |
| `02_ddpm_cosine_joint_finetune` | Same, encoder also trains | DDPM + encoder |
| `03_latent_guided_sampler` | Latent guidance at sampling time | Neither |

See [section3-cycle/integracion/README.md](section3-cycle/integracion/README.md).

---

## Physical Metrics — the canonical set

Exactly three observables, all computed on the nanodot disk (R_d = 18.25 MUC,
N = 1049 px) of the **cropped 39×39** image:

| Observable | Symbol | Meaning |
|---|---|---|
| Average magnetisation | M | Net spin polarisation |
| Nearest-neighbour spin correlation | C_nn | Local alignment of the s_z projection |
| Peak wave vector | q_peak | Dominant spatial frequency of the texture |

**Crop before mask.** The DDPM canvas is 40×40, reflect-padded on the right and
bottom edges. `topleft_crop` recovers the original 39×39 pixels exactly; masking
the canvas instead would average duplicated pixels on a half-pixel-displaced
disk. `metrics.py` rejects a 40×40 array to make that mistake impossible.

|M|, χ, C_v and E were removed — see [common/docs/07_metrics.md](common/docs/07_metrics.md)
for the reasoning.

---

## Theoretical Background

| Document | Content |
|---|---|
| [01_hamiltonian.md](section0-mcmc.gpu/docs/01_hamiltonian.md) | Full Hamiltonian with all 8 parameters, physical ranges, emergent phases |
| [02_monte_carlo.md](section0-mcmc.gpu/docs/02_monte_carlo.md) | MH-MC algorithm, simulated annealing, image generation pipeline |
| [03_inverse_problem.md](section2-inverse/docs/03_inverse_problem.md) | Regression formulation, Xception, RAMs interpretability, identifiability analysis |
| [04_direct_problem.md](section1-generative/docs/04_direct_problem.md) | CVAE (ELBO, KLD, β-scheduling) and DDPM (forward/reverse process) |
| [05_complete_cycle.md](section3-cycle/docs/05_complete_cycle.md) | Full cycle pipeline, preprocessing rules, evaluation protocols |
| [06_datasets.md](common/docs/06_datasets.md) | Dataset descriptions, splits, MinMaxScaler alignment rules |
| [07_metrics.md](common/docs/07_metrics.md) | All metrics with formulas: regression, image, physical |

---

## Quick Start (Google Colab)

Each notebook is self-contained and designed to run on **Google Colab with GPU** (A100 recommended for DDPM training).

**Prerequisites:**
1. Upload your `kaggle.json` API key when prompted
2. Select **Runtime → Change runtime type → GPU**
3. Run cells top to bottom

**Dependency installation** is handled inside each notebook via `!pip install`.

For a local environment, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Hamiltonian Parameters at a Glance

The 8 free parameters of the extended Heisenberg Hamiltonian (with $\tilde{J}_1 = 1.0$ meV fixed):

| # | Symbol | Role | Range |
|---|---|---|---|
| 0 | $T^{(0)}$ [K] | Annealing temperature | 0.2–20.0 |
| 1 | $\tilde{J}_2$ [meV] | 2nd-shell exchange | −0.66–0.66 |
| 2 | $\tilde{J}_3$ [meV] | 3rd-shell exchange | −0.29–0.29 |
| 3 | $\tilde{J}_4$ [meV] | 4th-shell exchange | −0.23–0.23 |
| 4 | $\tilde{K}_{an1}$ [meV/atom] | Bulk anisotropy | 0.0–0.2 |
| 5 | $\tilde{K}_{anS}$ [meV/atom] | Surface anisotropy | 0.0–0.2 |
| 6 | $\tilde{H}_{ex}$ [meV/atom] | External Zeeman field | 0.0–0.05 |
| 7 | $\tilde{K}_{DM}$ [meV] | DMI strength | 0.0–1.2 |

---

## Authors

- J. Méndez-Rondón — AI-LAB/Signal Processing and Recognition Group, Universidad Nacional de Colombia
- C. Canamejoy-Alegría — Signal Processing and Recognition Group, Universidad Nacional de Colombia
- M. García-Quimbayo — Signal Processing and Recognition Group, Universidad Nacional de Colombia
- A. Álvarez-Meza — AI-LAB/Signal Processing and Recognition Group, Universidad Nacional de Colombia
- J. Montes-Monsalve — Dirección Académica, Universidad Nacional de Colombia, Sede La Paz
- J. Agudelo-Giraldo — Departamento de Física y Matemáticas, Universidad Autónoma de Manizales

**Funding:** Hermes 62642, Universidad Nacional de Colombia; Code 111908, Minciencias (951-2024).
