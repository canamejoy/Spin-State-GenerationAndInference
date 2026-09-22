# Structural outline: companion Materials & Design article

## 1. Bibliographic front matter (source text)

### Title

Regression-based explainable deep learning for estimating Hamiltonian parameters from magnetic nanodot images

### Authors and affiliation letters

- Juan Sebastián Méndez-Rondón a iD
- María Isabel García-Quimbayo a iD
- Andrés Marino Álvarez-Meza a,∗ iD
- Jorge Iván Montes-Monsalve a iD
- Jose Darío Agudelo-Giraldo b, c iD

### Affiliations

- a AI-Lab, Signal Processing and Recognition Group, Universidad Nacional de Colombia, Sede Manizales, 170003 Manizales, Colombia
- b Grupo de Investigación en Física y Matemáticas con Énfasis en la Formación de Ingenieros, Departamento de Física y Matemáticas, Facultad de Ingeniería, Universidad Autónoma de Manizales, Antigua Estación del Ferrocarril, 170002 Manizales, Colombia
- c Departamento de Física, Facultad de Ciencias Exactas y Naturales, Universidad de Caldas, 170002 Manizales, Colombia

## 2. Highlights (verbatim)

- Atomistic Hamiltonian generates relaxed 2D magnetic domain images for training.
- Deep learning regression recovers Hamiltonian parameters from nanodot images.
- Regression Activation Maps explain continuous prediction errors spatially.
- Framework isolates degenerate textures linking black-box AI to magnetic physics.

## 3. Keywords (verbatim)

Explainable AI

Magnetic nanodots

Hamiltonian parameters

Regression activation maps

Deep learning

## 4. Abstract (verbatim)

Magnetic nanodots host topological spin textures whose stability is dictated by an extended Heisenberg Hamiltonian combining symmetric exchange, the Dzyaloshinskii–Moriya interaction, magnetocrystalline anisotropy, and the Zeeman field. Inferring these atomistic parameters from two-dimensional magnetization images underpins the design of functional magnetic materials, yet it remains an ill-posed inverse problem: under certain external conditions and transient states, configurations relax into noisy, weakly discriminative textures that render the dataset ill-conditioned. Deep-learning estimators act as opaque black boxes and rely on slow atomistic simulations. We propose an explainable regression methodology whose novelty lies in data generation and interpretability: a matrix decomposition of the extended Hamiltonian enables GPU-accelerated Metropolis–Hastings Monte Carlo with simulated annealing, accelerating computation without altering the underlying physics, while Regression Activation Maps (RAMs), a novel error-based attribution metric, turn the network into a traceable gray-box model. Trained on several configurations, convolutional and Vision Transformer regressors reliably recover the physical parameters from simulated nanodot images, while the higher-order exchange constants remain bounded by the intrinsic identifiability limits of the 𝑠𝑧 projection. Comparing both families through their attention maps and accuracy-versus-size trade-off, RAMs isolate the chiral domain walls and skyrmion cores driving predictions.

## 5. Numbered section and subsection hierarchy (exact heading wording/order)

1. `1 . Introduction`
2. `2 . Related work`
3. `3 . Materials and methods`
   1. `3.1 . Atomistic Hamiltonian framework`
   2. `3.2 . GPU-accelerated image dataset generation`
   3. `3.3 . Deep regression framework`
   4. `3.4 . Spatial interpretability via RAMs`
      1. `3.4.1 . RAMs for CNN-based approaches`
      2. `3.4.2 . RAMs for ViT-based approaches`
      3. `3.4.3 . Normalized RAMs for layer-wise analysis`
4. `4 . Experimental set-up`
   1. `4.1 . Atomistic simulations details`
   2. `4.2 . Deep learning training`
   3. `4.3 . Spatial interpretability: implementation details`
5. `5 . Results and discussion`
   1. `5.1 . Thermodynamic simulation of topologies`
   2. `5.2 . Deep learning regression results`
   3. `5.3 . RAMs-based interpretability results`
   4. `5.4 . Interpretability benchmark: RAMs vs. baselines`
   5. `5.5 . Limitations`
6. `6 . Conclusions`

Unnumbered back matter visible in the extraction: `CRediT authorship contribution statement`; `Declaration of Generative AI and AI-assisted technologies in the writing process`; `Declaration of competing interest`; `Acknowledgements`; `Appendix A . Supplementary data`; `Data availability`; `References`.

## 6. Figures (number and full caption, in order)

1. **Fig. 1.** Flowchart of the GPU-accelarated image-based magnetic dataset generation. Foundational inputs (green), the Hamiltonian parameters 𝜽, the unit-cell geometry, and the external directional vectors, drive the atomistic Metropolis Monte Carlo simulation with simulated annealing (blue), relaxing the system into a converged 3D topological state. The resulting vector field is then projected, rendered, and preprocessed (yellow) into the labeled dataset D mapping 2D 𝑠𝑧 images to their generating parameters. The bottom inset Details the three levels of GPU parallelization: (a) spins, within a single lattice, the nearest-neighbor Hamiltonian makes the two checkerboard sublattices A/B conditionally independent, so all sites of one sublattice are updated simultaneously (with the other held fixed) before switching, reproducing a sequential sweep; (b) replicas, the same parameter set 𝜽 is run with different random seeds as independent Markov chains, batched on a GPU; and (c) experiments: different parameter sets 𝜽 are distributed across GPUs via single-program multiple-data (pmap) execution. The annealing temperature and the Monte Carlo sweeps 𝜇(𝑡) →𝜇 (𝑡+1) remain strictly sequential, so the parallelization introduces no approximation to the underlying physics. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
2. **Fig. 2.** RAMs extraction pipeline. Latent feature maps and continuous predictions from the network backbone are explicitly coupled via an error-decay objective. Gradient-based importance weights combine these features to produce a normalized, physically traceable spatial saliency map for the target Hamiltonian parameter 𝜃𝑝.
3. **Fig. 3.** Thermodynamic convergence of a representative nanodot during simulated annealing. The main panel shows the total Hamiltonian energy H [meV] versus the annealing temperature 𝑇 (0) [K]; each inset shows the 𝑠𝑧 field of the central layer at that state, tracing the transition from a disordered paramagnetic phase at high 𝑇 (0) to a relaxed labyrinthine configuration at 𝑇 (0) = 0.1 K.
4. **Fig. 4.** Two-dimensional phase diagram of the magnetization texture 𝐗𝑛 as a function of the annealing temperature 𝑇 (0) and the DMI strength ̃𝐾𝐷𝑀. Each cell shows an enlarged representative image on a coarse sampling grid; the background color encodes ̃𝐾𝐷𝑀. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
5. **Fig. 5.** Mean 𝑅2 over the identifiable Hamiltonian parameters versus the number of trainable weights (log scale) for the five architectures. The nearly flat trend, with the compact CNN already matching models two orders of magnitude larger, indicates that accuracy is limited by the physical information in the 𝑠𝑧 projection, not by model capacity.
6. **Fig. 6.** UMAP-based 2D projection of the Xception Latent space on the held-out test set, colored by each of the eight Hamiltonian parameters 𝜽= [𝑇 (0), ̃𝐽2, ̃𝐽3, ̃𝐽4, ̃𝐾𝑎𝑛1, ̃𝐾𝑎𝑛𝑆, ̃𝐻𝑒𝑥, ̃𝐾𝐷𝑀]. Each point represents a magnetization image 𝐗𝑛 projected from the high-dimensional feature space of the Xception backbone into two dimensions via UMAP.
7. **Fig. 7.** Representative magnetization image 𝐗𝑛 for each of the five magnetic phase categories identified by clustering the Xception Latent space (the sample nearest the cluster centroid in the UMAP embedding); color encodes the out-of-plane component 𝑚𝑧 (+1 spin up, −1 spin down). The phases span ordered helical stripes, thermally disordered others, mixed-chirality bimerons, dense skyrmion lattices, and sharp Ising domain walls. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
8. **Fig. 8.** Sensitivity analysis of the RAM precision factor 𝛾. Left: shape of the Gaussian objective O𝑝 =𝑒 −𝛾𝜀² versus the prediction error 𝜀=|𝜃𝑝 − ̂𝜃𝑝| for 𝛾∈ {0.1,…,100} (selected 𝛾= 10 in red; vertical dotted line = empirical mean error). Right: O𝑝 evaluated at each parameter’s mean error as a function of 𝛾; the horizontal dotted line marks the gradient-collapse threshold O𝑝 = 0.05. The value 𝛾= 10 is the selected operating point. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
9. **Fig. 9.** RAMs ̃R(𝑙,𝑝)RAM for the Xception architecture in the helical phase. Rows represent three Hamiltonian parameters {𝑇 (0), ̃𝐽2, ̃𝐾𝐷𝑀}; columns correspond to three representative network layers, block1 (entry, B1), block6 (middle, B6), and block13 (exit, B13), with the original 𝑠𝑧 image in the leftmost column. Each panel overlays the RAM intensity (locally normalized) on the magnetization; the value below each original image indicates the ground-truth parameter.
10. **Fig. 10.** Heatmap of the mean maximum RAM activation per network layer (columns, B1–B13) and Hamiltonian parameter 𝜽 (rows), for the five magnetic phases, averaged over 𝑁= 300 test samples per phase (𝑁= 211 for Ising; Xception CNN) and measured within the nanodot disk (masked RAM). Activations are normalized row-wise (per parameter, across layers) via a sigmoid 𝜎((𝑥−𝜇)/𝜎𝑥), so warm cells mark the layers where each parameter’s gradient response is relatively strongest. The identifiable parameters peak at the exit-flow layer B13, whereas ̃𝐽3 and ̃𝐽4 show a flat, unstructured profile with no dominant layer.
11. **Fig. 11.** Physical criterion for the SLIC segmentation used by the perturbation-based baselines. The superpixel equivalent diameter is plotted against the number of SLIC segments; the selected value 𝑁= 100 (dashed red) yields superpixels whose size matches the characteristic magnetic domain width (≈ 4.9 px, dashed green), so that each interpretable region corresponds to a single magnetic domain. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
12. **Fig. 12.** Deletion curves (most relevant first) for RAM, LIME, SHAP, and SF-SHAP, with a random-relevance Lower bound, for the three representative parameters 𝑇 (0), ̃𝐽2, and ̃𝐾𝐷𝑀. The vertical axis is the induced deviation |̂𝜃pert − ̂𝜃0| as the most relevant pixels are progressively replaced by the baseline; the area under each curve (AUC, in the legend) quantifies faithfulness. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
13. **Fig. 13.** Computational cost of the interpretability methods. Left: measured wall-clock time per relevance map (log scale). Right: number of model evaluations per map (log scale). A RAM requires a single backward pass, compared to ∼ 10³ evaluations for LIME and SHAP and 128 for SF-SHAP. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

## 7. Tables (number and full caption, in order)

1. **Table 1.** Hamiltonian parameters sampling ranges. The first-shell exchange ̃𝐽1 = 1.0 meV is fixed as the energy reference. Parameters are drawn with a quasi-random (low-discrepancy) scheme within the specified bounds. Physical units: temperature in Kelvin [K], exchange and DMI constants in meV, anisotropy constants in meV/atom, external field in meV/atom.
2. **Table 2.** Configuration of evaluated deep learning architectures for inverse Hamiltonian parameter estimation. All models share a standardized regularized output head mapped to the ̆𝑃= 8-dimensional physical parameter space: BN → Drop(0.4) → Dense(256, ReLU, 𝓁2) → BN → Drop(0.3) → Dense(8). BN: Batch Normalization; Drop(𝑝): Dropout with rate 𝑝.
3. **Table 3.** Per-variable 𝑅2 and MAE across the five evaluated architectures (Xception, DenseNet, ResNet, ViT, and the conventional CNN baseline) on the held-out test set. Each parameter shows 𝑅2 (top row) and MAE (bottom row, in its physical units: K for 𝑇 (0), meV for exchange/DMI, meV/atom for anisotropy/field); the best 𝑅2 per parameter is in bold. Negative 𝑅2 indicates predictions no better than the dataset mean.
4. **Table 4.** Per-phase 𝑅2 and MAE of Xception across the Hamiltonian parameters 𝜽 for the five magnetic phases. Each parameter shows 𝑅2 (top row) and MAE (bottom row, in its physical units); the best 𝑅2 per parameter is in bold. N/A denotes cases with near-zero target variance where 𝑅2 is undefined, and a dash (–) indicates that no valid metric is available.

## 8. Main-section summaries and rough lengths (summaries, not source text)

### 1 . Introduction — roughly 135 lines / ~1,800 words

**Summary:** Establishes magnetic-nanodot Hamiltonian inference from 2D magnetization as an ill-posed inverse problem and contrasts atomistic modeling with experimental imaging and black-box deep regression. It motivates GPU-accelerated simulation and continuous-regression interpretability, then states three contributions: tensorized Hamiltonian simulation, RAMs, and CNN/ViT comparison with identifiability boundaries. It closes by previewing Sections 2–6.

### 2 . Related work — roughly 105 lines / ~1,400 words

**Summary:** Reviews indirect magnetic imaging, simulation-based parameter fitting, and their ambiguity when image-space metrics stand in for physical equivalence. It surveys deterministic, probabilistic, cycle-consistent, and physics-informed learning approaches, emphasizing computational cost, degeneracy, and opacity. The section positions the present work around atomistic modeling, GPU tensor decomposition, and RAM-based spatial attribution.

### 3 . Materials and methods — roughly 788 lines / ~7,400 words including displayed equations

**Summary:** Defines an eight-target extended Heisenberg formulation, decomposes its interaction matrices, and describes vectorized Metropolis–Hastings Monte Carlo with simulated annealing to produce normalized 2D central-layer 𝑠𝑧 images. It specifies regression models (CNN, ResNet50, DenseNet, Xception, ViT) and introduces RAMs for CNN and ViT feature representations, including normalized layer-wise analysis. It provides formal objectives for simulation, learning, RAMs, and the interpretability baselines.

### 4 . Experimental set-up — roughly 445 lines / ~4,200 words including tables/equations

**Summary:** Specifies nanodot geometry, annealing schedule, parameter ranges, dataset construction, target identifiability rationale, train/validation/test split, preprocessing, shared regression head, and optimization protocol. It details RAM extraction, comparison with LIME/SHAP/SF-SHAP, MoRF deletion evaluation, and compute/software environment. The section records the public Kaggle dataset and GitHub source-code locations.

### 5 . Results and discussion — roughly 661 lines / ~6,100 words including tables/figures

**Summary:** Validates simulated thermodynamic behavior and shows that ordered phases are more discriminative than high-temperature, noisy configurations. It compares five regressors, identifies ̃𝐽3 and ̃𝐽4 as intrinsically unidentifiable from 2D 𝑠𝑧, clusters Xception latent representations into five phases, and reports phase-specific performance. RAMs are analyzed by layer and benchmarked against perturbation methods, followed by limitations on partial observability, simulation-to-experiment transfer, sampling design, and local-gradient explanations.

### 6 . Conclusions — roughly 65 lines / ~750 words

**Summary:** Concludes that the framework combines GPU atomistic simulation, multi-architecture regression, UMAP/HDBSCAN analysis, and RAMs for continuous regression. It reports broadly comparable models with Xception strongest overall, six effectively recoverable targets, and ̃𝐽3/̃𝐽4 limited by observability rather than capacity. It proposes volumetric inputs, domain adaptation, attribution uncertainty, and active learning as future directions.

## 9. Equations, named quantities, terms, units/ranges, and metric definitions

### Equation numbering style

Parenthesized consecutive Arabic numbers, `(1)` through `(25)`, are used. Equations (1)–(4) define the Hamiltonian; (5)–(7) the thermodynamic distribution and Metropolis acceptance; (8)–(12) regression and architectures; (13)–(18) RAMs; (19)–(20) regression metrics; and (21)–(25) attribution baselines/deletion metrics.

### Main quantities and definitions recoverable from the extraction

- `H ∈ ℝ [meV]`: total effective energy/Hamiltonian; `𝐒 ∈ ℝ^{N×3}` gathers normalized spin vectors `𝐬ᵢ ∈ ℝ³`, with `‖𝐬ᵢ‖₂ = 1`.
- Extended target vector: `𝜽 = [̃𝐽1, ̃𝐽2, ̃𝐽3, ̃𝐽4, ̃𝐾𝐷𝑀, ̃𝐻𝑒𝑥, ̃𝐾𝑎𝑛1, ̃𝐾𝑎𝑛𝑆]`; the regression target is reported as `𝜽 = [𝑇(0), ̃𝐽2, ̃𝐽3, ̃𝐽4, ̃𝐾𝑎𝑛1, ̃𝐾𝑎𝑛𝑆, ̃𝐻𝑒𝑥, ̃𝐾𝐷𝑀] ∈ ℝ⁸`, with `̃𝐽1 = 1.0 meV` fixed as the energy reference.
- `̃𝐽ᵣ`: symmetric exchange at neighbor shell `r`; positive values favor parallel alignment and negative values antiparallel alignment. `̃𝐾𝐷𝑀`: DMI magnitude, first-shell only. `̃𝐻𝑒𝑥`: Zeeman field magnitude. `̃𝐾𝑎𝑛1` and `̃𝐾𝑎𝑛𝑆`: bulk and surface cubic magnetocrystalline anisotropy constants.
- `𝑇 [K]`, `k_B [meV/K]`, `Z`: temperature, Boltzmann constant, and canonical partition function. `ΔH = H(𝜇(t+1)) − H(𝜇(t))`; Eq. (7) accepts `ΔH < 0` with probability 1 and otherwise with `exp(−ΔH/(k_B T(t)))`.
- `𝐗ₙ ∈ [0,1]^{̆H×̆W}`: normalized two-dimensional `𝑠_z` magnetization image; the final dataset is `D = {𝐗ₙ, 𝜽ₙ}_{n=1}^{̆N}`. The rendered observation is the central layer’s `𝑠_z`, not recoverable as a full 3D observation.
- `f_Θ: ℝ^{̆H×̆W} → ℝ^{̆P}`: learned regressor; `L` is stated as MSE or Huber in Eq. (8), while the training protocol explicitly uses MSE. Adam is the optimizer.
- RAM objective: `Oₚ = exp(−𝛾ₚ(𝜃ₚ − ̂𝜃ₚ)²)` (Eq. 13). Feature-map weights `𝛼ₖ^(p)` are global averages of `∂Oₚ/∂Aₖᵢⱼ` (Eq. 14); `R_RAM^(p) = ReLU(Σₖ 𝛼ₖ^(p) Aₖ)` (Eq. 15). ViT tokens are reshaped to `A_ViT` (Eq. 16), and Eq. (18) globally normalizes RAMs over layers/parameters.
- `𝑅² = 1 − Σ(𝜃ₙ,ₚ − ̂𝜃ₙ,ₚ)² / Σ(𝜃ₙ,ₚ − ̄𝜃ₚ)²` (Eq. 19): proportion of target variance captured; reported N/A for near-zero target variance. `MAE = (1/N_test)Σ|||𝜃ₙ,ₚ − ̂𝜃ₙ,ₚ|||` (Eq. 20): absolute prediction error; normalized-space MAE is dimensionless.
- LIME uses a sparse, proximity-weighted linear surrogate (Eq. 21). SHAP uses Shapley values (Eq. 22) and a Shapley kernel (Eq. 23). `SF-SHAP` uses `Q = 7` radial frequency bands, so `2^Q = 128` coalitions can be exactly enumerated.
- MoRF deletion: `Δₚ^(m)(𝜌) = |||fₚ(𝐗^(m)(𝜌)) − fₚ(𝐗)|||` (Eq. 24); `AUCₚ^(m) = ∫₀^{𝜌max} Δₚ^(m)(𝜌)d𝜌` (Eq. 25). Larger AUC is defined as more spatially localized and physically faithful explanation.

### Units and sampled ranges (Table 1)

| Parameter | Physical role | Min | Max |
|---|---|---:|---:|
| `𝑇(0) [K]` | Annealing temperature | 0.0 | 20.0 |
| `̃𝐽2 [meV]` | 2nd-shell exchange | −0.33 | 0.66 |
| `̃𝐽3 [meV]` | 3rd-shell exchange | −0.29 | 0.29 |
| `̃𝐽4 [meV]` | 4th-shell exchange | −0.23 | 0.24 |
| `̃𝐾𝑎𝑛1 [meV/atom]` | Bulk anisotropy | 0.0 | 4.57 |
| `̃𝐾𝑎𝑛𝑆 [meV/atom]` | Surface anisotropy | 0.0 | 0.20 |
| `̃𝐻𝑒𝑥 [meV/atom]` | External Zeeman field | 0.0 | 1.20 |
| `̃𝐾𝐷𝑀 [meV]` | DMI strength | 0.0 | 1.20 |

Additional recoverable conversions: with `g ≈ 2`, `̃𝐻𝑒𝑥` corresponds to `≈ 0.116 meV per Tesla`; the sampled field maximum maps to `≈ 10 T`. The text reports `̃𝐾𝐷𝑀/̃𝐽1 ∼ 0.3–0.7` for typical interfacial materials and samples up to `≈ 1.2`; it reports `k_B T/̃𝐽1` up to `≈ 1.7` at `T(0) = 20 K`.

## 10. References

### Style and count

Numeric bracketed reference list, ordered `[1]` through `[55]`: approximately **55 references**. Entries use author initials/surnames, quoted title, source, volume/issue/year, and pages or article number where present.

### First 15 references (verbatim, line-wrap hyphenation repaired only)

[1] A. Heinrich, W. Oliver, L. Vandersypen, A. Ardavan, R. Sessoli, D. Loss, A. Jayich, J. Fernandez, A. Laucht, A. Morello, “Quantum-coherent nanoscience, Nat. Nanotechnol. 16 (12) (2021) 1318–1329.

[2] A. Fert, R. Ramesh, V. Garcia, F. Casanova, M. Bibes, “Electrical control of magnetism by electric field and current-induced torques, Rev. Mod. Phys. 96 (1) (2024) 015005.

[3] X. Batlle, C. Moya, M. Escoda, O. Iglesias, A. Fraile, A. Labarta, “Magnetic nanoparticles: from the nanostructure to the physical properties, J. Magn. Magn. Mater. 543 (2022) 168594.

[4] I. Charalampidis, J. Barker, “Metadynamics calculations of the effect of thermal spin fluctuations on skyrmion stability, arXiv preprint arXiv:2310.03169, 2023.

[5] X. Yu, “Magnetic imaging of various topological spin textures and their dynamics, J. Magn. Magn. Mater. 539 (2021) 168332.

[6] Y. Tokura, N. Kanazawa, “Magnetic skyrmion materials, Chem. Rev. 121 (5) (2020) 2857–2897.

[7] V. Kuchkin, B. Barton, F. Rybakov, S. Blügel, B. Schroers, N. Kiselev, “Magnetic skyrmions, chiral kinks, and holomorphic functions, Phys. Rev. Lett. 125 (18) (2020) 187202.

[8] J. Kong, Y. Ren, M. Tey, P. Ho, K. Khoo, X. Chen, A. Soumyanarayanan, “Quantifying the magnetic interactions governing chiral spin textures using deep neural networks, ACS Appl. Mater. Interfaces 16 (1) (2023) 1025–1032.

[9] M. Goerzen, S. Malottki, G. Kwiatkowski, P. Bessarab, S. Heinze, “Atomistic spin simulations of electric-field-assisted nucleation and annihilation of magnetic skyrmions in Pd/Fe/Ir(111), Phys. Rev. B 105 (21) (2022) 214435.

[10] A. Aldarawsheh, M. Sallermann, M. Abusaa, S. Lounis, “Intrinsic néel antiferromagnetic multimeronic spin textures in ultrathin films, J. Phys. Chem. Lett. 14 (40) (2023) 8970–8978.

[11] C. Back, V. Cros, H. Ebert, K. Everschor, A. Fert, M. Garst, T. Ma, S. Mankovsky, T. Monchesky, M. Mostovoy, et al., “The 2020 skyrmionics roadmap, J. Phys. D: Appl. Phys. 53 (36) (2020) 363001.

[12] R. Murakami, M. Mizumaki, I. Akai, H. Shouno, “Inverse estimation of parameters for the magnetic domain via dynamics matching using visual-perceptive similarity, Sci. Technol. Adv. Mater.: Methods 2 (1) (2022) 139–152.

[13] H. Kwon, H. Yoon, C. Lee, G. Chen, K. Liu, A. Schmid, Y. Wu, J. Choi, C. Won, “Magnetic Hamiltonian parameter estimation using deep learning techniques, Sci. Adv. 6 (39, p. eabb0872, 2020).

[14] T. Nguyen, L. Rasabathina, O. Hellwig, A. Sharma, G. Salvan, S. Yochelis, Y. Paltiel, L. Baczewski, C. Tegenkamp, “Cooperative effect of electron spin polarization in chiral molecules studied with non-spin-polarized scanning tunneling microscopy, ACS Appl. Mater. Interfaces 14 (33) (2022) 38013–38020.

[15] J. Zhao, L. Bai, S. Li, Z. Cao, Y. Peng, J. Bai, X. Cai, X. Shi, X. Lin, G. Wei, et al., “An overview of advanced instruments for magnetic characterization and measurements, Front. Electron. 6 (2025) 1645594.

## 11. Explicit recoverable dataset, image, split, architecture, and reported-metric evidence

### Dataset and images

- Direct quote: “It contains a total of **166,141 samples**, where each input is a grayscale magnetic domain image of size **39 × 39 × 1 pixels**, and each corresponding target is a vector of eight physical parameters `𝜽∈ℝ⁸` as defined above.”
- Direct quote: “the spatial magnetization field was extracted from the central discrete layer of the nanodot (`𝑧=⌊̆𝐿∕2⌋`). The out-of-plane spin components (`𝑠𝑧 ∈ [−1,1]`) were normalized to the strict interval `[0,1]` and mapped onto a Cartesian pixel grid”.
- Direct quote: “The dataset is publicly available and can be accessed through the Kaggle repository.” URL in the extraction: `https://www.kaggle.com/datasets/carloscanamejoy/dataset-spines-complete`.
- Direct quote: “The cylindrical geometry was constrained by a radius of `𝑅𝑑 = 18.25 [MUC]` and a thickness of `̆𝐿= 5 [MUC]`… yielding an active interacting volume of approximately **5200 localized spins** per simulated dot.”
- Direct quote: “the external field orientation was varied within the dataset: in addition to the in-plane configuration, an out-of-plane field was employed”. `̃𝐽2` and `̃𝐾𝐷𝑀` were “activated in a mutually exclusive fashion”; their reported anti-correlation is `𝜌≈ −0.53`, while remaining pairs are “essentially uncorrelated (`|𝜌|≲0.3`)”.

### Splits and preprocessing

- Direct quote: “This procedure yields an effective split of **70% for training, 15% for validation, and 15% for testing**.” The test set is first isolated at 15%; exactly 15% of the total initial data is then allocated to validation from the remaining 85%.
- A fixed random seed was used; its numeric value is **not recoverable** from the extraction.
- Direct quote: “the single-channel `𝑠𝑧` matrices are spatially resized to **224 × 224** pixels and duplicated across three channels (grayscale-to-RGB conversion).” The conventional CNN uses native `39 × 39 × 1` inputs.
- Targets use training-set Min-Max normalization to `[0,1]`; random `90°` rotations are applied on the fly during training.

### Architectures and training

- Evaluated architectures: conventional CNN, ResNet50, DenseNet121, Xception, and ViT-B/16; all are trained from scratch with no pretrained weights.
- The baseline CNN has four convolutional stages; each has two `3 × 3` convolutions with batch normalization, ReLU, spatial dropout, and max pooling, followed by global average pooling and the shared regression head.
- Shared head: `BN → Drop(0.4) → Dense(256, ReLU, 𝓁2) → BN → Drop(0.3) → Dense(8)`; the text specifies `𝓁2` weight regularization `λ=10⁻⁴`.
- Training uses Adam, MSE, initial learning rate `η=10⁻⁴`, batch size `256`, maximum `100` epochs; `ReduceLROnPlateau` reduces the rate by `0.3` after five stagnant validation epochs, and `EarlyStopping` restores the best weights after 12 without improvement.
- Environment: RunPod with an NVIDIA H200 GPU; TensorFlow for training/inference and multi-GPU JAX/pmap for atomistic simulation. Source code URL: `https://github.com/deathperminut/PaperInverseProblemEstimation`.

### Reported metrics and identifiability

- Table 3 test results (`R²`; MAE in stated physical units):

| Parameter | Xception | DenseNet | ResNet | ViT | CNN |
|---|---|---|---|---|---|
| `𝑇(0)` | 0.96; 0.34 | 0.95; 0.39 | 0.94; 0.42 | 0.95; 0.39 | 0.94; 0.49 |
| `̃𝐽2` | 0.89; 0.018 | 0.87; 0.021 | 0.88; 0.018 | 0.89; 0.018 | 0.88; 0.019 |
| `̃𝐽3` | 0.01; 0.030 | −0.21; 0.031 | −0.24; 0.032 | −0.29; 0.032 | 0.00; 0.028 |
| `̃𝐽4` | 0.00; 0.022 | −0.06; 0.024 | −0.67; 0.028 | −0.30; 0.025 | 0.00; 0.022 |
| `̃𝐾𝑎𝑛1` | 0.94; 0.052 | 0.90; 0.057 | 0.92; 0.054 | 0.92; 0.055 | 0.86; 0.060 |
| `̃𝐾𝑎𝑛𝑆` | 0.64; 0.027 | 0.62; 0.028 | 0.66; 0.027 | 0.63; 0.027 | 0.68; 0.026 |
| `̃𝐻𝑒𝑥` | 0.96; 0.013 | 0.95; 0.013 | 0.96; 0.012 | 0.97; 0.012 | 0.95; 0.013 |
| `̃𝐾𝐷𝑀` | 0.99; 0.019 | 0.99; 0.032 | 0.99; 0.032 | 0.99; 0.037 | 0.99; 0.028 |

- Direct quote: “Six of the eight parameters, the annealing temperature `𝑇 (0)`, the second-shell exchange `̃𝐽2`, the DMI strength `̃𝐾𝐷𝑀`, the bulk and surface anisotropy constants `̃𝐾𝑎𝑛1` and `̃𝐾𝑎𝑛𝑆`, and the external field `̃𝐻𝑒𝑥`, are effectively recovered from the two-dimensional projection.”
- Direct quote: “the higher-order exchange constants `̃𝐽3` and `̃𝐽4` remain essentially unidentifiable”.
- Direct quote: “the compact conventional CNN (∼ **1 M parameters**) to the ViT (∼ **86 M parameters**)”; the mean-`R²` trend is described as “essentially flat”.
- Per-phase Xception Table 4 metrics are recoverable in the source (including N/A and dashes); full numeric values are included in the source extraction but a separate phase-by-phase table is not required for this structural outline beyond the requested explicit reported-metric evidence above.

### Items not recoverable from the extraction

- Exact image-file names, file formats, and individual sample identifiers: **not recoverable**.
- The numeric fixed random seed: **not recoverable**.
- Exact dataset split counts after rounding: **not recoverable**; only 70/15/15 percentages and total sample count are stated.
- Exact trainable-parameter counts for ResNet50, DenseNet121, and Xception: **not recoverable**; only approximate CNN and ViT counts are stated.
