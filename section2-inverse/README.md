# Sección 2 · El problema inverso (Paper: Inverse Problem Estimation)

> [!IMPORTANT]
> **Repositorio original:** El código, modelos y notebooks de reproducibilidad de esta sección corresponden al repositorio público [deathperminut/PaperInverseProblemEstimation](https://github.com/deathperminut/PaperInverseProblemEstimation/tree/main).
> Se incorporan aquí dentro del proyecto general *Spin State Generation and Inference* exclusivamente por motivos de organización, archivo e integración en el flujo global. La versión de referencia y desarrollo principal se encuentra en el repositorio original enlazado.

---

## Overview / Resumen

**Paper asociado:** *Regression-Based Explainable Deep Learning for Estimating Hamiltonian Parameters from Magnetic Nanodot Images*, Méndez-Rondón et al., *Materials & Design* (2026), [doi:10.1016/j.matdes.2026.116816](https://doi.org/10.1016/j.matdes.2026.116816).

El problema inverso consiste en: dada una imagen 2D de la magnetización fuera del plano $s_z$ ($\mathbf{X}_n$), recuperar los **8 parámetros del Hamiltoniano** extendido de Heisenberg $\boldsymbol{\theta}_n$ que la generaron, formulado como una regresión continua multi-salida:

$$f_\Theta : \mathbb{R}^{\breve{H} \times \breve{W}} \to \mathbb{R}^{\breve{P}}$$

El problema está **mal condicionado por naturaleza**: combinaciones distintas de parámetros pueden dar configuraciones de espín casi indistinguibles (estados degenerados), y la proyección escalar $s_z$ descarta la información en el plano ($s_x, s_y$).

### Los ocho parámetros del Hamiltoniano

$$\boldsymbol{\theta} = [\,T^{(0)},\ \tilde{J}_2,\ \tilde{J}_3,\ \tilde{J}_4,\ \tilde{K}_{an1},\ \tilde{K}_{anS},\ \tilde{H}_{ex},\ \tilde{K}_{DM}\,]$$

con el intercambio de primera capa fijado como referencia de energía ($\tilde{J}_1 = 1.0\text{ meV}$). El DMI se muestrea solo en valores no negativos, ya que el signo del intercambio antisimétrico no es identificable únicamente a partir de la proyección $s_z$ (invierte la quiralidad en el plano pero deja invariante $s_z$).

---

## Pipeline

```
 01_data_generation            02_models                      03_results
 ┌────────────────────┐        ┌────────────────────┐         ┌───────────────────────────┐
 │ Monte Carlo (JAX)  │        │ CNN baseline       │         │ dataset_analysis          │
 │ extended Heisenberg│  ───▶  │ ResNet-50          │  ───▶   │ model_metrics             │
 │ + simulated        │ Xn, θ  │ DenseNet-121       │  f_θ    │ model_comparison          │
 │ annealing (GPU)    │        │ Xception           │         │ rams_interpretability     │
 │                    │        │ Vision Transformer │         │ interpretability_benchmark│
 └────────────────────┘        └────────────────────┘         └───────────────────────────┘
```

---

## Estructura de la Sección

| Ruta | Descripción |
|---|---|
| [`01_data_generation/`](01_data_generation/) | Simulación Monte Carlo acelerada en GPU (JAX) para generación del dataset etiquetado. |
| [`02_models/`](02_models/) | Notebooks de entrenamiento de las 5 arquitecturas: CNN baseline, ResNet-50, DenseNet-121, Xception y Vision Transformer. |
| [`03_results/`](03_results/) | Análisis del dataset, métricas por modelo, comparación cruzada, mapas de atribución RAMs (Regression Activation Maps) y benchmark contra LIME/SHAP. |
| [`cuadernillos/`](cuadernillos/) | Cuadernillos de trabajo locales y notas previas: entrenamiento histórico de Xception y porte a PyTorch. |
| [`docs/`](docs/) | Documentación de fondo ([`03_inverse_problem.md`](docs/03_inverse_problem.md)). |
| [`CITATION.cff`](CITATION.cff) | Metadatos de citación del paper. |
| [`LICENSE`](LICENSE) | Licencia MIT original del repositorio. |

---

## Disponibilidad de Datos

El dataset sintético está disponible públicamente en Kaggle:
- <https://www.kaggle.com/datasets/carloscanamejoy/dataset-spines-united-v2> (~166,141 muestras de imágenes $s_z$ de $39\times39$ con sus 8 parámetros generadores en formato `.npz`).

---

## Cita

Si utilizas este código o los modelos, por favor cita el artículo correspondiente (ver [`CITATION.cff`](CITATION.cff)).
