# Sección 1 · El problema directo (generativo)

Parámetros del Hamiltoniano → imagen de magnetización. Dado un vector de
**8 parámetros θ**, sintetizar una imagen 2D de `s_z` que sea físicamente fiel.

Es un mapeo **uno-a-muchos y estocástico**: el mismo θ puede producir varias
configuraciones termodinámicas distintas (estados degenerados, fases
metaestables). Por eso el objetivo no es predecir una imagen sino aprender a
muestrear de `p(X | θ)`.

> Documentación de fondo: `docs/04_direct_problem.md`
> Paper: *Image-Driven Estimation of Magnetic Configurations in Nanodots*,
> Méndez-Rondón, Canamejoy-Alegría et al., en preparación.

## Contenido

| carpeta | qué hay |
|---|---|
| `modelos/` | las dos familias de arquitecturas entrenadas: **CVAE** (8 parámetros, ViT, Xception, y sus reentrenamientos) y **DDPM** (base, estilo Lee, función de pérdida nueva), más un NCSN con AdaGN |
| `evaluacion/` | la comparación entre modelos: métricas físicas, comparación de 4 modelos, barrido de pasos del DDPM, robustez de las métricas de imagen |

## Nota

Este README describe lo que hay, no lo que se concluyó. El informe consolidado
de esta sección todavía no está escrito — en `section0-mcmc.gpu/informe/` está
el modelo a seguir.
