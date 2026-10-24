# Sección 2 · El problema inverso

Imagen → parámetros del Hamiltoniano. Dada una imagen 2D de `s_z`, recuperar
los **8 parámetros θ** que la generaron, como una regresión continua
multi-salida.

El problema está **mal condicionado por naturaleza**: combinaciones distintas
de parámetros pueden dar configuraciones de espín casi indistinguibles (estados
degenerados), y la proyección `s_z` descarta la información de las componentes
en el plano `s_x` y `s_y`.

> Documentación de fondo: `docs/03_inverse_problem.md`
> Paper: *Regression-Based Explainable Deep Learning for Estimating Hamiltonian
> Parameters from Magnetic Nanodot Images*, Méndez-Rondón et al., enviado a
> Elsevier.

## Contenido

| archivo | qué hay |
|---|---|
| `cuadernillos/XceptionFullDataBaseV3100.ipynb` | el entrenamiento sobre la base completa |
| `cuadernillos/xception-keras-to-torch.ipynb` | el porte de Keras a PyTorch |
| `cuadernillos/xception_inverse.md` | las notas del modelo |

## Nota

Este README describe lo que hay, no lo que se concluyó. El informe consolidado
de esta sección todavía no está escrito — en `section0-mcmc.gpu/informe/` está
el modelo a seguir.
