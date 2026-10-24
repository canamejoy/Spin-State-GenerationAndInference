# Sección 0 · MCMC en GPU

Por qué el número de réplicas por GPU cambia el costo de la simulación, qué lo
causa, y el kernel que se escribió para dejar de pagarlo.

## Empezar por aquí

**`informe/informe.pdf`** — 32 páginas. El registro completo: lo que se buscó,
lo que se midió, cada hipótesis descartada y cada error cometido. Es la versión
canónica de esta sección. El mismo contenido en HTML está en
`informe/informe.html`.

## El resultado en una tabla

Escalera de producción (200 temperaturas × 20 000 barridos = 4 000 000 de
barridos por réplica), un estado con 50 réplicas:

| hardware | estándar | optimizado |
|---|---|---|
| CPU, 24 hilos | 31.4 h | — |
| 1× T4 | 56.4 min | **24.3 min** |
| 1× H100 | 8.1 min | **2.9 min** |
| 4× H100, 100 estados | 2.7 h | **57.6 min** |

Y la causa, en una línea: con `elementos = B·L·N²·3` y `N` impar,

```
ancho = min(4, 2^(v2(B)+v2(L)))     →  castigo si v2(B) + v2(L) ≥ 2
```

89 de 89 rejillas predichas sin un fallo, sobre 5 GPU y 4 arquitecturas.

## Contenido

| carpeta | qué hay |
|---|---|
| `informe/` | el informe consolidado, HTML y PDF |
| `paginas/` | las 6 páginas interactivas de las que sale |
| `codigo/` | `sim_core.py`, `triton_halo.py`, `triton_bench.py`, `app.py` y los módulos de ablación |
| `cuadernillos/` | 14 notebooks, incluidos los de Kaggle autocontenidos |
| `resultados/` | 74 JSON de Modal, 16 de Kaggle, la tasa de CPU |

## Dos avisos

**Las páginas están copiadas, no movidas.** `section0-mcmc.gpu/paginas/` sigue siendo lo que
sirve GitHub Pages; moverlas rompería el sitio vivo. Eso significa que esos
archivos están dos veces en el repositorio, igual que los notebooks y los JSON.
Está así a propósito, como paso intermedio: cuando se decida el destino final
de la sección se resuelve de un lado o del otro.

**El informe es el documento canónico de la sección, pero las páginas no son
redundantes:** llevan las gráficas interactivas y los widgets, que un PDF no
puede. El informe las resume y las cita; no las reemplaza.

## Reproducir

```
# el paso, por lote
modal run codigo/app.py::triton_halo
# la escalera completa, los dos motores
modal run codigo/app.py::triton_full
# un estado en 1, 2 y 4 GPU
modal run codigo/app.py::state_ladder --reps 50
# la regla del ancho contra el kernel
modal run codigo/app.py::triton_width
```

Las credenciales de Modal van en `~/.modal-accounts/<cuenta>/.modal.toml` y se
usan con `MODAL_CONFIG_PATH`; las de Kaggle en
`~/.kaggle-accounts/<cuenta>/kaggle.json` con `KAGGLE_CONFIG_DIR`.

Los cuadernillos de Kaggle son autocontenidos: embuten `sim_core.py`,
`triton_halo.py` y `triton_bench.py` como literales con su SHA-256 verificado,
así que corren código byte a byte idéntico al de `codigo/`.
