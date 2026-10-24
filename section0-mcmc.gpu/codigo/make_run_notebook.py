"""Build the ready-to-run notebook for section 0.

One notebook, no repository needed: it embeds `sim_core.py`, `triton_halo.py`
and `triton_bench.py` as literals with their SHA-256 checked, installs what the
host lacks, pins Triton's ptxas to the one XLA assembles with, and proves the
optimised engine reproduces the standard one before it runs anything.

The parameters are an array of arrays. The notebook counts the arrays and that
is how many states there are -- nothing else declares it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "cuadernillos", "simulation_ready_for_r.ipynb")

import sys
sys.path.insert(0, HERE)      # the cell bodies live beside this file
import celdas_run as C


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(True)}


def embed(name):
    src = io.open(os.path.join(HERE, name), encoding="utf-8").read()
    h = hashlib.sha256(src.encode()).hexdigest()
    return code(
        f"# {name} — idéntico byte a byte al del repositorio. sha256 {h}\n"
        f"import hashlib\n_src = {src!r}\n"
        f'assert hashlib.sha256(_src.encode()).hexdigest() == "{h}", '
        f'"{name} alterado"\n'
        f'open("{name}", "w").write(_src)\n'
        f'print("{name}  {{}} bytes  sha256 ok".format(len(_src)))\n')


CELLS = [
 md("""# Simulación Monte Carlo del nanodisco — lista para correr

Sube este archivo a Kaggle, Colab o córrelo en tu máquina. **No necesita el
repositorio**: lleva dentro los tres módulos que usa, con su `sha256`
verificado, así que ejecuta código idéntico byte a byte al del repositorio.

## Qué hace

Enfría un nanodisco magnético desde una temperatura caliente hasta la
temperatura objetivo de cada estado, con Metropolis de un solo sitio sobre un
tablero de ajedrez, y reporta las métricas físicas en cada escalón.

## Cómo usarlo

**Edita sólo la primera celda.** Los parámetros son un arreglo de arreglos: un
arreglo por estado, y el cuadernillo cuenta cuántos hay. Después, *Ejecutar
todo*.

## Qué motor usa

Hay dos, y la celda de configuración elige:

| motor | qué es |
|---|---|
| **estándar** | el de JAX: consigue los seis vecinos creando seis copias desplazadas de la red con `jnp.roll` |
| **optimizado** | un *stencil* escrito a mano en Triton: lee su parcela más un borde, hace todo en registros y escribe una vez — las seis copias nunca existen |

El optimizado corre entre un **57 % y un 79 % más rápido** según el ancho del
lote, y **no se usa sin verificarlo**: antes de cronometrar nada reproduce el
*stencil* del estándar con la misma aleatoriedad (cero espines distintos de
182 520) y compara las dos cadenas a equilibrio.

> Con GPU. Sin GPU corre igual, en CPU, pero mucho más despacio: la escalera
> completa de un estado son unas 31 horas contra 2.9 minutos en una H100.
"""),
 md("## 1 · Parámetros\n\nLo único que hay que editar."),
 code(C.PARAMS_CELL),
 md("## 2 · Entorno\n\nDetecta la GPU, instala lo que falte y deja coherente "
    "la cadena de herramientas."),
 code(C.SETUP_CELL),
 md("## 3 · Los módulos\n\nSe escriben a disco desde los literales de abajo, "
    "con su `sha256` verificado."),
 embed("sim_core.py"),
 embed("triton_halo.py"),
 embed("sim_multishell.py"),
 embed("triton_bench.py"),
 md("## 4 · Lectura y verificación de los parámetros"),
 code(C.VALIDA_CELL),
 md("## 5 · La corrida"),
 code(C.CORRE_CELL),
 md("## 6 · Las métricas físicas"),
 code(C.METRICAS_CELL),
 md("## 7 · Las figuras"),
 code(C.FIGURA_CELL),
]

for i, c in enumerate(CELLS):
    c["id"] = f"c{i:02d}"

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"escrito {os.path.normpath(OUT)}  "
      f"{os.path.getsize(OUT)//1024} KB  {len(CELLS)} celdas")
