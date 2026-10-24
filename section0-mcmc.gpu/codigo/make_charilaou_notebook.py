"""The Kaggle notebook for the three controlled comparisons against Charilaou.

Reuses the environment cells of `make_kaggle_notebooks` -- the ptxas pinning in
particular, since Triton ships a CUDA minor version ahead of XLA's and nothing
compiles without it -- and embeds `bench_charilaou.py` byte for byte.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_kaggle_notebooks import (PTXAS_CELL, SMOKE_CELL, OUT, code, embed,
                                   md, write)

HEAD = """# Nosotros contra Charilaou, con todo lo demas fijo

La comparacion de la seccion 0 contra **arXiv:2106.11286** vale poco tal como
esta: su cifra de **1.55-1.63 ns por volteo** esta medida en una **Tesla V100**
con un Hamiltoniano mas liviano que el nuestro, y nuestra cifra de 0.129 ns
esta medida en una **H100** con 5000 replicas de un disco de 5245 sitios.
Varian el modelo, el hardware y el metodo a la vez, asi que el cociente de 12x
no mide ninguno de los tres.

Este cuadernillo los fija de a uno.

| | que se fija | que queda libre |
|---|---|---|
| **T1** | mismo dispositivo, mismo modelo, misma geometria | **el metodo** |
| **T2** | mismo modelo, nuestro metodo | el dispositivo, acotado entre P100 y A100 |
| **T3** | su modelo, nuestro metodo, nuestro dispositivo | contra su cifra publicada |

## Su metodo, que no es checkerboard

El paper **rechaza** la descomposicion en tablero, con el argumento de que
restringe la simetria de la red, y a cambio acepta una carrera de
lectura/escritura que mitiga solo manteniendo `n` suficientemente grande:

> *three linearised 1-D arrays, one per component, in global memory; N_b blocks
> of n spins; one spin per block chosen at random per step; dE computed locally;
> cuRAND Mersenne-Twister*

Nuestro metodo paga un coloreo y a cambio tiene un halo sin carreras y sin
barrera. **T1 es exactamente lo que cuesta o rinde ese intercambio.**

La carrera se **reproduce, no se arregla**: arreglarla mediria algo que el
paper no afirma.

## Tres cosas que juegan en nuestra contra, a proposito

1. **Una sola replica.** Nuestro 0.129 ns sale de 5000 replicas y el piso por
   dispositivo es real. Aqui todos los brazos corren **un solo sistema** en
   **su** geometria: cubica simple 128x128x128 con PBC en los tres ejes.
2. **Su `n` se re-optimiza.** El optimo de `n=8` lo hallo en una V100;
   reutilizarlo en otra GPU lo perjudicaria, asi que se barre `n` y se reporta
   **su mejor**, igual que se barre nuestra altura de bloque.
3. **Actualizacion en sitio.** Su kernel escribe sobre los mismos arreglos. Si
   el anfitrion asignara un buffer nuevo por paso habria que copiar 2.1 millones
   de espines en cada uno y su metodo mediria mucho peor de lo que es. El
   driver **alinea entradas con salidas y se niega a cronometrar** si el
   `jax_triton` instalado no lo permite.

## Nada se cronometra antes de que la fisica pase

`verify_host` comprueba las formulas solo con numpy (corre sin GPU);
`verify_device` compara el kernel contra un motor de referencia en JAX puro con
**la misma aleatoriedad**, donde la unica diferencia admisible es aritmetica y
el conteo de espines distintos tiene que ser **cero**; `verify_theirs` es
estadistico, porque su carrera lo hace no determinista por construccion.
"""

SMI = """import subprocess
print(subprocess.run(["nvidia-smi",
  "--query-gpu=name,memory.total,clocks.max.sm,driver_version,compute_cap",
  "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
"""

INSTALL = """import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'jax[cuda12]', 'jax-triton'], check=False)
print('instalado')
"""

ALIAS = '''import inspect, jax_triton as jt
sig = inspect.signature(jt.triton_call)
has = "input_output_aliases" in sig.parameters
print("jax_triton.triton_call acepta input_output_aliases:", has)
if not has:
    print("\\nSIN ALIAS su metodo no se puede medir con justicia:")
    print("cada paso copiaria los 2.1e6 espines intactos.")
    print("parametros disponibles:", list(sig.parameters))
'''

RUN = '''import os, json
os.environ.setdefault("TRITON_PTXAS_PATH", os.environ.get("TRITON_PTXAS_PATH", ""))
import bench_charilaou as BC

res = BC.run(N=128, LZ=128, sweeps=SWEEPS, steps=STEPS,
             label=LABEL, out="/kaggle/working/" + LABEL + ".json")
print(json.dumps({k: res[k] for k in
                  ("device", "best_ours_ns", "best_theirs_ns",
                   "ratio_theirs_over_ours")}, indent=1))
'''

SIZES = '''# T3 wants his two reported sizes, not just one, because his own figure
# falls from 1.63 ns at 128^3 to 1.59 at 512^3 -- the rate depends on the
# lattice and a single point would not be comparable. 512^3 needs 1.6 GB for
# the three arrays, so it is attempted and allowed to fail on a small card.
import json, bench_charilaou as BC

sizes = []
for n in (64, 128, 192, 256):
    try:
        r = BC.time_ours(n, n, sweeps=max(20, 2_000_000 // (n * n * n // 50)),
                         BY=BEST_BY)
        r["N"] = n
        sizes.append(r)
        print("  %4d^3  %8.4f ns/update   %9.1f e6 upd/s"
              % (n, r["ns_per_update"], r["updates_per_s"] / 1e6))
    except Exception as e:
        print("  %4d^3  rechazado: %s: %s" % (n, type(e).__name__, str(e)[:120]))
json.dump(sizes, open("/kaggle/working/" + LABEL + "_sizes.json", "w"), indent=1)
'''

TAIL = """## Como leer el resultado

- **`theirs/ours` en el mismo dispositivo** es T1, y es el unico numero de este
  cuadernillo del que el hardware ya no es excusa para ninguna de las dos
  partes. Si sale por debajo de 1, su metodo es mas rapido que el nuestro con
  el mismo modelo y la misma GPU, y eso habria que decirlo en la seccion.
- **`best_ours_ns` contra sus 1.55-1.63 ns** es T2/T3. La T4 esta *por debajo*
  de una V100 y la P100 mas todavia, asi que si nuestro numero en T4 ya le gana
  a su numero en V100, la ventaja no puede ser del hardware.
- El barrido de tamanos existe porque **su propia cifra depende de la red**
  (1.63 ns en 128^3, 1.59 en 512^3). Comparar un solo punto contra otro solo
  punto no seria comparar.

Los `.json` quedan en `/kaggle/working/` y se bajan al repositorio.
"""


def build(label, sweeps, steps, best_by=4):
    cells = [
        md(HEAD),
        code(SMI),
        code(INSTALL),
        code(PTXAS_CELL),
        code(SMOKE_CELL),
        code(ALIAS),
        embed("bench_charilaou.py"),
        code(f'LABEL = {label!r}\nSWEEPS = {sweeps}\nSTEPS = {steps}\n'
             f'BEST_BY = {best_by}\nprint(LABEL, SWEEPS, STEPS)\n'),
        md("## T1, T2 y T3 en este dispositivo\n\n"
           "Primero la verificacion; si algo falla, el script aborta y no "
           "reporta ningun tiempo."),
        code(RUN),
        md("## El barrido de tamanos, porque su cifra depende de la red"),
        code(SIZES),
        md(TAIL),
    ]
    path = os.path.join(OUT, f"kaggle_{label}.ipynb")
    write(path, cells)
    return path


if __name__ == "__main__":
    build("charilaou_t4", sweeps=120, steps=1200)
