"""A Kaggle-ready copy of the ready-to-run notebook, with fewer states.

The canonical notebook carries four states and the full production schedule,
which is hours of T4 time. This makes a copy with the first `--estados` of them
so the device split can be watched end to end, and nothing else changes: the
geometry, the schedule and the engine selection stay exactly as the canonical
notebook has them, so what runs is the real thing and not a toy.

Why a script and not an edit: the notebook is generated from `celdas_run.py`,
so a hand edit would be lost the next time anyone regenerates it. This reads
the generated notebook and rewrites only the two lines of the parameter block
that declare the states.
"""

from __future__ import annotations

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "cuadernillos", "simulation_ready_for_r.ipynb")


def build(n_states: int, out: str) -> str:
    nb = json.load(open(SRC, encoding="utf-8"))
    src = "".join(nb["cells"][2]["source"])

    # the two declarations that set how many states there are
    mp = re.search(r"PARAMETROS = \[\n(.*?)\n\]\n", src, re.S)
    mn = re.search(r'NOMBRES = \[(.*?)\]', src, re.S)
    assert mp and mn, "the parameter block is not where it was"

    rows = [l for l in mp.group(1).split("\n") if l.strip().startswith("[")]
    head = [l for l in mp.group(1).split("\n") if not l.strip().startswith("[")]
    names = [x.strip().strip('"') for x in mn.group(1).split(",") if x.strip()]
    assert len(rows) == len(names), (len(rows), len(names))
    assert 1 <= n_states <= len(rows), f"pick 1..{len(rows)}"

    src = src.replace(mp.group(0),
                      "PARAMETROS = [\n"
                      + "\n".join(head + rows[:n_states]) + "\n]\n", 1)
    src = src.replace(mn.group(0),
                      "NOMBRES = ["
                      + ", ".join(f'"{x}"' for x in names[:n_states]) + "]", 1)
    nb["cells"][2]["source"] = src.splitlines(True)

    # a note at the top saying what this copy is, so it is never confused for
    # the canonical notebook
    kept = ", ".join(names[:n_states])
    nb["cells"].insert(0, {
        "cell_type": "markdown", "metadata": {}, "id": "runnote",
        "source": [
            f"> **Copia de corrida.** Igual al cuadernillo del repositorio, "
            f"salvo que lleva **{n_states} estado(s)** (`{kept}`) en vez de "
            f"{len(rows)}. La geometria, el cronograma y la seleccion de motor "
            f"quedan intactos, asi que lo que corre es lo real.\n",
            ">\n",
            "> Con 1 estado y 2 GPU el reparto es por **replicas**: las 50 se "
            "dividen en 25 por dispositivo. La celda de la corrida imprime que "
            "modo eligio y por que.\n",
        ]})

    nb["metadata"]["accelerator"] = "GPU"
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estados", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = build(a.estados, a.out)
    print(f"escrito {p}  {os.path.getsize(p) // 1024} KB")
