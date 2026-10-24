import hashlib, io, json, os

ROOT = "/home/andres/Spin-State-GenerationAndInference"
MOD = f"{ROOT}/notebooks/simulation/modal"
OUT = f"{ROOT}/notebooks/simulation"
CUR = f"{MOD}/../results/curves"

PTXAS_CELL = r'''import glob, os, re, subprocess

def ver(p):
    try:
        o = subprocess.run([p, "--version"], capture_output=True,
                           text=True).stdout
        m = re.search(r"release (\d+)\.(\d+)", o)
        return tuple(map(int, m.groups())) if m else (0, 0)
    except Exception:
        return (0, 0)

pats = ["/usr/local/lib/python3*/dist-packages/nvidia/cuda_nvcc/bin/ptxas",
        "/usr/local/lib/python3*/site-packages/nvidia/cuda_nvcc/bin/ptxas",
        "/opt/conda/lib/python3*/site-packages/nvidia/cuda_nvcc/bin/ptxas",
        "/usr/local/lib/python3*/dist-packages/triton/backends/nvidia/bin/ptxas",
        "/opt/conda/lib/python3*/site-packages/triton/backends/nvidia/bin/ptxas",
        "/usr/local/cuda*/bin/ptxas"]
found = sorted({p for q in pats for p in glob.glob(q)})
rows = [(ver(p), p) for p in found]
print("ptxas encontrados:")
for v, p in sorted(rows):
    tag = "   <- el de triton" if "triton" in p else ""
    print("  CUDA %d.%-3d %s%s" % (v[0], v[1], p, tag))

xla = [(v, p) for v, p in rows if "triton" not in p and v > (0, 0)]
if xla:
    # XLA's is the limiting half, so Triton gets pinned to it.
    v, p = min(xla)
    os.environ["TRITON_PTXAS_PATH"] = p
    print("\nTRITON_PTXAS_PATH -> CUDA %d.%d  %s" % (v[0], v[1], p))
else:
    print("\nAVISO: no se hallo un ptxas fuera de triton; se deja como esta.")
'''

SMOKE_CELL = r'''import sys
for m in [k for k in list(sys.modules) if k.split(".")[0] in
          ("jax", "jaxlib", "triton", "jax_triton")]:
    del sys.modules[m]
import jax, jax.numpy as jnp
print("jax", jax.__version__, jax.devices())
try:
    import triton, triton.language as tl, jax_triton as jt
    print("triton", triton.__version__,
          "| jax_triton", getattr(jt, "__version__", "?"))
except Exception as e:
    print("FALTA:", type(e).__name__, e)
    raise SystemExit

# One tiny kernel, so a toolchain failure shows HERE and not buried forty
# minutes into the benchmark.
@triton.jit
def _add1(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, tl.load(x_ptr + i) + 1.0)

try:
    x = jnp.arange(8, dtype=jnp.float32)
    y = jt.triton_call(x, kernel=_add1, grid=(1,), N=8,
                       out_shape=jax.ShapeDtypeStruct((8,), jnp.float32))
    ok = bool((y == x + 1).all())
    print("\nprueba minima de Triton:", "PASA" if ok else "FALLA", "->", y)
except Exception as e:
    print("\nprueba minima de Triton: FALLA  %s: %s"
          % (type(e).__name__, str(e)[:500]))
'''

Q = chr(39) * 3   # triple quote, built so it never nests in this file

def md(s):  return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(True)}

def embed(name):
    src = io.open(f"{MOD}/{name}", encoding="utf-8").read()
    # Parsing only proves the file is syntactically Python. Two Kaggle runs
    # died on things a static check sees: a module that was never written and
    # a call passing a name the function does not take. pyflakes is checked
    # here, at embed time, because a GPU run is an expensive linter.
    try:
        import subprocess, sys
        r = subprocess.run([sys.executable, "-m", "pyflakes", f"{MOD}/{name}"],
                           capture_output=True, text=True)
        bad = [l for l in r.stdout.splitlines()
               if "undefined name" in l or "imported but unused" not in l
               and "missing placeholders" not in l
               and "never used" not in l and l.strip()]
        assert not bad, f"{name} no pasa pyflakes:\n" + "\n".join(bad)
    except ImportError:
        print(f"  AVISO: pyflakes no disponible, {name} sin revisar")
    h = hashlib.sha256(src.encode()).hexdigest()
    return code(
        f"# {name} -- byte-identical to the repository copy. sha256 {h}\n"
        f"import hashlib\n_src = {src!r}\n"
        f'assert hashlib.sha256(_src.encode()).hexdigest() == "{h}", '
        f'"{name} alterado"\n'
        f'open("{name}", "w").write(_src)\n'
        f'print("{name}  {{}} bytes  sha256 ok".format(len(_src)))\n')

NBMETA = {"kernelspec": {"display_name": "Python 3", "language": "python",
                         "name": "python3"},
          "language_info": {"name": "python", "version": "3.11"},
          "accelerator": "GPU"}

def write(path, cells):
    # nbformat_minor 5 requires an id per cell; without one Kaggle logs
    # MissingIDFieldWarning and warns it will become a hard error. The ids are
    # deterministic so regenerating does not churn the diff.
    for i, c in enumerate(cells):
        c["id"] = f"c{i:02d}"
    json.dump({"cells": cells, "metadata": NBMETA, "nbformat": 4,
               "nbformat_minor": 5}, io.open(path, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("escrito", path, os.path.getsize(path) // 1024, "KB")

# ---------------------------------------------------------------- A: portability
A = [
 md("""# El kernel de Triton fuera de la H100

Todo el resultado de este kernel se midió en H100 alquiladas por hora. Eso
deja dos cosas sin saber.

**¿Compila en otra parte?** Triton exige capacidad de cómputo **7.0 o más**.
La **T4** de Kaggle es 7.5 y debería funcionar; la **P100** es 6.0 y lo más
probable es que se niegue. Si se niega, eso *es* un resultado — elige T4.

**¿La ganancia es una propiedad de la H100?** La H100 tiene 132 SMs y una L2
enorme. Un *stencil* que vive en registros podría comportarse muy distinto en
una T4 de 40 SMs, y esa es justo la pregunta.

Orden del cuadernillo, y no se cronometra nada antes de que la física
coincida:

1. entorno y capacidad de cómputo
2. **exactitud** — el *stencil* contra `sim_core.substep` con la **misma**
   aleatoriedad: debe dar **cero** espines distintos
3. **física** — las dos cadenas a equilibrio, cada una con su propio RNG
4. **ritmo del paso**, y la regla del ancho puntuada por separado para cada
   motor
5. **escalera corta**, para que la cifra citada sea el *anneal* completo y no
   un paso suelto

> Acelerador: **GPU T4 x2** (usa una sola). Tiempo estimado: 10-15 min,
> casi todo compilación.
"""),
 code("""import subprocess
print(subprocess.run(["nvidia-smi",
  "--query-gpu=name,memory.total,clocks.max.sm,driver_version",
  "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())"""),
 md("""## Instalación, y el `ptxas` que hace fallar esto

La primera corrida en T4 murió aquí, y vale la pena dejarlo escrito, porque
**no era la GPU ni el kernel**:

```
ptxas fatal : Unsupported .version 8.8; current version is '8.7'
```

Triton **trae su propio `ptxas`** (CUDA 12.9, ISA 8.8) y emite PTX para esa
versión. El `ptxas` que usa XLA para ensamblar el módulo es el de la rueda
`nvidia-cuda-nvcc-cu12` (CUDA 12.8, ISA 8.7), y no entiende 8.8. Una versión
menor de diferencia, y nada compila.

La T4 es capacidad **7.5**, de sobra para Triton. El arreglo es que **las dos
mitades de la cadena usen el mismo `ptxas`**: se inventarían todos los que
haya y se le fija a Triton el de XLA con `TRITON_PTXAS_PATH`. Bajar el ISA no
cuesta nada para este kernel.

Tiene que pasar **antes de importar `jax`**, porque el backend fija su cadena
al inicializarse.
"""),
 code("import sys, subprocess\n"
      "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
      "                'jax[cuda12]', 'jax-triton'], check=False)\n"
      "print('instalado')\n"),
 code(PTXAS_CELL),
 code(SMOKE_CELL),
 embed("sim_core.py"),
 embed("triton_halo.py"),
 embed("triton_bench.py"),
 md("""## La medición

La rejilla es `L ∈ {5, 6} × B ∈ {64, 65, 100, 101}`. Esos cuatro lotes están
elegidos para que la regla del ancho quede puntuada dentro de cada capa:
`v₂(64)=6` y `v₂(100)=2` la disparan, `v₂(65)=v₂(101)=0` no. En la H100 el
castigo fue **+17.8 %** para producción y **+0.3 %** para el kernel.
"""),
 code("""import sys, json
sys.path.insert(0, ".")
import triton_bench

res = triton_bench.run(
    batches=(64, 65, 100, 101), layers=(5, 6),
    repeats=5, steps=200, Rd=18.30, by=4, warps=4,
    replicas=50, n_temps=20, n_therm=300, n_meas=150,
    out="/kaggle/working/triton_bench_kaggle.json", label="kaggle")"""),
 md("""## Comparación contra la H100

La fila de la H100 son los números medidos allí con la misma rejilla. Lo que
se compara **no** es el tiempo absoluto —una T4 es mucho más lenta y eso no
dice nada— sino **la ganancia relativa** y **si la regla del ancho sigue
muerta** en el kernel.
"""),
 code("""H100 = {"paso_cambio_pct": -66.6, "escalera_cambio_pct": -62.8,
        "castigo_produccion_pct": 17.8, "castigo_kernel_pct": 0.3}

if not res.get("ok"):
    print("La fisica no coincidio o Triton no compilo: no hay nada que "
          "comparar.")
    print("  error:", res.get("error", "ver avisos arriba"))
else:
    import statistics as st
    ch = st.mean([r["change_pct"] for r in res["rows"]])
    lad = res["ladder"]["arms"]["change_pct"]
    pk = [v for k, v in res["penalty"].items() if "kernel" in k]
    pp = [v for k, v in res["penalty"].items() if "produccion" in k]
    print(f"{'':<34}{'esta GPU':>12}{'H100':>10}")
    print(f"{'paso, kernel vs produccion':<34}{ch:>11.1f}%"
          f"{H100['paso_cambio_pct']:>9.1f}%")
    print(f"{'escalera completa':<34}{lad:>11.1f}%"
          f"{H100['escalera_cambio_pct']:>9.1f}%")
    if pp:
        print(f"{'castigo del ancho, produccion':<34}"
              f"{st.mean(pp):>+11.1f}%{H100['castigo_produccion_pct']:>+9.1f}%")
    if pk:
        print(f"{'castigo del ancho, kernel':<34}"
              f"{st.mean(pk):>+11.1f}%{H100['castigo_kernel_pct']:>+9.1f}%")
    print()
    print("Lectura: si la ganancia se sostiene y el castigo del kernel sigue")
    print("cerca de cero, el resultado NO es una propiedad de la H100.")"""),
]
write(f"{OUT}/kaggle_triton_portability.ipynb", A)

# ---------------------------------------------------------------- B: results
L4 = json.load(open(f"{CUR}/triton_ladder4.json"))
L1 = json.load(open(f"{CUR}/triton_full.json"))
W = json.load(open(f"{CUR}/triton_width.json"))
C = json.load(open(f"{CUR}/triton_check_fixed.json"))
A1 = json.load(open(f"{CUR}/triton_fixed_acc.json"))
H = json.load(open(f"{CUR}/triton_halo.json"))

DATA = dict(
    ladder4=dict(n_temps=L4["n_temps"], n_therm=L4["n_therm"],
                 n_meas=L4["n_meas"], chains=L4["total_chains"],
                 per_device=L4["per_device"], devices=L4["n_devices"],
                 arms={k: dict(minutes=v["minutes"],
                               us=v["us_per_replica_sweep"],
                               vs_A=v["vs_A_pct"]) for k, v in L4["arms"].items()},
                 C_vs_B=L4["C_vs_B_pct"]),
    ladder1=dict(n_temps=L1["ladder"]["n_temps"],
                 n_therm=L1["ladder"]["n_therm"],
                 n_meas=L1["ladder"]["n_meas"],
                 sweeps=L1["ladder"]["total_sweeps"],
                 replicas=L1["replicas"],
                 arms={k: dict(minutes=v["minutes"],
                               us=v["us_per_replica_sweep"])
                       for k, v in L1["ladder"]["arms"].items()},
                 recovered=L1["ladder"]["recovered_pct"]),
    width=[dict(L=r["L"], B=r["B"], suma=r["suma"], prod=r["prod_us"],
                kernel=r["halo_us"], change=r["change_pct"])
           for r in W["rows"]],
    penalty=W["penalty"],
    substep=[dict(B=r["B"], prod=r["prod_us"], stencil=r["stencil_only_us"],
                  kernel=r["halo_us"], change=r["change_pct"],
                  stencil_pct=r["stencil_pct"]) for r in H["timing"]],
    substep_summary=H["summary"],
    obs_check=L1["obs_check"],
    exact=H["correctness"],
    physics=H["physics"],
    control=[dict(therm=r["n_therm"], meas=r["n_meas"],
                  ctl_mean=r["control_mean"], ctl_max=r["control_max"],
                  k_mean=r["halo_mean"], k_max=r["halo_max"],
                  M_ctl=r["M_control_max_abs_diff"], M_k=r["M_max_abs_diff"])
             for r in C["rows"]],
    acceptance=[dict(T=r["T"], prod=r["acc_prod"], kernel=r["acc_halo"],
                     ratio=r["ratio"]) for r in A1["acceptance"]],
)

B = [
 md("""# Una escalera de 400 réplicas en 4 minutos

Resultados medidos en **H100**. Este cuadernillo **no calcula nada**: trae
los números y los dibuja, para poder leerlos de un golpe.

## El titular

La escalera de producción —200 temperaturas, 10 000 + 5 000 barridos,
400 cadenas— con tres motores distintos:

| | reloj |
|---|---|
| producción, tal como está | ~15.8 min |
| producción + protocolo de partición | ~11.5 min |
| **kernel de Triton escrito a mano** | **~3.9 min** |

Lo medido fueron 60 temperaturas; la cifra de 200 es el escalado lineal en
barridos, que es exacto por construcción. Las tres filas corrieron en las
**mismas 4 GPU, con el mismo trabajo y la misma física**.
"""),
 # json.dumps emits true/false/null, which are not Python literals, so the
 # payload is PARSED rather than pasted as code.
 code("import json, numpy as np\n"
      "import matplotlib.pyplot as plt\n"
      "DATA = json.loads(r" + Q + json.dumps(DATA, indent=1) + Q + ")\n"
      'print("datos cargados:", list(DATA))\n'),
 md("""## 1 · Cuatro GPU, 400 cadenas

`A` es lo que hacía el proyecto. `B` es la mejor opción **sin tocar el
kernel**: partir el lote en dos olas impares, que esquiva la vectorización de
ancho 4 de XLA. `C` es el kernel, en **una sola ola**, porque no tiene nada
que esquivar.
"""),
 code("""l4 = DATA["ladder4"]
esc = 200 / l4["n_temps"]          # a las 200 temperaturas de produccion
names, meas, full = [], [], []
for k, v in l4["arms"].items():
    names.append(k); meas.append(v["minutes"]); full.append(v["minutes"] * esc)

fig, ax = plt.subplots(figsize=(10, 4.2))
y = np.arange(len(names))
b = ax.barh(y, full, color=["#c2c2c2", "#8ab4f8", "#2e7d32"], height=.6)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("minutos de la escalera completa (200 T x 10000+5000, "
              f"{l4['chains']} cadenas, {l4['devices']} GPU)", fontsize=11)
for r, m, f_ in zip(b, meas, full):
    ax.text(f_ + .2, r.get_y() + r.get_height() / 2,
            f"{f_:.1f} min   (medido {m:.2f} min a {l4['n_temps']} T)",
            va="center", fontsize=10)
ax.set_xlim(0, max(full) * 1.45)
ax.set_title("La misma escalera, tres motores", fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()

print(f"{'brazo':<38}{'medido':>9}{'a 200 T':>10}{'vs A':>9}")
for k, v in l4["arms"].items():
    print(f"{k:<38}{v['minutes']:>8.2f}m{v['minutes']*esc:>9.1f}m"
          f"{v['vs_A']:>+8.1f}%")
print(f"\\nEl kernel contra el protocolo (la barra real): "
      f"{l4['C_vs_B']:+.1f}%")"""),
 md("""## 2 · Una GPU: de dónde sale cada punto

Dos cambios, medidos por separado. Quitar las **seis matrices desplazadas**
que `jnp.roll` materializaba vale la mitad; generar los números aleatorios
**dentro** del kernel vale otro tanto sobre lo que queda.
"""),
 code("""s = DATA["substep"]; sm = DATA["substep_summary"]
Bs = [r["B"] for r in s]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(Bs, [r["prod"] for r in s], "o-", label="produccion (6 rolls + Threefry)",
        color="#b00020", lw=2)
ax.plot(Bs, [r["stencil"] for r in s], "s-", label="solo el stencil",
        color="#8ab4f8", lw=2)
ax.plot(Bs, [r["kernel"] for r in s], "^-", label="stencil + su propio RNG",
        color="#2e7d32", lw=2)
ax.set_xlabel("replicas por lanzamiento (B)", fontsize=11)
ax.set_ylabel("us / replica-barrido", fontsize=11)
ax.set_title("El paso de Monte Carlo, por lote", fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=.25)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()

print(f"stencil solo      : {sm['mean_stencil_pct']:+.1f}% de media")
print(f"+ su propio RNG   : {sm['mean_change_pct']:+.1f}% "
      f"(gana {sm['wins']}/{sm['n']}, mejor {sm['best_change_pct']:+.1f}%)")
print(f"\\nFijate en B=100 contra B=101 en la curva roja: +39% por una sola")
print(f"replica de diferencia. En la verde no se distingue.")
print(f"\\ndispersion entre lotes:  produccion {sm['dispersion_prod_pct']:.1f}%"
      f"   kernel {sm['dispersion_halo_pct']:.1f}%")"""),
 md("""## 3 · Los múltiplos de 4 dejan de importar

La regla medida en este proyecto era: **hay castigo si `v₂(B) + v₂(L) ≥ 2`**,
donde `v₂` es cuántas veces entra el 2. Venía de que XLA decidía vectorizar de
cuatro en cuatro el bucle que le entregábamos.

El kernel no le entrega ningún bucle: la forma del bloque la fija el kernel y
el número de capas solo dice cuántos bloques hay. Así que la regla no tiene
por dónde actuar — y se midió sobre `L ∈ {4,5,6,8}` para comprobarlo.

En `L = 4` y `L = 8` **no se puede puntuar** un castigo: `v₂(4) = 2` y
`v₂(8) = 3` ya disparan la regla para *cualquier* `B`, así que no queda grupo
de control. Eso no es un hueco en la medición: es la regla comportándose como
está documentada, y por eso solo `L = 5` y `L = 6` tienen fila.
"""),
 code("""w = DATA["width"]; pen = DATA["penalty"]
print(f"{'L':>3}{'B':>6}{'v2(B)+v2(L)':>13}{'produccion':>13}{'kernel':>9}"
      f"{'cambio':>9}")
for r in w:
    mark = " <-" if r["suma"] >= 2 else ""
    print(f"{r['L']:>3}{r['B']:>6}{r['suma']:>13}{r['prod']:>13.3f}"
          f"{r['kernel']:>9.3f}{r['change']:>+8.1f}%{mark}")

print(f"\\n=== castigo por v2 >= 2, por motor ===")
ks = sorted(pen)
fig, ax = plt.subplots(figsize=(8, 3.4))
lab = [k.replace("_", "  ") for k in ks]
col = ["#b00020" if "produccion" in k else "#2e7d32" for k in ks]
bb = ax.bar(range(len(ks)), [pen[k] for k in ks], color=col, width=.55)
ax.set_xticks(range(len(ks))); ax.set_xticklabels(lab, fontsize=10)
ax.set_ylabel("castigo (%)", fontsize=11); ax.axhline(0, color="k", lw=.8)
ax.set_title("La regla del ancho, por motor", fontsize=13)
for r, k in zip(bb, ks):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .4,
            f"{pen[k]:+.1f}%", ha="center", fontsize=10)
ax.set_ylim(0, max(pen.values()) * 1.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
for k in ks:
    print(f"  {k:<22}{pen[k]:+7.1f}%")
print("\\nMuerta. Con este kernel el numero de replicas se elige por fisica")
print("o por logistica, no para esquivar al compilador.")"""),
 md("""## 4 · ¿Y la física es la misma?

Cinco pruebas, y ningún tiempo se miró antes de que pasaran. Las dos primeras
son **exactas** (no hay aleatoriedad involucrada); las tres últimas son
estadísticas.
"""),
 code("""e = DATA["exact"]; o = DATA["obs_check"]; p = DATA["physics"]
c = DATA["control"]; a = DATA["acceptance"]

print("1. el stencil contra sim_core.substep, MISMA aleatoriedad")
for r in e:
    print(f"   {r['sublattice']:>5}: dE max |dif| {r['dE_max_abs']:.2e}   "
          f"espines distintos {r['spin_mismatch']}/{r['spin_total']}")
print("\\n2. los observables contra los de JAX, mismo estado")
print(f"   E {o['E_rel']:.2e} relativo | |M| {o['M_rel']:.2e} | "
      f"Q {o['Q_abs']:.2e} absoluto")
print("\\n3. las dos cadenas a equilibrio, cada una con su RNG")
for r in p:
    print(f"   T={r['T']:>5.1f}  E/sitio {r['E_prod']:+9.5f} / "
          f"{r['E_halo']:+9.5f}  ({r['E_sigma']:.2f} sigma)   "
          f"|norma-1| {r['norm_err']:.1e}")
print("\\n4. control: produccion contra SI MISMA, la vara de medir")
print(f"   {'ventana':>14}{'control':>16}{'kernel':>16}")
for r in c:
    print(f"   {r['therm']:>6}+{r['meas']:<7}"
          f"{r['ctl_mean']:>8.2f}/{r['ctl_max']:<7.2f}"
          f"{r['k_mean']:>8.2f}/{r['k_max']:<7.2f}")
print("\\n5. tasa de aceptacion")
for r in a:
    print(f"   T={r['T']:>5.1f}  {r['prod']:.4f} / {r['kernel']:.4f}   "
          f"razon {r['ratio']:.4f}")
print("\\nLa cuarta es la clave metodologica: correr produccion contra si")
print("misma con otra semilla da el RUIDO PROPIO DE LA VARA. Fue la que")
print("destapo un bug de RNG mio -- claves Philox adyacentes sobre un")
print("contador compartido -- y, tras arreglarlo, la que lo cerro.")"""),
 md("""## 5 · Qué cambió en el código

Para actualizar un espín hacen falta sus 6 vecinos. `jnp.roll` los conseguía
**creando seis copias completas de la red**, cada una corrida un paso: seis
escrituras y seis lecturas de toda la red en cada medio paso. HLO le atribuía
el **31.5 %** del tiempo, y otro **17.1 %** al generador aleatorio.

El kernel lee **su parcela más un borde de un sitio**, hace todo en registros
y escribe una vez. Las seis copias nunca existen.

**El detalle que lo hace funcionar** —y que falló tres veces antes, en Pallas—
no es hacer menos lecturas: sigue emitiendo 21. Es la **forma de la
dirección**:

| | la dirección del vecino es… | el compilador emite… |
|---|---|---|
| intentos en Pallas | una por carril, calculada por carril | accesos **dispersos** |
| este kernel | un **vector** desplazado sobre un tramo contiguo | accesos **coalescidos** |

La GPU trae 32 bytes de golpe: ocho hilos pidiendo ocho números seguidos es
**una** lectura; ocho hilos calculando su propia dirección son **ocho**.

Y hay un detalle de física que lo hace legal sin sincronización: el **tablero
de ajedrez** garantiza que todo vecino de un sitio actualizado tiene la
paridad opuesta, así que ningún bloque lo escribe en ese medio paso. El halo
se lee de memoria global **sin barrera**.

### Cómo usarlo

```python
import sim_core as S, triton_halo as TH

geo = S.build_geometry(Rd=18.30, L=5)
ann = TH.build_anneal_full(geo, BY=4, num_warps=4)

k1  = jnp.where(surf_j, P["KanS"], P["Kan1"])   # igual que hoy
par = TH.pack_params(P, B)                      # (B, 6)
sp  = TH.planar(S.init_spins(key, geo, B))      # (B,3,L,N,N)

obs, final, snaps = ann(sp, seed, betas, par, k1, n_therm, n_meas)
resultados = S.reduce_obs(np.asarray(obs), T, geo["n_active"])  # SIN cambios
```

Tres avisos: la imagen necesita `jax-triton`; `pack_params` descarta `Kan1` a
propósito, porque `sim_core._an` lo recibe y no lo usa; y `seed` es un entero,
no una `PRNGKey`, porque el kernel es por contador.
"""),
 code("""l1 = DATA["ladder1"]
print(f"Escalera en UNA H100: {l1['n_temps']}T x {l1['n_therm']}+"
      f"{l1['n_meas']} = {l1['sweeps']:,} barridos, {l1['replicas']} replicas")
print()
for k, v in l1["arms"].items():
    print(f"  {k:<34}{v['minutes']:7.2f} min   {v['us']:7.3f} us/replica-barrido")
print(f"\\nLos kernels de observables recuperan {l1['recovered']:+.1f} puntos.")
print()
print("Y una enmienda que conviene leer: predije que los 8 puntos entre el")
print("paso suelto (-66.6%) y la escalera eran el bucle de medicion.")
print("Recuperan 4.5 de los 8. El resto esta en los ocho acumuladores, en el")
print("scan sobre 200 temperaturas, y en que la escalera corre con lote 50")
print("mientras el banco del paso corria entre 64 y 128.")"""),
]
write(f"{OUT}/kaggle_triton_h100_results.ipynb", B)

# ------------------------------------------------- C: one state vs ten states
# Reuses A's setup cells verbatim: install, the ptxas pin, the smoke test and
# the three checksummed modules. Only the final call differs.
# A's cells 1..8: smi, install prose, pip, the ptxas pin, the smoke
# test, and the THREE checksummed modules. Slicing to 8 dropped
# triton_bench and the notebook died on its import.
SETUP = A[1:9]
assert sum(1 for c in SETUP
           if 'sha256 ok' in ''.join(c['source'])) == 3, \
    'los tres modulos deben quedar embutidos'
C = [
 md("""# Un estado contra diez, en una T4 y en dos

Kaggle reparte **dos T4**, que es justo el par que hace falta aquí. Este
cuadernillo mide las cuatro celdas que faltaban para las gráficas:

| | 1 estado (50 réplicas) | 10 estados (500 réplicas) |
|---|---|---|
| **1× T4** | ✓ | ✓ |
| **2× T4** | ✓ | ✓ |

Los dos conteos dividen limpio entre los dos conteos de dispositivo —
500 → 500/250 y 50 → 50/25— así que no hay que aproximar nada, al contrario
de lo que pasó con 50 réplicas entre cuatro tarjetas.

## Los dos motores

**estándar** es el motor que el proyecto ha usado siempre:
`sim_core.build_engine(...)["anneal"]`, en JAX. Para cada espín consigue sus
seis vecinos creando **seis copias desplazadas de la red** con `jnp.roll`, y
sortea con Threefry.

**optimizado** es el nuevo: un *stencil* escrito a mano en Triton crudo que
lee su parcela más un borde de un sitio, hace toda la aritmética en registros
y escribe una vez — las seis copias nunca existen. Los observables también son
kernels. Sortea con Philox dentro del kernel.

Misma física, verificada: **0 de 182 520** espines distintos contra
`sim_core.substep` con la misma aleatoriedad.

## La escalera se acorta a propósito

Lo que esto mide es una **tasa** — microsegundos por réplica-barrido — y el
reloj de las 200 temperaturas se reporta como esa tasa multiplicada por el
número real de barridos. Una T4 corriendo diez estados a longitud completa
tardaría horas, así que las celdas largas se **extrapolan** en vez de
esperarse. El escalado en barridos es exacto por construcción; lo que no es
exacto es el ruido, que para celdas cortas ronda el ±2.5 %.

> Acelerador: **GPU T4 x2**. Unos 60-70 min.
"""),
] + SETUP + [
 code("import sys, json\n"
      "sys.path.insert(0, '.')\n"
      "import triton_bench\n"
      "\n"
      "res = triton_bench.run_states(\n"
      "    reps=(50, 500), devices=(1, 2), n_temps=10, sweeps=20000,\n"
      "    Rd=18.30, L=5, by=4, warps=4,\n"
      "    out='/kaggle/working/triton_states_t4.json',\n"
      "    label='kaggle-t4-states')\n"),
 md("""## Lectura

La columna que importa es **cambio**: cuánto baja el motor optimizado respecto
al estándar en cada celda.

En las H100 apareció un **piso de lote por dispositivo**: con 12-13 réplicas
por tarjeta el motor optimizado *pierde*, porque su rejilla se queda en unos
600 bloques y deja la tarjeta medio vacía. Aquí las celdas más flacas son
25 réplicas por T4 (1 estado, 2 tarjetas) y las más anchas 500 (10 estados,
1 tarjeta), así que la tabla debería mostrar la ganancia **creciendo con el
ancho del lote** — que es la razón por la que más estados aprovechan mejor el
hardware.
"""),
 code("if not res['rows']:\n"
      "    print('no hubo filas: revisa la cadena de herramientas arriba')\n"
      "else:\n"
      "    print(f\"{'disp':>5}{'repl':>6}{'/disp':>7}{'us/repl-barrido':>32}\")\n"
      "    print(f\"{'':>18}{'estandar':>14}{'optimizado':>14}{'cambio':>9}\")\n"
      "    for r in res['rows']:\n"
      "        print(f\"{r['devices']:>5}{r['replicas']:>6}{r['per_device']:>7}\"\n"
      "              f\"{r['estandar']['us_per_replica_sweep']:>14.3f}\"\n"
      "              f\"{r['optimizado']['us_per_replica_sweep']:>14.3f}\"\n"
      "              f\"{r['change_pct']:>+8.1f}%\")\n"
      "    print()\n"
      "    w = sorted(res['rows'], key=lambda r: r['per_device'])\n"
      "    print('ordenado por replicas por dispositivo:')\n"
      "    for r in w:\n"
      "        print(f\"  {r['per_device']:>4} por T4 -> {r['change_pct']:+6.1f}%\")\n"),
]
write(f"{OUT}/kaggle_triton_states.ipynb", C)
