# -*- coding: utf-8 -*-
"""Cell bodies for the ready-to-run notebook, kept out of the generator so
neither file has to nest triple quotes."""

PARAMS_CELL = r'''# ══════════════════════════════════════════════════════════════════════════
#  LOS PARÁMETROS — lo único que hay que editar
# ══════════════════════════════════════════════════════════════════════════
#
# Un arreglo por estado. El cuadernillo cuenta los arreglos y de ahí sale
# cuántos estados hay: no hay que declararlo en ningún otro sitio.
#
# Cada arreglo lleva los OCHO parámetros del conjunto de datos, EN ESTE ORDEN:
#
#   índice  nombre   qué es                                        unidades
#   ──────  ──────   ───────────────────────────────────────────   ────────
#     0     T        temperatura objetivo: a esta se enfría        K
#     1     Jex2     intercambio de la 2.ª capa de vecinos        (ver abajo)
#     2     Jex3     intercambio de la 3.ª capa                   (ver abajo)
#     3     Jex4     intercambio de la 4.ª capa                   (ver abajo)
#     4     Kan1     anisotropía cúbica, constante de 1.er orden   meV
#     5     KanS     anisotropía de superficie                     meV
#     6     Hex      campo externo (magnitud)                      meV
#     7     KDM      Dzyaloshinskii–Moriya interfacial             meV
#
# El intercambio de PRIMEROS vecinos está normalizado a Jex = 1, que es la
# escala de energía del modelo; los demás se dan relativos a él.
#
# ⚠  Jex2, Jex3 y Jex4 DEBEN ser 0. No es una limitación del código sino del
#    algoritmo: la actualización por tablero de ajedrez es un reordenamiento
#    EXACTO del barrido secuencial sólo si los enlaces conectan sitios de
#    paridad opuesta. Las capas 2 y 4 conectan sitios de la MISMA paridad, así
#    que con ellas encendidas el tablero deja de ser exacto y la cadena
#    simulada ya no es la que se pretende. El cuadernillo lo verifica y se
#    detiene si no se cumple.

PARAMETROS = [
    # [  T  , Jex2, Jex3, Jex4,  Kan1  , KanS ,  Hex   ,  KDM   ]
    [  2.10,  0.0,  0.0,  0.0,  0.2867,  0.0 ,  0.7691,  0.6535],  # uniform
    [  2.10,  0.0,  0.0,  0.0,  0.1770,  0.139,  0.4830,  0.8800],  # helical
    [  5.00,  0.0,  0.0,  0.0,  0.3500,  0.0 ,  0.2000,  0.9500],  # points
    [  1.00,  0.0,  0.0,  0.0,  0.1000,  0.200,  0.9000,  0.3000],  # lines
]

# Nombres para las tablas y las figuras. Si se deja vacío se numeran solos.
NOMBRES = ["uniform", "helical", "points", "lines"]

# ══════════════════════════════════════════════════════════════════════════
#  LA GEOMETRÍA
# ══════════════════════════════════════════════════════════════════════════
RD     = 18.30   # radio del disco, en pasos de red. Define N = 2(⌊Rd⌋+1)+1
CAPAS  = 5       # capas en z. ⚠ ver la nota del ancho de vector más abajo
ANGULO = 90.0    # dirección del campo externo, en grados sobre el plano x–z

# ══════════════════════════════════════════════════════════════════════════
#  EL CRONOGRAMA
# ══════════════════════════════════════════════════════════════════════════
REPLICAS      = 50     # réplicas independientes POR ESTADO
T_INICIAL     = 20.0   # temperatura caliente de partida del enfriamiento
N_TEMPERATURAS = 200   # escalones del enfriamiento, de T_INICIAL a la T del estado
BARRIDOS      = 20000  # barridos por temperatura (⅔ termalizan, ⅓ miden)
SEMILLA       = 42

# ══════════════════════════════════════════════════════════════════════════
#  EL MOTOR
# ══════════════════════════════════════════════════════════════════════════
# "auto"       -> usa el kernel optimizado de Triton si se puede, si no el de JAX
# "optimizado" -> exige Triton; falla si no está disponible
# "estandar"   -> el motor de JAX de siempre, sin Triton
MOTOR = "auto"

# ⚠  SOBRE EL NÚMERO DE RÉPLICAS Y DE CAPAS
#    Con el motor "estandar" el costo depende de la divisibilidad por dos:
#    hay castigo (hasta +49 % según la tarjeta) cuando
#        v₂(REPLICAS) + v₂(CAPAS) ≥ 2
#    siendo v₂(n) cuántas veces n se divide por 2. Con CAPAS = 5 (impar) basta
#    con que REPLICAS no sea múltiplo de cuatro. El motor "optimizado" NO paga
#    ese castigo (+0.3 % medido), así que ahí el número es libre.
#    El cuadernillo avisa cuál es tu caso antes de arrancar.
'''

SETUP_CELL = r'''# Entorno: detecta la GPU, instala lo que falte y deja la cadena de
# herramientas coherente. Pensado para Kaggle, Colab o una máquina propia.
import subprocess, sys, os, glob, re

def _sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()

print("GPU:", _sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                  "--format=csv,noheader") or "no se detectó ninguna")

def _tiene(mod):
    try:
        __import__(mod); return True
    except Exception:
        return False

faltan = [p for p, m in (("jax[cuda12]", "jax"), ("jax-triton", "jax_triton"))
          if not _tiene(m)]
if faltan:
    print("instalando:", " ".join(faltan))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *faltan],
                   check=False)

# Triton trae su propio ptxas, a veces una versión de CUDA por delante del que
# usa XLA para ensamblar. Si difieren, nada compila. Se fija Triton al de XLA
# ANTES de importar jax, porque el backend congela su cadena al inicializarse.
def _ver(p):
    m = re.search(r"release (\d+)\.(\d+)", _sh(p, "--version"))
    return tuple(map(int, m.groups())) if m else (0, 0)

_pats = ["/usr/local/lib/python3*/dist-packages/nvidia/cuda_nvcc/bin/ptxas",
         "/usr/local/lib/python3*/site-packages/nvidia/cuda_nvcc/bin/ptxas",
         "/opt/conda/lib/python3*/site-packages/nvidia/cuda_nvcc/bin/ptxas",
         "/usr/local/cuda*/bin/ptxas"]
_xla = sorted({p for q in _pats for p in glob.glob(q)})
_xla = [(v, p) for v, p in ((_ver(p), p) for p in _xla) if v > (0, 0)]
if _xla:
    v, p = min(_xla)
    os.environ["TRITON_PTXAS_PATH"] = p
    print(f"TRITON_PTXAS_PATH -> CUDA {v[0]}.{v[1]}")

import jax
print("jax", jax.__version__, "|", jax.devices())
try:
    import triton, jax_triton as jt
    print("triton", triton.__version__, "| jax_triton",
          getattr(jt, "__version__", "?"))
except Exception as e:
    print("Triton no disponible:", type(e).__name__, e)
'''

VALIDA_CELL = r'''# Lee los parámetros, cuenta los estados y verifica lo que no es negociable.
import numpy as np

v2 = lambda n: (n & -n).bit_length() - 1 if n else 0
P = np.asarray(PARAMETROS, dtype=float)
if P.ndim != 2 or P.shape[1] != 8:
    raise SystemExit(f"PARAMETROS debe ser un arreglo de arreglos de 8 "
                     f"elementos; llegó con forma {P.shape}")

N_ESTADOS = P.shape[0]
nombres = list(NOMBRES) if NOMBRES else []
nombres += [f"estado-{i+1}" for i in range(len(nombres), N_ESTADOS)]
nombres = nombres[:N_ESTADOS]

# Las capas 2 y 4 conectan sitios de la MISMA paridad: con ellas el tablero de
# ajedrez deja de ser un reordenamiento exacto del barrido secuencial.
malos = [(nombres[i], P[i, 1:4]) for i in range(N_ESTADOS) if np.any(P[i, 1:4] != 0)]
if malos:
    for nm, v in malos:
        print(f"  {nm}: Jex2,Jex3,Jex4 = {v}")
    raise SystemExit("Jex2, Jex3 y Jex4 deben ser 0: con ellos el tablero de "
                     "ajedrez no es exacto y la cadena simulada no es la que "
                     "se pretende.")

print(f"{N_ESTADOS} estado(s) leídos de PARAMETROS")
print(f"{'':>3} {'estado':<12}{'T':>7}{'Kan1':>9}{'KanS':>8}{'Hex':>9}{'KDM':>9}")
for i, nm in enumerate(nombres):
    T, _, _, _, K1, KS, H, D = P[i]
    print(f"{i+1:>3} {nm:<12}{T:>7.2f}{K1:>9.4f}{KS:>8.3f}{H:>9.4f}{D:>9.4f}")

s = v2(REPLICAS) + v2(CAPAS)
print(f"\nv₂({REPLICAS}) + v₂({CAPAS}) = {v2(REPLICAS)} + {v2(CAPAS)} = {s}")
if s >= 2:
    print("  ⚠ con el motor 'estandar' este lote paga el castigo del ancho de")
    print(f"    vector. Con CAPAS={CAPAS}, una REPLICAS que no sea múltiplo de")
    print("    cuatro lo evita. El motor 'optimizado' no lo paga.")
else:
    print("  ✓ libre del castigo del ancho de vector en los dos motores.")

total = N_ESTADOS * REPLICAS
print(f"\ntrabajo: {N_ESTADOS} estados × {REPLICAS} réplicas = {total} cadenas")
print(f"         {N_TEMPERATURAS} temperaturas × {BARRIDOS} barridos "
      f"= {N_TEMPERATURAS*BARRIDOS:,} barridos por cadena")
'''

CORRE_CELL = r'''# Construye el motor y corre el enfriamiento de cada estado.
import time, json
import jax, jax.numpy as jnp
import sim_core as S

N_TERM = BARRIDOS * 2 // 3
N_MED  = BARRIDOS - N_TERM
geo = S.build_geometry(Rd=float(RD), L=int(CAPAS))
eng = S.build_engine(geo)
n_act = float(geo["n_active"])
print(f"red: N={geo['N']}  capas={geo['L']}  sitios activos={geo['n_active']}"
      f"  de {geo['L']*geo['N']**2} de la caja")

# ---- elegir el motor
usar_triton = False
if MOTOR in ("auto", "optimizado"):
    try:
        import triton_halo as TH
        _ = TH.build_anneal_full(geo, BY=4, num_warps=4)
        usar_triton = True
    except Exception as e:
        if MOTOR == "optimizado":
            raise
        print(f"motor optimizado no disponible ({type(e).__name__}); se usa el estándar")
print("motor:", "optimizado (Triton)" if usar_triton else "estándar (JAX)")

# ---- verificación de física: el optimizado debe reproducir al estándar
if usar_triton:
    import triton_bench as TB
    chk = TB.verify(geo, by=4, warps=4)
    if not chk["ok"]:
        raise SystemExit("el motor optimizado no reproduce la física del "
                         "estándar: no se corre nada")

# ---- parámetros de un estado, difundidos a las réplicas
def columnas(fila, b):
    T, _, _, _, K1, KS, H, D = fila
    c = lambda v: jnp.full((b, 1, 1, 1), float(v), jnp.float32)
    return dict(Jex=c(1.0), Kan1=c(K1), Kan2=c(0.0), KanS=c(KS),
                Hex=c(H), KDM=c(D), g=c(float(np.deg2rad(ANGULO)))), float(T)

kB = 0.086173404
surf = jnp.asarray(geo["surface"][None, ...])
salida = {}
t_total = time.perf_counter()

for i, nm in enumerate(nombres):
    Pcols, T_obj = columnas(P[i], REPLICAS)
    k1 = jnp.where(surf, Pcols["KanS"], Pcols["Kan1"])
    temps = np.linspace(T_INICIAL, T_obj, int(N_TEMPERATURAS))
    betas = jnp.asarray(1.0 / (kB * temps), jnp.float32)
    sp = S.init_spins(jax.random.PRNGKey(SEMILLA + i), geo, REPLICAS)

    if usar_triton:
        ann = TH.build_anneal_full(geo, BY=4, num_warps=4)
        par = TH.pack_params(Pcols, REPLICAS)
        f = jax.jit(lambda s: ann(s, SEMILLA + i, betas, par, k1, N_TERM, N_MED))
        arg = TH.planar(sp)
    else:
        f = jax.jit(lambda s: eng["anneal"](s, jax.random.PRNGKey(SEMILLA + i),
                                            betas, Pcols, N_TERM, N_MED))
        arg = sp

    f = f.lower(arg).compile()          # compila sin ejecutar
    t0 = time.perf_counter()
    obs, final, snaps = jax.block_until_ready(f(arg))
    dt = time.perf_counter() - t0

    red = S.reduce_obs(np.asarray(obs), temps, n_act)
    salida[nm] = dict(T=temps.tolist(),
                      **{k: np.asarray(v).tolist() for k, v in red.items()
                         if k != "T"},
                      reloj_s=dt,
                      us_por_replica_barrido=1e6 * dt
                        / (int(N_TEMPERATURAS) * BARRIDOS * REPLICAS))
    # la capa media de la réplica 0 en la última temperatura, para la figura
    salida[nm]["_snap"] = np.asarray(snaps[-1]).tolist()
    print(f"  {nm:<12}{dt:7.1f} s   "
          f"{salida[nm]['us_por_replica_barrido']:6.3f} µs/réplica-barrido")

print(f"\ntotal: {(time.perf_counter()-t_total)/60:.2f} min para "
      f"{N_ESTADOS} estado(s)")
'''

METRICAS_CELL = r'''# Las métricas físicas, estado por estado, en su temperatura objetivo.
#
# Cómo se calcula cada una:
#
#   E/sitio   energía total del Hamiltoniano dividida por los sitios activos.
#             Cada enlace se cuenta UNA vez, usando sólo los desplazamientos
#             +x, +y y +z, nunca los seis.
#   |M|       norma del vector de magnetización promedio sobre los sitios
#             activos. Mz es su componente z.
#   Cv        capacidad calorífica, de la varianza de la energía:
#                 Cv = ⟨(ΔE)²⟩ / (k_B T² N)
#   χ         susceptibilidad, de la varianza de |M|:  χ = ⟨(Δ|M|)²⟩ / (k_B T)
#   Q         carga topológica de la capa media por la fórmula de
#             Berg–Lüscher, que es cuantizada — a diferencia de una diferencia
#             finita, que no lo es.
#
# Cv y χ son varianzas, y acumular E y E² en bruto las destruye en float32:
# a T = 0.2 una ulp de ⟨E²⟩ es mayor que la varianza misma y Cv sale negativa.
# Por eso se acumula la fluctuación respecto a una referencia tomada una vez
# por temperatura; un desplazamiento constante deja la varianza intacta.

print(f"{'estado':<12}{'T':>7}{'E/sitio':>11}{'|M|':>9}{'Mz':>9}"
      f"{'Cv':>10}{'χ':>10}{'Q':>9}")
for nm in nombres:
    r = salida[nm]
    j = -1                                   # la temperatura objetivo
    print(f"{nm:<12}{r['T'][j]:>7.2f}{r['E'][j]:>11.5f}{r['M'][j]:>9.4f}"
          f"{r['Mz'][j]:>9.4f}{r['Cv'][j]:>10.4f}{r['chi'][j]:>10.4f}"
          f"{r['Q'][j]:>9.4f}")

with open("resultados.json", "w") as f:
    json.dump(dict(parametros=P.tolist(), nombres=nombres,
                   geometria=dict(Rd=RD, capas=CAPAS, N=int(geo["N"]),
                                  sitios=int(geo["n_active"])),
                   cronograma=dict(replicas=REPLICAS, n_temperaturas=N_TEMPERATURAS,
                                   barridos=BARRIDOS, n_term=N_TERM, n_med=N_MED),
                   motor="optimizado" if usar_triton else "estandar",
                   resultados=salida), f, indent=1)
print("\nguardado resultados.json")
'''

FIGURA_CELL = r'''# Las curvas termodinámicas de cada estado al enfriar.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for nm in nombres:
    r = salida[nm]
    for a, k, t in ((ax[0, 0], "E", "energía por sitio"),
                    (ax[0, 1], "M", "|M|"),
                    (ax[1, 0], "Cv", "Cv"),
                    (ax[1, 1], "chi", "χ")):
        a.plot(r["T"], r[k], lw=1.4, label=nm)
        a.set_xlabel("T (K)"); a.set_ylabel(t)
        a.grid(alpha=.25); a.invert_xaxis()
ax[0, 0].legend(fontsize=9)
fig.suptitle(f"{N_ESTADOS} estado(s) · {REPLICAS} réplicas · "
             f"{N_TEMPERATURAS}×{BARRIDOS} barridos", y=1.0)
plt.tight_layout(); plt.show()

# La configuración final de la capa media, una réplica por estado.
fig, axs = plt.subplots(1, N_ESTADOS, figsize=(3.1 * N_ESTADOS, 3.3))
axs = np.atleast_1d(axs)
for a, nm in zip(axs, nombres):
    im = a.imshow(np.asarray(salida[nm].get("_snap", np.zeros((geo["N"], geo["N"])))),
                  cmap="jet", vmin=-1, vmax=1)
    a.set_title(f"{nm}  ($s_z$)", fontsize=10); a.axis("off")
fig.colorbar(im, ax=axs, fraction=.025)
plt.show()
'''
