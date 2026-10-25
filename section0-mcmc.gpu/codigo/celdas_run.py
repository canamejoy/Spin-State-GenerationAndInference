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
#     0     Jex2     intercambio de la 2.ª capa de vecinos        (ver abajo)
#     1     Jex3     intercambio de la 3.ª capa                   (ver abajo)
#     2     Jex4     intercambio de la 4.ª capa                   (ver abajo)
#     3     Kan1     anisotropía cúbica, constante de 1.er orden   meV
#     4     KanS     anisotropía de superficie                     meV
#     5     Hex      campo externo (magnitud)                      meV
#     6     KDM      Dzyaloshinskii–Moriya interfacial             meV
#
# LA TEMPERATURA NO ES UN PARÁMETRO. El enfriamiento es siempre el mismo,
# de 20 K a 0.1 K en pasos de 0.1 K, para todos los estados — son 200
# escalones, y np.linspace(20, 0.1, 200) los da exactos porque 19.9/199 = 0.1.
# Fijarla hace que los estados sean comparables entre sí, y convierte la
# temperatura de TRANSICIÓN en algo que se mide al final en vez de algo que se
# declara al principio.
#
# El intercambio de PRIMEROS vecinos está normalizado a Jex = 1, que es la
# escala de energía del modelo; los demás se dan relativos a él.
#
# LAS CAPAS DE VECINOS Y LAS SUBREDES
#
# Dejar un Jex2, Jex3 o Jex4 distinto de cero enciende esa capa de vecinos, y
# el cuadernillo lo detecta solo: cuenta cuáles están activas y elige el
# coloreo que corresponda. No hay nada más que configurar.
#
# Por qué hace falta elegir. Actualizar varios sitios a la vez es un
# reordenamiento EXACTO del barrido secuencial sólo si ningún par de sitios
# del mismo color interactúa. Con primeros vecinos basta un tablero de dos
# colores, porque todo primer vecino está a un desplazamiento de suma impar y
# por tanto cae en el otro color. Las capas 2 y 4 tienen desplazamientos de
# suma par y rompen eso, así que piden más subredes:
#
#   capas activas        coloreo                        subredes
#   ──────────────────   ────────────────────────────   ────────
#   1, o 1+3             (x+y+z) mod 2                      2
#   cualquiera con la 2  (x mod 2, y mod 2, z mod 2)         8
#   cualquiera con la 4  (x mod 3, y mod 3, z mod 3)        27
#
# La capa 4 llega a distancia 2 sobre un eje, y ningún coloreo de paso 2 puede
# separar eso: de ahí el paso 3. El cuadernillo VERIFICA el coloreo elegido
# recorriendo los desplazamientos de cada capa activa, en vez de darlo por
# bueno.
#
# Lo que cuesta. Un barrido pasa a ser un pase por subred, y cada pase
# actualiza 1/n de los sitios. El trabajo total por barrido no cambia, pero el
# paralelismo disponible en cada pase sí se divide por n — que en una GPU es lo
# que manda. Y el kernel optimizado de Triton está escrito para la primera
# capa: con capas superiores el cuadernillo usa el motor de JAX y lo avisa.

PARAMETROS = [
    # [ Jex2, Jex3, Jex4,  Kan1  , KanS ,  Hex   ,  KDM   ]
    [  0.0,  0.0,  0.0,  0.2867,  0.0 ,  0.7691,  0.6535],  # uniform
    [  0.0,  0.0,  0.0,  0.1770,  0.139,  0.4830,  0.8800],  # helical
    [  0.0,  0.0,  0.0,  0.3500,  0.0 ,  0.2000,  0.9500],  # points
    [  0.0,  0.0,  0.0,  0.1000,  0.200,  0.9000,  0.3000],  # lines
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
T_INICIAL     = 20.0   # inicio del enfriamiento, en K
T_FINAL       = 0.1    # fin del enfriamiento, en K
N_TEMPERATURAS = 200   # escalones: 200 da pasos de exactamente 0.1 K
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
if P.ndim != 2 or P.shape[1] != 7:
    extra = ("  (¿sobra la temperatura? ya no es un parámetro: el "
             "enfriamiento es siempre 20 K -> 0.1 K)" if P.ndim == 2
             and P.shape[1] == 8 else "")
    raise SystemExit(f"PARAMETROS debe ser un arreglo de arreglos de 7 "
                     f"elementos; llegó con forma {P.shape}{extra}")

N_ESTADOS = P.shape[0]
nombres = list(NOMBRES) if NOMBRES else []
nombres += [f"estado-{i+1}" for i in range(len(nombres), N_ESTADOS)]
nombres = nombres[:N_ESTADOS]

print(f"{N_ESTADOS} estado(s) leídos de PARAMETROS")
print(f"{'':>3} {'estado':<12}{'Jex2':>7}{'Jex3':>7}{'Jex4':>7}"
      f"{'Kan1':>9}{'KanS':>8}{'Hex':>9}{'KDM':>9}")
for i, nm in enumerate(nombres):
    J2, J3, J4, K1, KS, H, D = P[i]
    print(f"{i+1:>3} {nm:<12}{J2:>7.3f}{J3:>7.3f}{J4:>7.3f}"
          f"{K1:>9.4f}{KS:>8.3f}{H:>9.4f}{D:>9.4f}")

# ---- las capas de vecinos activas, y el coloreo que piden
# Todos los estados de una corrida comparten el coloreo, así que las capas se
# toman como la unión sobre los estados: un Jex2 distinto de cero en un solo
# estado ya obliga a las ocho subredes para todos.
import sim_multishell as MS

activas = sorted({k for i in range(N_ESTADOS)
                  for k in MS.shells_of(*P[i, 0:3])})
COLOREO, N_SUBREDES, _f = MS.pick_colouring(activas)
print(f"\ncapas de vecinos activas: {activas}")
print(f"  coloreo: {COLOREO}  ->  {N_SUBREDES} subredes")
print(f"  verificado sobre los desplazamientos de cada capa: "
      f"{'si' if MS.check_colouring(_f, activas) else 'NO'}")
if N_SUBREDES > 2:
    print(f"  un barrido son {N_SUBREDES} pases y cada uno toca 1/{N_SUBREDES}"
          f" de los sitios:")
    print(f"  el trabajo por barrido es el mismo, el paralelismo por pase se"
          f" divide por {N_SUBREDES}")
    print(f"  y el kernel de Triton es de primera capa, así que se usará el"
          f" motor de JAX")

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
# Con sólo la primera capa se usa `sim_core`, que es el motor verificado de
# referencia. Con capas superiores se usa `sim_multishell`, que lo reproduce
# EXACTAMENTE en ese caso (0 de 13500 espines distintos con la misma semilla)
# y además generaliza el coloreo.
if activas == [1]:
    geo = S.build_geometry(Rd=float(RD), L=int(CAPAS))
    eng = S.build_engine(geo, with_mvec=True)
    eng_mvec = True
else:
    geo = MS.build_geometry(Rd=float(RD), L=int(CAPAS), shells=activas)
    eng = MS.build_engine(geo)
    # el motor multicapa aun no emite el vector por replica
    eng_mvec = False
n_act = float(geo["n_active"])
print(f"red: N={geo['N']}  capas={geo['L']}  sitios activos={geo['n_active']}"
      f"  de {geo['L']*geo['N']**2} de la caja")

# ---- elegir el motor
usar_triton = False
if activas != [1] and MOTOR == "optimizado":
    raise SystemExit(f"el kernel optimizado cubre la primera capa de vecinos; "
                     f"con las capas {activas} usa MOTOR='auto' o 'estandar'")
if MOTOR in ("auto", "optimizado") and activas == [1]:
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
# El enfriamiento es el mismo para todos los estados, asi que se construye
# UNA vez y no uno por estado.
TEMPS = np.linspace(T_INICIAL, T_FINAL, int(N_TEMPERATURAS))
print(f"enfriamiento: {TEMPS[0]:.1f} K -> {TEMPS[-1]:.1f} K en "
      f"{len(TEMPS)} escalones de {TEMPS[0]-TEMPS[1]:.3f} K")


def columnas(fila, b):
    J2, J3, J4, K1, KS, H, D = fila
    c = lambda v: jnp.full((b, 1, 1, 1), float(v), jnp.float32)
    return dict(Jex=c(1.0), Jex2=c(J2), Jex3=c(J3), Jex4=c(J4),
                Kan1=c(K1), Kan2=c(0.0), KanS=c(KS),
                Hex=c(H), KDM=c(D), g=c(float(np.deg2rad(ANGULO))))

kB = 0.086173404
surf = jnp.asarray(geo["surface"][None, ...])
salida = {}
t_total = time.perf_counter()

# ---- cuantas GPU hay, y sobre que eje se reparten
#
# Esto existe porque la version anterior usaba `jax.jit` a secas, que coloca
# TODO en el dispositivo 0: en una maquina con dos T4 usaba una sola y la otra
# quedaba ociosa. Hay dos ejes sobre los que repartir y no son equivalentes.
#
#   estados   un estado entero por GPU. Es el eje que escala -- medido, 3.29x
#             sobre 4 GPU con 10 estados contra 1.10x con uno solo -- porque
#             cada estado lleva su propia escalera de temperaturas y sus
#             propios parametros, asi que no hay nada que comunicar.
#   replicas  parte las replicas de UN estado entre las GPU. Es lo que sirve
#             cuando hay menos estados que GPU. Medido en T4: 50 replicas en
#             2 T4 (25 por dispositivo) dan 3.413 us/replica-barrido contra
#             7.282 en una sola, es decir 2.13x.
#
# Los repartos DESIGUALES se serializan (+83.8% medido), asi que se exige
# divisibilidad exacta y, si no la hay, se usa una sola GPU y se dice por que.
n_dev = jax.local_device_count()
if n_dev > 1 and N_ESTADOS >= n_dev and N_ESTADOS % n_dev == 0:
    MODO, por_dev = "estados", REPLICAS
elif n_dev > 1 and REPLICAS % n_dev == 0:
    MODO, por_dev = "replicas", REPLICAS // n_dev
else:
    MODO, por_dev = "uno", REPLICAS
    if n_dev > 1:
        print(f"AVISO: {n_dev} GPU visibles, pero ni {N_ESTADOS} estados ni "
              f"{REPLICAS} replicas se dividen exacto entre {n_dev}; un "
              f"reparto desigual se serializa, asi que se usa 1 GPU")

print(f"GPU: {n_dev} visible(s) -> "
      f"{sorted({d.device_kind for d in jax.devices()})}")
if MODO == "estados":
    print(f"reparto: un estado por GPU, {N_ESTADOS // n_dev} tanda(s) "
          f"de {n_dev}, {REPLICAS} replicas cada uno")
elif MODO == "replicas":
    print(f"reparto: las {REPLICAS} replicas de cada estado entre {n_dev} "
          f"GPU, {por_dev} por dispositivo")
else:
    print(f"reparto: 1 GPU, {REPLICAS} replicas")


def _arma(i, b):
    """Lo que un dispositivo necesita para el estado `i` con `b` replicas."""
    Pc = columnas(P[i], b)
    temps = TEMPS
    return dict(Pcols=Pc, temps=temps,
                betas=jnp.asarray(1.0 / (kB * temps), jnp.float32),
                k1=jnp.where(surf, Pc["KanS"], Pc["Kan1"]),
                par=(TH.pack_params(Pc, b) if usar_triton else None))


mvectores = {}          # nombre -> (n_T, replicas, 3), o None


def _guarda(nm, obs, snap, temps, dt, repl, mvec=None):
    mvectores[nm] = None if mvec is None else np.asarray(mvec, np.float32)
    red = S.reduce_obs(np.asarray(obs), temps, n_act)
    salida[nm] = dict(T=temps.tolist(),
                      **{k: np.asarray(v).tolist() for k, v in red.items()
                         if k != "T"},
                      reloj_s=dt,
                      us_por_replica_barrido=1e6 * dt
                        / (int(N_TEMPERATURAS) * BARRIDOS * repl))
    salida[nm]["_snap"] = np.asarray(snap).tolist()
    print(f"  {nm:<12}{dt:7.1f} s   "
          f"{salida[nm]['us_por_replica_barrido']:6.3f} us/replica-barrido")


if MODO == "estados":
    # Un estado entero por dispositivo. La escalera, los parametros y la
    # semilla cambian por estado, asi que viajan como ejes del pmap.
    for base in range(0, N_ESTADOS, n_dev):
        idx = list(range(base, base + n_dev))
        pz = [_arma(i, REPLICAS) for i in idx]
        betas = jnp.stack([q["betas"] for q in pz])
        k1 = jnp.stack([q["k1"] for q in pz])
        sem = jnp.asarray([SEMILLA + i for i in idx], jnp.int32)
        sp = [S.init_spins(jax.random.PRNGKey(SEMILLA + i), geo, REPLICAS)
              for i in idx]
        if usar_triton:
            ann = TH.build_anneal_full(geo, BY=4, num_warps=4,
                                       with_mvec=True)
            par = jnp.stack([q["par"] for q in pz])
            g = lambda s, sd, bt, pr, kk: ann(s, sd, bt, pr, kk,
                                              N_TERM, N_MED)
            arg = (jnp.stack([TH.planar(x) for x in sp]), sem, betas, par, k1)
        else:
            Pst = {k: jnp.stack([q["Pcols"][k] for q in pz])
                   for k in pz[0]["Pcols"]}
            g = lambda s, sd, bt, pc: eng["anneal"](
                s, jax.random.PRNGKey(sd), bt, pc, N_TERM, N_MED)
            arg = (jnp.stack(sp), sem, betas, Pst)
        t0 = time.perf_counter()
        _o = jax.block_until_ready(jax.pmap(g)(*arg))
        # only the Triton engine returns the magnetisation vector
        obs, final, snaps, mvec = _o if (usar_triton or eng_mvec) \
            else (*_o, None)
        dt = time.perf_counter() - t0
        for j, i in enumerate(idx):
            _guarda(nombres[i], np.asarray(obs)[j],
                    np.asarray(snaps)[j][-1], pz[j]["temps"], dt, REPLICAS,
                    None if mvec is None else np.asarray(mvec)[j])

elif MODO == "replicas":
    # Las replicas de un estado repartidas. Todos los dispositivos comparten
    # la escalera y los parametros; solo cambian los espines y la semilla.
    for i, nm in enumerate(nombres):
        q = _arma(i, por_dev)
        sem = jnp.asarray([SEMILLA + i * 1000 + d for d in range(n_dev)],
                          jnp.int32)
        sp = S.init_spins(jax.random.PRNGKey(SEMILLA + i), geo, REPLICAS)
        sp = sp.reshape(n_dev, por_dev, *sp.shape[1:])
        if usar_triton:
            ann = TH.build_anneal_full(geo, BY=4, num_warps=4,
                                       with_mvec=True)
            g = lambda s, sd: ann(s, sd, q["betas"], q["par"], q["k1"],
                                  N_TERM, N_MED)
            arg = (jnp.stack([TH.planar(x) for x in sp]), sem)
        else:
            g = lambda s, sd: eng["anneal"](s, jax.random.PRNGKey(sd),
                                            q["betas"], q["Pcols"],
                                            N_TERM, N_MED)
            arg = (sp, sem)
        t0 = time.perf_counter()
        _o = jax.block_until_ready(jax.pmap(g)(*arg))
        # only the Triton engine returns the magnetisation vector
        obs, final, snaps, mvec = _o if (usar_triton or eng_mvec) \
            else (*_o, None)
        dt = time.perf_counter() - t0
        # `anneal` devuelve (n_T, 6, replicas) POR dispositivo, asi que las
        # replicas se reunen concatenando por el ULTIMO eje. Aplanar el eje
        # principal mezclaria temperaturas con observables, que es justo el
        # error que `reduce_obs` advierte en su docstring.
        obs = np.concatenate(list(np.asarray(obs)), axis=-1)
        # the vector is (n_dev, n_T, por_dev, 3); the replicas live on axis 1
        # of each device's slice, so they rejoin there and nowhere else
        mv = (None if mvec is None
              else np.concatenate(list(np.asarray(mvec)), axis=1))
        _guarda(nm, obs, np.asarray(snaps)[0][-1], q["temps"], dt, REPLICAS,
                mv)

else:
    for i, nm in enumerate(nombres):
        q = _arma(i, REPLICAS)
        sp = S.init_spins(jax.random.PRNGKey(SEMILLA + i), geo, REPLICAS)
        if usar_triton:
            ann = TH.build_anneal_full(geo, BY=4, num_warps=4,
                                       with_mvec=True)
            f = jax.jit(lambda s: ann(s, SEMILLA + i, q["betas"], q["par"],
                                      q["k1"], N_TERM, N_MED))
            arg = TH.planar(sp)
        else:
            f = jax.jit(lambda s: eng["anneal"](
                s, jax.random.PRNGKey(SEMILLA + i), q["betas"], q["Pcols"],
                N_TERM, N_MED))
            arg = sp
        f = f.lower(arg).compile()          # compila sin ejecutar
        t0 = time.perf_counter()
        _o = jax.block_until_ready(f(arg))
        obs, final, snaps, mvec = _o if (usar_triton or eng_mvec) \
            else (*_o, None)
        dt = time.perf_counter() - t0
        _guarda(nm, obs, snaps[-1], q["temps"], dt, REPLICAS, mvec)

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
    j = -1                                   # el fondo de la escalera, 0.1 K
    print(f"{nm:<12}{r['T'][j]:>7.2f}{r['E'][j]:>11.5f}{r['M'][j]:>9.4f}"
          f"{r['Mz'][j]:>9.4f}{r['Cv'][j]:>10.4f}{r['chi'][j]:>10.4f}"
          f"{r['Q'][j]:>9.4f}")

# ══════════════════════════════════════════════════════════════════════════
#  LA TRANSICIÓN
# ══════════════════════════════════════════════════════════════════════════
#
# Con la escalera fija, la temperatura de transición deja de ser un dato de
# entrada y pasa a ser algo que se mide. Se localiza por el PICO de las dos
# funciones de respuesta, que es la definición operativa habitual en un
# sistema finito:
#
#   Cv = ⟨(ΔE)²⟩ / (k_B T² N)     pico -> transición de ORDENAMIENTO
#   χ  = ⟨(Δ|M|)²⟩ / (k_B T)      pico -> transición MAGNÉTICA
#
# Tres advertencias que hay que leer junto al número:
#
#   · La malla es de 0.1 K, así que el pico no se puede situar mejor que eso.
#     Aquí NO se interpola: un ajuste parabólico sobre tres puntos ruidosos
#     da una precisión aparente que el dato no tiene.
#   · Esto es un sistema FINITO (5245 sitios). Lo que se ve es un máximo
#     redondeado, no una divergencia, y su posición se desplaza con el tamaño:
#     no es la T_c del límite termodinámico.
#   · Si el pico cae en un extremo de la escalera, la transición está FUERA
#     del rango recorrido y el número no significa nada. Se avisa.

print()
print(f"{'estado':<12}{'T(pico Cv)':>12}{'Cv máx':>10}"
      f"{'T(pico χ)':>12}{'χ máx':>10}  aviso")
transiciones = {}
for nm in nombres:
    r = salida[nm]
    T = np.asarray(r["T"])
    fila = {}
    msg = []
    for k, etiq in (("Cv", "Cv"), ("chi", "chi")):
        v = np.asarray(r[k], dtype=float)
        v = np.where(np.isfinite(v), v, -np.inf)
        j = int(np.argmax(v))
        fila[f"T_{etiq}"] = float(T[j])
        fila[f"{etiq}_max"] = float(v[j])
        if j == 0 or j == len(T) - 1:
            msg.append(f"el pico de {etiq} cae en el borde")
    transiciones[nm] = fila
    print(f"{nm:<12}{fila['T_Cv']:>12.1f}{fila['Cv_max']:>10.4f}"
          f"{fila['T_chi']:>12.1f}{fila['chi_max']:>10.4f}"
          f"  {'; '.join(msg) if msg else 'ok'}")

print()
print(f"malla de {TEMPS[0]-TEMPS[1]:.1f} K, asi que cada pico vale "
      f"+-{(TEMPS[0]-TEMPS[1])/2:.2f} K; sistema finito de "
      f"{int(n_act)} sitios, el maximo es redondeado y no una divergencia")

with open("resultados.json", "w") as f:
    json.dump(dict(parametros=P.tolist(), nombres=nombres,
                   geometria=dict(Rd=RD, capas=CAPAS, N=int(geo["N"]),
                                  sitios=int(geo["n_active"])),
                   cronograma=dict(replicas=REPLICAS, n_temperaturas=N_TEMPERATURAS,
                                   barridos=BARRIDOS, n_term=N_TERM, n_med=N_MED),
                   motor="optimizado" if usar_triton else "estandar",
                   resultados=salida, transiciones=transiciones), f, indent=1)
print("\nguardado resultados.json")

# ══════════════════════════════════════════════════════════════════════════
#  LOS ESTADOS, UN .npz POR ESTADO
# ══════════════════════════════════════════════════════════════════════════
#
# Hasta aqui lo unico que sobrevivia a la corrida eran las curvas YA
# promediadas sobre replicas, mas una sola imagen (la capa media de la replica
# 0 en la ultima temperatura). El registro por replica se perdia.
#
# Cada archivo lleva:
#
#   parametros    (7,)           los siete del estado, en el orden de
#                                PARAMETROS: Jex2, Jex3, Jex4, Kan1, KanS,
#                                Hex, KDM
#   M             (50, 200, 3)   la magnetizacion media, indexada
#                                [replica, temperatura, componente]
#   temperaturas  (200,)         la escalera, que indexa el eje 1 de M
#
# La forma es (replicas, temperaturas, 3) y no una lista plana de 10000 filas
# a proposito: asi no hay ningun orden que documentar. M[r] es la curva de
# enfriamiento COMPLETA de la replica r, M[:, t] son las 50 replicas a la
# temperatura t, y temperaturas[t] dice cual es esa temperatura. Una lista
# plana obliga a recordar si el orden era por temperatura o por replica, y ese
# es justo el tipo de detalle que se pierde entre quien escribe el archivo y
# quien lo lee.
#
# Que es M. No es una configuracion instantanea: es el promedio sobre los
# N_MED barridos de medicion de esa temperatura, para esa replica. Es decir un
# promedio termico, uno por replica, no una muestra suelta.

import os

guardados = []
for nm in nombres:
    mv = mvectores.get(nm)
    if mv is None:
        print(f"  {nm}: sin vector de magnetizacion "
              f"(lo da el motor optimizado, y aqui corrio el estandar)")
        continue
    i = nombres.index(nm)
    nT, nR, _ = mv.shape
    # el motor lo entrega (temperatura, replica, 3); se transpone, no se
    # aplana, para que el archivo no tenga un orden que haya que explicar
    M = np.transpose(mv, (1, 0, 2)).astype(np.float32)
    fn = f"estado_{nm}.npz"
    np.savez_compressed(
        fn,
        parametros=np.asarray(P[i], np.float32),
        M=M,
        temperaturas=np.asarray(salida[nm]["T"], np.float32))
    guardados.append(fn)
    print(f"  {fn}  parametros (7,)  M {M.shape} "
          f"[replica, temperatura, componente]  "
          f"{os.path.getsize(fn)/1024:.0f} KB")

if guardados:
    print(f"\nguardados {len(guardados)} .npz")
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
