# Auditoría técnica del kernel Triton de Monte Carlo para nanodots magnéticos

Auditoría realizada contra `inspection_kernel_mcmc.md`. Solo lectura: no se
modificó, sobrescribió ni eliminó ningún archivo del proyecto, y no se ejecutó
ninguna simulación de producción. Las pruebas nuevas son deterministas, de
segundos, y se ejecutaron en CPU.

**Niveles de evidencia** usados en todo el documento, según el §2 del encargo:

| nivel | significado |
|---|---|
| **Verificado** | respaldado directamente por código o por un resultado reproducible que se ejecutó aquí |
| **Hallazgo confirmado** | inconsistencia concreta, con la evidencia que la demuestra |
| **Riesgo** | posibilidad real que requiere una prueba adicional para cerrarse |
| **No verificado** | no hay evidencia suficiente; se declara como tal |

---

## 1. Resumen ejecutivo

**La física está bien y el rendimiento está mal contado.** Esa es la conclusión
en una línea.

El Hamiltoniano implementado en Triton coincide **término por término** con la
implementación de referencia en JAX, y lo hace con una prueba de exactitud, no
estadística: alimentadas con la misma aleatoriedad, las dos rutas producen
**0 de 182 520 espines distintos** en ambas paridades. El coloreado es correcto
para las interacciones realmente activas, y lo es por una razón estructural que
verifiqué y que el manuscrito no declara.

Lo que no se sostiene tal como está escrito son las cifras de producción. Las
tres cifras insignia del PDF —57.6 min, 1.9 h y 3.8 h— **no son relojes de la
escalera completa**: son extrapolaciones ×100 de una corrida de **2 de las 200
temperaturas**. La línea base de CPU, 130.7 días, es una extrapolación ×500 000
de una corrida de 400 barridos con 100 réplicas. El texto de la sección las
presenta como «the wall clock of the complete production ladder», y el §11 del
encargo prohíbe explícitamente presentar estimaciones como mediciones.

El segundo hallazgo de severidad alta es del generador: el espaciado entre
semillas es **ocho veces menor** que lo que una corrida completa consume, así
que dos corridas con semillas contiguas comparten el 87.5 % de su espacio de
claves con contadores idénticos. Eso afecta a la independencia entre réplicas
cuando se reparten entre dispositivos o entre estados.

Nada de lo encontrado invalida el Hamiltoniano ni la cadena de Markov dentro de
una sola llamada. No hay hallazgos de severidad crítica.

---

## 2. Inventario de archivos y versiones

| Componente | Archivo y función | Implementación activa | Evidencia |
|---|---|---|---|
| Referencia JAX | `codigo/sim_core.py` · `build_engine.substep` (l. 83–107) | **sí** — es el patrón de todas las pruebas de exactitud | 248 líneas, sha256 `649973371a82` |
| Kernel Triton | `codigo/triton_halo.py` · `_substep_kernel` (l. 60–231) | **sí** — es el motor que corre en producción | 807 líneas, sha256 `f397f10c5870` |
| Motor multicapa | `codigo/sim_multishell.py` · `build_engine` | **parcial** — único con J2/J3/J4; **sin ruta Triton** | 296 líneas, sha256 `f7416703e62f` |
| ΔE local | `sim_core.substep` l. 96–104 · `_substep_kernel` l. 208–222 | ambas | comparadas término por término en §3 |
| Propuesta y aceptación | `_substep_kernel` l. 140–161 (RNG) y l. 224 (Metropolis) | Triton con RNG interno | `HAS_RNG=True` es la variante de producción |
| Subredes | `sim_multishell.pick_colouring` / `check_colouring` (l. 85–92) | sí, pero **solo la verifica**; el kernel fija paridad | `_substep_kernel` l. 121: `par = (xc+yc+z) % 2` |
| Geometría y máscaras | `sim_core.build_geometry` (l. 93–127 en multishell) | sí | prueba determinista en §5 |
| Observables | `triton_halo` `_eM_kernel` / `_topo_kernel`; `sim_core.energy/magnetization/topo_charge` | ambas | §7 |
| Reducción a escalares | `sim_core.reduce_obs` (l. 209–236) | sí, **compartida por las dos rutas** | 10+ llamadas en el repositorio |
| Banco de pruebas | `codigo/triton_bench.py` · `verify()` (l. 536+) | sí | produce `exact_ok` / `physics_ok` |
| Producción Modal | `codigo/app.py` (10 llamadas a `build_anneal_full`) | sí | `grep build_anneal_full` |
| Parámetros | `codigo/theta4.json` **y** `codigo/celdas_run.py::PARAMETROS` | **dos juegos que discrepan** | hallazgo M3 |
| Resultados del manuscrito | `resultados/curvas/states100_{1,2,4}gpu.json` | sí | hallazgo A1 |

**Entorno registrado en la corrida de validación** (`resultados/kaggle/triton_bench_t4.json`):
Tesla T4, capacidad de cómputo 7.5, Python 3.13.15, JAX 0.11.1, Triton 3.8.0,
jax_triton 0.4.1. Precisión **float32** en todo el camino.

**Versión activa vs histórica.** `sim_core` es la referencia y `triton_halo` el
candidato; `sim_multishell` generaliza el coloreo pero **no tiene kernel Triton**,
así que no participa en ninguna cifra de rendimiento. Los `resultados/` contienen
135 archivos JSON, de los cuales los tres `states100_*gpu.json` son los que
alimentan la figura insignia.

---

## 3. Tabla de equivalencia del Hamiltoniano

Reconstruido desde el código ejecutado, no desde los comentarios.

| Término | Manuscrito (ec. 2) | JAX (`sim_core.substep`) | Triton (`_substep_kernel`) | ¿Equivalentes? | ¿Activo en producción? |
|---|---|---|---|---|---|
| Intercambio J₁ | `−J₁ Σ s_i·s_j` | l. 96–97: `−Jex·Σ dS·sv` con `sv` = 6 vecinos | l. 208: `−Jex*(d0*sv0+d1*sv1+d2*sv2)` | **sí** | **sí** |
| Intercambio J₂ | `−J₂ Σ_⟨ij⟩₂` | **ausente** | **ausente** | n/a | **no** (`Jex2 = 0`) |
| Intercambio J₃ | `−J₃ Σ_⟨ij⟩₃` | **ausente** | **ausente** | n/a | **no** (`Jex3 = 0`) |
| Intercambio J₄ | `−J₄ Σ_⟨ij⟩₄` | **ausente** | **ausente** | n/a | **no** (`Jex4 = 0`) |
| Anisotropía cúbica K₁ | `K₁/S₀⁴ · Σ(s_a s_b)²` | l. 80 vía `_an` | l. 211–217, `i4 = 1/SO⁴` | **sí** | **sí** |
| Anisotropía K₂ | `K₂/S₀⁶ ·(s_x s_y s_z)²` | l. 81 | l. 214, `i6 = 1/SO⁶` | **sí** | **no** (`Kan2 = 0` siempre) |
| Anisotropía de superficie | `K_S` en `∂Λ` | campo `K1_eff` precalculado en el anfitrión | `k1_ptr`, campo por sitio | **sí** | **no** (`KanS = 0`, hallazgo M2) |
| DMI interfacial | solo enlaces en plano | l. 100–103 | l. 220, con `dxz/dxy/dyx/dyz` | **sí**, signo a signo | **sí** |
| Campo externo | `−H·Σ s_i` | l. 104: `−Hex(dS₀cos g + dS₂ sin g)` | l. 221: `−Hex*(d0*cosg + d2*sing)` | **sí** | **sí** |
| Norma del espín | `‖s‖ = S₀` | l. 92: normaliza la normal | l. 154–158: `g/nrm*SO` | **sí** | **sí** · medido `‖norma−1‖ ≤ 2.4e−7` |
| Frontera en z | abierta | l. 87–88: pone a cero el plano que envuelve | l. 112–117: compuertas `gp/gm` | **sí** | **sí** |

**Verificación del DMI, término a término.** JAX calcula
`dS₁(S_xp₂−S_xm₂) − dS₂(S_xp₁−S_xm₁) + dS₂(S_yp₀−S_ym₀) − dS₀(S_yp₂−S_ym₂)`
(l. 100–103). Triton calcula `d1*dxz − d2*dxy + d2*dyx − d0*dyz` (l. 220), donde
`dyx` se define en el bloque de la componente 0 (l. 184), `dxy` en el de la 1
(l. 194) y `dxz`/`dyz` en el de la 2 (l. 204–205). Sustituyendo, las dos
expresiones son **idénticas carácter a carácter**.

### Conclusión sobre el alcance físico

> **Verificado.** El kernel Triton implementa, exactamente, un Heisenberg de
> **primeros vecinos** con DMI interfacial, anisotropía cúbica (K₁ como campo
> por sitio, K₂ escalar) y campo Zeeman, sobre un disco cilíndrico con frontera
> periódica en plano y abierta fuera de plano.
>
> **Hallazgo confirmado (M4).** No implementa J₂, J₃ ni J₄. El manuscrito lo
> declara en su sección de limitaciones —«does not yet implement beyond the
> first shell»— así que el alcance está correctamente acotado, pero la ecuación
> (2) presenta la suma sobre `S ⊆ {1,2,3,4}` como el modelo, y el lector puede
> concluir que el kernel la realiza. **Debe matizarse en la ecuación, no solo en
> las limitaciones.**
>
> **Verificado.** La restricción no tiene consecuencia práctica para estos
> experimentos: los cuatro estados de `theta4.json` llevan `Jex2 = Jex3 = Jex4 = 0`.

---

## 4. Auditoría del algoritmo de Metropolis y del coloreado

### 4.1 Metropolis

| Aspecto | Hallazgo | Nivel |
|---|---|---|
| Distribución de la propuesta | Tres normales independientes normalizadas a `S₀` → uniforme en la esfera. `_substep_kernel` l. 152–158 | Verificado |
| Conservación de la norma | Medido `max‖‖s‖−S₀‖ = 2.4e−7` (f32) | Verificado |
| Criterio de aceptación | `(dE < 0) | (ru < exp(−β dE))` en **ambas** rutas | Verificado |
| `dE < 0` vs `dE ≤ 0` | En `ΔE = 0` cae al test `ru < exp(0) = 1`, cierto con probabilidad 1 porque `ru ∈ [0,1)`. **Equivalente**, no es un defecto | Verificado |
| Definición de β | `β = 1/(k_B T)`, `k_B = 0.086173404` meV/K, en el anfitrión | Verificado |
| Protección numérica | Triton acota `exp(−β·min(dE, 80))`; JAX **no** acota | Riesgo, severidad baja |
| Barridos por sweep | 2 sub-pasos (par, impar). `build_anneal_full` l. 466–467 | Verificado |
| Estado entre temperaturas | Se arrastra: el `scan` pasa `s` de una T a la siguiente | Verificado |
| Sitios inactivos | `upd = live & (disk > 0.5) & (par == PARITY)`, l. 121–122 | Verificado |

**Sobre el acotado asimétrico.** Triton evalúa `exp(−β·min(ΔE, 80))` y JAX
evalúa `exp(−β·ΔE)` sin cota. Con β grande (T → 0.1 K, β ≈ 116) y ΔE > 0.7 meV,
el argumento supera −80 y JAX produce un `exp` subdesbordado a 0, mientras
Triton satura. **Las dos dan rechazo**, así que el comportamiento coincide; la
diferencia es de robustez, no de física. No es un hallazgo, es una asimetría que
conviene documentar.

### 4.2 Coloreado — prueba determinista ejecutada

El §6 del encargo pide no aceptar comentarios como demostración. Ejecuté la
comprobación directa sobre la **geometría real de producción** (Rd = 18.30,
L = 5, N = 39, 5245 sitios activos), recorriendo los seis desplazamientos de
primera capa con x,y periódicos y z abierto:

```
pares interactuantes (1ª capa, dentro del disco):  28 632
  pares del MISMO color:                                0
  de ellos, en la costura periódica:                    0
sitios del disco sobre el borde de la caja:             0
```

> **Verificado.** Ningún par interactuante comparte clase. El halo sin barrera
> es seguro para las interacciones activas.

**Y hay aquí un detalle que el manuscrito no declara y que debería.** `N` es
**impar** (39). Con frontera periódica, la paridad `(x+y+z) mod 2` **no cierra**
en la costura: `x = 38` y `x = 0` son vecinos y ambos son pares. El coloreado
solo sobrevive porque **ningún sitio del disco toca el borde de la caja**.

Verifiqué que eso es **estructural y no casualidad**: la construcción usa
`L₂ = ⌊Rd⌋ + 1`, así que el sitio `(L₂, 0)` dista `L₂ > Rd` y queda siempre
fuera del disco. Lo comprobé en diez radios:

| Rd | 2.0 | 2.5 | 3.0 | 4.3 | 7.0 | 10.5 | 18.0 | 18.3 | 19.0 | 25.7 |
|---|---|---|---|---|---|---|---|---|---|---|
| sitios en el borde | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| choques de color | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

> **Verificado**, y es un argumento de corrección no obvio que merece estar en
> el artículo: el texto actual menciona que N es impar como un hecho de
> *rendimiento* («the box extent carries no factor of two»), sin señalar que con
> frontera periódica rompería el tablero si la máscara no lo impidiera.

### 4.3 Accesos de memoria

- **No cruzan réplicas.** `sca = sp_ptr + b*sb` con `sb = 3·LZ·N²`; los
  desplazamientos en plano se calculan dentro de `o_c ∈ [0, N²)` (l. 106–110) y
  z se acota con `min/max` (l. 112–113). Cota máxima `(N−1)·N + N−1 = N²−1`.
  **Verificado** por aritmética de índices.
- **Carriles fuera de rango.** `BX` es la potencia de dos ≥ N, así que `xs`
  excede `N−1`; la máscara `live` los anula y `xc = where(xs<N, xs, 0)` mantiene
  la dirección dentro de rango incluso enmascarada (l. 93–98). **Verificado.**

---

## 5. Auditoría del generador aleatorio

### 5.1 Lo que está bien

`_substep_kernel` l. 141–161 usa **una clave y cuatro contadores distintos**:

```python
off = ((b * LZ + z) * N + yc[:, None]) * N + xc[None, :]
o4  = off * 4
g0, g1, g2 = randn(sd, o4+0), randn(sd, o4+1), randn(sd, o4+2)
ru         = rand(sd, o4+3)
```

> **Verificado.** Dentro de una llamada no hay colisión: `off` es único por
> `(réplica, capa, fila, columna)` y los cuatro sorteos difieren en el contador.
> El máximo en producción es `off·4 ≈ 1.52e8`, holgadamente dentro de int32.

El comentario del propio código (l. 143–150) documenta que una versión anterior
variaba la **clave** en vez del contador y sesgaba la aceptación un 1.1 %. Ese
defecto está corregido.

### 5.2 Hallazgo confirmado A2 — el espaciado entre semillas es insuficiente

`triton_halo.py:509` fija la clave inicial como `c0 = seed · 1000003`, y
`l. 467` la avanza **+2 por barrido**. Una escalera de producción son
200 × 20 000 = 4·10⁶ barridos, es decir **8·10⁶ de avance de clave**, contra un
espaciado entre semillas contiguas de **1 000 003**.

```
semilla 42: claves [42 000 126 .. 50 000 126]
semilla 43: claves [43 000 129 .. 51 000 129]
SOLAPE: 6 999 997 claves = 87.5 % de la corrida
```

Dentro del solape la **clave coincide y el contador también**, porque `off` no
depende de la semilla. Dos corridas con semillas contiguas producen entonces las
**mismas direcciones de prueba y las mismas uniformes** para el mismo índice de
réplica dentro del shard, desfasadas en el tiempo.

**Dónde muerde esto.** No dentro de una llamada —ahí las réplicas sí están
separadas por contador—, sino **entre llamadas con semillas vecinas**:

| ruta | semillas | consecuencia |
|---|---|---|
| cuadernillo, reparto por réplicas | `SEMILLA + i·1000 + d`, d = 0,1 | las 25 réplicas del dispositivo 0 y las 25 del 1 comparten flujo |
| cuadernillo, reparto por estados | `SEMILLA + i` | estados contiguos comparten flujo |
| `app.py` producción | 42, 43, … | corridas contiguas comparten flujo |

> **Consecuencia científica.** La afirmación de que las 50 réplicas son
> independientes no se sostiene cuando se reparten entre dispositivos. Las
> barras de error calculadas suponiendo independencia están subestimadas, en una
> cantidad que no he medido.
>
> **Nivel: Hallazgo confirmado por aritmética.** Lo que verifiqué es que los
> rangos de clave se solapan y que el contador no depende de la semilla. **No
> ejecuté** la comprobación de que dos corridas producen sorteos idénticos
> —requiere GPU— y la dejo como prueba recomendada P1.

**Corrección sugerida** (no aplicada, según el §2 del encargo): usar un
espaciado mayor que el avance máximo previsto, p. ej. `c0 = seed · 2³²` en
int64, o mejor, derivar la clave con una función de mezcla en vez de una
multiplicación lineal.

### 5.3 Lo que no está demostrado

> **No verificado.** No existe en el repositorio ninguna prueba de la
> distribución de los sorteos: ni isotropía de la dirección propuesta, ni
> uniformidad de `ru`, ni independencia entre ambas. El §8 del encargo advierte
> exactamente contra confundir «contadores distintos» con una demostración, y
> esa demostración no está. Lo más cercano es la tasa de aceptación contra la
> referencia (tabla 3 del PDF), que es una prueba indirecta y de potencia
> limitada.

---

## 6. Auditoría de observables y precisión

| Observable | Fórmula implementada | Normalización | ¿Coincide JAX/Triton? |
|---|---|---|---|
| E | cada enlace una vez, solo traslaciones +x,+y,+z | `/n_a` en `reduce_obs` | sí, `1.2e−7` relativo |
| ‖M‖ | `‖Σ s_i·disco / n_a‖` | por sitio | sí |
| M_z | componente z de lo anterior | por sitio | sí |
| Q | Berg–Lüscher, dos triángulos por plaqueta, `/4π` | solo plaquetas con las 4 esquinas en el disco | sí, `8.4e−8` absoluto |
| C_v | `var(E_total) / (k_B T² n_a)` | **por sitio, estándar** | comparten `reduce_obs` |
| χ | `var(m) / (k_B T)` | **falta un factor `n_a`** | comparten `reduce_obs` |

### 6.1 Hallazgo confirmado M1 — χ y C_v no comparten convención

`sim_core.energy` (l. 121) devuelve la energía **total**; `magnetization`
(l. 123) devuelve la magnetización **por sitio**. `reduce_obs` (l. 233–234)
divide C_v por `n_a` y **no** multiplica χ por `n_a`:

```
C_v = var(E_total) / (k_B T² n_a)      <- coincide con la definición por sitio
χ   = var(m)       / (k_B T)
    la definición por sitio sería n_a · var(m) / (k_B T)
```

> **Consecuencia.** La χ reportada está **5245 veces por debajo** de la
> susceptibilidad por sitio. Es un factor **constante**, así que la posición del
> pico y la forma de la curva no cambian: **la temperatura de transición que
> reporta el cuadernillo no se ve afectada**. Lo que no es interpretable es la
> escala absoluta de χ, ni compararla con C_v.
>
> **Severidad: Media.** No invalida ninguna conclusión física del manuscrito,
> porque el manuscrito no cita valores absolutos de χ. Afecta a cualquiera que
> reutilice las curvas.

### 6.2 Determinismo y float32

- **Verificado.** La reducción por parciales de bloque es independiente del
  orden de retiro, lo que hace admisible la comparación exacta de observables.
  Las cifras de `1.2e−7` y `8.4e−8` existen **por eso**.
- **Verificado.** El centrado de las varianzas respecto a una referencia por
  temperatura es exacto (un desplazamiento constante no cambia una varianza) y
  es lo que evita un C_v negativo en f32. El código lo documenta en
  `sim_core.py` l. 157–167 con el cálculo del ulp.
- **Distinción que pide el §9.** El determinismo de la reducción **no** implica
  reproducibilidad de la trayectoria: dos corridas con semillas distintas son
  cadenas de Markov distintas y solo pueden coincidir en distribución.

---

## 7. Resultados de validación existentes

| Prueba | Condiciones | Métrica | Resultado | Criterio | Limitaciones |
|---|---|---|---|---|---|
| Exactitud, clase par | misma aleatoriedad, T4 | espines distintos | **0 / 182 520** | == 0 | una geometría, un lote |
| Exactitud, clase impar | misma aleatoriedad | espines distintos | **0 / 182 520** | == 0 | ídem |
| ΔE, par / impar | misma aleatoriedad | max abs | 1.91e−6 / 1.43e−6 | resolución f32 | ídem |
| Observables, E | configuración común | relativo | 1.2e−7 | f32 | una configuración |
| Observables, Q | configuración común | absoluto | 8.4e−8 | f32 | una configuración |
| Equilibrio, T = 20 | RNG propio cada motor | σ | 0.09 | < 3 | 16 réplicas |
| Equilibrio, T = 5 | ídem | σ | 0.17 | < 3 | 16 réplicas |
| Equilibrio, T = 1 | ídem | σ | 0.23 | < 3 | 16 réplicas |
| Norma del espín | tras un paso con RNG interno | desviación | 2.4e−7 | f32 | — |
| Curvas reducidas | contra el control consigo mismo | σ media / máx | 0.99 / 1.90 | control: 0.87 / 2.53 | ver abajo |

**Lo que sostiene cada una.** Las filas de exactitud son las fuertes: con la
aleatoriedad fija la única diferencia admisible es aritmética, así que un conteo
no nulo señalaría un término mal transcrito. Esa es una prueba de **corrección
de la fórmula**, y la pasa.

**Lo que no sostienen.** Ninguna prueba de equilibrio aborda la autocorrelación.
El §10 del encargo lo pide explícitamente: *«no trates los sweeps correlacionados
como observaciones independientes»*. Las σ de la tabla se calculan sobre 16
réplicas sin estimar τ.

> **Riesgo.** Con τ desconocido, un σ de 0.23 puede corresponder a muchas menos
> observaciones efectivas de las que el denominador supone. La prueba de curvas
> reducidas **sí** lleva un control (la referencia contra sí misma, 0.87 / 2.53)
> y es metodológicamente la más sólida del conjunto, porque mide el suelo de
> ruido en vez de suponerlo.

> **No verificado.** No hay comparación de **distribuciones** —ni de energía ni
> de magnetización— entre los dos motores. La equivalencia estadística está
> respaldada por medias dentro de σ, no por una prueba de dos muestras.

---

## 8. Auditoría de benchmarks y escalabilidad

### 8.1 Hallazgo confirmado A1 — las cifras insignia son extrapolaciones

Los tres archivos que alimentan la figura principal declaran `n_temps = 2`:

| arma | T medidas | reloj **medido** | µs/réplica-barrido | ×100 | **en el PDF** |
|---|---|---|---|---|---|
| 1× H100 | **2** de 200 | 135.5 s | 0.6774 | 3.8 h | 3.8 h |
| 2× H100 | **2** de 200 | 66.9 s | 0.3343 | 1.9 h | 1.9 h |
| 4× H100 | **2** de 200 | 34.6 s | 0.1728 | 57.6 min | **57.6 min** |

Las tres cifras del manuscrito se reproducen **exactamente** multiplicando la
tasa medida por 4·10⁶ barridos y 5000 réplicas. Es decir: se midió el **1 %** de
la escalera y se escaló ×100.

La línea base de CPU es peor: `cpu_rate_repeat.json` registra **400 barridos con
100 réplicas**, y los 130.7 días salen de extrapolar eso por un factor de
**500 000**.

> **Severidad: Alta.** El texto de la sección dice «It presents the wall clock
> of the complete production ladder — 200 temperatures by 20 000 sweeps». Eso
> es una medición que no se hizo. El §11 del encargo lo prohíbe en estos
> términos: *«No presentes estimaciones como mediciones directas.»*
>
> **Lo que sí es medición:** la tasa de 0.129 ns por actualización de sitio. Eso
> está medido (sobre 2 temperaturas) y es legítimo como tasa. Lo que no está
> medido es el reloj de pared de la escalera completa.
>
> **Atenuante real, que debe citarse junto a la cifra:** el repositorio contiene
> una calibración de estas extrapolaciones contra medición (9 celdas, error
> máximo 7.9 %, 8 de 9 erraron por exceso). No es una extrapolación a ciegas.
> Pero una extrapolación calibrada sigue siendo una extrapolación.

### 8.2 Lo que el protocolo hace bien

- **Verificado.** La línea base de CPU toma el **mínimo** de 4 repeticiones
  (564.7 de un rango 564.7–675.5 µs). El mínimo es el denominador **más
  desfavorable para nosotros**: con la mediana el cociente subiría de 834× a
  ~926×. Es la elección conservadora, y conviene decirlo en el artículo.
- **Verificado.** `triton_bench` compila con `.lower().compile()` antes de
  arrancar el cronómetro, y sincroniza con `block_until_ready` antes de pararlo.
- **Verificado.** La física se comprueba **en la misma corrida** que produce los
  tiempos (`verify()` antes de cronometrar), no en otra.

### 8.3 Escalabilidad y la regla v₂(B) + v₂(L)

El §12 pide comprobar la hipótesis. `resultados/curvas/triton_width.json` mide
cuatro recuentos de capas (4, 5, 6, 8) × cuatro lotes (64, 65, 100, 101).

> **Hallazgo confirmado, y ya corregido en el PDF durante esta sesión.** La
> condición `v₂(B) + v₂(L) ≥ 2` **solo puede voltearse variando B cuando
> v₂(L) < 2**, lo que entre las capas medidas ocurre en L = 5 y L = 6, y **no**
> en L = 4 ni L = 8, donde se cumple para todo B. Citar las cuatro capas como
> evidencia era engañoso. Los residuos reales son 0.30 % y 1.70 % en H100, y
> 0.56 % y 0.97 % en T4: **como mucho 1.7 %, sin signo consistente**.
>
> **Severidad: era Media, ya resuelta.** La versión actual de `section0.tex`
> dice exactamente eso.

---

## 9. Tabla de afirmaciones del manuscrito frente a la evidencia

| Afirmación | Clasificación | Evidencia |
|---|---|---|
| «single fused kernel» | **Respaldada directamente** | un lanzamiento por clase de color; `_substep_kernel` hace stencil, RNG, Metropolis y escritura en una pasada |
| «synchronisation-free halo» | **Respaldada directamente** | la prueba determinista del §4.2: 0 de 28 632 pares comparten color, así que ningún bloque escribe lo que otro lee |
| «counter-based proposals» | **Respaldada parcialmente** | los contadores son correctos dentro de una llamada (verificado), pero **entre** llamadas las claves se solapan 87.5 % (hallazgo A2), y no hay prueba de la distribución |
| «deterministic observables» | **Respaldada directamente** | la reducción por parciales es independiente del orden; es lo que hace posibles las cifras de 1.2e−7 y 8.4e−8 |
| Equivalencia numérica | **Respaldada directamente** | 0/182 520 espines distintos, ambas paridades |
| Equivalencia estadística | **Requiere experimento adicional** | medias dentro de σ, pero sin control de autocorrelación y sin prueba de distribuciones |
| 0.129 ns por actualización | **Respaldada directamente** como *tasa* | medida en `states100_1gpu.json` |
| «the complete production ladder» 57.6 min | **Debe matizarse** | extrapolación ×100 de 2 de 200 temperaturas (A1) |
| 834× sobre la CPU | **Debe matizarse** | el denominador es una extrapolación ×500 000; el numerador, ×100. El cociente es defendible (misma física, mismo código) pero **ninguno de los dos lados es un reloj completo** |
| Escalado 3.91× sobre 4 GPU | **Respaldada directamente** | 0.6774 / 0.1728 = 3.92, de relojes medidos |
| Hamiltoniano extendido `S ⊆ {1,2,3,4}` | **Debe matizarse** | el kernel es solo primera capa (M4); está dicho en limitaciones pero no en la ecuación |
| Anisotropía de superficie como campo por sitio | **Debe matizarse** | implementada y correcta, pero `KanS = 0` en los cuatro estados de producción (M2): **nunca se ejercita** |

---

## 10. Hallazgos clasificados por severidad

### Crítica

Ninguno. No se encontró nada que invalide el Hamiltoniano, la dinámica de
Metropolis ni la comparación principal.

---

### A1 · Alta · Las cifras de producción son extrapolaciones presentadas como mediciones

- **Archivo:** `resultados/curvas/states100_{1,2,4}gpu.json`; texto en
  `tesis/section0.tex` §4.1.
- **Evidencia:** los tres archivos declaran `n_temps = 2`. Las cifras del PDF se
  reproducen exactamente como `tasa × 4·10⁶ × 5000`. La CPU se extrapola ×500 000
  desde 400 barridos y 100 réplicas.
- **Consecuencia:** una afirmación de rendimiento central está mal etiquetada.
  El cociente de 834× no compara dos relojes completos.
- **Recomendación:** etiquetar las tres cifras como estimaciones y citar la
  calibración de extrapolaciones existente (error máximo 7.9 %), o medir una
  escalera completa de al menos un brazo.

### A2 · Alta · El espaciado entre semillas del RNG es 8× menor que el avance de una corrida

- **Archivo y función:** `codigo/triton_halo.py:509` (`c0 = seed·1000003`) y
  `:467` (`c + 2` por barrido).
- **Evidencia:** avance por corrida 8·10⁶ contra espaciado 1 000 003 →
  **87.5 % de solape** entre semillas contiguas, con contadores idénticos porque
  `off` no depende de la semilla.
- **Consecuencia:** las réplicas repartidas entre dispositivos, y los estados
  con semillas contiguas, **no son independientes**. Las barras de error que
  suponen independencia están subestimadas.
- **Prueba necesaria (sigue siendo hipótesis en su parte empírica):** P1 abajo.
- **Recomendación:** espaciar las semillas por encima del avance máximo, o
  derivar la clave con una función de mezcla.

---

### M1 · Media · χ y C_v usan convenciones de normalización distintas

- **Archivo:** `codigo/sim_core.py:233-234`, compartido por ambas rutas.
- **Evidencia:** `energy()` devuelve el total y `magnetization()` el valor por
  sitio; C_v divide por `n_a` y χ no multiplica por `n_a`.
- **Consecuencia:** χ está `n_a = 5245` veces por debajo de la susceptibilidad
  por sitio. **La posición del pico no cambia**, así que la transición que
  reporta el cuadernillo es válida; la escala absoluta no.
- **Recomendación:** fijar una convención y declararla junto a la fórmula.

### M2 · Media · La anisotropía de superficie nunca se ejercita en producción

- **Archivo:** `codigo/theta4.json`.
- **Evidencia:** `KanS = 0.0` en los cuatro estados.
- **Consecuencia:** el manuscrito presenta el campo por sitio como un rasgo
  distintivo del modelo y lo usa para argumentar que ninguna implementación
  publicada lleva esa combinación. La ruta de código existe y es correcta, pero
  **ningún resultado publicado la ejercita**.
- **Recomendación:** o correr un estado con `KanS ≠ 0`, o declarar que el
  término está implementado y no activado en estos experimentos.

### M3 · Media · Dos juegos de parámetros «de producción» que discrepan

- **Archivos:** `codigo/theta4.json` vs `codigo/celdas_run.py::PARAMETROS`.
- **Evidencia:**

  | estado | Kan1 (json / cuadernillo) | KanS | Hex | KDM |
  |---|---|---|---|---|
  | uniform | 0.2867 / 0.2867 | 0.000 / 0.000 | 0.7691 / 0.7691 | 0.6535 / 0.6535 |
  | helical | **0.2591 / 0.1770** | **0.000 / 0.139** | **0.0926 / 0.4830** | **1.0250 / 0.8800** |
  | points | **0.2777 / 0.3500** | 0.000 / 0.000 | **0.3889 / 0.2000** | **0.9998 / 0.9500** |
  | lines | **0.5085 / 0.1000** | **0.000 / 0.200** | **0.0223 / 0.9000** | **0.5123 / 0.3000** |

- **Consecuencia:** afecta a la reproducibilidad. Un lector que corra el
  cuadernillo no reproduce los estados del manuscrito.
- **Recomendación:** declarar cuál es canónico y que el otro lo importe.

### M4 · Media · El kernel no implementa J₂, J₃ ni J₄

- **Evidencia:** `grep` de `Jex2|Jex3|Jex4` da 0 en `sim_core.py` y en
  `triton_halo.py`; solo `sim_multishell.py` los tiene, y ese módulo **no tiene
  ruta Triton**.
- **Consecuencia:** la ecuación (2) del manuscrito promete más de lo que el
  kernel medido realiza. Está dicho en limitaciones, no en la ecuación.
- **Recomendación:** acotar el alcance en la propia ecuación.

---

### B1 · Baja · Parámetro muerto en `_an`

`sim_core.py:77` declara `_an(s, Kan1, Kan2, K1_eff)` y **nunca usa `Kan1`**
(l. 80 usa solo `K1_eff`). Confunde al lector sobre cuál constante manda.

### B2 · Baja · Acotado de la exponencial asimétrico entre rutas

Triton acota `exp(−β·min(ΔE,80))`, JAX no. El resultado coincide (ambos
rechazan), pero la asimetría no está documentada.

### B3 · Baja · La corrección del coloreado con N impar no está declarada

Verificada y estructural (§4.2), pero el artículo presenta N impar como un hecho
de rendimiento sin señalar que la máscara circular es lo que salva el tablero
bajo frontera periódica. Es un argumento que **fortalece** el artículo.

---

## 11. Pruebas adicionales recomendadas

| id | prueba | qué cerraría | coste |
|---|---|---|---|
| **P1** | Dos corridas cortas con semillas 42 y 43, misma geometría; comparar los espines de prueba sorteados en el rango de claves solapado | Confirma empíricamente A2, hoy demostrado solo por aritmética | minutos de GPU |
| **P2** | Medir τ (tiempo de autocorrelación) de E y ‖M‖ a 3 temperaturas y recalcular las σ de la tabla de validación con observaciones efectivas | Cierra la laguna del §10: hoy las σ suponen independencia | minutos |
| **P3** | Prueba de dos muestras (Kolmogorov–Smirnov o Anderson–Darling) sobre las distribuciones de E y ‖M‖ entre los dos motores | Convierte «equivalencia estadística» de parcialmente respaldada a respaldada | minutos |
| **P4** | χ² sobre la isotropía de la dirección propuesta y Kolmogorov–Smirnov sobre `ru`, más correlación entre `ru` y ΔE | Cierra la laguna del §8: hoy no hay prueba de la distribución | segundos |
| **P5** | Una escalera completa de 200 temperaturas en **un** brazo (el de 4× H100, ~58 min estimados) | Convierte A1 de extrapolación en medición para al menos un punto | ~1 h de 4 H100 |
| **P6** | Un estado de producción con `KanS ≠ 0` | Cierra M2 | igual que un estado normal |

---

## 12. Plan de corrección ordenado por prioridad

1. **Reetiquetar las cifras de producción** (A1). Es edición de texto, coste
   cero, y es el hallazgo de mayor impacto sobre la credibilidad del artículo.
2. **Arreglar el espaciado de semillas** (A2) y volver a correr lo que dependa
   de independencia entre réplicas. Es una línea de código y una re-corrida.
3. **Ejecutar P1, P2 y P4** — las tres son baratas y cierran las dos lagunas
   metodológicas que el encargo señala por nombre.
4. **Unificar los parámetros** (M3) y declarar el canónico.
5. **Fijar la convención de χ** (M1) y escribirla junto a la fórmula.
6. **Acotar la ecuación (2)** al alcance real del kernel (M4) y declarar M2.
7. **Añadir el argumento de N impar** (B3) al artículo: es evidencia a favor que
   hoy no se está usando.
8. Limpiezas B1 y B2.

---

## 13. Limitaciones que deben declararse en el artículo

1. El kernel medido implementa **primeros vecinos únicamente**; J₂, J₃ y J₄ no
   tienen ruta Triton.
2. La anisotropía de superficie está implementada pero **vale cero en todos los
   experimentos publicados**.
3. Los relojes de pared de producción son **extrapolaciones calibradas**, no
   mediciones; lo medido es la tasa sobre 2 de 200 temperaturas.
4. La equivalencia estadística se apoya en **medias dentro de σ sin control de
   autocorrelación**, no en una comparación de distribuciones.
5. Las conclusiones de rendimiento valen para **una geometría** (disco
   Rd = 18.30, L = 5) y **dos arquitecturas** (T4 y H100); no se generalizan.
6. Toda la aritmética es **float32**.

---

## 14. Criterio final de evaluación

**1. ¿El kernel Triton implementa el Hamiltoniano que declara el manuscrito?**
Parcialmente. Implementa **exactamente** —verificado término por término y con
0 de 182 520 espines distintos— el Heisenberg de primeros vecinos con DMI
interfacial, anisotropía cúbica y campo. **No** implementa las capas 2, 3 y 4
que la ecuación (2) incluye. Como esas capas valen cero en todos los
experimentos, la restricción no afecta a ningún resultado publicado, pero la
ecuación promete más que el código.

**2. ¿La dinámica de Metropolis y el coloreado son correctos para las
interacciones activas?**
Sí, verificado. La propuesta es isótropa y de norma conservada (2.4e−7), el
criterio de aceptación coincide en ambas rutas, y el coloreado no admite ningún
par interactuante del mismo color: 0 de 28 632 en la geometría de producción,
por una razón estructural que comprobé en diez radios.

**3. ¿La evidencia disponible respalda la equivalencia numérica y estadística?**
La **numérica**, sí y de forma contundente: con aleatoriedad idéntica el
resultado es exacto, no estadístico. La **estadística**, solo parcialmente: las
medias coinciden dentro de σ, pero las σ no controlan la autocorrelación y no
hay comparación de distribuciones. Añádase que A2 compromete la independencia
entre réplicas repartidas.

**4. ¿Los benchmarks comparan el mismo trabajo y justifican las aceleraciones?**
El **trabajo** sí es el mismo: mismo código, misma física, misma geometría, y el
cociente por actualización de sitio es la unidad correcta. Los **relojes** no
son comparables como se presentan: ambos lados del cociente de 834× son
extrapolaciones, de ×100 y de ×500 000 respectivamente. El escalado de 3.91×
sobre cuatro GPU **sí** sale de relojes medidos.

**5. ¿Qué afirmaciones deben corregirse, matizarse o demostrarse mejor?**
Corregirse: «the wall clock of the complete production ladder». Matizarse: el
alcance de la ecuación (2), la anisotropía de superficie como rasgo distintivo,
y el 834×. Demostrarse mejor: la equivalencia estadística y la calidad del
generador.

**6. ¿Cuál es el principal riesgo científico que queda abierto?**
**El solape de claves del generador (A2).** Es el único hallazgo que puede
afectar a un número físico publicado y no solo a cómo se describe. Si las
réplicas repartidas entre dispositivos comparten flujo aleatorio, las medias
siguen siendo correctas pero sus incertidumbres están subestimadas en una
cantidad que nadie ha medido. Todo lo demás es de etiquetado, alcance o
presentación.

---

## Las cinco acciones más importantes antes de enviar el artículo

1. **Reetiquetar las tres cifras de producción y el 834× como estimaciones
   calibradas**, citando el error máximo del 7.9 % de la calibración existente.
2. **Corregir el espaciado de semillas** y volver a correr cualquier resultado
   que dependa de independencia entre réplicas.
3. **Medir τ y recalcular las σ de validación** con observaciones efectivas.
4. **Acotar la ecuación (2)** a primeros vecinos y declarar que `KanS = 0` en
   todos los experimentos.
5. **Correr una escalera completa en un brazo** para que al menos un punto de la
   figura insignia sea una medición.
