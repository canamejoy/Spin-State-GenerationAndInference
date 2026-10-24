# Sección 0 escrita para la tesis

**El entregable es `section0.pdf`** (12 páginas, compilado con
`~/texlive/2026/bin/x86_64-linux/pdflatex`). Su fuente es `section0.tex`, que
transpone las tres secciones en markdown y les añade las cinco figuras de
`figuras/` y la bibliografía resuelta.

Las tres secciones en markdown están redactadas contra los contratos de
papersmith (`~/papersmith-ai/sections/`). Están en **inglés** porque el paper de
`main/main.tex` lo está (`elsarticle`); la conversación y las páginas
interactivas son otra cosa.

| archivo | contrato | prosa | estado |
|---|---|---|---|
| `01-materials-and-methods.md` | `sections/01` | 1883 palabras (560–3950) | ✅ dentro |
| `03-results-and-discussion.md` | `sections/03` | 1803 palabras (1170–4150) | ✅ dentro |
| `05-related-work.md` | `sections/05` | 1153 palabras (820–1700) | ✅ dentro |

## Cómo se recompila

```bash
export PATH="$HOME/texlive/2026/bin/x86_64-linux:$PATH"
cd section0-mcmc.gpu/codigo && python make_tesis_figs.py   # las 5 figuras
cd ../tesis && pdflatex section0.tex && pdflatex section0.tex
```

Dos pasadas, por las referencias cruzadas. `microtype` no está en esta
instalación de TeX Live y va comentado; es cosmético.

## Qué lleva el PDF y qué no

El PDF lleva **Materials and Methods** completa, **Validation**, **Performance**
y **Limitations**. En concreto, por pedido explícito:

| se pidió | dónde está |
|---|---|
| la **descomposición matricial**, clara | §2.1 `The matrix decomposition` — la configuración como arreglo de rango 5, el operador de traslación con su compuerta en z, la suma de vecinos, ΔE contrayendo el eje de componentes, y la actualización como un `where` enmascarado. Ecs. (7)–(11) |
| explicar las **subredes** | §1.2 — ahora desglosa capa por capa **qué le hace cada desplazamiento a la paridad de x+y+z**, que es lo que decide la admisibilidad, y por qué el coloreo se ensancha en dos pasos (2 → 8 → 27) |
| el **desglose de cada métrica física** | §2.6 `The observables, one by one` — E, ‖M‖, M_z, Q, C_v y χ, cada una con sus ejes de reducción, su normalización y el cuidado numérico que exige. Ecs. (13)–(16) |

**La comparación contra el estado del arte está FUERA por ahora**, a petición
del autor, hasta hablarlo con el tutor. El código y las mediciones siguen en el
repositorio (`codigo/bench_charilaou.py`,
`resultados/kaggle/charilaou_t4*.json`, `resultados/modal/charilaou_a*.json`);
solo no se redacta en la sección. Con ella fuera, la única línea base que queda
en el PDF es la **CPU**, que es la comparación realmente controlada: mismo
código, misma física, mismo workload.

## Por qué son tres y no una

Pediste «la descripción de la simulación más optimizada, desglosando cómo se
hace, desglosando las matrices, cómo se calculan las métricas físicas». Eso es
**Materials and Methods**, y su contrato dice literalmente:

> **Does not carry** any result, measured number, or comparison

Así que la gráfica insignia de los 100 estados y la comparación contra el estado
del arte **no pueden ir ahí**. Van en Results and Discussion, que es la sección
cuyo contrato las pide. Y el panorama de la literatura va en Related Work, que
es la que no lleva ecuaciones y sí lleva un artefacto de síntesis. Meter las
tres cosas en una sección habría violado los tres contratos a la vez.

## Alcance: sólo lo optimizado

Las secciones reportan **el simulador final y nada más**. No aparecen el motor
estándar como línea base de rendimiento, ni versiones intermedias, ni intentos
abandonados, ni la secuencia en que se llegó. Eso queda en
`informe/informe.pdf`, que es donde el proceso sí es el contenido.

La implementación de referencia **sí** aparece, pero sólo donde la corrección
exige algo contra lo que ser correcto: las filas de la Tabla 1 que comparan
espín a espín con la misma aleatoriedad. Nunca como comparación de velocidad.
Las únicas líneas base de rendimiento son la **CPU** y el **estado del arte
publicado**.

Verificado: Results no contiene ninguna de las cifras del motor estándar
(−51.3 %, −31.4 %, +47.7 %, +17.8 %, +49.9 %) ni las palabras «reference
engine» o «standard engine».

## Las cuatro capas de vecinos

El modelo ya no está restringido a primeros vecinos. La Sección 2 define un
coloreo general $\gamma$ y su condición de admisibilidad, y da el coloreo más
estrecho para cada conjunto de capas activas:

| capas activas | coloreo | clases |
|---|---|---|
| {1} o {1,3} | $(x+y+z)\bmod 2$ | 2 |
| con la 2 | paridad por eje | 8 |
| con la 4 | paso 3 por eje | 27 |

Las capas 1 y 3 tienen desplazamientos de suma impar y dejan el grafo bipartito;
las 2 y 4 no. La capa 4 llega a distancia 2 sobre un eje, que ningún coloreo de
paso 2 separa. La garantía de «sin barrera» de la Sección 3.2 se sigue de la
condición de admisibilidad y no del número de clases, así que vale igual con 8
y con 27.

El motor está en `codigo/sim_multishell.py` y **reproduce `sim_core`
exactamente** cuando sólo hay primera capa: 0 de 13 500 espines distintos con la
misma semilla, y energía, |M| y Q a 0.000e+00.

## Lo que cumple cada contrato

**Materials and Methods.** No lleva ningún número medido — se puede verificar
leyéndola. El conjunto de datos *no* es de esta sección (es del experimental
setup), así que abre directamente en la maquinaria y su única figura es el
diagrama de resumen, que es obligatorio. Las cuatro contribuciones están
definidas cada una **bajo la propiedad que entrega**, con ese nombre:

1. *Contiguous neighbour access* — el diseño component-planar
2. *A synchronisation-free halo* — el stencil de cargas desplazadas
3. *Counter-based proposals* — la generación dentro del kernel
4. *Deterministic observables* — la reducción por parciales de bloque

La ecuación (12) es la combinación general, y cada símbolo en ella traza a una
declaración anterior.

**Results and Discussion.** Cero ecuaciones numeradas, sin preámbulo, cada
artefacto citado antes de aparecer, y el movimiento de tres tiempos por
artefacto (señalar → leer en números → dar el mecanismo). Los tres registros de
discusión están repartidos: inferencial (*indicates, suggests*), asertivo
(*attributed to, because*) y corroborativo (*confirms*).

**Related Work.** Temática, no catálogo: dos líneas de literatura por lo que
cada una decide, no una lista de papers. Cierra con tres frentes abiertos y la
brecha, y la Tabla 4 la hace legible como fila.

## Las cinco figuras

Las genera `codigo/make_tesis_figs.py` leyendo los archivos de medición de
`resultados/`; nada está tecleado a mano salvo las etiquetas.

| figura | qué es | archivo |
|---|---|---|
| 1 | el diagrama de resumen del kernel, obligatorio por contrato | `fig1_diagram.pdf` |
| 2 | **la insignia**: 100 estados, CPU + T4 + H100, eje log | `fig2_flagship_cpu.pdf` |
| 3 | la misma sin CPU y con eje lineal | `fig3_flagship_gpu.pdf` |
| 4 | el piso de réplicas por dispositivo | `fig4_floor.pdf` |
| 5 | costo contra el lote, una línea por número de capas | `fig5_shape.pdf` |

Las barras son **verticales** en las figuras 2 y 3, y el eje va en horas, no en
segundos crudos.

## Lo que se corrigió al armar el PDF

Tres cosas que estaban mal en el markdown y ahora lo están en los dos lados:

**1. La afirmación sobre la condición de anchura estaba escogida a conveniencia.**
Decía «0.3 % en el dispositivo de centro de datos […] medido sobre 4, 5, 6 y 8
capas». El 0.3 % es sólo la celda L=5; la celda L=6 del mismo archivo
(`resultados/curvas/triton_width.json`, campo `penalty`) mide **1.70 %**. Y en
L=4 y L=8 la condición $v_2(B)+v_2(L)\ge 2$ **no se puede voltear variando B**,
porque $v_2(L)\ge 2$ ya la satisface, así que esas dos capas no informan de
nada. Ahora dice: 0.30 % y 1.70 % en el H100, 0.56 % y 0.97 % en el T4, como
mucho 1.7 %, sin signo consistente.

**2. Dos referencias de fila apuntaban mal.** La fila 2 de la tabla de
comparación es el T4, no el agregado de cuatro dispositivos (que esa tabla omite
a propósito, porque tabula ritmos por dispositivo). Y la línea base de CPU es la
fila 3, no la 4 —la 4 es Charilaou.

**3. El 12× sobre Charilaou se bajó a 2–3×.** El 12× crudo (0.129 contra
1.55 ns) compara un H100 con un Tesla V100, dos generaciones aparte.
Normalizando por nuestro T4 (1.75 ns) contra ese V100, que lleva el doble de
multiprocesadores y casi el triple de ancho de banda, el método va adelante
**entre 2× y 3×**, no 12×. Eso es lo que dice el PDF.

## Lo que sigue faltando

**Las claves de cita.** La bibliografía del PDF es un `thebibliography` a mano
con los siete arXiv resueltos por `paper_resolve.py` de papersmith (título y
autores reales, no de segunda mano). Para integrarla en `main/main.tex` hay que
pasarla a BibTeX con sus DOI.

**Results lleva 4 figuras y el contrato pide 5–14.** Con las 4 tablas son 8
artefactos. Falta la figura de validación —las curvas reducidas contra el
control del mismo motor consigo mismo—, que tiene dato y no gráfico.

**La geometría es una sola.** El comportamiento de costo de la Figura 5 está
caracterizado en una única geometría, y así lo dice la sección de limitaciones.

## Fuentes del estado del arte

Resueltas con `paper_resolve.py` de papersmith y leídas del PDF, no de un
resumen. Los siete PDF se bajaron para verificar las cifras de la Tabla 5.

- Block, Virnau & Preis (2010), Ising 2D multi-spin en GPU — `arXiv:1007.3726`
- Weigel (2010), *Simulating spin models on GPU* — `arXiv:1006.3865`
- Weigel & Yavors'kii (2011), *GPU accelerated MC of lattice spin models* — `arXiv:1107.5463`
- Fang et al. (2013), Edwards-Anderson 3D, multi-spin asíncrono — `arXiv:1311.5582`
- Müller et al. (2019), *Spirit*, marco atomístico con CUDA — `arXiv:1901.11350`
- Charilaou (2021), Heisenberg anisótropo en GPU — `arXiv:2106.11286`
- Bisson et al. (2025), Ising/Blume-Capel a escala de rack — `arXiv:2502.18624`

El primero se citaba antes como «Preis et al.»; el resolutor da los autores
reales, **Block, Virnau & Preis**. Y Charilaou **sí** declara la GPU —un Tesla
V100—, lo dice el PDF aunque no el abstract; la Tabla 5 ya no pone «unstated».
