# Sección 0 escrita para la tesis

Tres secciones redactadas contra los contratos de papersmith
(`~/papersmith-ai/sections/`). Están en **inglés** porque el paper de
`main/main.tex` lo está (`elsarticle`); la conversación y las páginas
interactivas son otra cosa.

| archivo | contrato | prosa | estado |
|---|---|---|---|
| `01-materials-and-methods.md` | `sections/01` | 1595 palabras (560–3950) | ✅ dentro |
| `03-results-and-discussion.md` | `sections/03` | 1793 palabras (1170–4150) | ✅ dentro |
| `05-related-work.md` | `sections/05` | 1170 palabras (820–1700) | ✅ dentro |

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

## Lo que FALTA, y no está disimulado

**1. Las citas no están resueltas.** El contrato de Related Work pide
`citations: resolution`, o sea claves resueltas por OpenAlex/Crossref con su
DOI. La sección nombra autores y años en prosa pero **no lleva claves de cita**.
Hay que pasarla por el resolutor de papersmith antes de que sirva para entregar.

**2. Las figuras no están dibujadas.**

| figura | qué es | estado |
|---|---|---|
| Figura 1 | el diagrama de resumen del kernel | **por dibujar** — su caption ya está escrito y enumera los cuatro componentes en orden |
| Figura 2 | costo por réplica contra el lote, los dos motores | el dato existe (`resultados/curvas/triton_halo.json`) |
| Figura 3 | **la insignia**: 100 estados, CPU + T4 + H100 | el SVG existe: `resultados/curvas/chart_100estados_concpu.svg` |
| Figura 4 | la misma sin CPU, eje lineal | `chart_100estados_sincpu.svg` |

**3. Results lleva 3 figuras y el contrato pide 5–14.** Con las 3 tablas son 6
artefactos, pero está por debajo. Faltan las figuras de validación (las curvas
reducidas contra el control) y la del piso de lote por dispositivo, que tienen
dato y no gráfico.

**4. Un dato del estado del arte está incompleto.** El trabajo más comparable
—Heisenberg anisótropo en GPU, 1.55 ns por volteo— **no declara en qué GPU** se
midió, y la página del abstract tampoco. La Tabla 3 lo dice («unstated GPU») en
vez de rellenarlo. Para la versión final hay que sacarlo del PDF.

**5. La comparación con el estado del arte es honesta, no favorable.** La Tabla 3
incluye las implementaciones de Ising que son **hasta cuatro órdenes de magnitud
más rápidas por volteo** que la nuestra, y dice por qué esa comparación no mide
nada: un volteo de Ising con empaquetado de bits son unas pocas operaciones
enteras, y uno nuestro son ~290 operaciones aritméticas, una exponencial, una
raíz inversa y cuatro sorteos. Lo que sí se sostiene es más estrecho y está
dicho así: alcanzamos el ritmo por sitio de un Ising multi-spin de 2010 cargando
un Hamiltoniano dos órdenes de magnitud más pesado, y le ganamos 12× al trabajo
publicado con el Hamiltoniano más parecido.

## Fuentes del estado del arte

Localizadas por búsqueda; los enlaces quedan para el resolutor de citas.

- Preis et al. (2010), Ising 2D multi-spin en GPU — `arXiv:1007.3726`
- Weigel (2011), *Simulating spin models on GPU* — `arXiv:1006.3865`
- Weigel & Yavors'kii (2011), *GPU accelerated MC of lattice spin models* — `arXiv:1107.5463`
- Charilaou (2021), Heisenberg anisótropo en GPU — `arXiv:2106.11286`
- Spirit (2019), marco atomístico con CUDA — `arXiv:1901.11350`
- Ising/Blume-Capel a escala de rack (2025) — `arXiv:2502.18624`
- Edwards-Anderson 3D, multi-spin asíncrono — `arXiv:1311.5582`
