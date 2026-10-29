# Auditoría técnica del kernel Triton de Monte Carlo para nanodots magnéticos

## 1. Objetivo

Realizar una auditoría técnica, física, numérica y de rendimiento de la implementación optimizada de Monte Carlo desarrollada con Triton, comparándola con la implementación de referencia en JAX y con las afirmaciones del manuscrito:

**"Monte Carlo simulation of magnetic nanodots on graphics hardware: a single fused kernel for the extended Heisenberg chain".**

El objetivo es determinar:

1. Si la implementación Triton simula correctamente el Hamiltoniano y la dinámica de Monte Carlo previstos.
2. Si existe equivalencia numérica y estadística con la implementación de referencia.
3. Si el acceso a vecinos, el coloreado, las condiciones de frontera y la aleatoriedad son correctos.
4. Si los benchmarks justifican las aceleraciones reportadas.
5. Qué afirmaciones del manuscrito están respaldadas por el código y los experimentos, cuáles requieren evidencia adicional y cuáles deben corregirse.

Esta es una auditoría, no una tarea de refactorización.

## 2. Reglas de trabajo

- No modifiques, sobrescribas ni elimines archivos del proyecto.
- No cambies parámetros, semillas, kernels, notebooks ni configuraciones durante la inspección.
- No ejecutes simulaciones de producción costosas sin autorización explícita.
- No asumas que el código implementa lo que dice el README, los comentarios o el manuscrito: inspecciona las funciones realmente ejecutadas.
- No declares que una prueba pasó si no se ejecutó o si no existe evidencia verificable.
- Distingue entre:
  - **Verificado:** respaldado directamente por código o resultados reproducibles.
  - **Hallazgo confirmado:** se ha identificado una inconsistencia concreta.
  - **Riesgo:** existe una posibilidad que requiere una prueba adicional.
  - **No verificado:** no hay evidencia suficiente.
- Toda conclusión debe incluir evidencia: archivo, función, líneas de código, fórmula, prueba o resultado relevante.
- Si existen varias versiones del kernel o de los notebooks, identifica cuál es la versión activa y cuáles son las referencias históricas.
- No uses conocimientos generales para rellenar silenciosamente información ausente. Declara las incertidumbres.
- No implementes correcciones automáticamente. Primero entrega los hallazgos y las recomendaciones.

## 3. Inventario del proyecto

Identifica los archivos relevantes para:

- Implementación de referencia en JAX.
- Kernel Triton optimizado.
- Funciones de energía y variación de energía local.
- Generación de propuestas aleatorias y aceptación de Metropolis.
- Actualización de subredes o clases de color.
- Construcción de geometrías, máscaras y condiciones de frontera.
- Cálculo de observables.
- Pruebas de validación numérica y estadística.
- Benchmarks de CPU y GPU.
- Scripts de producción y configuración de parámetros.
- Resultados experimentales utilizados en el manuscrito.

Busca las llamadas reales a las funciones y determina qué código se ejecuta en cada benchmark. No basta con encontrar funciones declaradas que no se utilizan.

Entrega una tabla con:

| Componente | Archivo y función | Implementación activa | Evidencia |
|---|---|---|---|

## 4. Auditoría del Hamiltoniano

Reconstruye el Hamiltoniano efectivo a partir del código ejecutado y compáralo término por término con el Hamiltoniano definido en el manuscrito.

Inspecciona:

- Intercambio de primeros vecinos \(J_1\).
- Intercambio de segundos vecinos \(J_2\).
- Intercambio de terceros vecinos \(J_3\).
- Intercambio de cuartos vecinos \(J_4\).
- Interacción Dzyaloshinskii–Moriya (DMI).
- Anisotropía magnetocristalina.
- Anisotropía superficial, si corresponde.
- Campo externo y su orientación.
- Restricción de norma de los espines.
- Constantes físicas, signos, factores de doble conteo y unidades.

Para cada término, indica:

1. La expresión matemática esperada según el manuscrito.
2. La expresión implementada en JAX.
3. La expresión implementada en Triton.
4. Si ambas expresiones son equivalentes.
5. Si el término se encuentra realmente activo en los experimentos de producción.
6. Qué prueba permite confirmarlo.

No supongas que \(J_2\), \(J_3\) y \(J_4\) están implementados solo porque aparecen en la definición de los parámetros.

Presta especial atención a las diferencias entre energía total, energía local y variación de energía \(\Delta E\). Comprueba que el criterio de aceptación utiliza la misma convención energética en ambas implementaciones.

**Resultado obligatorio:** una tabla de equivalencia del Hamiltoniano y una conclusión explícita sobre el alcance físico que puede afirmarse para el kernel actual.

## 5. Auditoría del algoritmo de Metropolis

Comprueba la implementación real del algoritmo y verifica:

- Distribución de las propuestas de nuevos espines.
- Conservación de la norma de los espines.
- Cálculo de \(\Delta E\).
- Criterio de aceptación \(p=\min(1,\exp(-\beta\Delta E))\).
- Definición de \(\beta\) y uso de \(k_B\).
- Tratamiento de temperaturas bajas y de \(T=0\).
- Número de actualizaciones por sweep.
- Orden de actualización de las clases de color.
- Uso de configuraciones anteriores y posteriores al subpaso.
- Conservación del estado entre temperaturas de la escalera.
- Máscara de sitios activos y tratamiento de sitios inactivos.

Verifica que el algoritmo implementado corresponda al descrito en el manuscrito. Identifica cualquier diferencia que pueda alterar la cadena de Markov o la distribución estacionaria.

## 6. Auditoría del coloreado y las condiciones de frontera

El kernel utiliza actualizaciones por clases de color para evitar sincronizaciones globales entre espines que interactúan.

Comprueba:

- La definición matemática de cada clase de color.
- Los desplazamientos de interacción para todos los términos activos del Hamiltoniano.
- Que ningún par de espines que interactúa pueda pertenecer a la misma clase cuando ambos se actualizan en el mismo subpaso.
- Que el coloreado siga siendo válido con las fronteras periódicas en \(x,y\).
- Que las fronteras abiertas en \(z\) no introduzcan vecinos inexistentes.
- Que los accesos desplazados y el halo lean índices válidos.
- Que las cargas de memoria no lean datos de otra réplica.
- Que la máscara circular del nanodot se aplique correctamente.
- Que las dimensiones impares o pares de la caja no generen errores de indexación.

No aceptes como demostración suficiente una explicación en comentarios. Contrasta las reglas de coloreado con los desplazamientos realmente usados.

Cuando sea viable, diseña pruebas pequeñas y deterministas que verifiquen los vecinos y las clases de color sin ejecutar una simulación de producción.

## 7. Auditoría del acceso a vecinos y la fusión del kernel

Compara la ruta computacional de JAX con la ruta Triton.

En JAX, inspecciona las operaciones `jnp.roll`, las funciones compiladas con `jit`, los bucles de `lax` y las operaciones de distribución entre dispositivos.

En Triton, inspecciona:

- Distribución de espines entre programas y bloques.
- Cargas de memoria para los vecinos.
- Cálculo de los índices.
- Uso de registros y valores temporales.
- Máscaras de carga y escritura.
- Escritura de la configuración aceptada.
- Número de lanzamientos de kernels por sweep.
- Operaciones que se fusionan y operaciones que permanecen fuera del kernel.
- Reducciones de observables y transferencias de datos.

Determina si la eliminación de los desplazamientos explícitos reduce realmente el trabajo o el tráfico de memoria.

No afirmes que cada `jnp.roll` genera necesariamente una copia materializada: considera las optimizaciones que puede realizar XLA.

Separa las ventajas demostradas de las ventajas teóricas. No atribuyas una mejora de rendimiento a un componente concreto sin mediciones o un estudio de ablación.

## 8. Auditoría del generador aleatorio

Inspecciona el generador aleatorio de ambas implementaciones.

Para Triton, identifica:

- La clave inicial.
- La construcción de los contadores.
- El índice global de cada espín.
- El índice de réplica.
- El paso de Monte Carlo.
- La clase de color.
- Los contadores usados para generar las componentes de la propuesta y la variable de aceptación.
- La posibilidad de colisiones entre contadores.
- La conversión de los números generados a las distribuciones requeridas.
- La reproducibilidad entre ejecuciones.

Comprueba si las réplicas utilizan secuencias diferenciadas y si existe evidencia suficiente para afirmar independencia estadística. No confundas contadores diferentes con una prueba matemática de independencia.

Verifica que el generador produce propuestas con la distribución correcta sobre la esfera y que la variable de aceptación sigue la distribución uniforme esperada.

## 9. Auditoría de observables y precisión numérica

Revisa las implementaciones de:

- Energía.
- Magnetización vectorial.
- Magnitud de la magnetización.
- Componente \(M_z\).
- Calor específico \(C_v\).
- Susceptibilidad magnética \(\chi\).
- Carga topológica \(Q\), si está activa.

Para cada observable, documenta la fórmula, las unidades, la normalización y la implementación en ambas versiones.

Verifica específicamente:

- Número de espines activos utilizado en las normalizaciones.
- Diferencia entre el número de sitios de la caja y el número de sitios del nanodot.
- Factores de \(N\), \(k_B\) y \(T\) en \(C_v\) y \(\chi\).
- Si la susceptibilidad utiliza la varianza de \(|M|\), de \(M_z\) o de otra magnitud.
- Si las definiciones de ambas implementaciones coinciden.
- Si los momentos centrados se calculan correctamente.
- Si las sumas parciales y reducciones son deterministas para una configuración fija.
- Si el uso de FP32 genera errores relevantes.
- Si las fórmulas se evalúan correctamente a temperaturas muy bajas.

Distingue entre determinismo de una reducción numérica y reproducibilidad de toda la trayectoria estocástica.

## 10. Auditoría de validación numérica y estadística

Localiza todas las pruebas existentes y determina qué afirmación respalda cada una.

Revisa, cuando estén presentes:

- Comparación con aleatoriedad idéntica.
- Comparación de configuraciones completas de espines.
- Comparación de energías locales y totales.
- Comparación de observables sobre una configuración común.
- Comparación entre cadenas independientes.
- Tasas de aceptación.
- Distribuciones de magnetización y energía.
- Evolución térmica de \(C_v\) y \(\chi\).
- Dependencia respecto a la temperatura y los parámetros del Hamiltoniano.

Para cada prueba, registra:

| Prueba | Condiciones | Métrica | Resultado observado | Criterio de aceptación | Limitaciones |
|---|---|---|---|---|---|

Comprueba si las tolerancias se definieron antes de interpretar los resultados o si se eligieron después de observarlos.

Cuando haya cadenas independientes, revisa si los errores estándar consideran la autocorrelación temporal. No trates los sweeps correlacionados como observaciones independientes.

Si los datos disponibles no permiten demostrar equivalencia estadística, propón pruebas adicionales concretas. No inventes resultados ni afirmes que una distribución es equivalente basándote solamente en una gráfica visual.

## 11. Auditoría del protocolo de rendimiento

Reconstruye cómo se obtuvieron los tiempos reportados en el manuscrito.

Los resultados descritos incluyen un benchmark de producción con:

- 100 estados.
- 5.000 réplicas por estado.
- 200 temperaturas.
- 20.000 sweeps por temperatura.

También se reportan comparaciones de CPU, T4 y H100 con diferentes cantidades de dispositivos.

Verifica estos valores en los scripts y configuraciones reales; no los des por confirmados solo porque aparezcan en el manuscrito.

Inspecciona:

- Hardware y número de dispositivos.
- Número de hilos de CPU.
- Versiones de Python, JAX, Triton, CUDA y controladores.
- Precisión numérica.
- Geometría y número de espines activos.
- Cantidad exacta de actualizaciones.
- Tamaño de lote y réplicas por dispositivo.
- Número de lanzamientos de kernels.
- Tiempo de compilación.
- Tiempo de inicialización y transferencias.
- Sincronización del dispositivo antes de detener el cronómetro.
- Calentamiento previo.
- Número de repeticiones.
- Variabilidad de los tiempos.
- Cálculo del tiempo por actualización.
- Método utilizado para extrapolar los tiempos de producción.

Distingue explícitamente entre:

1. Tiempo medido de una ejecución completa.
2. Tiempo medido de una ejecución parcial.
3. Estimación extrapolada a producción.

Recalcula las aceleraciones usando los tiempos disponibles y comprueba que los denominadores representen el mismo trabajo computacional.

No presentes estimaciones como mediciones directas.

## 12. Auditoría de escalabilidad y tamaño de lote

Evalúa si las conclusiones sobre réplicas por GPU y escalabilidad están respaldadas por los experimentos.

Revisa:

- Tiempo por actualización según el número de réplicas.
- Rendimiento con una, dos y cuatro GPU cuando existan datos.
- Diferencias entre T4 y H100.
- Distribución de estados y réplicas entre dispositivos.
- Sobrecoste de los lanzamientos de kernels.
- Coste de transferencias y sincronizaciones.
- Utilización y ocupación de la GPU, si hay métricas disponibles.
- Dependencia del rendimiento con la geometría y el número de sitios activos.

Comprueba la hipótesis sobre \(v_2(B)+v_2(L)\), donde \(v_2(n)\) es el exponente de 2 en la factorización de \(n\). Verifica la fórmula, los tamaños utilizados y la magnitud de las diferencias observadas.

Si el efecto es pequeño o inconsistente, indícalo. No conviertas una correlación experimental en una regla general sin suficiente evidencia.

## 13. Reproducibilidad y alcance de las conclusiones

Comprueba si el repositorio permite reproducir los resultados publicados.

Identifica si están disponibles:

- Código de referencia.
- Código Triton.
- Configuraciones de los experimentos.
- Semillas o claves iniciales.
- Parámetros del Hamiltoniano.
- Geometría.
- Scripts de benchmark.
- Datos o logs de las ejecuciones.
- Versiones del software y del hardware.
- Instrucciones de ejecución.

Evalúa si las conclusiones se limitan correctamente a las geometrías, interacciones y arquitecturas realmente probadas.

No generalices el rendimiento a otras geometrías ni a otros términos del Hamiltoniano que no hayan sido evaluados.

## 14. Contraste con el manuscrito

Busca las afirmaciones técnicas y numéricas del manuscrito y vincula cada una con su evidencia en el código o en los resultados.

Presta especial atención a:

- "single fused kernel".
- "synchronisation-free halo".
- "counter-based proposals".
- "deterministic observables".
- Equivalencia numérica y estadística.
- Aceleración por actualización.
- Resultados de producción.
- Escalabilidad entre dispositivos.
- Limitaciones de la implementación.

Para cada afirmación, clasifícala como:

- Respaldada directamente.
- Respaldada parcialmente.
- Requiere experimento adicional.
- Debe matizarse.
- Contradicha por el código o los resultados disponibles.

## 15. Entregable final

Genera un informe Markdown llamado `AUDITORIA_TECNICA_MONTE_CARLO.md`, sin modificar el código original.

El informe debe contener:

1. Resumen ejecutivo.
2. Inventario de archivos y versiones.
3. Tabla de equivalencia del Hamiltoniano.
4. Auditoría del algoritmo y del coloreado.
5. Auditoría del generador aleatorio.
6. Auditoría de observables y precisión.
7. Resultados de validación existentes.
8. Auditoría de benchmarks y escalabilidad.
9. Tabla de afirmaciones del manuscrito frente a la evidencia.
10. Hallazgos clasificados por severidad.
11. Pruebas adicionales recomendadas.
12. Plan de corrección ordenado por prioridad.
13. Limitaciones que deben declararse en el artículo.

Usa las siguientes categorías de severidad:

- **Crítica:** puede invalidar el Hamiltoniano, la dinámica o la comparación principal.
- **Alta:** puede invalidar una afirmación de equivalencia o rendimiento.
- **Media:** afecta la reproducibilidad, la interpretación o el alcance de las conclusiones.
- **Baja:** mejora de documentación o presentación.

Para cada hallazgo, incluye:

- Identificador.
- Severidad.
- Descripción.
- Archivo y función afectados.
- Evidencia concreta.
- Consecuencia científica o computacional.
- Prueba necesaria para confirmarlo, si todavía es una hipótesis.
- Recomendación de corrección.

Termina con una lista de las cinco acciones más importantes antes de enviar el artículo.

## 16. Criterio final de evaluación

Al terminar, responde explícitamente:

1. ¿El kernel Triton implementa el Hamiltoniano que declara el manuscrito?
2. ¿La dinámica de Metropolis y el coloreado son correctos para las interacciones activas?
3. ¿La evidencia disponible respalda la equivalencia numérica y estadística?
4. ¿Los benchmarks comparan el mismo trabajo y justifican las aceleraciones reportadas?
5. ¿Qué afirmaciones del artículo deben corregirse, matizarse o demostrarse mejor?
6. ¿Cuál es el principal riesgo científico que todavía queda abierto?

No declares la implementación correcta o incorrecta de forma global si la evidencia solo permite responder para determinados parámetros, geometrías o pruebas.

**La prioridad es establecer la corrección física y estadística antes de optimizar más el rendimiento.**