# Sección 3 · El ciclo completo

Cierra el lazo entre la sección 1 (DDPM: parámetros → imagen) y la sección 2
(Xception: imagen → parámetros), para evaluar **cuánta información física
sobrevive** a la vuelta completa.

```
θ ∈ ℝ⁸  →  DDPM  →  imagen 40×40  →  recorte 39×39  →  Xception  →  θ̂
```

Se mide por tres lentes complementarias: la **calidad de regresión** sobre los
parámetros recuperados, la **fidelidad de la imagen** a nivel de píxel, y tres
**observables físicos** de la configuración de espines.

> Documentación de fondo: `section3-cycle/docs/05_complete_cycle.md`

## Contenido

| carpeta | qué hay |
|---|---|
| `ciclo/` | el ciclo completo: `ciclo_completo_v2`, fidelidad de textura, el ciclo con métricas nuevas |
| `integracion/` | la línea de integración, en dos mitades: `training/` (DDPM con codificador congelado, ajuste conjunto por coseno, muestreador guiado por latentes) y `evaluation/` (ocho experimentos, del DDPM base al canal y la cabeza de marca de agua, con el mensaje interactivo) |
| `archivo/` | las versiones anteriores del ciclo, incluido el ciclo externo y su diagnóstico. Se conservan porque documentan el camino, no porque sigan vigentes |

## Nota

Este README describe lo que hay, no lo que se concluyó. El informe consolidado
de esta sección todavía no está escrito — en `section0-mcmc.gpu/informe/` está
el modelo a seguir.
