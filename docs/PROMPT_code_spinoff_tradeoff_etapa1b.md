# PROMPT Code — Spin-off trade-off, Etapa 1b: residual en la normalización original del plan

> Sigue a la Etapa 1 (`results/spinoff_tradeoff/etapa1/`). Mismo encuadre: spin-off exploratorio,
> sin GPU, sin entrenar, sin tocar configs, splits, armonización ni `contexto_fase_vigente.md`.
> Salidas a `results/spinoff_tradeoff/etapa1b/`; scripts en `scripts/spinoff_tradeoff/`.
> Reusar todo lo cacheado de la Etapa 1 (inferencia, DVHs, descriptores, métrica B).

## 0. Por qué

En la Etapa 1 la dosis real se renormalizó a D95(PTV_High)=100%. Eso borró la decisión de cobertura
del planificador, que vive en la normalización original del plan. El chequeo con la covariable lo
confirmó: el componente 1 de Próstata correlaciona r=0.69 con D95 absoluto / Rx, y el de CyC r=−0.94.
O sea, el eje principal era en buena parte el factor de renormalización.

La Etapa 1b repite el análisis en el espacio del plan tal como se entregó, para medir directamente el
compromiso cobertura contra OAR.

## 1. Cambio de espacio

- **Real:** dosis en % de la Rx_High **del protocolo**, sin renormalizar. Se obtiene reescalando la
  dosis normalizada del NPZ por `D95_abs_Gy(PTV_High) / Rx_High_Gy` de cada paciente (el valor del QA
  gate de Fase 0).
  - Sanity: recalcular D95(PTV_High) sobre la dosis reescalada y verificar que coincide con el QA gate.
    Listar los pacientes con discrepancia >0.5 pp y excluirlos si la causa no es evidente.
- **Predicho:** la predicción de A4 renormalizada a D95(PTV_High)=100%, como en la Etapa 1. Representa
  "el plan típico para esta anatomía con cobertura completa a la Rx".
- **Residual = real − predicho**, en ese espacio. Ahora ΔD95(PTV_High) es una variable libre: es la
  cobertura que el planificador resignó o agregó.

## 2. Vector de resúmenes (cambios respecto de la Etapa 1)

Igual que en la Etapa 1 (vector compacto: OARs elegibles por banda baja/media/alta, ΔDmean, ΔDmax
donde corresponda; cáscaras de caída; BODY), con estos cambios:

- **Bloque PTV:** agregar ΔD95 de PTV_High. Mid/Low siguen con ΔD95 respecto de su Rx.
- **Bloque hotspot:** en la Etapa 1, hotspot y PTV correlacionaban 0.92–0.94 porque el máximo del BODY
  cae dentro del PTV. Redefinir el hotspot **fuera de los PTV**: D0.035cm³ de (BODY − unión de PTVs) y
  volumen >107% fuera de los PTV. Reportar la correlación HOT~PTV resultante.

## 3. Análisis (por sitio, val+test, fuera de muestra)

Mismo pipeline que la Etapa 1 (sesgo y desvío, PCA, loadings, estabilidad bootstrap, réplica sin
modelo con k=3 y k=5 sobre la métrica B), con estos cambios:

1. **Prueba directa de compromiso (resultado principal).** Correlación de ΔD95(PTV_High) con:
   - ΔV banda media y ΔDmean de cada OAR elegible;
   - el score de calidad de cada bloque (OAR, hotspot, caída).
   Con nula por permutación, IC bootstrap y corrección por número de pruebas dentro del sitio. Signo en
   calidad: menos cobertura con OAR mejor = antagónico.
2. **Acoplamiento entre bloques con la nula correcta.** La nula de la Etapa 1 permutaba columnas una
   por una, rompiendo la correlación interna de cada bloque, y por eso inflaba el número de componentes.
   Nueva nula: **permutar pacientes por bloque entero** (cada bloque conserva su estructura interna y
   solo se rompe la relación entre bloques). Usarla tanto para el análisis paralelo como para el
   acoplamiento.
3. **Acoplamiento por componente, no solo por el primer componente de cada bloque.** Para cada
   componente retenido, reportar la carga orientada a calidad de cada bloque (suma de loadings con
   signo de calidad) y si mezcla bloques con signo opuesto. Motivo: en la Etapa 1 el compromiso
   recto/vejiga contra homogeneidad del componente 2 de Pelvis no apareció en la prueba por bloques.
4. **¿La concesión de cobertura es anatómica?** R² con validación cruzada dejando un paciente afuera de
   ΔD95(PTV_High) a partir de los descriptores de la métrica B (ridge, como en la Etapa 1), y
   correlación de ΔD95 de cada paciente con el promedio de sus vecinos. Acá conviene usar **todos los
   elegibles del sitio**, no solo val+test: ΔD95 real no depende del modelo (el predicho vale 100 por
   construcción). Reportar ambos (todos y val+test).
5. **Comparación con la Etapa 1.** Coseno entre los loadings de los componentes de ambas etapas (en las
   variables comunes), y si el eje de derrame de Pelvis sobrevive al cambio de espacio.

## 4. Lectura predeclarada (por sitio)

| Resultado | Lectura |
|---|---|
| ΔD95 antagónico con OARs (significativo) y R² anatómico de ΔD95 ≥0.2 | La concesión de cobertura es un compromiso real y en parte predecible desde la anatomía: candidato a salida extra del modelo principal o del modelo clásico |
| ΔD95 antagónico con OARs y R² anatómico <0.2 | Compromiso real pero de criterio del planificador: candidato a coordenada de preferencia |
| ΔD95 sin acoplamiento con OARs | La cobertura varía por política o costumbre, no como compromiso |
| Eje de derrame de Pelvis sobrevive (coseno >0.7 con la Etapa 1), sinérgico, sin acoplamiento con ΔD95 | Eje de calidad del plan independiente de la normalización |
| El componente principal sigue cargando casi solo en ΔD95 | Reportar los componentes siguientes como el resultado de interés |

## 5. Entregables

`resumen.md` corto, `summary.json`, y tablas: sanity del reescalado; prueba directa ΔD95 contra cada
OAR y bloque; autovalores contra la nula por bloques; loadings y carga por bloque de cada componente;
R² anatómico y correlación con vecinos de ΔD95; comparación con la Etapa 1. Gráficos: ΔD95 contra el
score OAR por sitio (dispersión), scree con la nula nueva.

Mismas guardas que la Etapa 1: sin métricas agregadas de desempeño sobre test, solo `patient_id`
anónimo, IC y n en todo, decisiones no previstas en una línea en el resumen.

## 6. Fuera de alcance

Predicciones out-of-fold, planes múltiples (Etapa 2), cualquier entrenamiento.

Traer al chat `resumen.md` y `summary.json`.
