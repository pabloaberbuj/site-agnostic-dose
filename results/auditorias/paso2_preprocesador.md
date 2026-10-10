# Paso 2 — preprocesador nuevo (`src/preprocess/`)

> Detalle de cada decisión de armonización (con evidencia) en
> `paso2_decisiones_autonomas.md`. Este archivo es el resumen ejecutable.

## Qué se construyó

`src/preprocess/`: `geometry.py` (grilla RTDOSE, mundo→vóxel), `dicom_io.py` (carga
por `Modality`, nunca por nombre de archivo), `rasterize.py` (contorno→máscara,
regla par-impar para agujeros), `name_resolver.py` (cuatro salidas de CHARTER §5 +
reglas de duplicado numerado y crop-suffix sobre `exclude_names`), `dose_metrics.py`
(D95 + normalización), `pipeline.py` (orquesta todo por paciente, sin tocar disco).
`scripts/run_preprocess.py` corre el pipeline sobre un sitio completo y escribe NPZ
versionado (`npz_v1_paso2/<sitio>/<patient_id>.{npz,json}`, grilla nativa sin
padding) + `_preprocess_report.json`.

## Tests (datos sintéticos, `src/preprocess/test_preprocess.py`)

6/6 OK: resolver de nombres (4 salidas + duplicado numerado + crop-suffix),
fallback de PTV sin `-04`, round-trip D95 normalizado↔Gy, anti-leakage (intención
no depende de la dosis), volumen de máscara vs. contorno (escala de vóxel, rel.
diff < 15%), determinismo de rasterización.

**Sensibilidad del método de rasterización** (medida en 1 paciente por sitio,
nearest-pixel vs. supersample 4×): volumen ~+5-8% con nearest-pixel (inclusión de
borde en `cv2.fillPoly`), D95 ~1.3-1.5% más bajo. Consistente entre sitios, no
aleatorio — queda documentado, no bloquea.

## Resultado sobre los 503 pacientes reales

| sitio | OK | fallidos | total | % OK |
|---|---|---|---|---|
| CyC | 115 | 81 | 196 | 58.7% |
| Pelvis | 105 | 56 | 161 | 65.2% |
| Prostata | 133 | 13 | 146 | 91.1% |
| **total** | **353** | **150** | **503** | **70.2%** |

Ninguno de los fallos es un error del pipeline: todos son `estructuras desconocidas`
(nombres que no matchean canónico/alias/ignore/exclude) o, en 2 casos, ambigüedad
real (dos contornos con volumen>0 para el mismo rol — el pipeline frena en vez de
elegir al azar). El pipeline corrigió además 2 bugs de robustez propios en el camino
(contornos de BODY fuera del rango Z de la grilla de dosis; más de un candidato
BODY por paciente) — ninguno de los dos es una decisión de armonización.

**24 decisiones de armonización** se tomaron sin confirmación en vivo (alias de
formato/castellano/abreviatura, `exclude_names` para subestructuras de PTV y pares
órgano-llano/PRV, patrones de `ignore` para estructuras de trabajo) — todas con
evidencia verificable, detalladas en `paso2_decisiones_autonomas.md`. Ahí también
está la lista completa de lo que NO se tocó porque requiere tu criterio clínico.

## Las 3 decisiones que más mueven el número final

1. **`Lobe_Temporal_L/R` + PRV en CyC (~55 de los 81 fallos de CyC).** ¿Es un OAR
   real del protocolo que falta en la tabla? Si se agrega como rol canónico (mismo
   patrón que Brainstem/SpinalCord/etc.), CyC pasaría de 58.7% a ~86% OK.
2. **Campo extendido en Pelvis — `Kidney_L/R`, `Liver`, `SpinalCanal`(`_PRV`),
   `Heart`, `Lung_*`, `Ovary_*`, etc. (~45-48 de los 56 fallos de Pelvis).** ¿Es una
   subpoblación clínica real (nodal extendido/para-aórtico) que necesita sus propios
   OAR modelados, o son casos a marcar `out_of_convention`? Pelvis pasaría de 65.2%
   a ~95%+ con cualquiera de las dos decisiones (modelar vs. excluir).
3. **`PTV_Mid01`/`PTV_Mid02` en CyC (~20 fallos).** Paciente con más niveles de PTV
   que el protocolo declarado — es el caso que CHARTER pide frenar explícitamente.
   ¿Se unen en una sola máscara, se reservan como `out_of_convention`, o es otra
   cosa?

Con esas 3 resueltas, el pool entero quedaría arriba de ~90% OK en los tres sitios
combinados — el resto es cola larga de casos únicos (ver `paso2_decisiones_autonomas.md`),
ninguno sistemático.
