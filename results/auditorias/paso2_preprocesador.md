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

**Antes de las decisiones del usuario (2026-10-14):**

| sitio | OK | fallidos | total | % OK |
|---|---|---|---|---|
| CyC | 115 | 81 | 196 | 58.7% |
| Pelvis | 105 | 56 | 161 | 65.2% |
| Prostata | 133 | 13 | 146 | 91.1% |
| **total** | **353** | **150** | **503** | **70.2%** |

**Después de la primera ronda de decisiones** (Lobe_Temporal canónico + 49 Pelvis
`out_of_convention` en bloque + excepciones tolerando estructuras desconocidas):

| sitio | OK | fallidos | total | % OK |
|---|---|---|---|---|
| CyC | 158 | 38 | 196 | 80.6% |
| Pelvis | 154 | 7 | 161 | 95.7% |
| Prostata | 134 | 12 | 146 | 91.8% |
| **total** | **446** | **57** | **503** | **88.7%** |

**Después de revisar Pelvis y CyC caso por caso** (SpinalCord/Ovary/Kidney/Liver-Heart-Lung
en Pelvis con el criterio de tamaño de campo para Kidney; Trachea/PRV-convención-propia/
PTV_C2-66-Mid_3mm en CyC):

| sitio | OK | fallidos | total | % OK |
|---|---|---|---|---|
| CyC | 175 | 21 | 196 | 89.3% |
| Pelvis | 147 | 14 | 161 | 91.3% |
| Prostata | 134 | 12 | 146 | 91.8% |
| **total** | **456** | **47** | **503** | **90.7%** |

**Después de la ronda final** (bug de Ovary corregido; PTV_Mid01/Mid02 resuelto con
volumen real; SpinalCord_PRV1/PRVx; RectumCorregido/Colostomia/Protesis/PTV_ProstateBed/
BowelLargePRV01/etc. en Próstata; Contraste/PTV_Low!High en Pelvis):

| sitio | OK | fallidos | total | % OK |
|---|---|---|---|---|
| CyC | 187 | 9 | 196 | 95.4% |
| Pelvis | 151 | 10 | 161 | 93.8% |
| Prostata | 145 | 1 | 146 | 99.3% |
| **total** | **483** | **20** | **503** | **96.0%** |

Ninguno de los fallos es un error del pipeline: todos son `estructuras desconocidas`
(nombres que no matchean canónico/alias/ignore/exclude) o ambigüedad real (dos
contornos con volumen>0 para el mismo rol — recontoreos, el pipeline frena en vez de
elegir al azar; desempate automático si solo uno de los candidatos dice "PRV" o
"corregido" en el nombre). El pipeline corrigió 5 bugs de robustez propios en el
camino (BODY fuera de rango Z; más de un candidato BODY; `out_of_convention` no
alcanzaba para generar el NPZ; Ovary mezclado por error con Liver/Heart/Lung;
`SpinalCord_PRV1` reinterpretado por la regla de duplicado numerado) — ninguno es
decisión de armonización.

**46 decisiones de armonización** registradas con evidencia en
`paso2_decisiones_autonomas.md`. Nuevo artefacto reusable:
`paso2_estructuras_por_paciente.csv` (inventario de nombres de estructura crudos de
los 503 pacientes, para no tener que rescanear DICOM en cada ronda nueva).

**Después de resolver los 20 pendientes + el caso `Oral_Cavity`/`Cavity_Oral` surgido al
resolverlos** (decisiones del usuario sobre `paso2_pendientes_20_pacientes.csv`, 2
casos especiales con anulación de nombre por paciente puntual —
`PT_c83d88c040c3a900` y `PT_0e31af2ee9f039e5`):

| sitio | OK | fallidos | total | % OK |
|---|---|---|---|---|
| CyC | 196 | 0 | 196 | 100% |
| Pelvis | 161 | 0 | 161 | 100% |
| Prostata | 146 | 0 | 146 | 100% |
| **total** | **503** | **0** | **503** | **100%** |

## Lo que queda pendiente

Ningún fallo de pipeline. Único ítem abierto, sin fallo asociado:
`BowelLarge` sin PRV en Próstata (¿mismo órgano que `Bowel_Small`/`Sigmoid`, o rol
nuevo? distinto de `BowelLargePRV01`, que sí se resolvió) — no bloquea nada, es solo
una pregunta de armonización para una futura ronda si se quiere.
