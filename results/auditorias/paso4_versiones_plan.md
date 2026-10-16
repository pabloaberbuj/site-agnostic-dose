# Paso 4 (0.2) — Conteo de versiones de plan por paciente (ARIA)

> Script: `scripts/audit_plan_versions.py`, sobre `metadata_planes_<Sitio>.csv` del
> extractor. `NNoAprobados` ya viene filtrado por tratamiento (mismo curso, misma Rx,
> mismo structure set, mismo `TargetVolumeID`). Vacío = no calculado, NO es 0. Detalle
> en `paso4_versiones_plan.csv`.

## Distribución de `NNoAprobados` por sitio

| sitio | N | sin calcular | 0 | 1 | 2 | 3+ | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| CyC | 196 | 0 | 28.6% | 27.0% | 22.4% | 22.0% | 1 | 4 | 7 | 11 |
| Pelvis | 161 | 1 | 55.6% | 25.0% | 13.1% | 6.3% | 0 | 2 | 5 | 6 |
| Prostata | 146 | 0 | 52.7% | 21.9% | 15.8% | 9.6% | 0 | 2 | 6 | 7 |

**1 "sin calcular" en Pelvis** (`PT_abec38677519da80`) — ya documentado en
`patient_exceptions.yaml` (`sin_TargetVolumeID_PTV_nombre_PTV`): plan sin
`TargetVolumeID`, el conteo de versiones no se puede calcular por eso. Esperado, sin
novedad.

**3 pacientes con conteo "cota inferior"** (TargetVolumeID mal asignado, ver
`patient_exceptions.yaml`, motivo `*cota_inferior_versiones`): `PT_348d801fd34b2a81`
(CyC, NNoAprobados=1), `PT_2801fef819714530` (Prostata, NNoAprobados=1),
`PT_5787d1d3258de95d` (Prostata, NNoAprobados=0) — el valor real podría ser mayor.

## Análisis de concentración (¿vale un criterio tipo Pareto?)

| sitio | % pacientes que acumulan el 80% del total de `NNoAprobados` |
|---|---|
| CyC | 41.3% (81/196) |
| Pelvis | 29.4% (47/160) |
| Prostata | 28.8% (42/146) |

**No hay concentración tipo Pareto real** (80/20): haría falta excluir 29-41% de los
pacientes para capturar el 80% de las revisiones de plan — muy lejos del 20/80 clásico.
La distribución es suave y unimodal (la mayoría en 0-2, cola larga sin salto brusco),
consistente con iteración clínica normal de planificación (ajustes antes de aprobar),
no con un subgrupo claramente atípico.

## Recomendación: **NO** aplicar un criterio de exclusión basado en `NNoAprobados`

- No hay un punto de corte natural en la distribución (sin bimodalidad ni outliers
  aislados del resto — hasta el máximo de cada sitio, CyC=11, Pelvis=6, Prostata=7,
  es solo la cola de una distribución continua).
- La métrica mide iteración de planificación, no necesariamente calidad del plan final
  tratado (que es el que se exporta y procesa en Paso 2).
- Si se quiere usar como señal complementaria más adelante (ej. ponderar o marcar para
  revisión manual puntual los casos con `NNoAprobados` muy alto), los candidatos son
  los de cola larga: CyC `NNoAprobados` ∈ {9, 11} (2 pacientes), Prostata {7} (2
  pacientes) — pero no se marcan `out_of_convention` solo por esto.

**Confirmado por el usuario (2026-10-16): no se excluye ningún paciente por
`NNoAprobados` en esta etapa.** Un eventual criterio tipo Pareto es de una etapa
posterior y probablemente requiera otros tests y otra selección de pacientes — Paso 4
es solo para contabilizar y dejar el número como insumo hacia adelante, no para decidir
exclusiones ahora.

## Paso 4c: cerrado
