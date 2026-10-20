# Fase 0 — CERRADA. Infraestructura y datos

> **Tier 3 — Archivo de fase congelado.** No se edita; lo que cambie va a fases posteriores.
> Cerrada: 2026-10-16. Leer con `CHARTER.md` (Tier 1) y `aprendizajes_transversales.md`.
> Prompts ejecutables: `docs/PROMPT_code_fase0.md`, `docs/PROMPT_python_pendientes_armonizacion.md`,
> `docs/SPEC_profiling_0.0.md`. Prompts del extractor: `PROMPT_code_ajuste_extractor.md`,
> `ADDENDUM_code_extractor_export1.md`.

---

## Resultado

**Infraestructura de datos montada, auditada y limpia. No se entrenó nada** (era el objetivo de la
fase). 503/503 pacientes procesados al 100%, 0 errores de lectura, 0 sin BODY. Producto: envelope de
la GPU, preprocesador nuevo (no fork), tabla de armonización, NPZ versionado, auditorías de dosis y de
escala, baseline no-aprendida, y splits. Datos: DICOM anonimizados y NPZ en `SiteAgnostic\data\`
(fuera de git, CHARTER §6); NPZ en `npz_v1_paso2/<sitio>/`.

## Cohorte final

| Sitio | Exportados | `out_of_convention` | Elegibles | Train (split_v1) |
|---|---|---|---|---|
| CyC | 196 | 40 | **156** | 101 |
| Pelvis | 161 | 23 | 138 | 90 |
| Próstata | 146 | 7 | 139 | 90 |
| **Total** | **503** | **70** | **433** | **281** |

- **Excluidos del minado** (nunca exportados, 4): reirradiación CyC (1), monorreno con trasplante
  renal Pelvis (1), Próstata (2: remarcación con dos PTV conviven; replanificado con margen menor).
- **Los 70 `out_of_convention` NO entran al train; quedan como conjunto reservado** para evaluar
  generalización (zero-shot / hold-out). Todos revisados a mano en Eclipse.
- **CyC, 40 `out_of_convention` por motivo:** más niveles de PTV que el protocolo (PTV_Mid01/Mid02
  con volumen real) 15; protocolo de 1 nivel real sin PTV_Mid (7 ausente + 3 volumen=0) 10;
  extensión mediastínica/orbitaria real (Trachea/EyeBack/EyeFront) 9; convención de nombres propia de
  un paciente puntual (PTV_C2, PTV66, PTV_Mid_3mm) 3; QA gate D95 fuera de banda (2 PTV_Mid, 1 PTV_Low) 3.
- **Split:** `data/splits/split_v1.json`, seed 42, 65/15/20 estratificado por sitio. El piso de 100
  pacientes/sitio es sobre la **cohorte total**, no sobre train (CHARTER §2).
- Fuente de verdad de excepciones por paciente: `data/harmonization/patient_exceptions.yaml`
  (`excluded` / `out_of_convention` / `reviewed_ok`). Solo `AnonID` + motivo (va a git).

## 0.0 Profiling (RTX A2000 12.88 GB, usable 11.75, headroom exigido 1.0)

Envelope en `results/profiling/envelope.md` (corrió con tensores sintéticos; no depende del preprocesado).

| Config | Resultado |
|---|---|
| 2D, plano 256×256, batch | 64 (tope del barrido) entra con holgura: la memoria no limita el batch en 2D |
| U-Net 3D vanilla, 176×128×256, `base_feat=16`, batch 1 | entra **sin parches**: 8.56 GB reserved sin ckpt; 5.95 GB con ckpt (+38% tiempo) |
| U-Net 3D `base_feat=32` | OOM aun con ckpt |
| MedNeXt-k5 (stand-in) | sin ckpt OOM; con ckpt entra (9.47 GB, headroom 2.28) |

El MedNeXt medido es un **stand-in provisorio**, no la implementación de Fase 3. No se downsamplea
la grilla de dosis. **Se re-perfila en Fase 3 sobre los casos reales más grandes** (p.ej.
`PT_85558912476286d3`, 8.03 M de vóxeles tras recortar al BODY), no sobre una caja padeada.

## 0.1 Preprocesador y armonización

- **Preprocesador** `src/preprocess/`: site-paramétrico (la diferencia entre sitios vive en YAML, no
  en código duplicado). Resolución de nombres de 4 salidas (canónico / alias / ignore / desconocido;
  nunca adivina). Regla de normalización de nombres que preserva solo la marca de crop `04`.
- **Tabla de armonización** `data/harmonization/`: `harmonization_schema.yaml`, `sites_<sitio>.yaml`,
  `patient_exceptions.yaml`. Generada por `scripts/convert_constraints.py` y `build_harmonization.py`
  desde las plantillas in-house. 46+ decisiones de armonización con evidencia en
  `results/auditorias/paso2_decisiones_autonomas.md`.
- **Inventario de headers (Paso 1)**, `results/auditorias/paso1_headers.md`: spacing in-plane 2.5×2.5
  en los 503; Z 3.0 mm en 493 y **2.0 mm en 10** (CyC 9, Próstata 1), que **entran al train** con
  `spacing_z_mm` en la metadata. `DoseGridScaling` 4.87e-5–7.21e-5, sin outliers.
- **La grilla de dosis coincide con el bounding box del BODY** (481 de 503 dentro de ±3%), no con el
  campo de tratamiento. Consecuencia: un padding fijo global lo dictaría una cola de pocos pacientes
  (59/503 exceden 176×128×256 por eje, pero tras recortar al BODY 495/503 caben en 5.77 M de vóxeles).
  **Decisión:** el NPZ guarda la grilla nativa sin recortar ni rellenar, con el bounding box del BODY en
  la metadata. En 2D, plano fijo 256×256 (máximo nativo 248×242). El esquema 3D se define en Fase 3.
- 4 pacientes de grilla ~600×600 mm (3 Pelvis, 1 Próstata) **no** son `out_of_convention`: su BODY es
  normal; la grilla parece la de toda la imagen.

## 0.2 Auditorías y baseline

- **QA gate** (D95 por nivel vs Rx del protocolo, sobre dosis absoluta cruda): banda **por sitio**,
  Próstata [92%, 105%], CyC y Pelvis [95%, 105%]. El piso de 92% en próstata cubre un cluster revisado
  caso a caso en Eclipse (subdosado por OAR). **0 flags sin revisar** en los 503.
  `results/auditorias/paso3_qa_gate.md`.
- **Contaminación sobre la dosis** (extensión Z con dosis ≥X% Rx_High): confirma sin usar nombres de
  estructura los casos de campo extendido (retroperitoneo en Pelvis) que el criterio de tamaño de campo
  del RTPLAN ya había señalado, y encontró 1 caso adicional. `paso4_contaminacion_dosis.md`.
- **Escala de vóxel sobre los NPZ:** sin discrepancias sin explicar vs. Paso 1. Hallazgo: el volumen de
  PTV por contorno (Paso 1) sobreestimaba hasta 48% en estructuras con huecos; el validado es el
  rasterizado. `paso4_escala_voxel_npz.md`.
- **Versiones de plan (ARIA):** sin concentración tipo Pareto. **Decisión del usuario (2026-10-16):** no
  se excluye ningún paciente por número de planes no aprobados; se contabiliza como insumo para una
  etapa posterior. La pregunta del Pareto-conditioning (CHARTER §10) **sigue abierta**.
  `paso4_versiones_plan.md`.
- **Baseline no-aprendida** (DVH promedio por sitio y rol de estructura, solo con pacientes de train):
  `results/baseline/baseline_dvh_<sitio>.json` + `paso5_baseline.md`.

## Decisiones congeladas en la fase (viven en CHARTER §5 salvo indicación)

| Decisión | Detalle |
|---|---|
| Normalización | D95(PTV_High)=100%, recomputada desde la dosis cruda en Gy, sobre volumen nativo |
| Rx de intención | Del protocolo (externa a DICOM/Eclipse); ratios derivados de `rx_gy`, nunca de la dosis |
| PRV | No se ignoran: en seriales el PRV es la geometría |
| Resolución de PTV | Primer candidato de `geometry_candidates` con volumen ≠ 0 (CyC: `PTV_X-04`, luego `PTV_X`) |
| Fuera de la intención | `CTV_*` y estructuras de resta (son objetivos de optimización) |
| D0.035cm³ | ≡ Dmax. D1cm³ y D2% siguen abiertos (CHARTER §10) |
| NPZ | Solo geometría; los canales de goals se construyen al cargar desde la tabla (hash de la tabla en cada experimento) |
| Red | Sobre la grilla de dosis nativa, sin resamplear al CT |
| `out_of_convention` | Dos casos: nombre fuera de TG-263 → alias y entra; dosis o estructura atípica → reservado |
| Goals de OAR | Tres canales por tipo (máx / media / volumétrico), sin pseudo-DVH |
| Pareto | Pausado con insumo contabilizado (no se usa como criterio de exclusión) |

## Extractor (DicomExtract), ajustes de esta fase

- El CSV de metadata es **auditoría y gestión del dataset, no input del modelo**: sin HC (solo
  `AnonID`; el mapeo HC↔AnonID vive junto a los DICOM, fuera de git), con `Sitio`, `PTV_Resuelto`,
  `FlagRevision`, y conteos de versiones. La maquinaria de métricas/normalización/constraints quedó
  detrás de un flag desactivado (reactivable para otros proyectos).
- **Conteo de no-aprobados** con cadena de filtros: mismo curso → solo `UnApproved`/`Rejected` → misma
  prescripción → mismo structure set por UID → mismo `TargetVolumeID`. Criterio deliberado: mejor perder
  un plan que contaminar con otra localización. Vacío = "no calculado" (plan sin `TargetVolumeID`), no 0.
- Duplicados por paciente: se conserva el plan más antiguo, con reporte de lo descartado.

## Hallazgos curados

En `aprendizajes_transversales.md`, sección "Fase 0": resolución de nombres con desempate explícito;
reglas genéricas que se comen una excepción puntual; verificar antes de aplicar; inventario de nombres
una sola vez; auditorías independientes que se validan entre sí; aproximaciones "no validadas" que hay
que validar. Más: cobertura de PTV por debajo de banda ≠ contaminación (cavum, próstata hipo); y el piso
de 100 pacientes es sobre la cohorte total.

## Dependencias y deuda que pasa a Fase 1+ (no bloquearon el cierre)

- **Alias duplicados:** la lista de alias de PTV existe en dos copias (extractor y repo Python). Los
  alias `ptv_intermediate`→mid y `ptv_high1`→high están solo en Python: replicarlos en el extractor si
  se quiere que resuelva igual. La columna `PTV_Resuelto` contra lo que resuelve Python detecta divergencias.
- **`BowelLarge` sin PRV** en Próstata: sin decisión de armonización (no falla, queda sin rol).
- **Extractor, próximo minado:** remapeo consistente de UIDs y limpieza de tags privados/texto libre
  (datos locales, prioridad media). `PENDIENTES_extractor.md`. Pendiente también validar la cadena de
  no-aprobados con un paciente de tratamiento simultáneo en otra localización.
- **Punto ciego del audit de contaminación:** la reirradiación en otro centro no deja rastro en ARIA ni
  en el DICOM. Mitigación aceptada: revisión manual de lo que caiga fuera de banda en el QA gate.
- **Versionado de splits:** si la tabla de armonización mueve pacientes entre elegible y
  `out_of_convention`, decidir entre regenerar `split_v1.json` (rompe reproducibilidad de lo ya corrido)
  o versionar `split_v2.json`.
- **MedNeXt** del profiling es provisorio; los pacientes de `TargetVolumeID` mal asignado
  (`PTV_Prostate`, `PTVp`, `zRing`/`zOpt`) tienen conteo de versiones como cota inferior.
