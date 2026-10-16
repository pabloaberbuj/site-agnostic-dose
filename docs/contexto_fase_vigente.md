# Contexto Fase Vigente

> **Tier 2 — Documento vivo.** Se reescribe completamente al fin de cada fase.
> Refleja siempre "qué es verdad ahora", no la traza histórica.
>
> Última actualización: 2026-10-16 (cierre de Fase 0)

---

## Estado actual

- **Fase activa:** 0 (Infraestructura y datos) — **CERRADA**. Próxima: Fase 1
  (Reproducción del baseline — portar exp002 U-Net 2D + PSDM, MAE, al framework nuevo
  sobre próstata VMAT).
- **Datos:** 503/503 pacientes procesados al 100% (CyC 196, Pelvis 161, Prostata 146).
  NPZ en `npz_v1_paso2/<sitio>/` (fuera de git, carpeta hermana). Cohorte elegible
  (total procesado menos `out_of_convention`, el reservado nunca entra a ningún split):
  CyC 156/196 (41 removidos: 1 `excluded` sin DICOM + 40 `out_of_convention`), Pelvis
  138/161 (23 `out_of_convention`), Prostata 139/146 (7 `out_of_convention`). Desglose
  de motivos por sitio en `paso2_decisiones_autonomas.md` / `patient_exceptions.yaml`.

## Fase 0 — cerrada, qué queda construido

- **0.0 Profiling:** envelope en `results/profiling/envelope.md`. U-Net 3D vanilla
  176×128×256, `base_feat=16`, batch 1, entra sin parches. Re-perfilar en Fase 3 sobre
  casos reales grandes (no sobre caja padeada).
- **0.1 Preprocesador** (`src/preprocess/`): site-paramétrico (diferencia entre sitios
  vive en YAML, no en código por sitio). Resolución de nombres de 4 salidas
  (canónico/alias/ignore/exclude/desconocido, nunca adivina). Tabla de armonización en
  `data/harmonization/` (`harmonization_schema.yaml`, `sites_<sitio>.yaml`,
  `patient_exceptions.yaml`). 46+ decisiones de armonización documentadas con evidencia
  en `results/auditorias/paso2_decisiones_autonomas.md`.
- **0.2 Auditorías:**
  - QA gate (D95 vs Rx por nivel, banda por sitio — Prostata [92%,105%], resto
    [95%,105%]): 0 flags sin revisar en los 503. `results/auditorias/paso3_qa_gate.md`.
  - Contaminación sobre dosis (extensión Z con dosis ≥X% Rx_High): confirma
    independientemente los casos de campo extendido ya conocidos.
    `results/auditorias/paso4_contaminacion_dosis.md`.
  - Escala de vóxel sobre NPZ: sin discrepancias sin explicar vs. Paso 1.
    `results/auditorias/paso4_escala_voxel_npz.md`.
  - Conteo de versiones de plan (ARIA): sin concentración tipo Pareto; no se usa como
    criterio de exclusión en esta etapa (decisión del usuario, 2026-10-16); queda como
    insumo contabilizado para una etapa posterior. `results/auditorias/paso4_versiones_plan.md`.
  - Baseline no-aprendida (DVH promedio por sitio/rol, solo train): en
    `results/baseline/baseline_dvh_<sitio>.json` + `paso5_baseline.md`.
- **Splits:** `data/splits/split_v1.json` (seed=42, 65/15/20 estratificado por sitio).
  Train: CyC 101, Pelvis 90, Prostata 90 (total 281). El "piso de 100 pacientes/sitio"
  del CHARTER §2 es sobre la **cohorte total** (train+val+test), no sobre train solo —
  las 3 cohortes elegibles (156/138/139) están por encima; no es una limitación.

## Pendiente / abierto (no bloquea el cierre de Fase 0)

- `BowelLarge` sin PRV en Próstata: estructura sin decisión de armonización (no genera
  fallo, solo queda sin rol asignado). Ver `paso2_preprocesador.md`.
- Mirror pendiente en el extractor C# (`normalize_structure_name.py`): alias
  `ptv_intermediate`→mid y `ptv_high1`→high agregados solo en el repo Python — alguien
  con acceso al repo del extractor debe replicarlos si se quiere que `DicomExtract`
  resuelva igual.
- `data/splits/split_v1.json` es el primer split; si cambia la tabla de armonización de
  forma que mueva pacientes entre elegible/`out_of_convention`, hay que decidir si se
  regenera (rompe reproducibilidad de experimentos que ya lo usan) o se versiona
  `split_v2.json`.

## Métricas de éxito de Fase

- [ver CHARTER.md §2 — adaptadas para Fase 0: terminaron siendo "todo el pipeline
  corre sobre los 503 reales sin fallos, QA gate limpio" en vez de una métrica de
  modelo, porque Fase 0 no entrena nada]

## Notas vivas

Vacío — Fase 0 cerrada. Los hallazgos curados de esta fase están en
`docs/aprendizajes_transversales.md`.
