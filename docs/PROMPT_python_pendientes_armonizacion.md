# Prompt para Claude Code (repo Python) — integrar normalización de nombres + ignore

> Pegar en la sesión de Code abierta sobre el repo Python del proyecto site-agnostic
> (`site-agnostic-dose`). Son dos tareas que quedaron pendientes de una sesión del extractor
> (otra PC): llevar al repo Python una regla de normalización de nombres ya escrita y probada, y
> aplicar un cambio al generador de la tabla de armonización. Más dos notas de contexto que NO
> hay que perder.

---

## Contexto

La capa de resolución de nombres de estructura tiene cuatro salidas (CHARTER §5): canónico → usa;
alias conocido → remapea y loguea; ignore → saltea; desconocido → falla ruidoso y va a revisión,
nunca adivina. Un export real de CyC mostró que la resolución de PTV era demasiado laxa (matcheaba
por substring y agarraba `zRingPTV_High`, un ring de optimización) y que hay PTV con el sufijo de
crop `-04` mal tipeado (`PTV_High-0.4`, `PTV_High!04`). En vez de enumerar cada tipeo, se definió
una **regla de normalización** que absorbe el ruido de separadores y preserva solo la distinción
"tiene 04 o no". Esa regla ya está escrita y probada (45 casos) en el repo del extractor, en
`docs/normalize_structure_name.py`. Hay que traerla al repo Python.

## Tarea 1 — integrar `normalize_structure_name.py`

`normalize_structure_name.py` (de la sesión del extractor; si no está en este repo, pedírselo al
usuario o reimplementar según la especificación de abajo). Qué hace la regla:

- Pasa a minúsculas.
- Detecta y preserva la marca de crop `04` como flag semántico: un separador de ruido
  (`-`, `.`, `!`, `_`, espacio o combinación) seguido de `04` al final → `cropped=True`, y se
  quita ese sufijo del string base. Cubre `-04`, `-0.4`, `!04`, `_04`, ` 04`, etc. sin enumerar.
- Sobre el base ya sin sufijo de crop, colapsa los separadores de ruido restantes a una forma
  canónica, para que `ptv_high`, `ptv-high`, `ptv high` matcheen igual.
- Recompone a `PTV_<Nivel>` + flag `cropped`. `cropped=True` → candidato `-04` de
  `geometry_candidates`; `cropped=False` → candidato sin sufijo.
- **No** colapsa a `04` un sufijo numérico distinto (`-05`, `-10`, `-14`): eso es desconocido →
  falla ruidoso. El `0.4` SÍ es tipeo de `04` (marca de crop), no un margen real.
- Si dos estructuras caen en el mismo grupo normalizado → marcar **ambiguo**, no elegir una.
- Aplicar el mapa de alias sobre la **base ya normalizada** (sin el sufijo de crop), no sobre el
  nombre crudo: así `PTV-0.4` (alias `ptv` + crop) resuelve a `PTV_High-04`. Si el alias solo aplica
  al nombre exacto, ese PTV recortado queda sin reconocer. Orden de resolución: canónico → alias →
  ignore (un alias explícito gana sobre un patrón de ignore como `z_*`).
- Estructuras que NO deben aliasarse a `PTV_High` aunque su nombre lo sugiera, porque son otra
  estructura (más chica): `PTV_Prostate` (Próstata) y `PTVp` (Pelvis, es el PTV primario;
  `PTV_High` = `PTVp` + `PTVn`).

Integrarla en **dos lugares**, que es el punto de tener una sola regla:
1. El **preprocesador** (`src/preprocess/`): la resolución de nombres la usa para cada estructura,
   con las cuatro salidas. La resolución de PTV deja de ser substring; usa la regla + la selección
   de candidato por volumen ≠ 0 (recortado gana al original si tiene volumen).
2. El **generador** de la tabla (`scripts/build_harmonization.py` y/o `convert_constraints.py`):
   el matcheo de nombres de `nombresPosibles` contra roles pasa por la misma regla, para no
   depender de que los nombres de las plantillas estén perfectos.

Llevar también los 45 casos de prueba como test del repo Python (no reescribir, portar), y agregar
los que surjan de la Tarea 3.

## Tarea 2 — ignore `zring*` y `zopt*` en `build_harmonization.py`

Agregar `zring*` y `zopt*` a la **lista estática de ignore** de `build_harmonization.py` (el
patrón actual `z_*` no los atrapa: no tienen guion bajo). Son helpers de optimización → ignore
silencioso, correcto. Regenerar `harmonization_schema.yaml` y los `sites_*.yaml` con los scripts
(no editar los YAML generados a mano). Confirmar que tras regenerar, `zring*`/`zopt*` quedan en
`ignore_patterns` y que ningún `sites_*.yaml` cambió fuera de eso.

## Tarea 3 — al correr la auditoría de nombres (Paso 2 de Fase 0), loguear PTV crudos de TODOS los niveles

La metadata del extractor solo reporta el PTV_High (un PTV de referencia por paciente), así que las
variantes mal tipeadas de **Mid y Low** no se ven ahí. La regla de normalización debería cubrirlas
(no enumera por nivel), pero hay que **confirmarlo con los datos**: la auditoría de nombres del
preprocesador ve todas las estructuras de los RTSTRUCT. Al correrla, loguear la lista completa de
nombres crudos de PTV por nivel (High/Mid/Low) y marcar cuáles resolvió la normalización, cuáles
cayeron en ambiguo y cuáles en desconocido.

Vigilar un borde: un separador suelto al final sin `04` (p.ej. `PTV_High!`, `PTV_High_`,
`PTV_High ` con espacio). Debe resolver a `PTV_High` (ruido), no caer en desconocido. Si la regla
lo está rechazando de más, ese es el ajuste que sale de ver la lista real. Reportar antes de
cambiar la regla.

## Tarea 4 — ítems que habían quedado solo en el chat (si ya los pegaste, ignorar este bloque)

- **Alias manuales** en la extensión de `build_harmonization.py`: `ptv` → PTV_High (se perdió al
  regenerar el schema v2, que solo siembra alias desde `nombresPosibles`), `ptv_high_p` → PTV_High y
  `ptv_ptta` → PTV_High. Con el D95 de Eclipse los pacientes resueltos por estos alias dieron dentro
  de banda; repetirlo con el D95 de Python. El extractor tiene **su propia copia** de esta lista:
  mantenerlas iguales. La columna `PTV_Resuelto` de los CSV de metadata sirve de cruce.
- **`patient_exceptions.yaml`** (en `data/harmonization/`): leerlo ANTES de resolver nombres. Tres
  categorías: `excluded` (nunca exportados, solo registro), `out_of_convention` (exportados, fuera del
  train, conjunto reservado) y `reviewed_ok` (entran al train; solo evita re-revisarlos).
- **Conteos de versiones vacíos** (`NPlanesCurso`, `NAprobados`, …) significan "no calculado" (plan sin
  `TargetVolumeID`), no 0. Ningún análisis debe leer vacío como cero.
- **Columnas de los CSV de metadata:** `Sitio`, `PTV_Resuelto`, y `FlagRevision` con el nombre crudo del
  `TargetVolumeID` cuando no normaliza a PTV_High. Un `TargetVolumeID` distinto del PTV resuelto es
  **informativo** (`PTV_Prostate`, `PTVp`, `zRing`/`zOpt`): el PTV se resuelve por nombre de
  estructura, nunca por `TargetVolumeID`.

## Dos notas de contexto (de la revisión manual del export; NO son tareas de código)

- **Cobertura baja de PTV en CyC puede ser legítima, no contaminación.** En Ca de cavum el PTV se
  superpone con PRV de quiasma / nervios ópticos (prioridad mayor al PTV), y la cobertura baja es
  una decisión clínica deliberada. El QA gate (D95 vs Rx, banda [95%,105%]) va a flaggear estos
  casos sistemáticamente: son falsos positivos estructurales, se resuelven por revisión manual.
  **No bajar el piso del gate para acomodarlos** — dejaría pasar infracobertura real en otros
  sitios. El gate está haciendo su trabajo; la revisión manual decide.
- **Reirradiación: detección parcial, se acepta.** Un paciente del export era reirradiación con
  plan previo a 66 Gy (enfriado por el acumulado) → `out_of_convention: reirradiacion`, fuera del
  train, al conjunto reservado. La reirradiación **intra-ARIA** podría flaggearse mirando la
  historia de cursos (plan previo tratado a otra Rx), pero la **reirradiación en otro centro** no
  deja rastro en ARIA ni en el DICOM: no es detectable automáticamente. Decisión del usuario: son
  pocos casos, no van a dosis plena, y el que caiga fuera de banda lo agarra la revisión manual del
  QA gate. **No** construir detección automática de reirradiación; dejar que el QA gate + revisión
  manual sea la red.

## Cuándo frenar y preguntar
- Si `normalize_structure_name.py` no está en este repo y hay que reimplementar: confirmar la
  especificación antes, no improvisar la frontera "ruido vs desconocido".
- Si al regenerar los `sites_*.yaml` cambia algo más que el ignore.
- Si la lista de PTV crudos de Mid/Low muestra variantes que la regla no cubre.
