# Prompt de arranque para Claude Code — Fase 0 (resto: 0.1 y 0.2)

> Pegar esto como primer mensaje en Code. Antes de pegarlo, copiar al repo los archivos
> listados en "Antes de empezar" y completar los campos marcados `[COMPLETAR]`.

---

## Contexto

Trabajás en el repo del proyecto **site-agnostic dose prediction (KBP)**: una red única que
predice dosis 3D para próstata, pelvis ginecológica y cabeza y cuello (CyC), solo VMAT
normofraccionado/hipo en esta etapa. Estamos en **Fase 0 (infraestructura y datos)**. Fase 0
no entrena nada. Su producto es un preprocesador limpio, una tabla de armonización, datos
auditados y una baseline no-aprendida.

El diseño se hizo en un chat aparte. Vos ejecutás. **Si algo del diseño te parece mal o
ambiguo, no lo resuelvas en silencio: frená y reportalo** (ver "Cuándo frenar").

## Antes de empezar: leer, en este orden

1. `docs/CHARTER.md` (Tier 1, decisiones congeladas; no se modifica de facto. Si ves que algo
   hay que cambiarlo, señalalo explícitamente y yo actualizo el archivo).
2. `docs/ARRANQUE_fase0.md` (qué cierra Fase 0 y qué NO hacer).
3. `docs/SPEC_profiling_0.0.md` y `results/profiling/envelope.md` (0.0 ya está hecho, ver
   "Ya resuelto").
4. `data/harmonization/`: `harmonization_schema.yaml`, `sites_cyc.yaml`, `sites_pelvis.yaml`,
   `sites_prostata.yaml`, más `scripts/convert_constraints.py` y `scripts/build_harmonization.py`
   (generan esos YAML desde las plantillas del software in-house; son regenerables).

Rutas: repo en `C:\Pablo\SiteAgnostic\repo\site-agnostic-dose\`; datos pesados en la carpeta
**hermana** `C:\Pablo\SiteAgnostic\data\` (nunca dentro del repo, nunca a git).
DICOM crudos: `[COMPLETAR: ruta a los DICOM exportados por sitio/paciente]`.

## Ya resuelto (no re-discutir; implementar)

**Normalización de dosis:** D95(PTV_High)=100%, recomputada desde la grilla de dosis cruda
(ignorando la normalización de Eclipse). PTV_High = PTV de mayor Rx (`PTV_High`; en single-level
`PTV` o `PTV_High`). `Rx_High` en Gy se guarda como metadata por paciente para des-normalizar.

**Intención:** el canal de PTV codifica `Rx_nivel / Rx_High` con las Rx **del protocolo**
(`rx_gy` en el YAML del sitio), nunca leídas de la dosis ni de los objetivos de optimización
del plan (leakage). Los ratios se **derivan** de `rx_gy`, no se cargan a mano. Rx reales:
Próstata 70 Gy; Pelvis 50 Gy; CyC 69.96 / 59.4 / 54.45 Gy (High/Mid/Low).

**Red sobre la grilla de dosis nativa**, sin resamplear a la grilla del CT. Spacing esperado
en esta etapa 2.5×2.5×3 mm (confirmado en 6 pacientes; hay que confirmarlo en todos).
Padding a un tamaño fijo global (provisional 176×128×256, derivado de 6 pacientes; ver Paso 1).

**Resolución de PTV (aplica a normalización y a geometría de intención):** para cada nivel,
recorrer `geometry_candidates` en orden (en CyC `PTV_X-04` primero, luego `PTV_X`). Usar el primer
candidato que **exista y tenga volumen ≠ 0**. Una sola resolución por nivel por paciente.

**Goals de OAR: tres canales por tipo, sin pseudo-DVH** (CHARTER §3): *máximos* (Dmax), *medias*
(Dmean) y *volumétrico* (dosis del constraint solo sobre la fracción preservada, vóxeles más
distales al PTV). Valores en % de Rx_High. **Eso se construye en Fase 2, no en Fase 0**: acá solo
tiene que quedar lista y correcta la tabla. Los YAML ya clasifican cada goal por tipo.
**D0.035cm3 ≡ Dmax**: ya está unificado en los YAML (`unified_from` conserva el origen). Los
casi-máximo mayores (`D1cm3`, `D2%`) NO se unifican: su canal es la pregunta viva del CHARTER §10.
No los reclasifiques ni les asignes canal; reportá si te cruzás con alguno inesperado.

**Seriales:** los constraints de órganos seriales están sobre PRVs; el PRV es la geometría. Los
`*_PRV*` NO van al ignore.

**Excluidos de la intención:** `CTV_*` (objetivo de optimización) y estructuras de resta tipo
`Bone_Mand-PTV_Hi`. Se reportan, no se modelan.

**Resolución de nombres, cuatro salidas:** canónico → usa; alias conocido → remapea y loguea;
ignore explícito → saltea; **desconocido → falla ruidoso, va a lista de revisión, nunca adivina**.
Ojo: `nombresPosibles` de las plantillas tiene suciedad (strings vacíos, duplicados, nombres
concatenados tipo `PTV_HighPTV_High`); el consolidador ya la limpia, y el preprocesador debe
ser robusto a más casos de ese tipo.

**Pacientes fuera de convención:** dos casos distintos. (a) Nombre fuera de TG-263 → se resuelve
por alias y entra. (b) Dosis/estructura atípica → NO entra al train: se marca
`out_of_convention: <motivo>` en la metadata y queda como **conjunto reservado** para evaluación
futura de generalización. Nunca se descarta, nunca entra al train por default.

**QA gate (offline, nunca toca el modelo):** sobre **dosis absoluta cruda en Gy** (no la
normalizada: D95(PTV_High) normalizado es 100% por construcción, el chequeo sería circular),
D95 de cada nivel de PTV vs `rx_gy` del protocolo, banda única **[95%, 105%]** para todos los
niveles. Fuera de banda → flag para revisión manual, no descarte automático. Umbral provisional.

**Profiling (0.0) cerrado:** U-Net 3D vanilla entra a volumen entero (176×128×256) sin parches,
`base_feat=16`, batch 1 (`base_feat=32` no entra). El MedNeXt-k5 medido es un stand-in:
provisorio hasta tener la implementación real. No hay nada que hacer en 0.0 salvo no romperlo.

**Datos identificables:** los tres CSV de pacientes (CyC, PelvisGin, ProstataHipo) tienen nombre,
apellido y número de historia clínica. Cada CSV trae además modalidad y dosis total (todo VMAT;
un solo protocolo por sitio en este export).

## Plan de trabajo (en este orden; cada hito termina con un reporte al chat)

### Paso 0: Andamiaje y de-identificación
- Crear la estructura de directorios del charter §6 (repo y carpeta de datos hermana). Verificar
  que `.gitignore` no pueda arrastrar nada de `data\` pesado y que no hay datos identificables
  trackeados.
- Construir una tabla de de-identificación: `HC → patient_id` anónimo y estable. **La tabla de
  mapeo vive solo en la carpeta de datos, fuera de git.** Los CSV originales no se copian al repo.
  Los splits y la metadata que van a git usan solo `patient_id`.
- Colocar los YAML de armonización en `repo\...\data\harmonization\` y los scripts en `scripts\`.
- **Reporte:** estructura creada, N de pacientes por sitio tras de-identificar, confirmación de
  que nada identificable quedó en el repo.

### Paso 1: Inventario de headers (barato, antes de escribir el preprocesador)
Para **todos** los pacientes (no solo los 6 de muestra) leer solo headers DICOM y reportar por
sitio la distribución de: dimensiones de la grilla de RTDOSE, spacing (x, y, z), `DoseGridScaling`,
extent físico del BODY, y presencia/volumen de las estructuras que el YAML de cada sitio espera
(en particular `PTV_*-04`: existe / vacío / ausente).
- Este paso **es** la auditoría de escala de vóxel en su primera versión.
- Decide si el tamaño de padding fijo es válido: si algún paciente excede 176×128×256 o el spacing
  no es uniforme, **frená y reportá** antes de seguir.
- **Reporte:** tablas por sitio, outliers listados por `patient_id`, y recomendación de tamaño de padding.

### Paso 2: Preprocesador nuevo (`src/preprocess/`)
**No es un fork** de `preprocess.py` ni `preprocess_hipo.py` del proyecto de próstata. Diseño
site-paramétrico: la diferencia entre sitios vive en los YAML, no en código duplicado.

Debe, por paciente:
1. Cargar CT por `Modality` y `FrameOfReferenceUID` (precedente de bug: una función cargaba la
   serie RD como si fuera CT). Cargar RTDOSE aplicando `DoseGridScaling` → Gy.
2. Resolver nombres de estructura con las cuatro salidas, usando el schema global + el YAML del
   sitio. Loguear cada alias aplicado (`aliases_applied` en la metadata).
3. Resolver cada nivel de PTV con la regla de `geometry_candidates` (existe y volumen ≠ 0).
4. Rasterizar máscaras sobre la **grilla de dosis nativa**. Documentar el método y medir la
   sensibilidad del D95 al método de rasterización.
5. Calcular D95(PTV_High) en la dosis cruda y normalizar. Toda magnitud física (volúmenes,
   percentiles) usa el spacing nativo del DICOM, nunca `spacing_mm × voxels` de una grilla
   reducida (bug heredado de próstata, factor 1.1×–4.2×).
6. Armar el canal de PTV (intención de Rx) con `Rx_nivel/Rx_High` derivado de `rx_gy`.
   **El NPZ guarda solo geometría (decidido, CHARTER §5):** dosis normalizada, máscara por rol
   canónico ya resuelto y metadata. **No hornear canales de goals de OAR**: se construyen al
   cargar, a partir de la tabla, porque las ablaciones A1a/A1b y las augmentations de goals de
   A2 (omitir un goal, ruido, escalar ×[0.8,1.2]) necesitan variar los goals sin regenerar los
   NPZ.
7. Padear a tamaño fijo y escribir **NPZ versionado**: `npz_v1_<etiqueta>/<sitio>/`. Nunca pisar
   una versión existente; el config del experimento apunta explícitamente a una versión.
8. Guardar metadata por paciente según `patient_metadata_schema` (incluye `protocol_id`,
   `rx_high_gy`, `rx_by_level_gy`, `out_of_convention`, `qa_flags`) y el **hash de los YAML de
   armonización usados**. La tabla ahora define inputs del modelo, así que un cambio en ella
   (como la unificación D0.035cm3→Dmax) cambia el entrenamiento: cada experimento debe poder
   decir con qué versión de la tabla corrió.

Debe **fallar ruidoso** ante cualquier estructura desconocida, PTV no resoluble, o inconsistencia
de frame. Sin defaults silenciosos.

Tests mínimos (unitarios, con datos sintéticos o un paciente de muestra):
- el resolver de nombres cubre las cuatro salidas, incluidos los casos sucios de `nombresPosibles`;
- el resolver de PTV cae al original cuando el `-04` está vacío o ausente;
- tras normalizar, D95(PTV_High) = 100% dentro de tolerancia, y la des-normalización con
  `Rx_High` recupera Gy;
- el canal de intención no cambia si se altera la dosis (test anti-leakage);
- el volumen de una estructura calculado desde la máscara concuerda con el volumen de contornos
  del RTSTRUCT (test de escala de vóxel);
- determinismo: dos corridas dan NPZ idénticos.
- **Reporte:** resultados de los tests, un paciente por sitio procesado de punta a punta, y lista
  de pacientes que fallaron con el motivo.

### Paso 3: QA gate y pacientes fuera de convención
Implementar el QA gate como script offline sobre dosis cruda (ver "Ya resuelto"). Salida: tabla por
paciente y nivel con D95 en Gy, % de Rx, y flag. Correrlo sobre todos los pacientes y **no
descartar nada**: yo reviso los flags manualmente.
- **Reporte:** cuántos flags por sitio y nivel, distribución del D95/Rx por nivel (histograma o
  cuantiles). Con eso recalibramos la banda si hay demasiados falsos flags.

### Paso 4: Auditorías de 0.2
- **Contaminación sobre la dosis** (no sobre geometría ni volúmenes): detectar pacientes cuya dosis
  tenga naturaleza distinta a la del sitio (baños nodales, boosts no contemplados, niveles extra).
  Proponé qué indicadores usar y justificá; reportá candidatos antes de marcar nada como
  `out_of_convention`.
- **Escala de vóxel:** repetir la auditoría del Paso 1 sobre los NPZ ya generados.
- **Conteo de versiones de plan por paciente (ARIA):** **bloqueado**. Depende de que yo termine de
  actualizar el software de extracción, que hoy no exporta ese dato. No empezar; dejar el hook
  previsto en el esquema de metadata.

### Paso 5: Baseline no-aprendida
`scripts/baseline_dvh_promedio.py`: DVH promedio poblacional por `(sitio, rol de estructura)`.
Es un control no-aprendido (no ML clásico). Debe funcionar con cualquier cantidad de OARs (una
curva por rol). Calcularla **solo con pacientes de train** de cada split, nunca con test.
- Los splits (65/15/20 aprox., estratificado por sitio) se guardan como JSON en
  `repo\...\data\splits\`, con `patient_id` solamente.

## Qué NO hacer en Fase 0
No entrenar ninguna red. No incorporar IMRT ni mama. No tocar el stack PDRT / `commissioning\`.
No optimizar arquitectura ni loss. No modificar `CHARTER.md`. No resamplear la dosis a la grilla
del CT. No usar objetivos de optimización ni la dosis alcanzada para definir la intención.

## Cuándo frenar y preguntar
- Spacing no uniforme o grilla que excede el padding propuesto (Paso 1).
- Un nivel de PTV que no se resuelve por ninguna vía, o `Rx_High` que no matchea el protocolo.
- Estructuras desconocidas en más de un puñado de pacientes (puede faltar un alias).
- Cualquier paciente con más niveles de PTV que el protocolo de su sitio.
- Si algo del CHARTER choca con lo que ves en los datos.

## Formato de los reportes
Corto y con números: tablas, no prosa. Archivos en `results/` (auditorías en
`results/auditorias/`). Al volver al chat se trae **solo el delta** (el reporte del hito), no
documentos enteros. Al terminar la fase: actualizar `contexto_fase_vigente.md` (Tier 2) con el
estado actual, no con la traza, y anotar hallazgos nuevos en una línea en
`aprendizajes_transversales.md`.
