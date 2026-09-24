# Arranque — Fase 0: Infraestructura y datos

> **Para abrir el próximo chat del proyecto.** Leer junto con `CHARTER.md` (Tier 1).
> Este MD define qué hay que cerrar en Fase 0 y qué NO hacer todavía. Al terminar la fase,
> este chat genera: (1) un MD que congela Fase 0 y (2) el MD de arranque de Fase 1.

---

## Objetivo de la Fase 0

Dejar montada la infraestructura de datos y código **limpia y auditada** antes de tocar
cualquier modelo. Fase 0 no entrena nada. Su producto es: saber qué aguanta la PC, tener un
preprocesador nuevo confiable, una tabla de armonización, datos auditados por contaminación y
por escala, y la baseline no-aprendida (DVH promedio por sitio/rol) lista para Fase 1.

Recordatorio del porqué (aprendizajes heredados): en próstata, los bugs de datos (escala de
vóxel, carga de CT como serie RD) y la contaminación nodal costaron corridas enteras y un
hallazgo espurio que sobrevivió 3 entrenamientos. Fase 0 es la vacuna contra eso.

---

## Tareas

### 0.0 — Profiling de la PC (primero de todo, desde Code)

Determinar el *envelope* de experimentación en la RTX A2000 12GB antes de diseñar nada:

- Máximo tamaño de parche × batch_size × nº de features, para U-Net 2D y 3D.
- Con y sin gradient checkpointing.
- A resolución de grilla de dosis (no de CT).
- Estimar: ¿hace falta downsampling? ¿parches sí o sí en 3D? ¿qué batch es viable?

**Entregable:** un `results/profiling/envelope.md` con la tabla de configuraciones viables.
Adaptar el `profile_step.py` del proyecto de próstata.

**Decide:** el rango de configuraciones legítimas para las Fases 2–3 (para no diseñar
experimentos que no entran en la GPU).

### 0.1 — Preprocesador nuevo + armonización

- Preprocesador **nuevo y limpio** (NO fork de `preprocess.py` / `preprocess_hipo.py` del
  proyecto viejo). Sin reimplementar la lógica por-sitio (la deuda técnica de próstata fue
  justo esa duplicación). Diseño site-paramétrico: la diferencia entre sitios vive en config,
  no en código duplicado.
- Tabla de armonización con nomenclatura **TG-263**, estructura
  `sitio → {(OAR, métrica): (goal, prioridad)}`. Contemplar el caso 1-OAR-a-muchas-métricas
  desde el inicio aunque casi siempre sea 1-a-1.
- Definir y documentar la **convención de normalización de dosis multi-sitio** (pregunta viva
  del charter).
- Salida a NPZ **versionados por etapa**: `npz_v<N>_<etiqueta>/<sitio>/`. Nada de pisar.
- Evitar por diseño el bug de escala de vóxel: cualquier cálculo físico sobre máscaras
  downsampleadas se calibra con el volumen nativo, no con `spacing_mm` × voxels de la grilla
  reducida.

**Entregable:** `src/preprocess/`, `data/harmonization/` (a git), NPZ v1 en `data_root/`.

### 0.2 — Auditorías + baseline no-aprendida

- **Auditoría de contaminación sobre la dosis** (no sobre geometría/volúmenes — la lección
  nodal: la contaminación vive en la dosis). Para los 3 sitios VMAT: detectar pacientes cuya
  dosis tenga naturaleza distinta a la del sitio (p.ej. baños nodales, boosts no contemplados).
- **Auditoría de escala de vóxel** en el preprocesador nuevo (verificar que no reaparece el
  bug 1.1×–4.2×).
- **Conteo de versiones de plan por paciente en ARIA** (aprobados + no aprobados + intermedios).
  Es lo que decide la viabilidad del Pareto-conditioning de la Fase 5. Registrar cuántas
  versiones recuperables hay por paciente y por sitio.
- **Baseline no-aprendida** montada: DVH promedio poblacional/de protocolo por
  `(sitio, rol de estructura)` — no ML clásico (no aplica acá; ver charter §2 y §7). Sirve como
  piso mínimo de comparación con cualquier cantidad de OARs, listo desde Fase 1.

**Entregable:** reportes de auditoría en `results/auditorias/`, `scripts/baseline_dvh_promedio.py`,
y una decisión registrada sobre viabilidad del Pareto (sí / no / condicionada).

---

## Qué NO hacer en Fase 0

- No entrenar la U-Net (eso es Fase 1).
- No incorporar IMRT ni mama (solo VMAT, 3 sitios — ver charter §4).
- No tocar el stack PDRT / commissioning (en pausa).
- No optimizar arquitectura ni loss (Fases 3–4).

---

## Cómo cerrar la Fase 0

Al terminar, generar:

1. **MD de cierre de Fase 0** (a Tier 3 del Project): qué quedó montado, decisiones tomadas
   (normalización, viabilidad Pareto, envelope de la PC), auditorías con sus hallazgos.
2. **MD de arranque de Fase 1** (reproducción del baseline exp002 en el framework nuevo, sobre
   próstata VMAT).
3. Appendear a `aprendizajes_transversales.md` cualquier bug/hallazgo nuevo, curado, en una línea.

---

## Contexto mínimo para el chat de Fase 0

- Leer `CHARTER.md` completo (sobre todo §3 representación de input, §4 fases, §6 repo, §7
  aprendizajes).
- Arrancar por 0.0 (profiling) porque condiciona todo lo demás.
- Handoff con Code: traer al chat solo deltas (envelope.md, reportes de auditoría), no docs
  enteros.
