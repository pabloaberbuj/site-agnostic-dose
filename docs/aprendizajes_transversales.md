# Aprendizajes Transversales

> **Tier 3 — Append-only.** Memoria institucional curada. Las fases se cierran aquí.
> Cada entrada: línea con el hallazgo, luego puntero a dónde vive la evidencia/detalle.

---

## Heredados del proyecto de próstata

*(Ver CHARTER.md §7 para aplicabilidad a este proyecto)*

- **ML clásico vs. U-Net (próstata, constraints conocidos):** ML clásico ganó para clasificar constraints con features geométricos acotados. No transfiere a DVH completa con OARs variables. [Próstata proyecto, exp0XX]

- **Contaminación vive en dosis, no en máscaras:** Detectar con auditoría de dosis, no volumen/overlap. Lección nodal: 28 pacientes nodales enmascararon "techo de arcos" espurio. [Próstata proyecto, auditoría post-exp0XX]

- **Bugs de copy-paste en datos:** Spacing nativo sobre grilla downsampleada (1.1×–4.2×), cargar_ct cargaba RD. Argumenta preprocesador nuevo. [Próstata proyecto, Fase 0]

- **MAE puro ganó; DVH loss no aportó:** En próstata, MAE simple superó losses basadas en DVH/momentos. Baseline para comparación. [Próstata proyecto, Fase 4]

- **3D no aportó en próstata (contexto específico):** Próstata tiene gradiente axial suave. CyC/pelvis se espera que sí mejoren con 3D (estructura cráneo-caudal). [Próstata proyecto, Fase 3]

- **Higiene documental:** Tier 2 reescrito a verdad de hoy; append-only para cambios históricos. Evita acumulación lineal de correcciones. [Próstata proyecto, docs]

---

## De este proyecto

### Fase 0 (cerrada 2026-10-16)

- **Las reglas genéricas de resolución de nombres necesitan desempate explícito, no
  "el primero que matchee":** cuando dos nombres crudos alias al mismo rol y ambos
  tienen volumen>0 (recontoreos), elegir al azar oculta ambigüedad real. Desempate por
  marca explícita en el nombre ("prv", "corregido") resuelve la mayoría sin adivinar;
  lo que no se puede desempatar así queda como falla ruidosa para revisión en Eclipse,
  nunca se resuelve en silencio. [`src/preprocess/pipeline.py`, `paso2_decisiones_autonomas.md` #29-30, #44]

- **Una regla genérica puede "comerse" una excepción puntual sin que se note:** la
  regla de duplicado numerado (`Cochlea_L1`→`Cochlea_L`) reinterpretó
  `SpinalCord_PRV1` (margen no estándar) como "duplicado de `SpinalCord_PRV`" porque
  ese nombre sin dígito ya era alias válido — el paciente pasó de "falla ruidosa
  correcta" a "ambigüedad real" sin que la intención original (excluirlo) se cumpliera.
  Mismo patrón con `PTV_High`/`PTV_High1` y `Oral_Cavity`/`Cavity_Oral`: a veces hace
  falta una excepción scopeada a UN paciente puntual (`_PATIENT_RAW_NAME_OVERRIDES`),
  porque el nombre en sí resuelve bien para todos los demás. [`pipeline.py` #43, #51, #53]

- **Verificar antes de aplicar, nunca aplicar y asumir:** varias decisiones de
  armonización dependían de datos que no estaban precalculados (volumen real de una
  estructura, tamaño de campo del RTPLAN). Calcularlos antes de decidir evitó al menos
  2 bugs (recategorizar "Mid ausente" cuando en realidad tenía volumen real; confirmar
  que `PTV_High`/`PTV_High1` eran dos geometrías reales antes de fundirlas). [Paso 2, rondas 9-12]

- **Un inventario de nombres de estructura por paciente, generado una vez, evita
  releer DICOM en cada ronda de armonización nueva** (solo RTSTRUCT, sin dosis — barato).
  El volumen rasterizado sigue siendo caro y se calcula bajo demanda, no en bulk.
  [`results/auditorias/paso2_estructuras_por_paciente.csv`]

- **Auditorías independientes (dosis vs. estructura, RTPLAN vs. NPZ) se validan entre
  sí:** el indicador de contaminación por extensión de dosis confirmó, sin usar ningún
  nombre de estructura, los mismos pacientes de campo extendido que el criterio de
  tamaño de campo del RTPLAN (Paso 2) — y encontró 1 caso que el criterio scopeado a
  "pacientes con Kidney" no alcanzaba. Señal de que ambos métodos miden algo real, no
  ruido. [`paso4_contaminacion_dosis.md`]

- **Una aproximación documentada como "no validada" eventualmente hay que validarla
  con datos reales, no solo con la advertencia en el docstring:** el volumen de PTV
  por contorno de Paso 1 (suma de áreas, sin regla par-impar) sobreestimaba hasta 48%
  en estructuras con huecos (ej. PTV evitando SpinalCord en algunos cortes) — Paso 1 ya
  decía "no es el volumen validado" pero nadie había medido cuánto difería hasta Paso 4.
  [`paso4_escala_voxel_npz.md`]
