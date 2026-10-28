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

- **Cobertura de PTV por debajo de banda ≠ contaminación; es, a menudo,
  subdosificación legítima por OAR.** Dos clusters confirmados en Eclipse en Fase 0:
  cavum en CyC (PTV solapado con PRV de quiasma/ópticos) y un grupo en próstata hipo
  entre 92–95% de Rx (subdosado por OAR), que motivó bajar la banda del QA gate de
  próstata a [92%,105%]. El gate está haciendo su trabajo al marcarlos; la revisión
  manual decide. No bajar un piso de banda sin revisar los casos uno a uno en Eclipse.
  [`paso3_qa_gate.md`, CHARTER §5]

- **El "piso de 100 pacientes/sitio" es sobre la cohorte total, no sobre el train.**
  El 100 es el N total típico de la bibliografía KBP (train+val+test), fijado a ojo, no
  un mínimo de entrenamiento calculado. Un train de ~90 con cohorte de ~140 NO está
  bajo el piso. Evitar reportar como limitación del estudio algo que no lo es.
  [CHARTER §2]

### Fase 1 (cerrada 2026-10-20)

- **Reproducir como control de refactor se parte en dos niveles que localizan la capa
  que falla: inferencia (datos + forward + métrica) y fine-tuning (loop).** Juntos, un
  número que no da no dice qué capa falló; separados, sí. N1 reprodujo a 3e-4; N2
  reprodujo la trayectoria (`val/mae` 1.8758 ép.14 vs. 1.876 ép.14; early-stop 76 vs.
  77; LR coseno T=207 a 6e-16). [`fase1_cerrada.md`]

- **`strict=True` sin keys faltantes ni sobrantes es el control más fuerte por unidad
  de esfuerzo.** Un checkpoint entrenado no se reproduce de casualidad: si un canal
  estuviera cruzado o la normalización corrida, el MAE se iría lejos de 2.55 en vez de
  quedar a 3e-4. Cargar un ckpt viejo en el modelo nuevo exige preservar la estructura/
  nombres del `nn.Module` (o un remapeo); acá el modelo nuevo era calco del viejo.
  [`fase1_cerrada.md`, `PROMPT_code_fase1_nivel1.md` §5]

- **Dead config: replicar el comportamiento REAL, no "arreglarlo por prolijidad".** Dos
  casos en la misma corrida de referencia: `model.context_slices: 3` (lo lee
  `cfg.data.context_slices`, ausente → cae a 1, modelo de corte único 5 canales) y
  `scheduler.warmup_epochs: 5` (desconectado en `configure_optimizers()`). "Conectar"
  cualquiera cambia el grafo o el schedule de LR y rompe la reproducción. [`config
  exp_hipo_004`, `PROMPT_code_fase1_nivel2.md` §5]

- **Un entrenamiento `deterministic=False` igual puede reproducir casi bit-exacto** en
  la misma GPU con batch=1, workers seedeados e init fijo: la no-determinación de cuDNN
  es chica frente a la señal. El criterio "del mismo orden" fue conservador. No es
  excusa para forzar `use_deterministic_algorithms` (cambiaría la trayectoria vs. el
  original). [`fase1_cerrada.md`]

- **Separar observabilidad del loop:** un callback de logging con `import wandb` a nivel
  módulo y que logea `wandb.Image` crashea bajo CSVLogger. Dropearlo bajo `--no-wandb` no
  toca la trayectoria — no confundir un crash de logging con un bug de entrenamiento.
  [`src/callbacks/logging_callbacks.py`, `PROMPT_code_fase1_nivel2.md` §5.6]

- **Operacional:** no lanzar corridas GPU en background con `&` — un doble lanzamiento
  contendió por GPU y por el directorio de checkpoints. [Tier 2, notas vivas Fase 1]

## De Fase 2 (eje A)

- **Leer el set COMPLETO de métricas por sitio, nunca una sola.** El SSIM de A1 casi se descarta mirando
  solo DVH-OAR: los goals no movían la DVH pero subían fuerte la coherencia. Una métrica habría tirado un
  canal útil. (→ `fase2_cerrada.md`.)
- **Canales estáticos por protocolo son redundantes en-distribución.** Goals (y prioridad) tienen valor
  constante por sitio → la red no puede *aprender a responder* a ellos, solo usarlos de huella de sitio.
  Su payoff es fuera de distribución (multi-protocolo / zero-shot). Lo que mueve la aguja es lo
  **paciente-específico** (PSDM distance map). Vale para cualquier canal de "intención" a futuro.
- **Goals = prior de coherencia espacial, no de DVH.** Y el PSDM (PTV distance map) **sí aporta
  multi-sitio**, al revés que Xiong (donde no aportó) — recupera PTV cerca del target.
- **Un análisis estático puede reemplazar una corrida.** A6 (beam plate): con arcos coplanares ~360° el
  mapa colapsa a una rampa en z ≈ PSDM → resultado predeterminado, no se corrió. (§11 en acción.)
- **Bug train/eval de padding de canales (A4):** el canal nuevo se rellenaba distinto en train (1.0) vs
  eval (0) → números absurdos. Unificar el padding en un solo método. Familia: consistencia
  train/inferencia (como la dose-mask del dose_score y el `cargar_ct` de próstata).