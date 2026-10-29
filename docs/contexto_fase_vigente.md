# Contexto Fase Vigente

> **Tier 2 — Documento vivo.** Se reescribe completamente al fin de cada fase.
> Refleja siempre "qué es verdad ahora", no la traza histórica. El *por qué cambió* (p.ej. por qué
> v2b reemplazó a v1/v2) va al append-only de Tier 3, no acá.
>
> Última actualización: 2026-10-29 (Fase 2 cerrada — ver `fase2_cerrada.md`; Fase 3 / eje B en arranque — ver sección
> «Fase 3 / eje B» abajo. Las secciones de Fase 2 que siguen quedan como referencia hasta reescribir este doc al cierre de B).

---

## Fase 3 / eje B — estado (2026-10-29)

**Representación fija = A4** (goals uniformes fill 0 + PSDM de PTV_High, 7 canales). Eje B: U-Net 2D → vanilla-3D → (parches/MedNeXt solo si hace falta).

- **Contenedor in-plane nuevo: (176, 256) con crop-to-BODY** (2D y 3D por igual). Medido sobre los 432 del split: bbox del BODY máx **163×248**
  (grilla completa hasta 242×248; Z máx train 239 / val 214 / test 214). Con 256² vanilla-3D da OOM aun con gradient checkpointing; con
  (176,256) + ckpt entra (smoke sintético Z=239, `scripts/smoke_b1_vram.py`): pico asignado 8.17 GiB; con `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`
  reserva ~8.6 GiB (sin él ~10.4). Con tope de 9 GiB y sin el ajuste OOMea.
- **Código:** `Frame` en `src/datamodules/a0_datamodule.py` (crop centrado al bbox del BODY + pad rectangular, invertible con `Frame.undo`; `pad_hw` int
  sigue siendo el camino legacy, bit-exacto). Config: `data.container: {hw: [176,256], crop_body: true}`. `eval_a0.py` deshace pad y crop antes del DVH,
  verifica que ninguna estructura salga del box y escribe `crop_offsets.csv`. `train_a0.py` ganó `--resume` y `--report-vram`.
- **Verificación del crop (`scripts/smoke_container.py`, 432 pacientes): 0 fallas** — estructuras dentro del box, ida y vuelta exacta de dosis y canales, DVH de ida y vuelta idéntico.
  Probes con el ckpt viejo de A4 (epoch=105, val): (1) el camino legacy con el código nuevo reproduce el A4 viejo exacto; (2) con crop, el desplazamiento óptimo
  pred-vs-dosis es (0,0) con corr ≈0.98 en 9 pacientes → **no hay bug de offsets**; (3) el modelo viejo **no es invariante al marco**: solo más padding (288², sin crop) mueve
  OAR/PTV hasta ~1 pp y con crop hasta −10 pp en PTV alta de Pelvis (la escala predicha cambia hasta ~12 %; el SSIM casi no se mueve). Por eso la prueba real es reentrenar.
- **Corrida programada** (`scripts/run_b_chain.sh`, tarea `sa_b_chain`, desacoplada, reanudable con `--resume` y reintentos): canario (1 época de vanilla-3D sobre la cohorte real,
  mide VRAM; sin reintento, un OOM acá PARA y reabre parches) → A4-2D en el contenedor (`exp_a4_003_unet2d_psdm_c176x256`) → guardrail (`scripts/guardrail_a4_container.py`:
  OAR-media / PTV-alta vs A4-2D viejo, falla si |Δ|>1 pp; criterio fijado antes de correr, sin piso de ruido) → vanilla-3D (`exp_b1_001_unet3d_psdm`, 189/28, ckpt).
  `num_workers: 1` (la PC tiene ~6.5 GB de RAM libre por la VM Hyper-V). **Resultados: pendientes** (se escriben acá al leerlos).
- **Heads-up:** el contenedor alcanza para vanilla; probablemente NO para MedNeXt-k5. Parches se reabren ahí.

---

## Estado actual (histórico de Fase 2)

- **Fase activa:** 2 — primer site-agnostic (A-eje de input, SOLO VMAT). Plan A0–A6 en CHARTER §4
  y `ARRANQUE_fase2.md`.
- **Subestado:** HO#1 (infra + datamodule A0 + smoke test) **cerrado**, 0 fallas. No se entrenó nada
  todavía. HO#2 (loop de train de A0) y HO#2b (baseline no-aprendida) **especificados**, corriendo
  en paralelo. Siguiente lectura real de la fase: A0 vs baseline por sitio.

## Datos vigentes de Fase 2

- **NPZ:** `data/npz_v2b_body/<sitio>/<patient_id>.npz` (+ sidecar `.json`), 503 pacientes.
  Contenido: `dose_pct` (D95(PTV_High)=100%, grilla de dosis nativa) y `mask_<rol>` booleanas por
  rol canónico, **incluida `mask_BODY`** (v1 no la traía). Metadata (`protocol_id`, `rx_high_gy`,
  `rx_by_level_gy`, espaciados, `out_of_convention`, `harmonization_hash`) en el JSON, NO en el NPZ.
- **Split:** `data/splits/split_v2.json` = `split_v1` menos `PT_5787d1d3258de95d`. **432 pacientes:
  280 train / 65 val / 87 test** (Próstata 138, Pelvis 138, CyC 156).
- **Tabla de armonización:** hash `2a0c2037686c2978` (sha256 de nombre+contenido de todos los
  `.yaml` de `data/harmonization/`, 16 hex). Es el hash que va al record de cada experimento A0.
- `npz_v1_paso2/`, `npz_v2_body/` y `split_v1.json` quedan como histórico; no usar para entrenar.

## Datamodule A0 (`src/datamodules/a0_datamodule.py`)

- Item = paciente completo en cortes axiales: `input (Z,3,256,256)`, `dose (Z,256,256)` float32.
  Solo `batch_size=1` (Z varía; no hay pad en Z).
- Canales: **Ch0** PTV pintado con `Rx_nivel/Rx_High` PRESCRIPTO de la tabla (nunca de la dosis;
  el nivel alto gana el solape) · **Ch1** unión binaria de roles OAR (PRV incluidos) · **Ch2** body.
- Pad in-plane centrado a **256×256** (configurable). El HO decía 512, pero el máximo real es
  248×248; 512 era ~4× vóxeles de ceros. Un paciente que exceda `pad_hw` falla ruidoso.
- Falla ruidoso ante: rol desconocido en el NPZ, NPZ sin `mask_BODY`, paciente `out_of_convention`,
  nivel de PTV sin Rx en el protocolo.
- **Excepción registrada:** 16 pacientes de Próstata traen `PTV_Low` no declarado, idéntica a
  `PTV_High` (motivo `qa_ptv_low_identico_a_ptv_high` en `reviewed_ok`); el datamodule omite esa
  máscara inerte. `PT_5787d1d3258de95d` (PTV_Low distinta, IoU 0.865) pasó a `out_of_convention`
  (decisión 2026-10-20) y está fuera del split.
- `experiment_record()` devuelve el hash de la tabla, `npz_dir`, `splits_file`, `pad_hw`.
- Sin augmentations (A2 es de goals; flip/rotación geométricos no están pedidos para A0).

## Entorno

`requirements.txt` pinneado a las versiones con que corrió Fase 1 (Python 3.13.2, torch
2.12.1+cu130, PL 2.6.5, numpy 2.4.4). `.venv` en el repo (gitignored), independiente del proyecto
viejo. Instalar con `pip --no-cache-dir` (el sandbox bloquea el cache de pip). GPU: RTX A2000 12GB.

## Smoke test (`scripts/smoke_a0.py`)

`python scripts/smoke_a0.py --npz-dir <npz_v2b_body> --splits-file data/splits/split_v2.json [--all]`.
Estado: **0 fallas en los 432 del split** (`--all`); D95(PTV_High)=100.0% (tol ±2%), ratios de Ch0 =
tabla (CyC Mid 0.8491 / Low 0.7783), hash de sidecars = hash de la tabla.

## Evaluación (decidido — HO#2/HO#2b)

- **Métrica principal:** `dvh_curva_completa` = `mean|ΔV(D)|` por banda de dosis, por estructura,
  eje de dosis en **% de Rx_High**. **Banda media 40–80% la crítica para OARs**; banda alta para PTV
  (ahí concentra su error). DVH sobre **volumen nativo** (higiene de vóxel), no grilla downsampleada.
- **Reporte por sitio siempre**, con **IC95 bootstrap** resampleando pacientes dentro del sitio. El
  MAE global es secundario (promedia y esconde).
- **Módulo compartido `src/eval/`:** el mismo harness scorea red y baseline. El primero de HO#2/HO#2b
  que corra lo monta; el otro lo importa.
- **Secundarias de monitoreo (solo red):** MAE body, dose_score OpenKBP, SSIM/MS-SSIM. Se reportan,
  no son el norte.
- **Convención de lectura del sweep A0→A6:** decisiones sobre **val**; **test reservado** para el
  cierre del eje A (no filtrar test a través de las ablaciones incrementales).
- **Comparación que manda:** A0 vs baseline no-aprendida sobre v2b. La vara de Fase 1
  (`dvh_curva_completa`, banda media: Rectum 7.30, Bladder 3.34; PTV banda alta 4.82) es referencia
  de **orden de magnitud y forma, no umbral** (cambian cohorte y normalización).

## Baseline no-aprendida (control permanente — HO#2b)

DVH promedio poblacional por `(sitio, rol de estructura)`, construida **solo del split train**
(val/test sería leakage). La predicción para una estructura = la curva de su `(sitio, rol)`. Se
scorea **solo en espacio DVH** (no tiene dosis 3D → sin MAE body ni SSIM). Responde la pregunta de
§2: ¿la red aprende algo más que la forma típica del protocolo? Reemplaza al control de ML clásico
(descartado: no generaliza al target de DVH completa con N variable de OARs).

## HO#2 — referencia de entrenamiento verificada (2026-10-20, aún sin correr)

Fuente: `C:/Pablo/ProstateDoseProject/repo/configs/exp002_unet2d_psdm_ctfix_fov34.yaml` (run desde cero,
el único con horizonte propio, no fine-tune).

- U-Net 2D `base_features=16`, `depth=4`, GroupNorm, bilinear, dropout 0; `build_model` ya está en
  `src/models/unet2d.py` (solo cambia `in_channels: 3`).
- AdamW `lr=1e-4`, `wd=1e-5`; cosine `T_max = scheduler.max_epochs`; batch 1; clip 1.0; `16-mixed`;
  gradient checkpointing en el forward 2D.
- Horizonte de referencia: **200 épocas, patience 30, `N_train_ref=265`** (train normo de `split_v1`).
- **Propuesta (a confirmar):** `factor = 265/280 = 0.9464` → `max_epochs=scheduler.max_epochs=189`,
  `patience=28`. Registrar el factor en el record.
- `warmup_epochs` está desconectado en `configure_optimizers()` (dead config, se replica).
- **Factor confirmado (2026-10-20):** 0.946429 → 189 épocas, patience 28.
- **Montado:** `src/models/a0_module.py` (MAE body, forward por chunks de 32 cortes con checkpointing),
  `configs/exp_a0_001_unet2d.yaml`, `scripts/train_a0.py` (asserts de factor/N_train, escribe
  `results/<exp>/record.json`), `src/eval/dvh.py` (harness compartido: bandas, IC95 bootstrap por sitio,
  diferencia pareada), `scripts/eval_a0.py` (`net` | `baseline` | `compare`; default val).
- **Baseline v2b:** curvas en `results/baseline_v2b/` (solo train), scoreada en val en
  `results/baseline_v2b_eval/`. Val, OAR banda media / PTV banda alta (pp, IC95 en summary.json):
  CyC 9.03 / 5.66 · Pelvis 6.82 / 2.58 · Próstata 3.77 / 6.35.
- **Run `exp_a0_001_unet2d` terminado** (early stop; mejor ckpt `checkpoints/exp_a0_001_unet2d/epoch=097.ckpt`,
  val/mae 2.4236; ~8 min/época, ~4.4 GB VRAM). Evaluado en **val** (test intacto), 0 fallas de sanity.
- **Resultado A0 vs baseline (val, pp, IC95 pareado; >0 = red mejor):**
  OAR banda media — CyC +6.82 [5.76,8.08] · Pelvis +3.52 [2.39,4.55] · Próstata +1.57 [0.86,2.26].
  PTV banda alta — CyC −0.57 [−1.81,0.73] · Pelvis −1.58 [−3.58,−0.05] · Próstata +0.80 [−0.64,2.20]
  (empate o peor que la baseline en PTV). Red en pp: OAR media 2.21 / 3.30 / 2.20; PTV alta 6.23 / 4.15 / 5.54.
- **Secundarias (val):** MAE body CyC 2.37 · Pelvis 3.40 · Próstata 1.51; SSIM 0.38 / 0.66 / 0.67.
  `dose_score_openkbp` **removido del reporte** (idéntico al MAE body). MAE body incluye soporte/inmovilizadores
  (`mask_BODY` = External de Eclipse, correcto): no comparar MAE body entre pacientes/sitios.
- **IC degenerados (HO#3):** `bootstrap_ci` ahora devuelve `std` y `degenerate` (std entre pacientes = 0). Las cells
  per-estructura con `degenerate=true` (33/252 en la red) son bandas donde la estructura no recibe dosis (p.ej.
  CyC Cochlea_R / Eye_R: Dmax < 33% Rx en todo val → V(D≥40%)=0): IC trivial, no un bug. No leer como precisión.

## A-eje — estado (2026-10-24)

- **A1 cerrado:** representación ganadora = **fill 0 + goals uniformes** (6 canales). La modulación por distancia (A1b)
  no aportó: la coherencia (SSIM) la da *tener* goals, no modularlos.
- **A2 y A3 diferidos** a la evaluación fuera de distribución (multi-protocolo / zero-shot): son intención estática por
  protocolo, redundantes en distribución.
- **A4 corrido (HO#8):** A1-uniform + Ch6 = PSDM de `PTV_High` (7 canales). `exp_a4_001_unet2d_psdm`, lanzado 2026-10-24
  con `scripts/run_a4.sh` (train + eval en val; test intacto). Todo lo demás congelado e idéntico a A1-uniform
  (hash `7ab16f73fa632a3e`, 189 épocas / patience 28, seed 42). Comparador: `scripts/compare_a4.py` (A4 vs A1-uniform vs A0).
  - **PSDM portado de próstata** (`calcular_psdm`, proyecto viejo): EDT(fuera) − EDT(dentro) en mm nativos, /10 → cm,
    /15 cm (`PSDM_NORM_CM`), signo + afuera / − adentro, sin clip. Próstata usaba 3 canales (PTV, Rectum, Bladder); A4 solo el del PTV.
  - **Multi-PTV:** PSDM de `PTV_High` (la unión cruda mis-ordenaba CyC). Padding in-plane del canal = 1.0 (no 0 = borde).
  - Código: `src/datamodules/a4_datamodule.py`; smoke 0 fallas en los 432 pacientes.
  - **Resultado (val, mejor ckpt epoch=105, early stop en 133; `results/exp_a4_001_unet2d_psdm/`, comparación en `results/exp_a4_comparison.json`):**
    PTV alta CyC 6.23 (A0) / 8.41 (A1u) / **7.00** (A4); SSIM CyC 0.375 / 0.608 / 0.571, Pelvis 0.656 / 0.781 / 0.815, Próstata 0.674 / 0.776 / 0.856;
    OAR media plana en los 3 sitios; MAE body Pelvis 3.35→2.49 y Próstata 1.32→0.92, CyC sin cambio.
  - **Bug de eval corregido:** `eval_a0.py` rellenaba el PSDM con 0 en vez de 1.0 (train sí usaba 1.0) → primer eval inválido
    (archivado en `results/exp_a4_001_unet2d_psdm_evalbug_pad0/`, no usar). Ahora el padding sale de `ds.pad_channels()`, único punto para train y eval.
  - **HO#9 cerrado — dist-solo (A0 + PSDM, 4 canales, sin goals):** `exp_a4_002_unet2d_distsolo`, mejor ckpt epoch=118, corrido con
    `scripts/run_a4solo.sh`. PSDM y padding idénticos a A4 (verificado: Ch0-2 = A0 y PSDM = Ch6 de A4, pad 1.0). Cierra el 2×2;
    comparador `scripts/compare_2x2.py` (regla de poda: dist-solo ≈ A4 en SSIM/PTV → se dropean los goals).
    **Sin piso de ruido (decisión 2026-10-25, §11):** la réplica A0 seed 43 se canceló y sus artefactos se borraron (ver `cabos_sueltos.md`); el veredicto de HO#9 se lee con el rango entre las 4 celdas + IC95 pareado, por sitio, y se declara sin piso.
  - **Resultado 2×2 (val, `results/exp_a4_2x2_comparison.json`; celdas A0 / A1u / A4 / dist-solo):**
    SSIM CyC 0.375/0.608/0.571/**0.476**, Pelvis 0.656/0.781/0.815/**0.735**, Próstata 0.674/0.776/0.856/**0.733** →
    dist-solo cae claro vs A4 en los 3 sitios (−0.10 / −0.08 / −0.12, IC95 sin 0; 41–68% del rango de las 4 celdas) pero sigue sobre A0 (+0.06 a +0.10).
    PTV alta CyC 6.23/8.41/7.00/6.13 (dist-solo vs A4 +0.87 [0.00,1.79]), Pelvis ≈ igual, Próstata peor sin goals (−0.95 [−1.83,−0.08]).
    OAR media plana; MAE body dist-solo ≈ A4.
  - **Veredicto de poda (sin piso de ruido, por sitio):** los goals **se quedan** — aportan coherencia (SSIM) más allá de la geometría en los 3 sitios;
    **A4 (goals + PSDM) es la representación del bloque intención/geometría.** Costo conocido: en CyC el PTV alta queda ~0.9 pp peor que sin goals
    (IC borde-cero). Siguiente: A5 (densidad).
  - Rama opcional (hecha en HO#9): A4 sobre A0 (sin goals) para atribuir crédito, solo si el primario muestra que A4 ayuda.

## Pendiente / abierto (Fase 2)

- Correr **HO#2** (train A0: config + U-Net 2D fija, MAE, escala de épocas `factor=N_train_ref/280`,
  `scheduler.max_epochs` al horizonte) y **HO#2b** (baseline). `configs/` apunta a `npz_v2b_body` +
  `split_v2.json`; registrar hash de tabla.
- **Gate del MAE body (barato, antes de leer el número):** verificar que `mask_BODY` se rasteriza del
  **External** del paciente, sin couch ni inmovilización — alimenta Ch2 **y** la métrica a la vez. El
  MAE body nace con v2 (contorno real) y **no es comparable** con ningún "body" de la era v1 (bbox).
- Padding en Z si se quiere `batch_size>1`.
- Rama 3D rota (`build_model_3d` ausente): no bloquea Fase 2, la necesita Fase 3.
- Ruido de punto flotante en `spacing_z_mm` de ~30 pacientes (3.0000004 en vez de 3.0): inocuo,
  no usar como clave.
