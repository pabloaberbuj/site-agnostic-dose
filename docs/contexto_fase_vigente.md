# Contexto Fase Vigente

> **Tier 2 — Documento vivo.** Se reescribe completamente al fin de cada fase.
> Refleja siempre "qué es verdad ahora", no la traza histórica.
>
> Última actualización: 2026-10-19 (arranque de Fase 1)

---

## Estado actual

- **Fase activa:** 1 (Reproducción del baseline) — **ARRANCADA, nada corrido todavía**.
  Es un **control de refactor**, no un experimento: confirmar que el framework nuevo reproduce
  el pipeline de **regresión de dosis** del proyecto de próstata antes de construir Fase 2 encima.
- **Subestado:** Nivel 1 (inferencia) con prompt listo para Code (`docs/PROMPT_code_fase1_nivel1.md`),
  sin correr. Nivel 2 (fine-tuning) condicional a que Nivel 1 pase.
- **Datos de ESTA fase: proyecto VIEJO, no los de Fase 0.** NPZ `processed_hipo_ctfix/` (201,
  FOV 34 cm, D95(PTV)=100%, 3 Status: 132 TreatmentApproved + 38 Rejected + 31 UnApproved),
  split `splits_hipo_ctfix_v4.json` (198, 128/30/40). Los artefactos de Fase 0 (`npz_v1_paso2/`,
  `split_v1.json`, normalización D95(PTV_High)) NO se tocan en Fase 1 — entran en Fase 2.

## Qué es Fase 1 (y qué NO es)

- **ES:** cargar el checkpoint entrenado del proyecto viejo en el framework nuevo y verificar que
  da los mismos números → des-riesga el port (datamodule, lightning module, modelo, loss, métrica).
- **NO es:** reproducir el objetivo (a) del proyecto viejo (clasificación de constraints puntuales,
  ML clásico, AUC) — fuera de alcance. NO optimiza arquitectura/loss. NO cambia cohorte ni
  normalización.
- **Principio rector:** PORTAR VERBATIM, incluidos los dead config. "Refactorizar prolijo" es el
  riesgo que esta fase existe para atrapar.

## Corrida de referencia y target

`exp_hipo_004_finetune_ctfix_v4`, checkpoint `epoch=014.ckpt`. Target
(`results/exp_hipo_004_finetune_ctfix_v4_test_hipo/metrics_summary.json`, test n=40,
bootstrap 1000 seed 42):

| Métrica | Valor |
|---|---|
| MAE body | 2.552 [IC95 2.05–3.35] |
| MAE PTV | 1.804 |
| MAE Rectum | 6.120 |
| MAE Bladder | 3.976 |
| dose_score OpenKBP | 2.552 |
| PTV D95 (MAE pts) | 0.350 |
| PTV V70 (MAE pts) | 1.460 |
| Rectum V65 (MAE pts) | 2.100 |
| Bladder V65 (MAE pts) | 1.377 |

## Plan — dos niveles, en orden

- **Nivel 1 — INFERENCIA (control principal):** cargar `epoch=014.ckpt`, forward sobre el test de
  40, reproducir la tabla. No reentrena. Aísla datos + forward (lo que Fase 2 usa intensivo).
  Criterio: MAE body dentro de ~0.01–0.02 de 2.552. Detalle ejecutable en
  `docs/PROMPT_code_fase1_nivel1.md`.
- **Nivel 2 — FINE-TUNING (solo si N1 pasa):** init normo `epoch=127.ckpt`, fine-tune sobre el
  split de 198 (lr 1e-5, horizonte 207 / patience 62), valida el loop (optimizer, cosine, early
  stop, AMP). Criterio "del mismo orden" (val/mae ~1.88–1.90, test ~2.55), no idéntico al decimal.
  El prompt de N2 se escribe recién tras cerrar N1.

## Gotchas del port (verificados en el código viejo — REPLICAR, no arreglar)

1. **`model.context_slices: 3` es dead config:** `_build_input` lee `cfg.data.context_slices` →
   cae al default 1. El modelo es U-Net 2D de **corte único, 5 canales** (no 2.5D). No conectarlo
   ni moverlo a `data:`. (El `predict_one.py` viejo documenta la trampa.)
2. **`scheduler.warmup_epochs: 5` desconectado** en `configure_optimizers()` (solo AdamW +
   CosineAnnealingLR). Dejar sin cablear (importa en N2).
3. **Orden de 5 canales:** CT, BODY, PSDM_PTV, PSDM_Rectum, PSDM_Bladder (del orden
   `[ptv, rectum, bladder]` del anatomy schema). Reorden silencioso = bug más probable.
4. **`anatomy: prostate_hipo` explícito** (fallback va a `prostate` normo).
5. **Carga `strict=True`:** la `UNet2D` nueva (profiling) es calco estructural de la vieja →
   state_dict idénticos en keys/shapes. No pasar a `strict=False`.
6. **No recalcular la normalización:** el NPZ viejo trae `dose` ya a D95(PTV)=100% (+ `factor_norm`
   en meta). Usarla tal cual.

## Qué se porta y a dónde

Ruta **legacy/reproducción**: NPZ viejo + datamodule viejo. Mapa fuente→destino en
`docs/PROMPT_code_fase1_nivel1.md` §3. Reusa `src/models/unet2d.py` (ya está, es calco — solo
agregar la factory `build_model`). El datamodule site-agnostic (NPZ geometry-only + goals-al-cargar
+ canal de intención) es Fase 2, no se toca acá.

## Entregable extra

Primer `dvh_curva_completa` sobre las predicciones de N1 (banda media 40–80 % crítica para OARs).
exp_hipo_004 nunca lo corrió → **establece la vara** para Fase 2, no reproduce un número previo.

## Pendiente / abierto (Fase 1)

- Verificar en la PC que los NPZ del test-40 existen; si no, reprocesar con el pipeline **VIEJO**
  (`data/preprocess_hipo.py`), nunca con el de site-agnostic (filtraría Rejected/UnApproved y
  cambiaría la cohorte).
- Línea del CHARTER §4 "Portar exp002": el target real de Fase 1 es el fine-tune hipo
  `exp_hipo_004` (exp002 refiere a la arquitectura/pipeline). Ajustar la redacción al cierre de fase.

## Métricas de éxito de Fase

- **Nivel 1:** MAE body dentro de ~0.01–0.02 de 2.552 y resto de la tabla a la par. Desviación
  > ~0.05 = bug de port → triage sin GPU (prompt §9: correr la métrica nueva sobre las predicciones
  guardadas de exp_hipo_004, para localizar si falló la métrica o el forward) ANTES de reportar.
- **Nivel 2 (si corre):** converge a val/mae ~1.88–1.90, test vuelve a ~2.55.
- **Entregable:** primer `dvh_curva_completa` registrado (por estructura y por banda).

## Notas vivas

**2026-10-19 — Nivel 1 CORRIDO: REPRODUCE.** Precondiciones OK (ckpt abre, 201 OK / 29 ERROR en el
log, split 128/30/40, 201 NPZ en `processed_hipo_ctfix`, sin reprocesar). Composición del split de 198:
131 TreatmentApproved / 36 Rejected / 31 UnApproved (los 3 excluidos completan 132/38/31 de los 201).
`load_state_dict(strict=True)` limpio, sin missing/unexpected keys.

| Métrica | Nuevo | Target |
|---|---|---|
| MAE body | 2.5520 (IC95 2.054–3.345) | 2.552 (2.05–3.35) |
| MAE PTV / Rectum / Bladder | 1.8038 / 6.1201 / 3.9757 | 1.804 / 6.120 / 3.976 |
| dose_score OpenKBP | 2.5520 | 2.552 |
| MAE pts PTV D95 / V70 | 0.3504 / 1.4604 | 0.350 / 1.460 |
| MAE pts Rectum V65 / Bladder V65 | 2.0996 / 1.3774 | 2.100 / 1.377 |

Desviación máx. 3e-4 → triage §9 no necesario. Salidas en
`results/exp_hipo_004_finetune_ctfix_v4_repro_nivel1/` (`metrics_repro_nivel1.json`, CSV por paciente).

**Primer `dvh_curva_completa` (la vara para Fase 2)**, mean|ΔV| en pp, test n=40, pred clip a 0 y
renormalizada a D95(PTV_pred)=100%, 0 sanity checks fallidos
(`results/exp_hipo_004_finetune_ctfix_v4_dvh_curva_completa/`):

| Estructura | global | baja 0–40 | media 40–80 | alta 80–110 |
|---|---|---|---|---|
| PTV | 1.30 | 0.00 | 0.00 | 4.82 |
| Rectum | 4.73 | 4.20 | **7.30** | 2.02 |
| Bladder | 2.79 | 3.23 | **3.34** | 1.46 |
| BODY | 0.94 | 1.86 | 0.57 | 0.16 |

Detalles del port:
- GT de puntos DVH: `gt_dvh_hipo_256_ctfix_v4.csv` del repo viejo (el único que cubre val+test del split v4;
  `gt_dvh_hipo_256.csv` cubre solo 30/40 del test). Se lee desde el repo viejo, no está copiado.
- Se corre con el python del venv viejo (`ProstateDoseProject/repo/.venv`: torch 2.12+cu130, PL 2.6.5,
  omegaconf 2.3.1). El repo nuevo todavía no tiene requirements/venv propio.
- `src/models/unet3d.py` del repo nuevo no define `build_model_3d`; la rama `arch=="unet3d"` de
  `build_model` queda portada pero no sirve hasta que se agregue (irrelevante en Fase 1).
- `dvh_curva_completa.py` se adaptó (el viejo dependía de `analisis_angular.py` y del split v2): mismas
  funciones de curva/bandas, datos vía `DoseDataModule`. Se omitió el plot de overlay de 3 pacientes.
- Nada commiteado todavía.

**2026-10-19 — Nivel 2 CORRIDO: LOOP VALIDADO.** `scripts/train.py` + `logging_callbacks.py` portados
verbatim (único cambio: bajo `--no-wandb` se dropean DVHLoggingCallback/SliceLoggingCallback, §5.6 del
prompt). Init desde `exp002_unet2d_psdm_ctfix_fov34/epoch=127.ckpt` con `strict=True` OK (epoch 0 fresco).
- Mejor `val/mae` = **1.8758 en la época 14** (target 1.876 / época 14 del original; banda [1.85, 1.95]).
- Early stop en la época **76** (original 77; patience 62) — disparó antes de 207.
- LR: 1e-5 → 7.03e-6, coincide con cosine T=207 sin warmup (dif. rel. máx. 6e-16). Sin NaN.
- Test de `trainer.test(ckpt_path="best")`: MAE body 2.5526.
- Re-evaluación del mejor ckpt con `evaluate_repro_nivel1.py` (`results/exp_hipo_004_finetune_ctfix_v4_repro_nivel2/`):
  MAE body 2.5520 (IC95 2.054–3.345), PTV 1.8044, Rectum 6.1201, Bladder 3.9753, dose_score 2.5520;
  pts DVH PTV D95 0.347, PTV V70 1.451, Rectum V65 2.098, Bladder V65 1.377 — todo dentro de ±0.01 del target.
- Gotcha operativo: el primer lanzamiento (con `&` en un comando en background) no murió y quedó
  corriendo en paralelo con el segundo → dos procesos pisando GPU y checkpoints. Se mataron, se limpió
  `checkpoints/` y `lightning_logs/` y se relanzó UNA sola corrida limpia. Lanzar background sin `&`.
- Checkpoints del run en `checkpoints/exp_hipo_004_finetune_ctfix_v4/` (gitignored); logs en `lightning_logs/`.

**Fase 1 lista para cierre** (N1 + N2 OK). Pendiente: handoff al chat y MDs de cierre/arranque Fase 2.
