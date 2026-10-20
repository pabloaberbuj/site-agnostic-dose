# Fase 1 — CERRADA. Reproducción del baseline (des-riesgar el refactor)

> **Tier 3 — Archivo de fase congelado.** No se edita; lo que cambie va a fases posteriores.
> Cerrada: 2026-10-20. Leer con `CHARTER.md` (Tier 1) y `aprendizajes_transversales.md`.
> Prompts ejecutables: `docs/PROMPT_code_fase1_nivel1.md` y `docs/PROMPT_code_fase1_nivel2.md`.

---

## Resultado

**Reproducido: SÍ, en las dos capas.** El framework nuevo reproduce el pipeline de regresión de
dosis del proyecto viejo. Refactor des-riesgado para datos + forward + métrica (N1) y para el loop
de entrenamiento (N2). Corrida de referencia: `exp_hipo_004_finetune_ctfix_v4`, checkpoint
`epoch=014.ckpt`, sobre datos del **proyecto viejo** (`processed_hipo_ctfix`, 201 NPZ, FOV 34 cm,
D95(PTV)=100%, split `splits_hipo_ctfix_v4.json` 128/30/40).

## Nivel 1 — inferencia (desviación máx vs. target: 3e-4)

`epoch=014.ckpt` cargó con `strict=True` sin keys faltantes ni sobrantes. Test n=40, bootstrap 1000
seed 42.

| Métrica | Nuevo | Target |
|---|---|---|
| MAE body | 2.5520 (IC95 2.054–3.345) | 2.552 (2.05–3.35) |
| MAE PTV / Rectum / Bladder | 1.8044 / 6.1201 / 3.9753 | 1.804 / 6.120 / 3.976 |
| dose_score OpenKBP | 2.5520 | 2.552 |
| MAE pts PTV D95 / V70 | 0.347 / 1.451 | 0.350 / 1.460 |
| MAE pts Rectum V65 / Bladder V65 | 2.098 / 1.377 | 2.100 / 1.377 |

Detalle por paciente en `results/exp_hipo_004_finetune_ctfix_v4_repro_nivel1/`.

## Nivel 2 — fine-tuning (loop validado)

Init desde `exp002_unet2d_psdm_ctfix_fov34/epoch=127.ckpt` (normo ctfix), `strict=True`, full
finetune. Corrió de punta a punta, early stop activo, sin NaN.

- **Mejor `val/mae`: 1.8758 en época 14** (original: 1.876, época 14). Dentro de [1.85, 1.95].
- **Early stop: época 76** (original: 77).
- **LR:** 1e-5 → 7.03e-6, coseno T=207 sin warmup (dif. relativa máx 6e-16).
- `trainer.test(ckpt_path="best")`: MAE body 2.5526.
- Re-evaluación del mejor checkpoint con `evaluate_repro_nivel1.py`: tabla a la par de N1, todo
  **dentro de ±0.01 del target** (criterio del prompt era [2.50, 2.60]). Resultados en
  `results/exp_hipo_004_finetune_ctfix_v4_repro_nivel2/`.

La reproducción casi bit-exacta pese a `deterministic=False` es esperable (batch=1, workers
seedeados, init fijo, misma GPU) y la curva de LR (T=207) confirma entrenamiento fresco, no resume.

## `dvh_curva_completa` — primera corrida: la VARA para Fase 2

Métrica principal del CHARTER §2 (`mean|ΔV(D)|` en puntos porcentuales de volumen por banda de
dosis en % de Rx). exp_hipo_004 nunca la corrió → esto **establece la referencia**, no reproduce.
Metodología: dosis predicha clipeada a 0 y renormalizada a D95(PTV_pred)=100% antes del DVH; los 40
pacientes pasan los sanity checks. Bandas: baja 0–40, media 40–80, alta 80–110.

| Estructura | global | baja 0–40 | media 40–80 | alta 80–110 | dominante |
|---|---|---|---|---|---|
| PTV | 1.30 | 0.00 | 0.00 | 4.82 | alta |
| Rectum | 4.73 | 4.20 | 7.30 | 2.02 | media |
| Bladder | 2.79 | 3.23 | 3.34 | 1.46 | media |
| BODY | 0.94 | 1.86 | 0.57 | 0.16 | baja |

**Lectura:** en el PTV el error vive todo en la banda alta (coherente: el PTV solo tiene dosis
alta). En los OARs domina la **banda media 40–80%**, que el CHARTER marca como la crítica. **La vara
concreta a batir en Fase 2 es Rectum media 7.30 y Bladder media 3.34.** Detalle por paciente en
`results/exp_hipo_004_finetune_ctfix_v4_dvh_curva_completa/` y `dvh_fidelity_por_paciente.csv`.

## Gotchas de port confirmados (dead config — se replicaron, NO se "arreglaron")

- **`model.context_slices: 3` es dead config:** `_build_input` lee `cfg.data.context_slices` → cae
  a 1. El modelo es U-Net 2D de corte único, 5 canales. Orden: CT, BODY, PSDM_PTV, PSDM_Rectum,
  PSDM_Bladder.
- **`scheduler.warmup_epochs: 5` desconectado** en `configure_optimizers()` (solo AdamW + cosine).
- **`strict=True`** en ambos niveles (modelo nuevo = calco estructural del viejo).
- `anatomy: prostate_hipo` explícito; D95 no se recalcula (NPZ ya normalizado).

## Estado del repo nuevo al cierre

- **Portado verbatim** (verificado con `cmp` los .py): `lightning_module.py`, `dose_datamodule.py`,
  `losses.py`, `anatomy.py`, los dos yaml de anatomy, el yaml del experimento, el split.
- `build_model` agregada a `unet2d.py` (igual a la vieja). Scripts nuevos:
  `scripts/evaluate_repro_nivel1.py`, `scripts/dvh_curva_completa.py`, port de `scripts/train.py` y
  `src/callbacks/logging_callbacks.py`.
- Checkpoints y `lightning_logs/` en `.gitignore` (CHARTER §6: git = código + configs + docs).

## Dependencias y deuda que pasa a Fase 2 (no bloquearon Fase 1)

- **Entorno:** corrió sobre el venv del proyecto viejo (torch 2.12, PL 2.6.5). El repo nuevo aún no
  tiene `requirements` → primer insumo de infra de Fase 2.
- **Rama 3D rota:** `unet3d.py` no define `build_model_3d`; la rama `unet3d` de `build_model` no
  funciona. No afecta Fase 1; la necesita Fase 3 (B-eje).
- **GT de puntos DVH:** se usó `gt_dvh_hipo_256_ctfix_v4.csv` leído del repo viejo (único que cubre
  val+test del split v4), no copiado. No se arrastra: la métrica de record de Fase 2 es
  `dvh_curva_completa`, autocontenida (calcula V(D) directo de las dosis, sin CSV externo).
- **`dvh_curva_completa.py` adaptado, no verbatim:** mismas funciones de curva y bandas; cambió el
  origen de datos y se omitió el overlay de 3 pacientes (dependía de `analisis_angular.py` y otro
  split).
- **Operacional:** no lanzar corridas GPU en background con `&` (un doble lanzamiento contendió por
  GPU/checkpoint dir; anotado en su momento en el Tier 2).
