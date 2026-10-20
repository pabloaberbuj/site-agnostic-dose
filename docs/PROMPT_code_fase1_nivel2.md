# PROMPT Claude Code — Fase 1, Nivel 2: validación del LOOP por FINE-TUNING

> Pegar en Claude Code (corre en la PC de Pablo). Prerrequisito: **Nivel 1 cerrado y OK**
> (reprodujo la tabla con desviación máx 3e-4, `strict=True` sin keys faltantes/sobrantes).
> Leer en el repo nuevo: `docs/CHARTER.md`, `docs/ARRANQUE_fase1.md`, `docs/contexto_fase_vigente.md`,
> `docs/PROMPT_code_fase1_nivel1.md` (comparten gotchas y la ruta de eval que se reusa acá).
> Repo nuevo (destino): `C:\Pablo\SiteAgnostic\repo\site-agnostic-dose`
> Repo viejo (fuente a portar): `prostate-dose-prediction`.

---

## 0. Qué es esto (y qué NO es)

Nivel 2 valida la **única capa que Nivel 1 no ejerció: el loop de entrenamiento.** N1 confirmó
datos + forward + métrica; N2 confirma optimizer, scheduler cosine, early stopping, AMP 16-mixed,
gradient clipping, la rama de **gradient checkpointing** del `forward` (`if self.training` — nunca
corrió en inferencia), las **augmentations** (flip/jitter, solo activas en train) y que el backward
converge. Es lo que Fase 2 usa de forma intensiva: si hay un bug de loop latente, en Fase 2 no se
podría distinguir de "la representación de input no sirve".

**NO** es reproducir `epoch=014.ckpt` peso-a-peso. El entrenamiento es estocástico y el run viejo
corrió con `deterministic=False`; sumado a que acá corre sobre otra versión de librerías (venv
viejo, torch 2.12), la trayectoria **no** va a clonar bit a bit. El criterio es **"del mismo
orden"** (ver §6). **PORTAR VERBATIM, no refactorizar** — igual que N1.

---

## 1. Corrida de referencia y objetivo

**`exp_hipo_004_finetune_ctfix_v4`** — fine-tune full (sin freezing) desde el normo CT-corregido.

- **Init:** TODOS los pesos (state_dict) de
  `checkpoints/exp002_unet2d_psdm_ctfix_fov34/epoch=127.ckpt` (normo, CT corregido) vía
  `--init-weights`. **NO** `exp002_unet2d_psdm/epoch=191.ckpt` (ese es el normo viejo con CT
  corrupto). `--init-weights` carga solo state_dict con `strict=True`: el training arranca en epoch
  0, optimizer/scheduler/epoch frescos (NO es `--resume`).
- **Objetivo de convergencia (del run viejo):** early-stop en época ~77/207; **mejor
  `val/mae` ≈ 1.876** (época 15 del run viejo → `epoch=014.ckpt`). Re-evaluar ese mejor checkpoint
  en test debe volver a dar **MAE body ~2.55** (y la tabla de N1 a la par).

---

## 2. Precondiciones a verificar EN LA PC (parar y reportar si falla)

1. `checkpoints/exp002_unet2d_psdm_ctfix_fov34/epoch=127.ckpt` existe y abre (tiene `"state_dict"`).
2. Los ports de N1 ya están en el repo nuevo y verificados (lightning_module, dose_datamodule,
   losses, anatomy, unet2d con `build_model`, los yaml). NPZ `processed_hipo_ctfix/` + split
   `splits_hipo_ctfix_v4.json` ya validados en N1 (128/30/40, 201 NPZ) — no re-verificar.
3. GPU disponible (A2000 12GB). Costo esperado ~1 h.
4. Entorno: mismo venv del proyecto viejo que usó N1 (torch 2.12, PL 2.6.5) — tiene `wandb`
   instalado (importante para el import de los callbacks, ver §5.6).

---

## 3. Qué portar adicional (sobre lo de N1) y a dónde

| Fuente (repo viejo) | Destino (repo nuevo) | Notas |
|---|---|---|
| `scripts/train.py` | `scripts/train.py` | Verbatim (ajustar solo paths de import/salida si hace falta). Trae Trainer, callbacks, `--init-weights`, el parche de `torch.load(weights_only=False)` y el `trainer.test(ckpt_path="best")` final. |
| `src/callbacks/logging_callbacks.py` | `src/callbacks/logging_callbacks.py` | Verbatim (importa `wandb`/`matplotlib` a nivel módulo). Pero en la lista de callbacks, bajo `--no-wandb`, **dropear** `DVHLoggingCallback` y `SliceLoggingCallback` (ver §5.6). |

Lo demás (modelo, lightning, datamodule, loss, anatomy, config, split) ya está de N1. La ruta de
eval del scoring final (§6) reusa `scripts/evaluate_repro_nivel1.py` tal cual.

---

## 4. Config y comando

Config: `configs/exp_hipo_004_finetune_ctfix_v4.yaml` (ya portado en N1, verbatim con dead config).
Ya trae todo el escalado de horizonte: `scheduler.max_epochs: 207`, `training.max_epochs: 207`,
`early_stopping_patience: 62` (= round(100·265/128) y round(30·265/128), factor 2.0703).
**No recalcular ni ajustar esos números — usar el yaml tal cual.**

Comando (ajustar `--processed-dir` si el yaml no apunta a la ruta real de la PC):

```
python scripts/train.py \
  --config configs/exp_hipo_004_finetune_ctfix_v4.yaml \
  --init-weights checkpoints/exp002_unet2d_psdm_ctfix_fov34/epoch=127.ckpt \
  --no-wandb \
  --anatomy prostate_hipo        # (o el mecanismo que use load_experiment_config para inyectar el schema)
```

Tras el fit, scorear el mejor checkpoint con la ruta de N1:

```
python scripts/evaluate_repro_nivel1.py --config ... --checkpoint checkpoints/exp_hipo_004_finetune_ctfix_v4/<mejor epoch>.ckpt ...
```

(usar el mismo `evaluate_repro_nivel1.py` para que la tabla sea comparable con N1 y con el target).

---

## 5. Gotchas — REPLICAR, no "arreglar"

1. **`scheduler.warmup_epochs: 5` desconectado — EL gotcha de N2.** `configure_optimizers()` solo
   hace AdamW + CosineAnnealingLR. Si alguien "conecta" el warmup por prolijidad, cambia el
   schedule de LR y la trayectoria deja de ser comparable. Dejar desconectado.
2. **`scheduler.max_epochs` = 207, fijo al horizonte completo** (CHARTER §5), independiente de que
   el early stopping corte antes. Define la forma del coseno. NO bajarlo al epoch de corte esperado.
3. **Init desde `epoch=127.ckpt`** (exp002 normo **ctfix fov34**), NO `epoch=191` (CT corrupto).
   Full finetune sin freezing, vía `--init-weights` (state_dict, `strict=True`, epoch resetea a 0).
4. **Rama de gradient checkpointing** (`checkpoint(self.model, ...)` bajo `if self.training and
   torch.is_grad_enabled()`): se ejerce por primera vez en N2 (N1 era inferencia). El comentario
   del código la da por numéricamente idéntica al forward plano; validar que corre sin romper.
5. **Parche `torch.load(weights_only=False)`** para PyTorch 2.6+ (los ckpt traen `OmegaConf
   DictConfig`): ya está en `train.py`, aplica a `--init-weights` y a `trainer.test(ckpt_path="best")`.
   Portarlo. (N1 corrió en torch 2.12, así que hace falta.)
6. **Callbacks de logging bajo `--no-wandb`:** `logging_callbacks.py` hace `import wandb` a nivel
   módulo y `DVHLoggingCallback`/`SliceLoggingCallback` logean `wandb.Image` → con CSVLogger
   crashean en la 1ª época de validación que disparan (cada 5 / 10). **Dropearlos de la lista de
   callbacks** cuando se corre `--no-wandb`. Son observabilidad pura; no tocan la trayectoria.
   Mantener `ModelCheckpoint`, `EarlyStopping`, `LearningRateMonitor`, `EpochSummaryCallback`.
7. **`deterministic=False`** en el Trainer: replicarlo (el run viejo fue así). NO forzar
   `use_deterministic_algorithms` — cambiaría la trayectoria y no es lo que corrió el original.
8. `pl.seed_everything(42, workers=True)`; `torch.set_float32_matmul_precision('high')`; cap de
   VRAM `set_per_process_memory_fraction(0.88)` (override por env `PROSTATE_CUDA_MEM_FRACTION` si
   algún paciente de muchos cortes da OOM). Portar tal cual.
9. **Heredados de N1 (siguen):** `context_slices` dead → 5 canales, corte único; orden CT, BODY,
   PSDM_PTV, PSDM_Rectum, PSDM_Bladder; `anatomy: prostate_hipo` explícito; no recalcular D95.

---

## 6. Criterio de éxito (Nivel 2 — "del mismo orden", NO bit-exacto)

El loop se valida si:
1. **Corre de punta a punta sin crashear:** checkpoints y `val/mae` por época, LR siguiendo el
   coseno desde 1e-5 (verificar con `LearningRateMonitor`), early stopping activo (patience 62).
2. **Mejor `val/mae` en ~[1.85, 1.95]** (target 1.876).
3. **El early stopping efectivamente dispara** antes del epoch 207 (el run converge; no corre los
   207 planos). El epoch exacto de corte puede diferir del 77 del original — es esperable, no es
   señal de fallo.
4. **Re-evaluar el mejor checkpoint con `evaluate_repro_nivel1.py`** → **MAE body en ~[2.50, 2.60]**
   (target 2.552) y el resto de la tabla a la par.

## 7. Qué NO es señal de fallo (esperado por `deterministic=False` + versiones distintas)

- El epoch exacto de early-stop difiere del 77.
- `val/mae` difiere del 1.876 en el 2º/3er decimal (mientras caiga en la banda).
- La curva de entrenamiento no es idéntica corrida a corrida.

No "ajustar" nada para clonar los decimales del original.

## 8. Stop-and-ask (parar y avisar al chat)

- `--init-weights` no carga con `strict=True` (keys faltantes/sobrantes) — reportar las keys.
- Loss diverge / NaN, o `val/mae` nunca baja de ~2.0 (plateau alto) — probable bug de loop (LR,
  horizonte del scheduler, warmup conectado por error, init no cargado, grad-checkpointing roto).
- MAE body del mejor checkpoint fuera de ~[2.50, 2.60] de forma consistente.
- OOM no resuelto con el cap de VRAM / el env override.

## 9. Qué devolver al chat (handoff Code → chat)

Solo el delta: mejor `val/mae` + su epoch, epoch de early-stop, la **tabla de re-evaluación** del
mejor checkpoint (misma forma que N1/target), un chequeo de sanidad de la curva de LR, y un resumen
corto (convergió sí/no, en qué banda, cualquier gotcha que haya aparecido). Con N1 + N2 cierro la
fase: MD que congela Fase 1 (Tier 3) + arranque de Fase 2.
