# PROMPT Claude Code — Fase 1, Nivel 1: reproducción por INFERENCIA

> Pegar en Claude Code (corre en la PC de Pablo, con acceso a ambos repos y a los datos pesados).
> Leer primero, en el repo nuevo: `docs/CHARTER.md` (Tier 1), `docs/ARRANQUE_fase1.md`,
> `docs/aprendizajes_transversales.md` (Tier 3).
> Repo nuevo (destino): `C:\Pablo\SiteAgnostic\repo\site-agnostic-dose`
> Repo viejo (fuente a portar): `prostate-dose-prediction` (próstata, completo).

---

## 0. Qué es esto (y qué NO es)

Fase 1 es un **control**, no un experimento: confirmar que el framework nuevo reproduce el pipeline
de **regresión de dosis** del proyecto viejo, para des-riesgar el refactor antes de Fase 2. Nivel 1
= **inferencia pura**: cargar el checkpoint entrenado del proyecto viejo en el framework nuevo,
correr forward sobre el test de 40, y verificar que da los mismos números que
`metrics_summary.json`. NO se reentrena nada. NO se toca la cohorte ni la normalización. NO se
porta la capa de clasificación de constraints / AUC (ese es el objetivo (a) del proyecto viejo,
fuera de Fase 1).

**Principio rector: PORTAR VERBATIM, no refactorizar.** Copiar el comportamiento real del proyecto
viejo, incluidos sus "dead config" (ver §4). Cualquier "prolijidad" que cambie el grafo de cómputo
o el orden de canales rompe la reproducción y es exactamente lo que este nivel existe para atrapar.

---

## 1. Corrida de referencia y target numérico

**`exp_hipo_004_finetune_ctfix_v4`**, checkpoint `epoch=014.ckpt` (mejor val/mae, early-stop en
época 77/207). Config: `configs/exp_hipo_004_finetune_ctfix_v4.yaml` (repo viejo).

Target (de `results/exp_hipo_004_finetune_ctfix_v4_test_hipo/metrics_summary.json`, test n=40,
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

---

## 2. Precondiciones a verificar EN LA PC (parar y reportar si alguna falla)

1. `checkpoints/exp_hipo_004_finetune_ctfix_v4/epoch=014.ckpt` existe y abre
   (`torch.load(..., map_location="cpu")` sin error; tiene clave `"state_dict"`).
2. NPZ del test de 40 presentes en `processed_hipo_ctfix/` (los `anonid` de `splits['test']`).
   Si NO están (hay que reprocesar): reprocesar con el **pipeline VIEJO** (`data/preprocess_hipo.py`,
   FOV 34 cm, 3 Status, D95(PTV)=100%) — **nunca** con el preprocesador de site-agnostic (filtraría
   Rejected/UnApproved y cambiaría la cohorte). Log de referencia: `preprocess_hipo_log.json`
   (201 OK, 29 ERROR="no encontrado"). Confirmar 201 OK / 198 en split / composición por Status
   (132 TreatmentApproved, 38 Rejected, 31 UnApproved) antes de seguir.
3. `data/splits/splits_hipo_ctfix_v4.json` presente (198 pac, 128/30/40).
4. `metrics_summary.json` de la corrida vieja presente (target de verificación).

---

## 3. Qué portar y a dónde (verbatim desde el repo viejo)

| Fuente (repo viejo) | Destino (repo nuevo) | Notas |
|---|---|---|
| `src/models/unet2d.py` | **YA ESTÁ** en `src/models/unet2d.py` | Calco verificado. Solo falta la factory `build_model(cfg)` — agregarla igual a la vieja. NO tocar las clases. |
| `src/models/lightning_module.py` | `src/models/lightning_module.py` | Verbatim. Trae `_build_input` (orden de canales), `_calcular_metricas`, `_calcular_dvh_score`, `configure_optimizers`. |
| `src/datamodules/dose_datamodule.py` | `src/datamodules/dose_datamodule.py` | Verbatim. Lee NPZ viejo (keys `ct`,`dose`,`{s}_mask`,`psdm_{s}`,`meta`). |
| `src/losses/losses.py` | `src/losses/losses.py` | `CombinedLoss` (MAE puro: `use_moment_loss=false`). |
| `src/config/anatomy.py` + `configs/anatomy/{prostate,prostate_hipo}.yaml` | idem | Sistema de anatomy schema + `load_experiment_config`. |
| `scripts/evaluate_hipo.py` → **solo** `evaluar_dataset`, `mae_en_mascara`, `calcular_puntos_dvh_pred`, `bootstrap_ci`, y lo mínimo de carga | `scripts/evaluate_repro_nivel1.py` (nuevo, recortado) | Ver §6. NO portar: OPERATIONAL_CONSTRAINTS, capas 2/3 (AUC), NArcos, punto de operación. |
| `scripts/evaluate.py` → `dose_score_openkbp`, `cargar_modelo` | lo que importe el de arriba | — |
| `scripts/dvh_curva_completa.py` | `scripts/dvh_curva_completa.py` | Para el entregable extra (§8). |
| `configs/exp_hipo_004_finetune_ctfix_v4.yaml` | `configs/exp_hipo_004_finetune_ctfix_v4.yaml` | Verbatim, incluidos los dead config (§4). |

Esta es la ruta **legacy/reproducción**: NPZ viejo + datamodule viejo. El datamodule site-agnostic
(NPZ geometry-only + goals-al-cargar + canal de intención) es Fase 2 — no se toca acá.

---

## 4. Gotchas críticos — REPLICAR, no "arreglar"

1. **`context_slices` es dead config.** En el yaml vive bajo `model:`, pero `_build_input` lee
   `cfg.data.context_slices` → cae al default **1**. El modelo es U-Net 2D de **corte único, 5
   canales**. NO conectar context_slices, NO moverlo a `data:`, NO interpretar el `3` del yaml.
   (El `predict_one.py` viejo documenta esta trampa.)
2. **`scheduler.warmup_epochs: 5` desconectado.** `configure_optimizers()` solo hace AdamW +
   CosineAnnealingLR. Dejar el warmup sin cablear (importa en Nivel 2).
3. **`anatomy: prostate_hipo` explícito.** Pasar `--anatomy prostate_hipo` (o equivalente) para
   que `load_experiment_config` inyecte `cfg.anatomy_schema`. Si queda `None`, el fallback va a
   `prostate` (normo): para inferencia da el mismo orden de canales y `primary_target`, pero
   respetarlo igual.
4. **Orden de canales exacto:** CT, BODY, PSDM_PTV, PSDM_Rectum, PSDM_Bladder — derivado del orden
   `[ptv, rectum, bladder]` del anatomy schema. Un reorden silencioso es el bug más probable y es
   justo lo que el MAE≈2.55 detecta.
5. **No recalcular la normalización.** El NPZ viejo trae `dose` ya normalizada a D95(PTV)=100% (más
   `factor_norm` en `meta`). Usarla tal cual; el framework nuevo no debe re-derivar D95.

---

## 5. Carga del checkpoint

El `.ckpt` es un checkpoint de Lightning de `DosePredictionModule` (la U-Net vive en `self.model`,
así que los keys del state_dict están prefijados con `model.`). Recipe:

```python
cfg = load_experiment_config("configs/exp_hipo_004_finetune_ctfix_v4.yaml")  # inyecta anatomy_schema
module = DosePredictionModule(cfg)
sd = torch.load("checkpoints/.../epoch=014.ckpt", map_location="cpu")["state_dict"]
missing, unexpected = module.load_state_dict(sd, strict=True)   # DEBE ser strict=True
assert not missing and not unexpected, (missing, unexpected)
module.eval()
```

Si `strict=True` tira `missing`/`unexpected keys` → hay divergencia estructural en el port del
modelo/lightning. **Parar y reportar** las keys exactas; no pasar a `strict=False`.

---

## 6. Evaluación (ruta recortada, solo regresión de dosis)

Escribir `scripts/evaluate_repro_nivel1.py` que replique la lógica de `evaluar_dataset` del viejo,
recortada a lo que produce la tabla target:

- Por paciente del test (n=40), forward en modo eval (sin gradient checkpointing — la rama de
  `forward()` con checkpoint solo corre en training; replicar).
- Por paciente: `mae_body` y `mae_{ptv,rectum,bladder}` vía `mae_en_mascara` (en % de Rx);
  `dose_score_openkbp`; puntos DVH predichos vs. GT vía `calcular_puntos_dvh_pred` (PTV D95,
  PTV V70, Rectum V65, Bladder V65 → MAE de puntos).
- Agregar: media sobre los 40 + `bootstrap_ci(valores, n_boot=1000, rng=default_rng(42))`.
- Volcar a `results/<exp>/metrics_repro_nivel1.json` con el mismo layout que el target.

GT de los puntos DVH: usar el mismo origen que el viejo (`compute_gt_dvh_hipo.py` / el CSV de GT
que `evaluate_hipo.py` carga). Reproducir su fuente, no recalcular con otra definición.

---

## 7. Criterio de éxito (Nivel 1)

- **MAE body dentro de ~0.01–0.02 de 2.552** (debería ser prácticamente idéntico; única fuente de
  diferencia = numérica: orden de reducción, versión de librería).
- El resto de la tabla (PTV 1.804, Rectum 6.120, Bladder 3.976, dose_score 2.552, puntos DVH) a la
  par.
- Si MAE body se desvía **> ~0.05** → hay bug de port. Ir al triage (§9) ANTES de reportar, nunca
  "ajustar" para que dé.

---

## 8. Entregable extra barato (recomendado): primer `dvh_curva_completa`

Sobre las predicciones del Nivel 1, correr además `dvh_curva_completa.py` (métrica principal del
CHARTER §2: `mean|ΔV(D)|` por banda; bins baja 0–40 / **media 40–80 (crítica para OARs)** / alta
80–110). exp_hipo_004 nunca la corrió → esto **establece la vara** para Fase 2, no reproduce un
número previo. Registrar los valores por estructura y por banda en el MD de cierre.

---

## 9. Triage si Nivel 1 NO reproduce (gratis, sin GPU — hacer ANTES de reportar "no da")

Correr el código de métrica **nuevo** (`evaluate_repro_nivel1.py`) sobre las **predicciones
guardadas de exp_hipo_004** (las dosis predichas del proyecto viejo, no re-inferidas):
- Si **da 2.552** → la métrica portó bien; el bug está en forward/datamodule (canal cruzado,
  normalización, eje, orden de slices). Mirar `_build_input` y la carga del NPZ.
- Si **tampoco da** → el bug está en el port de la métrica. Mirar `mae_en_mascara` /
  `calcular_puntos_dvh_pred` / el GT.

Esto localiza la capa antes de abrir la stop-rule.

---

## 10. Stop-and-ask (parar y avisar al chat)

- `strict=True` no carga limpio (§5).
- NPZ reprocesado no da 201 OK / 198 en split, o cambia la composición por Status.
- MAE body se desvía > ~0.05 y el triage (§9) no lo explica como numérico.

## 11. Qué devolver al chat (handoff Code → chat)

Solo el **delta**, no el doc entero: `metrics_repro_nivel1.json` + los valores de
`dvh_curva_completa` + un resumen corto (reprodujo sí/no, desviación en MAE body, keys faltantes si
las hubo). Con eso el chat genera el MD de cierre de Fase 1 (Tier 3) y el de arranque de Fase 2.
