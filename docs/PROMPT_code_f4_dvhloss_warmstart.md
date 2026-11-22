# Prompt Code — Fase 4, DVH-loss: warm-start condicional tras el gate (autónomo)

Para dejar corriendo solo. Define qué hacer según el resultado del gate de la corrida L1 from-scratch que
está activa, **sin pausar a preguntar**. Track de visualización (se evalúa contra la vara MAE, no con el
harness de calibración).

## Disparo condicional (al escribirse el resultado del gate ep40 del L1 from-scratch)

- **Si el gate PASA** (~10%): dejar terminar el from-scratch hasta el final del cosine (189) y evaluar como
  cualquier challenger (sección Evaluación). No lanzar warm-start.
- **Si el gate FALLA** (~90%): cerrar la cadena L1 from-scratch y **lanzar automáticamente el warm-start** de
  abajo. Registrar el from-scratch como **"no concluido por convergencia lenta"**, no como resultado clínico
  negativo — son cosas distintas (el gate mide `val/mae` temprano, no el criterio DVH sobre el modelo
  convergido).

## Warm-start — especificación

**Por qué:** el problema es converger el MAE *desde cero* mientras el término DVH le compite el gradiente;
que cuadrática y L1 arranquen igual de lento lo confirma (es del from-scratch, no de la forma). Partir del
óptimo MAE y dejar que la DVH-loss solo **refine** responde la pregunta real: sobre el óptimo MAE, ¿la
DVH-loss mueve el DVH (pooled y PTV-alta CyC) o no?

- **Init:** cargar pesos de `checkpoints/exp_a4_003_unet2d_psdm_c176x256/epoch=113.ckpt` (la vara).
  **Reiniciar** estado de optimizer y scheduler (es fine-tune con LR/horizonte nuevos, no un resume).
- **Loss:** `masked_mae_body + λ·L_dvh`, forma **L1** de los momentos, **mismo λ que la corrida L1 que está en
  el gate** (no introducir perilla nueva; si con ese λ no mejora, es resultado válido).
- **Optimizer / LR:** AdamW, weight decay del baseline. **LR cosine 1e-5 → 1e-6** (piso del cosine original;
  bajo para no destruir el warm-start). Nivel 2, anotar.
- **Horizonte:** `scheduler.max_epochs = 50` (horizonte del fine-tune, explícito). `trainer.max_epochs = 50`.
  Early stopping monitor `val/loss`, **patience 15** (más corto que el 28 del from-scratch).
- **SIN el gate de 1.4×-ep40:** no aplica — el warm-start arranca en `val/mae` ≈ 2.3, lo pasaría trivial. El
  gate existe para abortar barato corridas que no convergen; ésta ya arranca convergida.
- Resto idéntico a la vara: A4 (7 canales), `npz_v2b_body`, `split_v2.json`, armonización `7ab16f73fa632a3e`,
  U-Net 2D base16/depth4, contenedor (176,256), pred × BODY, seed 42, 16-mixed, batch=1.

## Selección de checkpoint

Monitor `val/loss` (compuesta), guardar top-3. Checkpoint elegido = **menor criterio DVH (pooled OAR banda
media 40–80) sobre el top-3**. Reportar ambos. (Regla de fase, igual para todos.)

## Evaluación (pipeline de la vara, misma val, pred × BODY)

Set completo por sitio con IC95 bootstrap: criterio DVH (pooled OAR banda media + OAR-media + PTV-alta
80–110), MAE-body, SSIM_BODY + SSIM BODY∩dosis>5%, d2. Comparar contra la vara `exp_a4_003`.

**Veredicto por la regla de corte pre-registrada de la fase** — gana si **las tres**:
1. mejora el pooled OAR banda media;
2. ningún sitio empeora con IC95 que excluya 0 (OAR-media o PTV-alta);
3. al menos un sitio mejora con IC95 que excluya 0.

Reportar además **MAE-body warm-start vs vara**: es el riesgo propio del warm-start (la DVH-loss puede empujar
el MAE fuera de su óptimo). Que sea warm-start no cambia la regla de corte — solo etiquetarlo así.

## Límites de autonomía (hacer esto y parar)

Una corrida de warm-start (o dejar terminar la from-scratch si pasó el gate), evaluarla, reportar el veredicto,
**parar y avisar**. **NO** por tu cuenta: probar λ=0.1, otra forma de loss, otro from-scratch, ni re-tunear λ
para pasar ningún umbral. Si el warm-start no gana, la conclusión limpia es "la DVH-loss no aporta sobre el
óptimo MAE" — eso cierra la candidata, no abre un sweep.

## Salida / handoff

`results/<exp_id>/` (sugerencia `exp_f4_001_dvhmoment_sp_L1_warmstart`): `metrics.csv`, `summary.json`.
**Sin commit.** Avisar al escribirse el resultado del gate y de nuevo al terminar la evaluación, con el delta:
pooled por sitio + OAR-media/PTV-alta con IC + MAE-body + SSIM, y el veredicto de la regla de corte.
