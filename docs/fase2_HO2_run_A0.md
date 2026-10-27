# HO#2 — Fase 2: run de A0 + harness por sitio/bootstrap

> Prompt para Claude Code. Leer con `CHARTER.md` (§2 métricas, §4 A0, §5 congeladas),
> `ARRANQUE_fase2.md`, `fase1_cerrada.md`.
> **Alcance:** entrenar el primer site-agnostic en la config A0 (geometría + intención, SIN goals),
> MAE puro, U-Net 2D fija, y evaluarlo con el harness por sitio contra la baseline no-aprendida
> (HO#2b, en paralelo). A0 todavía NO toca goals (A1), augmentations (A2) ni prioridad (A3).

## Inputs ya montados (HO#1)

- Datamodule-A0 (smoke test OK, 432 pacientes 0 fallas). Datos: `data/npz_v2b_body/`,
  split `data/splits/split_v2.json`. Pad in-plane **256**. Hash de tabla **`2a0c2037686c2978`**.
- Cohorte: 280 train / 65 val / 87 test. Por sitio: Próstata 138, Pelvis 138, CyC 156.

## Antes de codear — reusar, no re-forkear

Confirmar del pipeline de Fase 1 (dose U-Net portada, linaje exp002): la arquitectura, el
optimizer, el LR scheduler y **`N_train` de referencia** + horizonte de épocas de la corrida que
reprodujo. Reusar ese módulo de modelo; A0 solo cambia los canales de entrada (los 3 de A0) y el N.

## Modelo + train (A0)

- **Un solo modelo site-agnostic pooled** sobre los 280 train con los 3 sitios mezclados; se
  reporta por sitio, pero se entrena uno solo (es el punto de la fase).
- Entrada: **3 canales de A0** (intención/PTV con ratios, OAR consolidado, body). Target: dosis
  normalizada del NPZ. Loss: **MAE puro** (§7: en próstata MAE ganó; las DVH-loss van en Fase 4).
- Congeladas (§5): sampling por paciente (ya en el datamodule), GroupNorm, bilinear upsample +
  conv 3×3. No tocar acá.
- **Escala de épocas por pasos de gradiente (§5):** `factor = N_train_ref / 280` (batch=1),
  aplicado al horizonte. `scheduler.max_epochs` fijo al horizonte completo, independiente de
  `trainer.max_epochs`, para que cortes tempranos sean comparables en LR. Registrar el factor usado.
- Selección de checkpoint: mejor val.

## Harness de evaluación — módulo compartido `src/eval/`

Portar `dvh_curva_completa` de Fase 1 y envolverlo con lo nuevo de este proyecto. **Compartido con
la baseline (HO#2b):** montarlo como módulo importable; el primero de los dos handoffs que corra lo
crea, el otro lo importa.

- **Métrica principal:** `mean|ΔV(D)|` por banda de dosis, por estructura, con el eje de dosis en
  **% de Rx_High**. Resaltar la **banda media 40–80% para OARs** (la crítica) y la **banda alta para
  PTV** (ahí concentra su error).
- **Reporte por sitio siempre**, con **IC95 bootstrap resampleando pacientes dentro de cada sitio**.
  El MAE global es secundario (promedia y esconde, §2).
- DVH computada sobre **volumen nativo** (higiene de vóxel, §5/§7), no sobre grilla downsampleada.
- **Secundarias de monitoreo (solo red, no baseline):** MAE body, dose_score OpenKBP, y una métrica
  de coherencia espacial SSIM/MS-SSIM. Se reportan, no son el norte.

## Lectura de A0

- **Comparación real: A0 vs baseline no-aprendida, por sitio** (¿la red aprende algo más que la
  forma típica del protocolo? §2).
- **Driver del sweep A0→A6: val.** Reservar **test** para el cierre del eje A, para no filtrar test
  a través de las decisiones incrementales de ablación. *(Default sensato; si querés otra cosa, se
  cambia — ver nota al handoff.)*
- La vara de Fase 1 (Rectum 7.30, Bladder 3.34 banda media; PTV banda alta 4.82) es **datos/norma
  viejos** → referencia de orden de magnitud y de "qué banda duele", **no** umbral a igualar.

## Record del experimento

`exp_id`, hash de tabla `2a0c2037686c2978`, datos `npz_v2b_body`, `split_v2.json`, pad 256, factor
de escala de épocas usado.

## Fuera de alcance

Goals/A1+, augmentations, prioridad, distance map, densidad, beam plate, 3D (Fase 3).

## Handoff de vuelta (§8)

Solo el delta: `metrics.csv` + `summary.json` por sitio (con IC95) + comparación vs baseline +
resumen corto. No el doc.
