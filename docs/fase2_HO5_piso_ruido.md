# HO#5 — Fase 2: piso de ruido seed-a-seed (A0)

> Prompt para Claude Code. Leer con `contexto_fase_vigente.md` (Tier 2) y `CHARTER.md` (§2).
> **Independiente de HO#6.** No es una ablación — es instrumentación: estimar cuánta de la
> variación que leemos entre configs es ruido de entrenamiento (seed) y no efecto real.

## Por qué

Todas las corridas del eje A tienen **un seed por config**. El bootstrap por paciente captura la
varianza de muestreo de pacientes, pero NO la de reinicialización + orden de batches. Sin esa vara,
diferencias de DVH-OAR de ~0.1 pp entre A0 y A1a (y entre fills) no son interpretables. El efecto que
manda en A1a (SSIM) está muy por encima de cualquier ruido, así que esto NO reabre A1a — es para poder
leer las ablaciones que vienen.

## Qué correr

- **Re-entrenar A0 con 3 seeds distintos** (p.ej. 43, 44, 45; el original fue 42 → quedan 4 en total).
  Config **idéntica** a `exp_a0_001_unet2d` salvo `experiment.seed`. Misma arquitectura, datos
  (`npz_v2b_body` + `split_v2.json`), hash `2a0c2037686c2978`, 6... **3 canales** (A0, NO los 6 de A1),
  epoch-scaling, early stopping, todo igual.
- Evaluar cada uno en **val** con el mismo harness. Test intacto.

## Qué reportar

- Por sitio, la **desviación estándar entre seeds** (4 valores: seed 42 + los 3 nuevos) de:
  `OAR_all.media_40_80`, `PTV_all.alta_80_110`, y `SSIM`. Es el piso de ruido por métrica.
- Una línea de lectura: ¿las diferencias A0↔A1a que vimos (OAR ~0.08, PTV CyC ~2.2, SSIM ~0.23) caen
  dentro o fuera de ese piso? (Esperado: OAR dentro = ruido; PTV y SSIM fuera = señal.)

No hace falta tocar checkpoints viejos ni la baseline. Solo 3 entrenamientos de A0 + eval.

## Handoff de vuelta (§8)

Delta: std entre seeds por sitio/métrica + la línea de lectura. Queda como vara de ruido para el resto
del eje A.
