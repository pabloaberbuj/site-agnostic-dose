# HO#3 — Fase 2: despejar flags baratos post-A0

> Prompt para Claude Code. Leer con `contexto_fase_vigente.md` (Tier 2) y `CHARTER.md` (§2, §5, §7).
> **Alcance:** dos items de higiene de reporte detectados al leer A0. **Ninguno requiere reentrenar** —
> el checkpoint `checkpoints/exp_a0_001_unet2d/epoch=097.ckpt` queda intacto y el harness `src/eval/`
> no cambia de contrato. Al cerrarlos pasamos a A1.

> **Nota — el chequeo de `mask_BODY` NO va.** `mask_BODY` = la External que usó Eclipse para calcular
> (incluye inmovilizadores en algunos casos, y eso es correcto: la dosis se calculó sobre ese volumen,
> es física que el Ch2 tiene que ver). No hay contaminación que limpiar. Única consecuencia, y es solo
> interpretativa: el **MAE body** como secundaria incluye vóxeles de soporte en fracción variable por
> paciente -> no sobre-interpretar comparaciones de MAE body entre pacientes/sitios. La métrica que
> manda (DVH de OAR/PTV) no toca el body.

## Flag A — `dose_score_openkbp` redundante con MAE body

Hoy `dose_score_openkbp` da idéntico al MAE body (misma dose-mask = body) -> no es una segunda señal.

- Confirmar que se computa sobre la misma máscara que el MAE body.
- Resolver, lo más barato aceptable primero: **(a)** sacarlo del reporte, o relabelarlo para que no
  se lea como métrica independiente; **(b)** solo si querés un número comparable a OpenKBP de verdad,
  computarlo sobre la *possible-dose mask* de la convención OpenKBP (no el body), que sí sería una
  cantidad distinta. Sin reentrenar.

## Flag B — IC por estructura degenerados (intervalos-punto)

Varios cells per-estructura tienen `ci95` colapsado a un punto, p.ej. CyC `Cochlea_R` media_40_80
`4.139 [4.139, 4.139]` (n=21) y `Eye_R` media `0.809 [0.809, 0.809]`.

- Diagnosticar en 2–3 cells la causa: **varianza entre pacientes = 0** (la estructura no recibe dosis
  en esa banda -> true/baseline/red constantes -> diff constante -> IC honesto pero trivial),
  `n_efectivo = 1`, o un **bug** del resampleo. (Hipótesis principal: el primer caso, no un bug.)
- Según la causa: si es varianza-cero/trivial, **anotar el output** (flag, o `n_efectivo` / varianza)
  para que no se lean como estimaciones precisas; si es bug, arreglar el bootstrap.
- Confirmar en cualquier caso que los agregados `OAR_all` / `PTV_all` **no** están afectados (tienen
  IC anchos y sanos, n=21–23). Son los que mandan la lectura.
- Solo eval/reporte, sin reentrenar.

## Handoff de vuelta (§8)

Delta: hallazgos de cada flag + cambios de reporte si los hubo. Con esto cerrado, el siguiente
handoff es A1.
