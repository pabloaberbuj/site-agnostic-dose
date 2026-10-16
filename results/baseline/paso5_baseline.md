# Paso 5 — Splits y baseline no-aprendida (cierra Fase 0)

> Scripts: `scripts/make_splits.py`, `scripts/baseline_dvh_promedio.py`. Splits en
> `data/splits/split_v1.json` (va a git, solo `patient_id`). Baseline por sitio en
> `results/baseline/baseline_dvh_<sitio>.json`.

## Splits (65/15/20 estratificado por sitio, seed=42)

| sitio | NPZ | out_of_convention | elegibles | train | val | test |
|---|---|---|---|---|---|---|
| CyC | 196 | 40 | 156 | 101 | 23 | 32 |
| Pelvis | 161 | 23 | 138 | 90 | 21 | 27 |
| Prostata | 146 | 7 | 139 | 90 | 21 | 28 |
| **total** | **503** | **70** | **433** | **281** | **65** | **87** |

**Nota CHARTER §2** ("piso operativo 100 pacientes/sitio, apuntar a 200 donde sea
recuperable"): CyC train=101 cumple el piso; Pelvis y Prostata quedan en 90 (debajo del
piso operativo) — consecuencia directa del tamaño del `out_of_convention` de cada sitio
(23 y 7 respectivamente) restado del total exportado. No es un error de split, es el
dato real disponible; queda como limitación conocida a reportar en cualquier resultado
de Fase 1+ sobre esos 2 sitios.

`out_of_convention` NUNCA entra a ningún split (conjunto reservado para evaluar
generalización, CHARTER). `excluded` (4 pacientes) ni siquiera tiene NPZ.

## Baseline DVH promedio (solo train, dosis en % Rx_High)

Calculada dinámicamente por rol presente en los NPZ de train — ningún rol hardcodeado.
Formato completo (curva 0-115% Rx_High, paso 1%) en los JSON; acá V50%/V80% como
lectura rápida de sanity check.

**CyC (N=101, 36 roles):** PTV_High/Mid/Low ≈100%/100%/88-100% en V80% (como se
espera de un target). OARs seriales cerca de PTV (`SpinalCord_PRV05`, `OpticChiasm_PRV3`,
`Brainstem_PRV05`) con V80% bajo (0.1-2.5%) — consistente con constraints de Dmax que
sí se respetan en la mayoría de los planes. `Lobe_Temporal_PRV3_L/R` y `Pituitary` con
cobertura baja (12-32%, N chico) — roles poco frecuentes en la plantilla, esperable.

**Pelvis (N=90, 10 roles):** `Bladder`/`Rectum` con V50% alto (97-98%) típico de OARs
adyacentes al PTV en pelvis. `Kidney_*`/`FemoralHead_*`/`SpinalCord_PRV05` con cobertura
baja (4-16%) — son los roles que solo aplican a la subpoblación con campo extendido o
casos puntuales (consistente con las decisiones de Paso 2).

**Prostata (N=90, 12 roles):** `Bladder`/`Rectum` V50% 32-56% (dosis más baja que en
Pelvis, geometría distinta). `Sigmoid`/`Bowel_Small`/`Kidney_*`/`PenileBulb` con
cobertura baja (1-38%) — subestructuras no presentes en todos los planes.

## Cierre de Fase 0

Con Paso 5 completo, el roadmap de `PROMPT_code_fase0.md` queda cerrado:
Paso 0 (andamiaje) → Paso 1 (headers) → Paso 2 (preprocesador, 503/503) → Paso 3
(QA gate) → Paso 4 (auditorías 0.2) → Paso 5 (splits + baseline). Pendiente fuera del
código: actualizar `docs/contexto_fase_vigente.md` (Tier 2) y sumar una línea a
`docs/aprendizajes_transversales.md` (Tier 3) — tarea editorial, no de script.
