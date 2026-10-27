# HO#9 — Fase 2: A4 sobre A0 (distance map SIN goals) — cierre del 2×2

> Prompt para Claude Code. Leer con `contexto_fase_vigente.md` (Tier 2), `CHARTER.md` (§4) y el record
> de A4 (`exp_a4_001_unet2d_psdm`).
> **Contexto:** A4 (goals + PSDM) quedó como mejor hasta ahora, aporte modesto pero real. Esta corrida
> completa el factorial 2×2 para decidir **si los goals siguen ganándose el lugar ahora que está el
> distance map** (los goals tienen un costo: dañan el PTV). Celda faltante = **PSDM solo, sin goals**.
> **Alcance:** una corrida. Decide poda de canales, no es atribución ociosa.

## La corrida

- **dist-solo = A0 (3 canales: intención/PTV, OAR consolidado, body) + PSDM del PTV.**
- El PSDM debe ser **idéntico** al de A4 (`exp_a4_001`): mismo encoding, mismo nº de canales, mismo
  manejo multi-PTV (distancia firmada a `PTV_High`), mismo `ds.pad_channels()` (el padding unificado
  que arregló el bug de eval en A4). **NO** tiene los 3 canales de goals (Ch3/4/5).
- Todo lo demás **congelado e idéntico** a A4: MAE, `npz_v2b_body` + `split_v2.json`, hash
  `7ab16f73fa632a3e`, epoch-scaling (N=280), early stopping, seed 42. Val decide; test intacto.

## El 2×2 y la decisión

Cuatro configs sobre las mismas métricas/val:

| | sin PSDM | con PSDM |
|---|---|---|
| **sin goals** | A0 | **dist-solo (esta)** |
| **con goals** | A1-uniform | A4 |

Regla de poda:

- Si **dist-solo ≈ A4** en SSIM y PTV (dentro del piso de ruido) → los goals no agregan sobre el
  distance map → **se dropean los goals**: representación más simple y recupera el PTV que costaban.
- Si **dist-solo cae claro en SSIM** vs A4 → los goals sí aportan coherencia más allá de la geometría
  → se quedan ambos (A4 es la representación).

## Lectura — set completo por sitio, IC95 bootstrap, CONTRA el piso de ruido

Acá el piso de ruido **sí es decision-relevant** (no solo sanity): la pregunta "¿goals redundantes?"
es un juicio de *cercanía* dist-solo vs A4, y esas diferencias pueden caer en zona de ruido. Contrastar
con los seeds de A0 disponibles (42, 43; rango crudo, alcanza para ubicar diferencias chicas).

1. **SSIM** — ¿dist-solo sostiene la coherencia que daban los goals? (celda central de la decisión).
2. **PTV banda alta** — ¿sin goals + PSDM el PTV queda mejor que A4? (los goals lo dañaban).
3. **OAR banda media** — plano esperado.
4. MAE body — monitoreo.

## Handoff de vuelta (§8)

Delta: `metrics.csv`/`summary.json` + la **tabla 2×2 completa** (A0 / A1-uniform / A4 / dist-solo) por
sitio con IC95, contrastada con el piso de ruido + veredicto de poda (se dropean goals o no) + resumen.
Con esto cierra la representación del bloque intención/geometría y sigue A5 (densidad).
