# HO#7 — Fase 2: A1b — goals uniformes vs. modulados por distancia (sobre fill 0)

> Prompt para Claude Code. Leer con `contexto_fase_vigente.md` (Tier 2), `CHARTER.md` (§3, §4 A1)
> y el cierre de A1a.
> **Contexto:** A1a cerró con `fill_sin_goal = 0` como ganador. Los goals NO movieron la banda media
> de OAR (nula, en el piso de ruido), pero SÍ mejoraron fuerte la **coherencia espacial (SSIM)** en los
> 3 sitios (CyC 0.375→0.608, Pelvis 0.656→0.781, Próstata 0.674→0.776), a costa del PTV en banda alta
> (CyC 6.23→8.41). El toggle `goal_mode` ya está construido; esto es un flip de config, sin reentrenar nada nuevo de infra.
> **Alcance:** correr A1b = `distance_modulated` vs. `uniform` (el de A1a fill 0), decidir, y cerrar A1.

## La corrida

- Una corrida nueva: `fill_sin_goal = 0` (fijo, ganador de A1a), `goal_mode = distance_modulated`.
- El comparador es **A1a fill 0** (`exp_a1a_001_unet2d_fill0`, `goal_mode=uniform`), ya existente. No
  hace falta re-correr el uniforme.
- Todo lo demás **congelado e idéntico** a A1a fill 0: U-Net 2D 6 canales, MAE, datos `npz_v2b_body` +
  `split_v2.json`, hash `7ab16f73fa632a3e`, epoch-scaling, early stopping, seed 42. Val decide; test intacto.

## Qué es `distance_modulated` (fijar la forma acá)

El charter (§3/§4 A1b) pide "canales de goal modulados por distancia al PTV, ponderada por nivel de
dosis en multi-PTV". Forma concreta propuesta (reportar si implementás otra):

- Los canales de **máximos (Ch3) y medias (Ch4)**, que en `uniform` pintan el techo constante sobre
  toda la estructura, pasan a **escalar el valor del goal por la distancia al PTV** (más cerca del PTV
  → el canal refleja que ahí se permite/espera más dosis; más lejos → decae). La EDT ya está (unión de
  PTVs, espaciado nativo).
- **Ch5 (volumétrico) ya es espacial** (fracción preservada distal): acá la modulación afina el *orden*
  dentro de la fracción, no la agrega de cero.
- **Multi-PTV (CyC):** la distancia debe ponderarse por nivel de dosis del PTV (no la unión cruda), o
  como fallback la distancia a `PTV_High`. La unión cruda de A1a mis-ordenaba CyC — este es el punto
  que A1b puede arreglar.

La decisión de diseño que importa: `distance_modulated` NO debe fundir intención con geometría pura
(§3) — la distancia *modula* el valor del goal, no lo reemplaza por un mapa de distancia. Si en los
hechos el canal termina siendo casi una función de la distancia sola, anotarlo: sería señal de que
colapsa a A4 (distance map) y no aporta como intención.

## Lectura — sobre el set COMPLETO de métricas, no la banda media de OAR sola

Es la lección de A1a: una sola métrica (DVH-OAR media) habría descartado un efecto real. Reportar por
sitio, con IC95 bootstrap, A1b vs A1a-uniforme vs A0:

1. **SSIM** (la señal que movió A1) — ¿la modulación mantiene o mejora la coherencia?
2. **PTV banda alta** — ¿recupera algo del PTV que los goals costaron (sobre todo CyC)? Hipótesis: la
   modulación por distancia podría concentrar los goals lejos del PTV y aliviar la competencia con Ch0.
3. **OAR banda media** — esperado nulo/plano; confirmar que no empeora.
4. MAE body como monitoreo.

## Regla de cierre de A1 (§4: "si empatan, queda la uniforme")

- Si `distance_modulated` **mejora** SSIM y/o recupera PTV sin perder OAR → es la representación de A1.
- Si **empata** con uniforme → queda **uniforme** (más simple, no funde intención con geometría).
- Con eso A1 cierra: representación ganadora = `fill 0` + (`uniform` o `distance_modulated` según esto).
  El chat genera el delta y define si el siguiente eje es A2 (augmentations de goals) o se saltea.

## Handoff de vuelta (§8)

Delta: `metrics.csv` + `summary.json` + comparación A1b vs A1a-uniforme vs A0 por sitio (SSIM, PTV alta,
OAR media, con IC95) + veredicto uniforme/modulado + resumen corto.
