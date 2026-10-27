# HO#8 — Fase 2: A4 — PTV distance map / PSDM

> Prompt para Claude Code. Leer con `contexto_fase_vigente.md` (Tier 2), `CHARTER.md` (§4 A4, §5)
> y el linaje de próstata (PSDM del piloto RD / exp002).
> **Contexto:** A1 cerró — ganador **fill 0 + uniform** (6 canales). La modulación por distancia (A1b)
> no aportó: la coherencia la da *tener* goals, no modularlos. **A2 y A3 se difieren** (son intención
> estática por protocolo → redundantes en distribución; su payoff es fuera de distribución —
> multi-protocolo / zero-shot— que todavía no evaluamos). A4 es el **primer canal paciente-específico
> desde A0**, ataca el PTV banda alta que los goals empeoraron, y testea un hallazgo abierto del charter.
> **Alcance:** una corrida — A1-uniform + distance map del PTV. No A2/A3.

## Antes de codear — reportar el PSDM de próstata

A4 testea el hallazgo de próstata (el PSDM **ayudó**) contra Xiong (el distance map **no aportó**) en
multi-sitio. Para que el test sea fiel, portar el encoding de próstata, no inventar uno. Reportar:

- Qué es exactamente el PSDM del piloto RD: cuántos canales, encoding (¿distancia firmada? ¿bandas?),
  unidades y normalización.
- La EDT ya existe (de A1b) — reusarla.

## La corrida

- **A4 = A1-uniform (6 canales) + PSDM del PTV.** Stem de la U-Net a 7 canales (o +N si el PSDM de
  próstata era multi-canal — reportar).
- **Multi-PTV (lo nuevo vs. próstata, que era 1 PTV):** reportar cómo se arma el mapa en CyC. Default
  simple y consistente con lo aprendido en A1b: **distancia firmada a `PTV_High`** (la unión cruda
  mis-ordenaba CyC). Mapa por-nivel como variante solo si CyC lo pide.
- Todo lo demás **congelado e idéntico** a A1-uniform: MAE, datos `npz_v2b_body` + `split_v2.json`,
  hash `7ab16f73fa632a3e`, epoch-scaling (N=280, factor 265/280), early stopping, seed 42. Val decide;
  test intacto.

## Lectura — set completo de métricas, por sitio, IC95 bootstrap

A4 vs A1-uniform vs A0:

1. **PTV banda alta — hipótesis central.** ¿El distance map recupera lo que los goals costaron, sobre
   todo en CyC (6.23 A0 → 8.41 A1)? Es un canal geométrico paciente-específico de conformidad del PTV.
2. **SSIM** — que no caiga (la coherencia de los goals se mantiene).
3. **OAR banda media** — plano esperado.
4. La pregunta del charter: ¿el distance map **aporta multi-sitio** o es redundante (como en Xiong)?
   Si no mueve nada, ese es el hallazgo — informativo, no un fracaso.

## Rama opcional (NO ahora)

Si el primario muestra que A4 ayuda y querés **atribuir el crédito** (¿necesita los goals o no?), una
segunda corrida A4 sobre A0 (sin goals) lo aísla. Se decide con el primario en mano, no por default.

## Cierre / Tier 2

Actualizar `contexto_fase_vigente.md`: A1 cerrado (ganador fill 0 + uniform), A2/A3 diferidos a eval
OOD, A4 en curso.

## Handoff de vuelta (§8)

Delta: definición del PSDM portado + manejo multi-PTV + `metrics.csv`/`summary.json` + comparación A4
vs A1-uniform vs A0 por sitio (PTV alta, SSIM, OAR media, IC95) + resumen corto.
