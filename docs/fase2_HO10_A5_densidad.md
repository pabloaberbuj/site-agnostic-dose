# HO#10 — Fase 2: A5 — densidad másica (NO CT crudo)

> Prompt para Claude Code. Leer con `contexto_fase_vigente.md` (Tier 2), `CHARTER.md` (§4 A5, §5, §7).
> **Contexto:** el bloque intención/geometría cerró — representación ganadora **A4 = goals (fill 0,
> uniform) + PSDM del PTV_High**, 7 canales. Costo conocido aceptado: ~0.9 pp de PTV alto en CyC (palanca
> futura A3/A6). A5 agrega **densidad másica** y testea si "el CT no aporta" (hallazgo de próstata) era en
> realidad "el **CT crudo** (HU) no aporta" — la dosis se deposita según densidad física, no según HU.
> **Alcance:** una corrida — A4 + canal de densidad. NO el CT en HU.

## Antes de codear — disponibilidad del dato (gate)

A diferencia de los goals (que se construyen de la tabla), la densidad sale del **dato crudo**, y el NPZ
`npz_v2b_body` guarda dosis + máscaras + metadata, **no el CT**. Reportar antes de construir:

- ¿El HU/CT está accesible (DICOM crudo, `dicom_extracted`, `cargar_ct`) y se puede llevar a la **grilla
  de dosis nativa** (resamplear CT→dosis, NO al revés — §5 higiene de vóxel)?
- Camino más barato para que la densidad llegue al canal: ¿array nuevo en una versión de NPZ (regen,
  `npz_v2c`), sidecar por paciente, o cómputo al cargar desde el CT? Proponer, no regenerar a ciegas.
- **Curva HU→densidad:** usar la **calibración clínica** (tabla CT-densidad de Eclipse del escáner/
  protocolo), la misma con la que se calculó la dosis — no una curva genérica. Reportar cuál se usa.

## La corrida

- **A5 = A4 (7 canales) + densidad (1 canal) = 8 canales.** Canal = densidad física (ρ, o ρ/ρ_agua),
  mapa completo sobre el cuerpo en la grilla de dosis, padding con el mismo `pad_channels()`. **NO HU.**
- Todo lo demás **congelado e idéntico** a A4: MAE, `split_v2.json`, hash `7ab16f73fa632a3e`,
  epoch-scaling (N=280), early stopping, seed 42. Val decide; test intacto.

## Lectura — set completo por sitio, IC95 bootstrap

A5 vs A4. Hipótesis dirigida: la densidad actúa vía **heterogeneidades** (hueso, aire, cavidades), así
que esperaríamos aporte **donde hay heterogeneidad** — CyC (mandíbula, cavidades aéreas, senos) más que
pélvicos (mayormente tejido blando, más homogéneo). Leer por sitio, no pooled.

1. **PTV alta y MAE body** — donde la física de deposición pega: ¿mejora la dosis cerca de
   heterogeneidades? CyC es el candidato.
2. **SSIM** — que no caiga.
3. **OAR banda media** — plano esperado.

Salidas: si la densidad aporta (sobre todo CyC) → "el CT crudo no aportaba, la densidad sí" (hallazgo
real multi-sitio). Si es plana → la info de heterogeneidad ya la captura la geometría + el target, o los
sitios son demasiado homogéneos para que pese. Ambas informativas.
Zona gris → recién ahí el piso de ruido (seed 43), pegado a la celda ambigua (ver `cabos_sueltos.md`).

## Handoff de vuelta (§8)

Delta: camino usado para la densidad + curva HU→densidad + `metrics.csv`/`summary.json` + comparación A5
vs A4 por sitio (PTV alta, MAE body, SSIM, OAR media, IC95) + resumen. Siguiente input sería A6 (beam
plate); después cierra el eje A y sigue el eje B (3D).
