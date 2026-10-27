# HO#11 — Fase 2: A6 — beam plate map (templado) — cierre del eje A

> Prompt para Claude Code. Leer con `contexto_fase_vigente.md` (Tier 2), `CHARTER.md` (§4 A6, §5) y el
> código público de Xiong (beam plate).
> **Contexto:** representación ganadora = **A4 = goals (fill 0, uniform) + PSDM del PTV_High**, 7 canales.
> Densidad (A5) descartada. A6 agrega el **beam plate map templado** — la única pista de *por dónde entra
> la dosis*. Es el último canal de input: al cerrarlo, se fija la representación del eje A y sigue el eje
> B (3D). **Alcance:** una corrida — A4 + beam plate = 8 canales.

## Expectativa (calibración de lectura)

En **VMAT-only** el beam plate es casi **estático por protocolo**: la geometría de arcos es fija por
sitio/técnica; lo único paciente-específico es el **isocentro** (≈ centroide del PTV_High) y la extensión
del arco. Y el iso ya está implícito vía PSDM + máscara de PTV. Así que el aporte real a medir es si la
**entrada anisotrópica** (por dónde entra el haz) agrega algo sobre la distancia **isotrópica** del PSDM.
Plano/redundante es un resultado plausible e informativo: el payoff grande del beam plate es la
**desambiguación de técnica** (fase IMRT futura), donde un VMAT en arco y un IMRT de 7 campos son
espacialmente muy distintos. No lo construimos por eso ahora, sino para cerrar el eje A y ver si la
geometría de entrada ayuda al PTV/cuerpo aun en VMAT.

## Antes de codear — gates

1. **Encoding del beam plate:** portar/reportar la definición del código público de Xiong (cómo se mapea
   la geometría de haces a un canal espacial), no inventar una.
2. **Geometría de arcos por protocolo:** ¿dónde está definida (ángulos start/stop, nº de arcos,
   coplanar)? Si no existe, hay que especificarla por sitio — es dato clínico del protocolo.
3. **Iso = centroide del PTV_High** (de las máscaras). Beam plate en la grilla de dosis, mismo
   `pad_channels()`.

## Templado vs. real (decisión §5)

- **Se entrena con el beam plate TEMPLADO** (sintetizado del protocolo: iso ≈ centroide PTV_High,
  ángulos/arcos del protocolo), NO del RTPLAN → consistencia train/inferencia (en inferencia no hay plan).
- **Además, medir y reportar la distancia beam-map real-vs-templado por sitio** (el RTPLAN real está en el
  DICOM) como diagnóstico: cuánto mismatch estamos aceptando entre el input templado y el target (dosis
  del plan real). Para VMAT se espera chico. Si sale grande en algún sitio → flaggear antes de confiar en
  el resultado (sería input-target mismatch contaminando la lectura).

## La corrida

A4 (7 canales) + beam plate templado (1 canal) = 8 canales. Todo lo demás **congelado e idéntico** a A4:
MAE, `split_v2.json`, hash `7ab16f73fa632a3e`, epoch-scaling (N=280), early stopping, seed 42. Val decide;
test intacto.

## Lectura — set completo por sitio, IC95 bootstrap

A6 vs A4:

1. **PTV alta** — ¿la geometría de entrada mejora la conformidad del envelope de alta dosis?
2. **MAE body** — caminos de entrada/salida del haz.
3. **SSIM** — que no caiga.
4. **OAR media** — plano esperado.

¿Agrega la entrada anisotrópica sobre el PSDM, o es redundante? Zona gris → piso de ruido (seed 43)
pegado a la celda, como en `cabos_sueltos.md`.

## Handoff de vuelta (§8)

Delta: encoding del beam plate + geometría de arcos usada + distancia real-vs-templado por sitio +
`metrics.csv`/`summary.json` + comparación A6 vs A4 por sitio (IC95) + resumen. **Con esto cierra el eje A**
(representación fija) y el siguiente bloque es el eje B (U-Net 2D → 3D con parches → MedNeXt).
