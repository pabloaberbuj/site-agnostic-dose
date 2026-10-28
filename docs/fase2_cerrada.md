# Fase 2 cerrada — eje A (representación de input), VMAT-only, U-Net 2D

> **Tier 3, congelado.** Freeze de Fase 2. El estado vivo pasa al `ARRANQUE_fase3.md` (eje B).
> Datos: `npz_v2b_body` + `split_v2.json`, hash de tabla `7ab16f73fa632a3e`, 432 pacientes
> (280/65/87; Próstata 138, Pelvis 138, CyC 156). U-Net 2D fija, MAE puro, pad 256, seed 42.
> Lectura siempre **por sitio**, IC95 bootstrap, sobre el **set completo de métricas**.

## Resultado principal (la pregunta de §2)

La red **aprende anatomía→dosis más allá del promedio del protocolo**: le gana a la baseline
no-aprendida en OARs (banda media) en los 3 sitios, IC95 lejos de 0 (A0 ya: +6.7 / +3.5 / +1.6 pp
CyC/Pelvis/Próstata). Eso se estableció en el **piso A0** y se sostuvo todo el eje.

## Representación final: A4 — 7 canales

1. Intención/PTV — ratios `Rx_nivel/Rx_High` prescriptos
2. OAR consolidado (unión binaria, PRV incluidos)
3. Body (External real)
4–6. Goals de OAR en 3 canales (máx / media / volumétrico), **fill 0, uniform**
7. **PSDM** — distance map firmado a `PTV_High` (norm 15 cm, pad 1.0)

## Qué aportó cada canal (y evidencia)

| Bloque | Veredicto | Evidencia / razón |
|---|---|---|
| A0 (intención+OAR+body) | **base** | le gana a la baseline en OAR; piso del eje |
| Goals ×3 (A1) | **entra** | no mueven DVH-OAR (estáticos por protocolo → redundantes) pero **suben SSIM fuerte** en los 3 sitios (coherencia). Costo: PTV alta peor, sobre todo CyC |
| fill 0 vs 95% (A1a) | **fill 0** | domina a 95%: mejor SSIM y menos daño al PTV. Ch1 ya da "hay OAR", así que 0 no pierde geometría |
| uniform vs distmod (A1b) | **uniform** | la modulación por distancia no aportó (λ/piso cerrados en cabos) |
| PSDM (A4) | **entra** | recupera PTV alta en CyC (8.41→7.00), sube SSIM+MAE body en pélvicos; **paciente-específico**. Aporta multi-sitio (contra Xiong). 2×2 confirmó que goals no son redundantes con PSDM |
| Densidad (A5) | **afuera** | no aportó; baja SSIM −0.07 en los 3. Hipótesis: bordes filosos ρ copiados a la dosis (2D). Revisar en 3D |
| Beam plate (A6) | **diferido** | análisis geométrico: arcos coplanares ~360° → mapa = rampa en z ≈ PSDM. No se corrió. Payoff real = desambiguar técnica en fase IMRT |

## Números finales por sitio (A0 → A4)

| | OAR media (DVH) | PTV alta (DVH) | SSIM |
|---|---|---|---|
| CyC | 2.21 → 2.29 | 6.23 → 7.00 | 0.375 → **0.571** |
| Pelvis | 3.30 → 3.21 | 4.15 → 4.35 | 0.656 → **0.815** |
| Próstata | 2.20 → 2.27 | 5.54 → 5.17 | 0.674 → **0.856** |

OAR-DVH plano todo el eje (el aporte fue al piso A0). La ganancia de A4 es **coherencia espacial**
(SSIM, uso visualización) + PTV-CyC vía PSDM.

## Costo conocido aceptado

A4 hornea ~**0.9 pp de PTV alta en CyC** (los goals; dist-solo lo recuperaba a ~6.13 ≈ A0). Aceptado
por la ganancia de coherencia. Palanca futura: **A3 (prioridad)** / A6, cuando entren.

## Diferidos (no omitidos — ver `cabos_sueltos.md`)

- **A2 (augment goals) y A3 (prioridad):** intención estática por protocolo → payoff solo fuera de
  distribución (multi-protocolo / zero-shot). Reabren con eval OOD.
- **A6 (beam plate):** a la fase IMRT/multi-técnica.
- **Densidad:** revisar en 3D (eje B).
- **Piso de ruido seed:** solo existe seed 42; se mide pegado a la primera comparación en zona gris.

## Punteros

Configs/exp: `exp_a0_001` (piso), `exp_a1a_001` (fill0), `exp_a4_001` (A4 final), `exp_a4_002`
(dist-solo, 2×2), `exp_a5_001` (densidad). Comparadores en `scripts/`. Bug corregido en el camino:
padding de canales distinto train vs eval (A4) → unificado en `pad_channels()`.
