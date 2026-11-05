# Fase 3 — Eje B (arquitectura) — CERRADA

> Tier 3, congelado. Reabrible (charter §8), no sellado.
> Input fijo: A4 (7 canales), `npz_v2b_body`, `split_v2.json`, hash armonización `7ab16f73fa632a3e`.
> Todo sobre **val** (n=65: CyC 23 / Pelvis 21 / Próstata 21). Test intacto. Seed única: 42.

## Veredicto

**La U-Net 2D queda como arquitectura de operación.** Ni la U-Net 3D vanilla ni MedNeXt-2D le ganan en
DVH (prioridad §1/§2). El 3D tiene una ganancia real pero acotada: mejor coherencia y MAE-body
**dentro del cuerpo en CyC**, sin mejora de DVH. Queda registrado como trade-off, no como arquitectura
de operación (ver cabos sueltos).

## Experimentos

| exp_id | Arquitectura | Ckpt (óptimo DVH) | Parámetros |
|---|---|---|---|
| `exp_a4_003_unet2d_psdm_c176x256` | U-Net 2D base16/depth4 | ep113 (= óptimo val/mae) | 1,812,593 |
| `exp_b1_001_unet3d_psdm` | U-Net 3D vanilla base16/depth4, ckpt ON | ep076 (= óptimo val/mae) | — |
| `exp_b2_001_mednext2d_psdm` | MedNeXt-2D S, n_channels=16, k3 | ep084 (val/mae: ep098) | 1,384,914 (0.76×) |

Todos: contenedor (176,256) crop-to-BODY, `masked_mae_body`, AdamW 1e-4/1e-5, cosine 189, patience 28,
16-mixed, módulo `a0_module`.

## Infraestructura resuelta

- **Contenedor.** El envelope de Fase 0 medía 176×128×256, pero el datamodule A4 padeaba a 256² (2× px
  por corte) → el 3D OOMeaba. Decisión: **crop al bbox del BODY + pad a contenedor fijo global
  (176,256)** (bbox máx real H=163, W=248, redondeado a múltiplo de 16), offsets por paciente,
  eval des-hace crop+pad hacia grilla nativa. Aplica a 2D y 3D por igual.
- **Guardrail del contenedor.** A4-2D re-corrido en el contenedor nuevo reproduce las métricas
  intra-estructura del viejo (6 deltas OAR-media/PTV-alta < 1 pp, todos los IC incluyen 0; máx
  CyC PTV −0.49). Sin bug de offsets.
- **Memoria 3D (RTX A2000 12 GB, Z=239, sintético).** (176,256)+ckpt: entra (pico 8.17 GiB asignado,
  10.66/11.99 nvidia-smi). (176,256) sin ckpt: OOM. 256²+ckpt: OOM.
- **Parches (§10): no hicieron falta** para U-Net 3D vanilla. Z nativo por paciente (110–239),
  pool-safety OK (cuello ≥6 cortes con depth-4).
- **Epoch-scaling:** batch=1 por paciente en 2D y 3D → pasos/época = N_train = 280 en ambos → se
  hereda 189/28 sin re-escalar.
- **MedNeXt:** vendorizado desde MIC-DKFZ/MedNeXt (Apache-2.0), solo la arquitectura, como
  `arch=mednext2d`. Desviaciones de §5 registradas: ConvTranspose depthwise, GroupNorm por canal,
  GELU, exp_r=2.

## Resultados

**DVH, delta pareado vs U-Net 2D** (pp; positivo = el nuevo es mejor; IC95 bootstrap por paciente):

| | 3D: OAR-media 40–80 | 3D: PTV-alta 80–110 | MedNeXt: OAR-media | MedNeXt: PTV-alta |
|---|---|---|---|---|
| CyC | −0.07 [−0.26, +0.14] | +0.32 [−0.77, +1.65] | −0.41 [−0.63, −0.18] | −4.22 [−5.47, −2.93] |
| Pelvis | +0.43 [−0.13, +1.10] | +1.75 [+0.31, +3.53] | +0.42 [−0.02, +0.98] | +0.44 [−0.71, +1.77] |
| Próstata | −0.43 [−0.94, +0.06] | −1.63 [−2.80, −0.55] | −0.19 [−0.58, +0.14] | −0.77 [−2.26, +0.91] |

**Criterio pooled** (mean|ΔV| OAR banda media, menor = mejor): 2D **2.487** (2.18/3.34/1.97) ·
3D 2.511 (2.25/2.92/2.39) · MedNeXt 2.556.

**Secundarias** (CyC / Pelvis / Próstata):

| | MAE-body | SSIM BODY | SSIM BODY∩dosis>5% |
|---|---|---|---|
| 2D | 2.324 / 2.530 / 0.917 | 0.923 / 0.960 / 0.978 | 0.889 / 0.951 / 0.949 |
| 3D | 1.719 / 2.732 / 0.922 | 0.954 / 0.964 / 0.981 | 0.926 / 0.953 / 0.949 |
| MedNeXt | 2.572 / 2.736 / 1.156 | 0.892 / 0.944 / 0.958 | 0.849 / 0.932 / 0.927 |

**d2, coherencia cráneo-caudal** (rugosidad pred/real, xy / dosis media / vfrac; 1 = igual a la real):

| | CyC | Pelvis | Próstata |
|---|---|---|---|
| 2D | 0.97 / 1.00 / 0.88 | 0.93 / 0.99 / 0.89 | 0.89 / 0.93 / 0.87 |
| 3D | 0.80 / 0.89 / 0.87 | 0.80 / 0.91 / 0.90 | 0.84 / 0.92 / 0.82 |
| MedNeXt | 1.15 / 1.12 / 0.91 | 1.21 / 1.10 / 1.03 | 1.04 / 1.03 / 1.06 |

## Lectura

- **La tesis de CyC quedó refutada:** el 2D ya era coherente en Z. El 3D sobresuaviza en Z; MedNeXt
  agrega rugosidad.
- **Selección:** óptimo val/mae = óptimo DVH en 2D y 3D (test débil: top-3 en épocas cercanas). En
  MedNeXt difieren (ep84 vs ep98).
- **Pelvis+ / Próstata− del 3D en PTV-alta: no atribuidos.** Sitios vecinos con efectos opuestos de
  igual tamaño, con una sola seed; el IC bootstrap no incluye la varianza entre corridas. No cambia la
  decisión (el 3D no gana en general en ninguna lectura).
- **Firma de piso en OARs.** Correlación del error por paciente 2D vs 3D: OAR-media CyC ρ=0.70
  (diferencia entre modelos = 0.34× la dispersión entre pacientes), Pelvis ρ=0.76 (0.71×). Los dos
  modelos fallan en los mismos pacientes. En PTV-alta no aparece (ρ 0.37–0.57, 0.9–1.5×).

## Validación del SSIM (resultado transversal de la fase)

- **El SSIM de frame completo medía sobre todo el aire.** Fuera del BODY está el 71% de los píxeles en
  CyC (~36% en los sitios pélvicos), sin supervisión. Error fuera del BODY en CyC: 2D 9.97 · 3D 1.20 ·
  MedNeXt 2.33 · A0 19.7 · A1 3.85. Las diferencias de frame las explicaban luminancia y contraste,
  no estructura.
- **Se retira el SSIM de frame completo.** Pasa a calcularse dentro del BODY (y BODY ∩ dosis>5% como
  variante). Ventanas sobre la imagen completa; la máscara solo elige qué píxeles se promedian.
- **MedNeXt:** su "mejor SSIM" era artefacto. Dentro del cuerpo es peor en los 3 sitios.
- **3D:** la ganancia dentro del cuerpo es real solo en CyC (+0.031 BODY, +0.037 región con dosis);
  en los sitios pélvicos ≈0.

**Auditoría de decisiones del eje A que citaban el SSIM:**

| Decisión | Sustento | Estado |
|---|---|---|
| PSDM adentro | DVH (recupera PTV de CyC) + MAE-body | Se sostiene |
| Densidad afuera | Sin aporte en DVH + "baja SSIM" | Se sostiene por DVH |
| Goals adentro | "SSIM sube fuerte" | **Re-justificada por diseño** (abajo) |

**Goals.** A1−A0 dentro del BODY: CyC −0.004 (ns) · Pelvis +0.002 (ns) · Próstata +0.020. En la región
con dosis: +0.010 / +0.008 (ns) / +0.018. Se mantienen, pero el fundamento cambia: son el **canal de
intención** (§3), lo único que permite que el modelo responda a otro protocolo, y la base del uso
"RapidPlan casero" y de A2/A3. Dentro de distribución (un protocolo por sitio) son casi neutros. Es una
apuesta: el riesgo es que funcionen como huella de sitio. Su test real es el protocolo apartado como
hold-out (§5).

## Post-proceso adoptado

La dosis predicha se multiplica por la máscara BODY en inferencia, evaluación y export. El BODY es
input, así que no hay leakage. Verificado con el 2D ep113: el DVH cambia de forma despreciable
(23/1104 filas, máx 0.049 pp; agregados ≤ 0.01 pp). Causa: 6 pacientes de val con estructuras que
tienen 1–4 vóxeles fuera del BODY (rasterización en el borde de la piel), ahora registradas en
`summary.json` (`estructuras_fuera_del_body`); no se recortan en el preprocesador. `mae_body` y
`ssim_body` prácticamente no cambian (+0.0009 de media). Motivo del post-proceso: la
red "pinta" dosis en el aire no supervisado, lo que se ve mal en la visualización médica (§1) y
contamina las métricas de frame.

## Pendiente que pasa a Fase 4

- **Revisión clínica de los pacientes de cola** (nivel 1, posible contaminación en la dosis, §7):
  error OAR alto compartido por 2D y 3D. CyC `PT_20011a605a19589c`, `PT_d41fa07c4f323437`,
  `PT_b55d907a5f2cc495`; Pelvis `PT_1236511005be4d68`, `PT_c687aea3fc02ef4f`.

Diferidos con condición de reapertura: en `cabos_sueltos.md`.
