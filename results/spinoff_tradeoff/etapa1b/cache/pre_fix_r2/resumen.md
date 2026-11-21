# Spin-off trade-off — Etapa 1b: residual en la normalización original del plan

Espacio: real en % Rx_High del protocolo (dosis NPZ x D95_abs/Rx); predicho A4 renormalizado a D95(PTV_High)=100%. Residual = real − predicho. Sin metricas agregadas de desempeno sobre test (ni MAE ni mean|dV| por banda); solo estructura del residual en val+test.

**Sanity del reescalado** (n=432): discrepantes >0.5 pp vs QA gate = 0; máx |dif| = 0.005 pp.

## Resultado por sitio (val+test)

| Sitio | n | Retenidos (bloques / columnas Et.1) | ΔD95 vs score OAR (r calidad [IC95], p Holm) | Pruebas sig. (de n) | R² anat. ΔD95 todos / val+test | r vecinos ΔD95 (todos) | HOT~PTV r |
|---|---|---|---|---|---|---|---|
| Prostata | 49 | 3 / 5 | -0.25 [-0.54,+0.06], 0.518 | 1 de 11 | 0.52 [0.34,0.62] (n=138) / 0.16 [-0.66,0.52] (n=49) | +0.35 [+0.23,+0.47] | -0.04 |
| Pelvis | 48 | 1 / 2 | -0.48 [-0.61,-0.31], 0.009 | 4 de 9 | -0.02 [-0.06,-0.02] (n=138) / -0.05 [-0.17,-0.04] (n=48) | +0.01 [-0.17,+0.19] | 0.94 |
| CyC | 55 | 4 / 8 | -0.67 [-0.95,-0.31], 0.502 | 4 de 48 | -38.42 [-177.28,-0.39] (n=156) / -12.41 [-140.22,-0.06] (n=55) | +0.67 [+0.43,+0.86] | 0.17 |

Signo en calidad: r<0 antagónico (menos cobertura ↔ OAR mejor).

## Lectura predeclarada

- **Prostata**: dD95 antagonico con OARs (1 pruebas sig., Holm) y R² anatomico 0.52>=0.2: concesion de cobertura = compromiso real, en parte predecible desde la anatomia -> candidato a salida extra del modelo principal / modelo clasico
- **Pelvis**: dD95 antagonico con OARs (3 pruebas sig., Holm) y R² anatomico -0.02<0.2: compromiso real pero de criterio del planificador -> candidato a coordenada de preferencia | Eje de derrame de Pelvis (Etapa 1 comp 1): coseno max 1.00 con comp 1 de 1b; sinergico=True; sin acoplamiento con dD95=False; no cumple el patron completo
- **CyC**: dD95 antagonico con OARs (4 pruebas sig., Holm) y R² anatomico -38.42<0.2: compromiso real pero de criterio del planificador -> candidato a coordenada de preferencia

## Componentes retenidos: carga por bloque (media orientada a calidad)

- Prostata comp 1: OAR=-0.33, PTV=+0.36, HOT=-0.18, FALL=-0.76; ΔD95 carga +0.59; mezcla de signos opuestos: True
- Prostata comp 2: OAR=-0.11, PTV=+0.42, HOT=-0.05, FALL=+0.28; ΔD95 carga +0.29; mezcla de signos opuestos: False
- Prostata comp 3: OAR=-0.03, PTV=-0.37, HOT=-0.12, FALL=-0.17; ΔD95 carga -0.48; mezcla de signos opuestos: False
- Pelvis comp 1: OAR=+0.57, PTV=+0.42, HOT=+0.85, FALL=+0.93; ΔD95 carga -0.44; mezcla de signos opuestos: False
- CyC comp 1: OAR=+0.31, PTV=-0.21, HOT=+0.18, FALL=+0.12; ΔD95 carga -0.77; mezcla de signos opuestos: True
- CyC comp 2: OAR=-0.06, PTV=+0.13, HOT=-0.14, FALL=+0.08; ΔD95 carga +0.52; mezcla de signos opuestos: False
- CyC comp 3: OAR=+0.13, PTV=+0.11, HOT=+0.49, FALL=+0.67; ΔD95 carga +0.02; mezcla de signos opuestos: False
- CyC comp 4: OAR=-0.05, PTV=+0.04, HOT=-0.07, FALL=+0.31; ΔD95 carga -0.08; mezcla de signos opuestos: False

## Sensibilidad por extremos de ΔD95 (no prevista en el prompt)

ΔD95 tiene colas pesadas; en CyC los extremos son los casos ya revisados "cavum con PTV solapado al PRV de quiasma" (QA gate `reviewed_ok`). Se repite la prueba directa y el R² anatómico (a) con rangos y (b) sin los pacientes con z robusto < −3.5.

| Sitio | Versión | n | ΔD95 vs score OAR (r calidad [IC95], p Holm) | Pruebas sig. (de n) | OAR sig. antagónicas | R² anat. ΔD95 (todos) |
|---|---|---|---|---|---|---|
| Prostata | rangos (Spearman) | 49 (pool 138) | -0.20 [-0.49,+0.10], 0.807 | 0 de 11 | — | 0.44 [0.29,0.55] |
| Prostata | sin extremos (z<-3.5) | 43 (pool 119) | -0.07 [-0.39,+0.28], 1 | 0 de 11 | — | 0.33 [0.17,0.45] |
| Pelvis | rangos (Spearman) | 48 (pool 138) | -0.45 [-0.65,-0.20], 0.0112 | 4 de 9 | Bowel_Small|dV_media_40_80, Bowel_Small|dDmean, score_OAR | -0.02 [-0.06,-0.02] |
| Pelvis | sin extremos (z<-3.5) | 48 (pool 138) | -0.48 [-0.61,-0.31], 0.0091 | 4 de 9 | Bowel_Small|dV_media_40_80, Bowel_Small|dDmean, score_OAR | -0.02 [-0.06,-0.02] |
| CyC | rangos (Spearman) | 55 (pool 156) | -0.33 [-0.58,-0.05], 0.616 | 0 de 48 | — | -340.45 [-1066.65,-4.08] |
| CyC | sin extremos (z<-3.5) | 51 (pool 147) | -0.16 [-0.34,+0.12], 1 | 0 de 48 | — | -60.38 [-130.49,-13.77] |

## Decisiones no previstas

- Predicho: la dosis predicha no estaba cacheada en la Etapa 1; se re-infirio en CPU solo val+test (152) para el hotspot fuera de PTV; verificado contra el factor de escala y variables cacheadas de la Etapa 1 (campo pred_check en cache).
- Hotspot redefinido fuera de los PTV: D0.035cm3 de (BODY - union de PTV) y volumen >107% fuera de PTV, ambos en real y predicho.
- Real = dosis NPZ x (D95_abs_Gy/Rx_High_Gy) de la metadata; sanity contra paso3_qa_gate.csv y recalculo de D95.
- Nula por bloques: cada bloque (OAR, PTV, HOT, FALL) se permuta entero entre pacientes con permutacion propia; el acoplamiento usa permutacion del score de un bloque (equivalente). Se reporta tambien el n retenido con la nula por columnas de la Etapa 1.
- Prueba directa: familia por sitio = (dV media + dDmean) de cada slot OAR + score de OAR/HOT/FALL; p por permutacion (10000), correccion Holm; significativo = p_holm<0.05 e IC95 bootstrap sin 0. Signo en calidad: r<0 antagonico.
- Lectura: "antagonico con OARs" = alguna prueba OAR (slot o score) significativa con r<0 tras Holm; R² anatomico de dD95 evaluado con TODOS los elegibles (se reporta tambien val+test).
- Carga por bloque = suma (y media) de loadings corr(score,var) con signo de calidad; "mezcla" si hay bloques con media > +0.15 y otros < -0.15.
- Eje de derrame de Pelvis = componente de la Etapa 1 con mayor |loading| medio sobre el bloque de caida de dosis (FALL); sobrevive si coseno>0.7, OAR y FALL con igual signo de calidad y sin acoplamiento con dD95.
- Comparacion con Etapa 1 en las variables comunes; el hotspot cambio de definicion (HOT|D0035_out) y no entra en el coseno.
- Sensibilidad por extremos de dD95 (ver sección arriba): corte z robusto < -3.5 fijado antes de correr la sensibilidad (tras ver la distribución de dD95).
