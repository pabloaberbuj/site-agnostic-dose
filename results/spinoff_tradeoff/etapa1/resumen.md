# Spin-off trade-off — Etapa 1: estructura del residual (real − A4)

Modelo `exp_a4_003_unet2d_psdm_c176x256 ep113`, hash armonización `7ab16f73fa632a3e`, 432 pacientes. Residual = real − predicho (renormalizado a D95(PTV_High)=100%).
**Uso del test:** Analisis no supervisado, sin seleccion de modelo. No se reportan metricas agregadas de desempeno (MAE ni mean|dV| por banda); solo estructura del residual (val+test, fuera de muestra).

## Métrica de similitud (k=3, spread del DVH real vs azar)

- Prostata: trabajo = **B**; ratio medio spread/azar {A: 0.67, B: 0.65} (C se reporta, no se usa; detalle en tablas/metrica_eleccion.csv)
- Pelvis: trabajo = **B**; ratio medio spread/azar {B: 0.50} (C se reporta, no se usa; detalle en tablas/metrica_eleccion.csv)
- CyC: trabajo = **B**; ratio medio spread/azar {B: 0.59} (C se reporta, no se usa; detalle en tablas/metrica_eleccion.csv)

## Resultado por sitio (val+test, fuera de muestra)

| Sitio | n | vars | retenidos | estabilidad (cos) | acoplamiento OAR vs PTV / HOT / FALL (r [IC95], calidad) | R² anat. | réplica cos k3/k5 | optimismo train/oos |
|---|---|---|---|---|---|---|---|---|
| Prostata (manual) | 49 | 30 | 4 | 0.96, 0.89, 0.87, 0.85 | -0.26 [-0.50,+0.07] / -0.28 [-0.49,+0.06] / -0.05 [-0.31,+0.22] | 0.38 [-0.13,0.55], -0.10 [-0.26,-0.04], -0.10 [-0.32,-0.01], -0.14 [-0.59,0.13] | 0.85/0.81, 0.83/0.76, 0.65/0.51, 0.36/0.44 | 0.76 [0.64,0.92] |
| Pelvis (manual) | 48 | 26 | 2 | 1.0, 0.95 | +0.48 [-0.15,+0.74] / +0.57 [+0.03,+0.76]* / +0.79 [+0.65,+0.88]* | -0.67 [-2.54,-0.32], -0.09 [-0.58,0.40] | 0.62/0.64, 0.69/0.69 | 0.66 [0.50,1.06] |
| CyC (RapidPlan (contraste)) | 55 | 126 | 4 | 0.93, 0.72, 0.51, 0.77 | +0.79 [+0.42,+0.96]* / +0.76 [+0.44,+0.97]* / +0.62 [+0.19,+0.88]* | 0.16 [-1.83,0.41], -0.18 [-2.04,-0.06], -1.24 [-9.45,-0.36], 0.04 [-0.48,0.40] | 0.49/0.48, 0.54/0.62, 0.25/0.31, 0.35/0.26 | 0.68 [0.42,0.96] |

`*` = significativo (perm p<0.05/3 e IC sin 0). Signo en calidad: r<0 antagónico, r>0 sinérgico.

## Lectura predeclarada

- **Prostata**: R² anatomico alto en componente estable: anatomia que A4 no captura (vuelve al eje principal como hallazgo)
- **Pelvis**: Eje de calidad global (sinergico OAR vs ['HOT', 'FALL']): rankea planes, no es coordenada de preferencia
- **CyC**: Eje de calidad global (sinergico OAR vs ['PTV', 'HOT', 'FALL']): rankea planes, no es coordenada de preferencia

## Contraste entre sitios (variables comunes)

Retenidos: {'Prostata': 4, 'Pelvis': 2, 'CyC': 4}; desvío mediano del residual en variables comunes: {'Prostata': 1.069, 'Pelvis': 3.471, 'CyC': 2.545}. Salvedad: CyC difiere de los otros dos en generador (RapidPlan) **y** en anatomía; no es un control limpio.

## Decisiones no previstas

- Bandas de dosis semiabiertas [lo,hi) salvo la ultima (80-100 incluye 100); dosis en %Rx_High (NPZ ya normalizado a D95(PTV_High)=100%).
- Dosis predicha: clip a 0, x BODY, renormalizada a D95(PTV_High)=100% con dvh.renorm_pred (misma funcion del harness); factor guardado por paciente en cache.
- PTV_Low de 16 pacientes de Prostata (nivel no declarado, identico a High) ignorado: solo niveles del protocolo.
- Bilaterales _L/_R: slot prox/dist por distancia minima al PTV_High; si falta uno de los dos en un paciente, ambos slots quedan NaN (imputados con la media tras centrar).
- Variables con >30% NaN o sin varianza se descartan (lista en summary); camaras con <50 voxeles = NaN. NaN restantes imputados con la media (0 tras centrar).
- D0.035cm3 = percentil 100*(1-0.035/V_cc) de las dosis de la estructura (maximo si V<0.035cc). ΔDmax solo en OARs con goal tipo Dmax.
- PTV Mid/Low: ΔD95 en % de la Rx del propio nivel. Homogeneidad = (D2-D98)/D50. PTV: D50 sin signo de calidad (ambiguo).
- Score por bloque = PC1 de las variables estandarizadas del bloque, orientado a calidad (+ = real mejor que predicho): OAR/hotspot/camaras: menos dosis=mejor; PTV: D98/D95 mas alto y D2/homogeneidad mas bajos = mejor. r<0 entre OAR y otro bloque = antagonico; r>0 = sinergico.
- Acoplamiento significativo: p de permutacion < 0.05/3 (Bonferroni sobre OAR vs PTV/HOT/FALL) E IC95 bootstrap sin 0.
- Sensibilidad "sin estandarizar": cada variable dividida por el desvio RMS de su grupo de unidades (pp, % dosis, cm3, razon), no por el propio.
- R² anatomico: ridge con alpha elegido por LOO interno dentro de cada pliegue (LOO externo), descriptores estandarizados en el pliegue; IC95 bootstrap sobre pares (y, pred LOO); p por permutacion de y (100).
- Vecinos del §5.5 buscados SOLO dentro de val+test (el score solo existe fuera de muestra), con la distancia de la metrica de trabajo, k=3.
- Replica §6: desviacion vs promedio de los k vecinos sobre TODOS los elegibles del sitio (real vs real); loadings comparables = corr(score, variables compactas). Criterio de replica: coseno k=3 > 0.7 (k=5 se reporta).
- Criterio de "R² anatomico bajo" para la lectura: R² LOO (estimacion puntual) < 0.2 en el componente; se evalua por componente estable.
- Optimismo: razon de RMS por variable (sin centrar) train / val+test, resumida por mediana; IC bootstrap de pacientes en ambos grupos.
- Candidata B: roles con peso igual por slot y por referencia (PTV_High / union); curvas y escalares con peso igual (cada grupo normalizado por la mediana de distancias entre pares del sitio). OVH en grilla 0-150 mm cada 2.5 mm; vol OAR/PTV en bruto (no log).
- Spread del azar: k=3 vecinos aleatorios del sitio, 100 repeticiones; spread comparado solo en celdas (paciente, slot) validas para todas las candidatas.
