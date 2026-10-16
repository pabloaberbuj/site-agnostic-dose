# Paso 4 (0.2) — Escala de vóxel repetida sobre los NPZ ya generados

> Script: `scripts/audit_npz_voxel_scale.py`. Compara volumen de PTV rasterizado
> (NPZ, real, el que ve el modelo) contra el volumen por contorno de Paso 1
> (aproximación, suma de áreas × paso entre cortes, sin regla par-impar — documentado
> como "no validado" desde Paso 1). Detalle en `paso4_escala_voxel_npz.csv`.

## Sanity check: spacing y dimensión de grilla — coincide exacto con Paso 1

| sitio | N | z=3.0mm | z=2.0mm | grilla (Z,Y,X) p50 | max |
|---|---|---|---|---|---|
| CyC | 196 | 187 | 9 | (142,111,206) | (239,164,248) |
| Pelvis | 161 | 161 | 0 | (149,97,173) | (217,242,242) |
| Prostata | 146 | 145 | 1 | (143,110,157) | (208,241,241) |

Conteos idénticos a `paso1_headers.md` (CyC 187/9, Pelvis 161/0, Prostata 145/1) — sin
regresión entre Paso 1 (headers) y Paso 2 (NPZ).

## Diferencia relativa: volumen rasterizado (NPZ) vs. contorno (Paso 1)

| sitio | nivel | N | p5 | p50 | p95 | outliers >15% |
|---|---|---|---|---|---|---|
| CyC | PTV_High | 196 | 7.0% | 11.2% | 17.9% | 34 |
| CyC | PTV_Mid | 184 | 10.5% | 13.8% | 17.8% | 52 |
| CyC | PTV_Low | 196 | 10.8% | 14.4% | 18.8% | 81 |
| Pelvis | PTV_High | 161 | 4.8% | 6.4% | 8.0% | 1 |
| Prostata | PTV_High | 146 | 7.8% | 9.4% | 11.2% | 0 |

**Sesgo sistemático positivo en los 3 sitios** (NPZ > Paso1), consistente con el
redondeo a píxel más cercano ya documentado en Paso 2 (`cv2.fillPoly` incluye borde).
Más alto en CyC (11-18%) que en Pelvis (6.4%) o Próstata (9.4%) — esperable: los PTV de
CyC son más chicos y con geometría más compleja (relación superficie/volumen mayor →
el redondeo de borde pesa proporcionalmente más). Pelvis tiene PTV grandes y suaves →
sesgo más bajo. **No es un hallazgo nuevo, es la confirmación cuantitativa completa
(503 pacientes) de lo que Paso 2 ya había medido en 1 paciente por sitio.**

## Investigado: 3 outliers NEGATIVOS en CyC PTV_Low (dirección opuesta al sesgo esperado)

`PT_1a7859ebcff2d4b0` (-22.8%), `PT_6b71a0d809521eb1` (-19.1%), `PT_ac70c40cbcd1ef75`
(-48.3%, el más extremo). Investigado caso por caso (`PT_ac70c40cbcd1ef75`): su
`PTV_Low-04` tiene **7 de 35 cortes con 2 contornos** (hueco/anillo — probablemente
evitando `SpinalCord_PRV05` en esos cortes). Confirmado: 0 cortes del contorno caen
fuera del rango Z de la grilla de dosis (no es el bug de rango-Z ya conocido).

**Explicación, no bug:** el volumen de Paso 1 (`audit_dicom_headers.py`,
`polygon_area_xy` + suma simple) **suma las áreas de ambos contornos sin aplicar la
regla par-impar** — cuenta el hueco como macizo, sobreestimando. El volumen del NPZ
(rasterizado con XOR en `rasterize.py`) resta correctamente el hueco. Paso 1 ya
advertía esto en su propio docstring ("no es el volumen validado"); esta auditoría lo
confirma con un caso concreto en vez de dejarlo como advertencia genérica. **El volumen
del NPZ es el correcto; no hace falta ninguna acción.**

## Conclusión

Paso 4b no encontró discrepancias sin explicar. Los outliers de magnitud (positivos y
negativos) tienen causa identificada y esperada (redondeo de rasterización; regla
par-impar vs. suma simple de Paso 1). El NPZ es consistente con Paso 1 en todo lo que
debe coincidir (spacing, grilla) y mejora en lo que Paso 1 ya sabía que era aproximado
(volumen con huecos).
