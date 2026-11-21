# Spin-off trade-off — Nota de lecturas corregidas (Etapas 1 y 1b)

Escrita para: quien decide los próximos pasos del proyecto (lectura rápida, sin releer los resúmenes completos).

## Qué cambió tras el arreglo del R² anatómico
Un defecto numérico (descriptores casi constantes en un pliegue LOO daban |z| hasta ~1900) producía R² absurdos. Se recortó z a ±5 y se rehicieron ambas etapas. Solo cambia el R² anatómico; PCA, estabilidad, acoplamientos, réplica y optimismo quedan idénticos.

| Etapa 1, R² anat. por componente | Antes | Después |
|---|---|---|
| Próstata | 0.37, −0.10, −0.12, −0.14 | 0.38, −0.10, −0.10, −0.14 |
| Pelvis | −1.33, −0.72 | −0.67, −0.09 |
| CyC | −1.25, −2.56, −0.53, −6.39 | 0.16, −0.18, −1.24, 0.04 |

Las lecturas predeclaradas de la Etapa 1 no cambian.

## Lecturas por sitio

**Próstata**
- Etapa 1: un componente estable (cos 0.96) con R² 0.38, pero el IC (−0.13 a 0.55) incluye 0 y el resto de los componentes tiene R² negativo. Acoplamiento OAR vs PTV/HOT/FALL antagónico pero no significativo.
- Etapa 1b: ΔD95 no muestra compromiso con los OAR (score OAR r=−0.25, p Holm 0.52). R² anatómico 0.52 con todos los elegibles, 0.35 en val+test con IC [−0.04, 0.57].
- Lectura: hay una señal anatómica moderada, no confirmada fuera de muestra. "Anatomía que A4 no captura" es una hipótesis, no un hallazgo.

**Pelvis**
- Es el único resultado robusto. ΔD95 es antagónico con los OAR (r=−0.48 [−0.61, −0.31], p Holm 0.009; 4 de 9 pruebas, sobre todo Bowel_Small). Sobrevive a rangos y a quitar extremos.
- El R² anatómico es ≈0, así que no se predice desde la anatomía: es criterio del planificador (coordenada de preferencia).
- Salvedad: el eje de derrame (HOT~PTV r=0.94) está muy correlacionado con ΔD95. Parte del efecto puede ser la escala global de dosis y no un compromiso puro.
- En la Etapa 1, el acoplamiento sinérgico (OAR con HOT/FALL) indica un eje de calidad global que rankea planes.

**CyC (contraste, RapidPlan)**
- Etapa 1: acoplamiento sinérgico y significativo en los tres bloques. R² anatómico por componente ≈0 o negativo.
- Etapa 1b: la lectura "antagónico, R² 0.56" no se sostiene. r=−0.67 tiene p Holm 0.50. Con rangos y sin extremos hay 0 de 48 pruebas significativas, y el R² baja a 0.22 y 0.09. En val+test es 0.05 con IC [−2.69, 0.27].
- Los extremos de ΔD95 (hasta −26 pp) son los casos de cavum con PTV solapado al PRV de quiasma (QA `reviewed_ok`).
- Lectura: sin evidencia de compromiso ni de predictibilidad anatómica. CyC no es un control limpio (generador y anatomía distintos).

## Otras salvedades
- Todo es val+test, con n≈50 por sitio: los IC son anchos.
- La inferencia de 1b para el predicho se hizo en CPU (152 pacientes), verificada contra los valores en GPU de la Etapa 1.
- La sensibilidad por extremos (z robusto < −3.5) no estaba en el prompt; el corte se fijó después de ver la distribución de ΔD95.
- Las líneas "Lectura predeclarada" de `etapa1b/resumen.md` para Próstata y CyC son más optimistas que la sensibilidad. Prevalece esta nota.
