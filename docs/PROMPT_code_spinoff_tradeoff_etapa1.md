# PROMPT Code — Spin-off trade-off, Etapa 1: estructura del residual de DVH (real − A4)

> **Spin-off exploratorio, fuera del eje principal.** No es Fase 4. No entrena nada. No toca
> `contexto_fase_vigente.md`, configs, splits ni la tabla de armonización. Todo lo que escribe va a
> `results/spinoff_tradeoff/etapa1/` y a scripts nuevos en `scripts/spinoff_tradeoff/`.
> Leer antes: `CHARTER.md` (§2, §5, §7, §11) y `fase3_cerrada.md`.

## 0. Pregunta que responde

El modelo determinista predice el DVH "típico" para una anatomía. La diferencia entre el DVH real y
el predicho es la parte que el promedio no explica. Queremos saber, **por sitio**:

1. ¿Cuánta estructura tiene ese residual (cuántos ejes por encima del ruido)?
2. ¿Qué cargan esos ejes? En particular: ¿acoplan estructuras entre sí (OAR contra PTV, hotspot o
   caída de dosis) o son error independiente por estructura?
3. ¿Es anatomía que el modelo no capturó, o es variación del plan a anatomía fija?

Contexto para leer el resultado: Próstata y Pelvis se planifican a mano; CyC con RapidPlan. La
expectativa es más residual estructurado en los dos manuales. No es un supuesto a confirmar: si no
sale, se reporta tal cual.

## 1. Insumos fijos

- Datos: `npz_v2b_body`, `split_v2.json`, hash de armonización `7ab16f73fa632a3e` (432 pacientes).
- Modelo: `exp_a4_003_unet2d_psdm_c176x256`, checkpoint ep113 (arquitectura de operación).
- Inferencia con el pipeline de evaluación existente: des-hacer crop y pad a grilla nativa, dosis
  predicha × BODY.
- Curvas DVH: las mismas funciones de `dvh_curva_completa` para real y predicho. Ninguna
  reimplementación paralela.

## 2. Paso A — DVH real y predicho para los 432

- Inferir los 432 y guardar, por paciente, la columna `split` (train / val / test).
- **Renormalizar la dosis predicha a D95(PTV_High)=100%** antes del DVH, igual que la real. Motivo: un
  error global de escala del modelo cargaría en todas las variables a la vez y aparecería como un eje
  espurio. Guardar el factor de escala por paciente como columna aparte.
- **Los análisis sobre el residual usan solo val+test** (fuera de muestra). En train el residual es
  optimista porque el modelo vio esos planes. Train se calcula igual, pero solo para el chequeo de
  optimismo del §5.6.
- Uso del test: este análisis es no supervisado y no selecciona modelo. **No reportar métricas
  agregadas de desempeño del eje principal sobre test** (ni MAE ni la métrica DVH por banda); solo la
  estructura del residual. Dejar esto asentado en el resumen.

## 3. Paso B — vector de resúmenes por paciente

Convención: `residual = real − predicho`. Todas las dosis en % de Rx_High. Volúmenes y distancias con
el spacing nativo de cada paciente (hay 10 con Z=2 mm).

**OARs.** Los que tienen rol y goal en `sites_<sitio>.yaml` y están presentes en ≥90% de los pacientes
del sitio (listar cuáles quedan y cuáles no). Estructuras bilaterales: ordenar como proximal/distal al
PTV_High por distancia, no como izquierda/derecha. Por OAR:

- ΔV(D) medio por banda de dosis, en puntos porcentuales de volumen, **sin normalizar por el goal**:
  - vector compacto: baja 5–40, media 40–80, alta 80–100;
  - vector fino: 5–20, 20–40, 40–60, 60–80, 80–100.
- ΔDmean.
- ΔD0.035cm³ si el OAR tiene goal de máximo.

La zona baja es parte central del análisis, no un agregado: ahí la optimización controla menos y se
espera la mayor dispersión.

**PTV (por nivel).** ΔD98, ΔD2, ΔD50 y homogeneidad (D2−D98)/D50. Con la normalización, D95(PTV_High)
vale 100% en real y predicho, así que la cobertura del nivel alto no es variable libre: se lee por
D98 y por la homogeneidad. En niveles Mid/Low, además, ΔD95 respecto de su Rx.

**Hotspot.** ΔD0.035cm³ del BODY y Δ del volumen (cm³) con dosis >107% fuera de los PTV.

**Caída de dosis en tejido sano.** Δ de dosis media en cáscaras del BODY por distancia a la unión de
los PTV (excluyendo los PTV): 0–5, 5–10, 10–20, 20–30, 30–50 y >50 mm. Aparte, Δ de dosis media del
BODY fuera del rango Z de los PTV, y Δ de dosis media del BODY completo.

**Covariable del plan real (sin contraparte predicha).** D95(PTV_High) en Gy dividido por la Rx del
protocolo, del QA gate de Fase 0. Es la única huella del compromiso de cobertura que la normalización
borra. No entra al PCA; se correlaciona con los scores (§5.5).

Guardar una tabla tidy por paciente con real, predicho y residual de cada variable.

## 4. Paso C — descriptores anatómicos y elección de la métrica de similitud

Solo geometría, sin dosis. Sobre los pacientes elegibles de cada sitio (no hace falta fuera de
muestra: no interviene el modelo, salvo en la candidata C).

- **Candidata A (solo Próstata, referencia):** las 7 features de `d3_sonda_vecinos.py` del repo de
  próstata (volúmenes de PTV, recto y vejiga; solapamientos absolutos y relativos), estandarizadas,
  distancia euclídea.
- **Candidata B (OVH por rol):** para cada OAR, la curva de fracción de volumen a menos de una
  distancia r del PTV, respecto de PTV_High y respecto de la unión de PTV. Agregar: volumen del OAR,
  volumen de cada PTV, fracción del OAR dentro del rango Z de los PTV, y área media de la sección del
  BODY en ese rango. Distancia entre pacientes: área entre curvas OVH por rol (en mm), combinada con
  las escalares estandarizadas. Documentar cómo se ponderan los roles.
- **Candidata C (cota):** distancia entre los DVH predichos por A4 (vector compacto).

**Criterio de elección, igual que D3:** k=3 vecinos dentro del sitio; spread (desvío estándar) del
DVH **real** dentro de cada grupo de 4, por OAR y por banda (baja, media, alta). Comparar las
candidatas entre sí y contra vecinos elegidos al azar (línea de base). La métrica de trabajo es la
geométrica (A o B) que menos spread deja; C se reporta pero no se usa para el §5.4.

## 5. Paso D — análisis del residual (por sitio, val+test, vector compacto)

1. **Sesgo y magnitud.** Media (sesgo sistemático del modelo) y desvío de cada variable del residual.
   El sesgo se reporta aparte y se resta antes de lo que sigue.
2. **PCA** sobre el residual centrado y estandarizado por variable. Como sensibilidad, repetir sin
   estandarizar dentro de cada bloque de unidades homogéneas.
3. **Cuántos ejes superan al ruido.** Análisis paralelo: permutar cada columna entre pacientes de forma
   independiente, rehacer el PCA muchas veces y retener solo los componentes cuyo autovalor supere el
   percentil 95 de esa distribución nula. Además, estabilidad de los loadings por bootstrap de
   pacientes (similitud coseno entre remuestreos).
4. **Acoplamiento entre bloques** (el resultado principal). Las bandas de un mismo OAR co-varían por
   construcción, así que pocos ejes no alcanzan como evidencia. Lo que distingue los casos es la
   relación entre bloques: OAR, PTV (homogeneidad, D98), hotspot y caída de dosis. Medir la
   correlación entre los scores principales de cada bloque, con nula por permutación, y reportar el
   **signo**:
   - antagónico (OAR mejor ↔ PTV, hotspot o caída peor): compromiso entre objetivos;
   - sinérgico (todo mejor o todo peor a la vez): eje de calidad global del plan;
   - sin acoplamiento: error independiente por estructura.
5. **¿Es anatomía?** Para cada componente retenido: R² con validación cruzada dejando un paciente
   afuera, prediciendo el score desde los descriptores anatómicos de la métrica de trabajo; y
   correlación del score de cada paciente con el promedio de sus vecinos. Correlacionar también los
   scores con la covariable de D95 absoluto.
6. **Optimismo en train.** Razón entre la magnitud del residual en train y en val+test, por sitio.
   Decide si la etapa siguiente necesita predicciones out-of-fold.

## 6. Paso E — réplica sin modelo (por sitio, todos los elegibles, vector fino)

Solo con DVH reales: para cada paciente, desviación respecto del DVH medio de sus k vecinos (métrica
de trabajo; probar k=3 y k=5). PCA sobre esas desviaciones, con el mismo análisis paralelo y el mismo
análisis de acoplamiento entre bloques. Comparar el subespacio con el del §5 (similitud coseno entre
loadings, llevando el vector fino al compacto). Si coinciden, los ejes no son un artefacto de A4.

## 7. CyC como contraste

Mismo pipeline. Comparar entre sitios la varianza del residual en las variables comunes (PTV_High,
hotspot, cáscaras, BODY) y la cantidad de ejes retenidos. Es un contraste, no un control limpio: CyC
difiere de los otros dos en generador **y** en anatomía. Reportarlo con esa salvedad.

## 8. Guardrails

- Nada de dosis en los descriptores anatómicos (A y B).
- Vecinos siempre dentro del mismo sitio.
- Mismo código de DVH para real y predicho; misma normalización.
- Solo `patient_id` anónimo en cualquier salida.
- Con ~50 pacientes fuera de muestra por sitio los intervalos van a ser anchos: reportar siempre el
  IC (bootstrap) y el n, y no concluir sobre un componente que no pase la nula.
- Si algo del pipeline obliga a una decisión no prevista acá, elegir lo razonable, anotarlo en una
  línea en el resumen y seguir.

## 9. Entregables

En `results/spinoff_tradeoff/etapa1/`:

- `resumen.md` (corto) y `summary.json`.
- Tabla de elección de métrica de similitud: spread por candidata, sitio y banda, contra azar.
- Por sitio: sesgo y desvío por variable; autovalores contra la nula; loadings de los componentes
  retenidos; matriz de acoplamiento entre bloques con signo e IC; R² anatómico por componente;
  coseno contra la réplica del §6; razón de optimismo train / fuera de muestra.
- Gráficos: scree con banda nula, heatmap de loadings, dispersión de scores entre bloques.

**Lectura predeclarada (umbrales provisorios, no ajustar después de ver los datos):**

| Resultado en un sitio manual | Lectura |
|---|---|
| ≥1 componente sobre la nula, estable (coseno >0.8), acoplamiento antagónico significativo, R² anatómico <0.2, replica en §6 (coseno >0.7) | Hay eje de compromiso utilizable como coordenada |
| Componente estable con acoplamiento sinérgico | Eje de calidad global: sirve para rankear planes, no como coordenada de preferencia |
| R² anatómico alto | Anatomía que A4 no captura: vuelve al eje principal como hallazgo, no al spin-off |
| Nada sobre la nula | Sin estructura detectable con este n: evaluar out-of-fold antes de descartar |

## 10. Fuera de alcance (etapas posteriores)

- Planes extra por paciente (no aprobados): no están extraídos ni procesados. La validación con pares
  del mismo paciente es la Etapa 2.
- Predicciones out-of-fold para los 432.
- Canal de conditioning, generador, selector y cualquier entrenamiento.

Al terminar, traer al chat solo `resumen.md` y `summary.json`.
