# Prompt Code — Fase 4, harness de evaluación del track de optimización (loss cuantílica)

Evaluación, no entrenamiento — no toca el modelo ni la loss. Mide un producto distinto al de visualización,
con otras métricas. **Actualizado tras el control MAE** (`exp_a4_003` ep113, val): el harness quedó validado,
con una jerarquía de lectura que corrige el criterio de sanity inicial.

## Qué es / qué no es

- **No usa la vara `exp_a4_003` ni la regla de corte de visualización.** La cuantílica empeora MAE y DVH
  contra el plan real **por construcción** (predice el mejor plan alcanzable, no el promedio); medirla con esa
  vara sería leer la métrica equivocada.
- Corre sobre **cualquier predicción de dosis** (pred × BODY, en % Rx_High). Esa independencia habilitó el
  control MAE de abajo.
- Sobre **val**. Test sellado hasta cerrar Fase 4. **Por sitio, IC95 bootstrap** sobre pacientes.

## Normalización: `none` (convención única, misma en MAE y en el cuantílico)

Fijado. El cuantílico entrena en el espacio normalizado D95(PTV_High)=100%, que es donde la loss no puede
comprar OARs resignando cobertura → la evaluación va en **ese** espacio (`none` = usar la dosis tal como
sale). `ptv95` queda descartada: reintroduce la perilla de normalización y **da vuelta los resúmenes por
órgano** (Pelvis DVH-point 0.52→0.29) mientras la métrica robusta (vóxel tejido sano) aguanta igual —
evidencia de que el movimiento es de la normalización, no del plan.

## Validación mecánica del harness (control MAE) — QUÉ pasó y QUÉ significa

**La única prueba de pass/fail es la calibración vóxel en tejido sano.** MAE predice la mediana por vóxel →
debe dar ≈0.50. **Dio 0.51 / 0.54 / 0.51 (CyC/Pelvis/Prost), los tres dentro de IC de 0.50.** Signo y cálculo
correctos. El harness sirve.

**DVH-point y OVH NO son pass/fail de 0.50 / p50.** Son el baseline de offset contra el que se difiere el
cuantílico, con dos sesgos conocidos que no son bugs:
1. **La fracción `real>pred` por órgano es hipersensible al signo:** la gobierna la *mediana* del sesgo, no su
   magnitud. En CyC-Dmean la fracción da 0.65 con un sesgo medio de **+0.015 %Rx** (trivial). Leer siempre la
   magnitud en %Rx junto a la fracción.
2. **Dmax (D0.035cc) está sesgado a la baja por el suavizado** de la predicción (el casi-máximo de un campo
   suave subestima el real): Próstata **+5.7 %Rx**, OVH en p20. En CyC el sesgo de magnitud es chico (+0.6).

No cambiar Dmax a D2%: no arregla nada que el differencing no maneje, rompe coherencia con el constraint
congelado D0.035cc≡Dmax (§5), y la métrica robusta es el vóxel de todos modos.

## Jerarquía de lectura para el cuantílico

- **PRIMARIA (decisión): calibración vóxel en tejido sano** (BODY−PTV). Robusta, baseline MAE ≈0.50, y es
  donde actúa la loss. Éxito = moverla hacia **(1−τ)** por sitio, con IC.
- **SEGURIDAD: PTV-alta (80–110) del cuantílico vs la vara MAE.** No debe degradarse material. Si se degrada,
  la loss compró OARs resignando cobertura y el espacio normalizado no bastó.
- **DIAGNÓSTICAS: DVH-point y OVH**, por **differencing contra el baseline MAE** y leyendo **magnitud en %Rx**,
  nunca la fracción/percentil contra 0.50/p50 pelado.

## Métrica 1 — Calibración por sitio (convención confirmada)

`calib = fracción de pacientes con dosis_real > dosis_pred`. Predictor τ bien calibrado → ≈ **(1−τ)**
(τ=0.35 → 0.65). Granularidades: **vóxel tejido sano** (primaria, robusta) y **DVH-point por (sitio, rol,
métrica)** (diagnóstica, con las cautelas de arriba). IC95 bootstrap sobre pacientes.

## Métrica 2 — OVH por rol (diagnóstica)

OVH (overlap volume histogram) = firma geométrica OAR–PTV_High. Para cada OAR de val: computar OVH → vecinos
del mismo sitio con OVH similar (k-NN, k=10; robusto a k=5/20) → percentil del valor predicho dentro de los
valores **reales** de los vecinos. Objetivo del cuantílico: caer bajo p50 (mejor que la mediana de vecinos),
hacia el percentil del τ pedido. Leer por differencing contra el baseline MAE.

## Hipótesis pre-registrada (del charter)

Gana **Pelvis**; Próstata y CyC neutros. Reportar por sitio para contrastarla antes de mirar el resultado.

## Salida / handoff

- `results/<harness_id>/`: calib vóxel + calib DVH-point + OVH-percentil, todo con IC95; `metrics.csv` +
  `summary.json` + CSV por paciente. Cache del pool en `results/f4_harness_pool/`. **Sin commit** salvo aviso.
- Delta al chat: calib vóxel tejido sano por sitio (lo primario) + PTV-alta vs MAE + DVH-point/OVH por
  differencing, para MAE (control) y para el cuantílico.

## Decisiones nivel 2 (fijadas)

- k vecinos OVH = 10 (resultado estable a 5/20). Puntos DVH = los de la tabla de armonización (Dmean
  paralelos, Dmax seriales); ampliar a dosis baja si hace falta.

## Lo que NO hace

No entrena, no selecciona checkpoint, no se compara con `exp_a4_003` por pooled/IC. Evalúa un producto
distinto (uso 2 del §1). El modelo MAE sigue siendo el de visualización.
