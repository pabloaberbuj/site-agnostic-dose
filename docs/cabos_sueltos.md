# Cabos sueltos — diferidos por criterio de corte (CHARTER §11)

> **Tier 3, append-only.** Acá aterriza lo que el §11 clasifica como "se decide y se anota en una
> línea": detalles que a lo sumo *afinan*, no invalidan. Cada entrada: **qué es**, **por qué se
> difirió**, **qué lo reabriría**. Se revisa al cerrar un bloque/fase o en un punto ciego — no en el
> momento. Un cabo registrado es alcance delimitado, no deuda.

---

### Fase 2

**Piso de ruido seed-a-seed.** Qué: no tenemos la varianza entre seeds (reinicialización + orden de
batches); solo existe **seed 42** (el A0 original). Las seeds 43–45 eran de HO#5, que cancelamos — quedó
un cadáver en `results/exp_a0_002_unet2d_seed43/` (checkpoints época 0–1, nunca terminó); Code lo borra o
lo marca para no volver a tropezarlo. Por qué se difirió: es instrumentación, hoy no decide nada — todo
lo que decidió el eje A (SSIM de A1a, fill0 vs fill95, uniform vs distmod, A4) estuvo lejísimos del ruido.
Reabre: la primera comparación que caiga en **zona gris** → medir el piso **pegado a esa comparación**,
no sobre A0 en abstracto. Candidata concreta: **CyC SSIM en HO#9** (dist-solo ≈ A4?) — si CyC queda
ambiguo, ahí se corre seed 43 completo (~14–17 h); si CyC es claro en cualquier dirección, no hace falta.

**A1b — λ=20 mm y piso=0.5 de la modulación por distancia. [CERRADO, no reabierto]** Qué: perillas de
Code que fijan alcance (~2–4 cm) y rango ([0.5,1.0]) del efecto. Resultado A1b: la modulación salió
**neutra-a-peor en las 3 métricas** (SSIM, PTV, OAR), no empatada-pero-prometedora. Por qué no se barre:
para cambiar la decisión (queda uniform), algún (λ,piso) tendría que volver la modulación *ganadora*,
improbable — la coherencia la da *tener* goals (A1b ≫ A0), y la modulación solo toca Ch3/Ch4, que aportan
poco frente al Ch5 ya escalonado. Argumento de mecanismo + dirección, no demostración. Reabre: si aparece
razón para creer que modular Ch3/Ch4 importa (p.ej. multi-protocolo con techos que varían) → barrido corto
(λ~40, piso~0.2).

**PenileBulb Dmax en Pelvis — entrada muerta de la tabla.** Qué: tiene goal pero nunca está presente →
no pinta nada. Por qué se difirió: inofensiva. Reabre: limpieza de doc opcional en el próximo toque de
tabla.

**CyC — redondeo 1.001 en 5 goals** (70 vs 69.96 Gy: Bone_Mandible, Joint_TM_L/R, Lobe_Temporal_PRV3_L/R).
Qué: techos levemente >1.0 por redondeo. Por qué se difirió: es redondeo, no ruido. Reabre: no se reabre
salvo que moleste; dejar como están.

**MAE body incluye soporte/inmovilizadores.** Qué: `mask_BODY` = External real (correcto), con soportes
en fracción variable por paciente. Por qué no es acción: no es error, es física que Ch2 debe ver.
Recordatorio de *lectura*: no comparar el MAE body entre pacientes ni entre sitios.

**A2 / A3 diferidos (no son cabo suelto, es reordenamiento registrado).** Qué: el sub-eje de intención
(augmentar goals, prioridad) es estático por protocolo → redundante en distribución (lo mostró A1). Por
qué se difirió: su payoff solo se mide fuera de distribución (multi-protocolo / zero-shot sobre el
conjunto reservado), que aún no está montado. Reabre: cuando exista eval OOD, o lleguen múltiples
protocolos por sitio. Mientras, el eje A sigue por A4+ (geometría/física paciente-específica).

**Veredicto de goals: leer por sitio, no pooled (posible confounder).** Qué: Pelvis es 1 de 3 sitios y
sus goals de OAR pesan poco (anatomía más tolerante, menos OARs críticos pegados al PTV); puede tirar
hacia abajo un veredicto pooled de "los goals no aportan". Por qué importa ahora: la decisión de poda de
HO#9 (dropear goals) hay que leerla **por sitio** — si dist-solo cae en SSIM en CyC pero empata en
Pelvis/Próstata, los goals se quedan (los justifica CyC). Reabre: ya activo en la lectura de HO#9;
revisar de nuevo al ampliar patologías (cambia el balance de sitios donde los goals importan).

**Check cráneo-caudal para CyC al entrar 3D (eje B).** Qué: CyC es cráneo-caudal y el modelo 2D no ve
ese eje; pista cuantitativa = el SSIM de CyC fue el más bajo de todo el eje A (0.375 en A0). Por qué se
difirió: es del eje B (U-Net 3D con parches), no del eje A actual. Reabre: al montar 3D — pensar un check
de coherencia que mire específicamente lo cráneo-caudal, no solo DVH/SSIM por corte; CyC es el sitio
donde esperaríamos el salto del 3D.

**Densidad (A5): descartada en 2D, revisar en 3D (eje B).** Qué: la densidad no aportó y restó coherencia
(SSIM −0.07 parejo en los 3 sitios). Hipótesis: un modelo 2D-MAE copia los bordes filosos de densidad
(hueso/aire) a la dosis, que es suave aun cruzándolos → textura espuria → SSIM abajo. Por qué se difirió:
el veredicto (no entra en A) es robusto; la réplica seed no se corrió porque no cambia la acción (§11).
Reabre: si el 3D pasa a arquitectura de operación (en Fase 3 el 3D no mejoró el DVH, ver `fase3_cerrada.md`) — un modelo con contexto volumétrico podría usar la densidad sin generar esa textura.
Misma familia que el check cráneo-caudal de CyC. Nota menor: para 54/503 pacientes (Brilliance 16, MX
16-slice, 1 ARIA) se usó la curva Big Bore por falta de la propia; conseguir las curvas si se reabre
densidad.

**A6 beam plate: no corrido en VMAT-only, diferido a fase IMRT.** Qué: análisis geométrico (Code) mostró
que con arcos coplanares ~360°, el beam plate templado (sin forma de campo) colapsa a |z−z_iso| →
redundante con PSDM. La única info direccional (CyC parciales = homolaterales, ~11%) es derivable de la
geometría de PTV mid/low que la red ya tiene; la forma de campo (MLC/BEV) viene del plan real, no se
templa ni existe en inferencia. Por qué se difirió: resultado predeterminado (plano); no se corrió (§11:
análisis estático > corrida confirmatoria). Reabre: fase IMRT/multi-técnica, donde el beam plate distingue
arco vs 7-campos (geometrías muy distintas). Nota: aun ahí, sin forma de campo templada queda un piso de
mismatch real-vs-templado.

### Fase 3 / eje B

**SSIM enmascarado por BODY (des-diluir el aire). — RESUELTA en Fase 3 (ver `fase3_cerrada.md`; el SSIM de frame completo quedó retirado, vara vigente en `results/ssim_validacion/`).** Entrada original: Qué: el SSIM se computa sobre el frame completo, que
incluye el aire (pred≈0, real≈0) matcheando trivialmente e inflando el índice. Verificado en el val de
exp_a4_003: ~47% del frame es aire en CyC (59–61% en pelvis/próstata) y, dentro de cada sitio,
rho(SSIM, fracción-aire) = +0.28 (CyC) / +0.47 (Pelvis) / +0.68 (Próstata). Alternativa: SSIM solo dentro
de la máscara BODY → frame-invariante y centrado en la estructura de dosis. Por qué se difirió: el delta
2D→3D cancela el aire en gran medida (mismo frame, ambos modelos predicen aire≈0), así que la comparación
de eje B es robusta sin enmascarar; y el instrumento primario de coherencia cráneo-caudal para CyC es d2,
no el SSIM (secundario, §2). Enmascarar ahora no cambia ninguna conclusión. Reabre: que el delta de SSIM
en CyC (3D−2D) caiga en zona gris Y d2 quede ambiguo. Ahí, recomputar SSIM enmascarado por BODY sobre las
predicciones existentes (barato, no entrena) para ver si el efecto estaba tapado por la dilución.

**Criterio de selección de checkpoint: val/mae enmascarado por BODY vs. métricas que sí deciden.** Qué: el
mejor checkpoint y el early stopping se eligen por `val/mae` (la misma pérdida de entrenamiento, medida en
val), no por las métricas que deciden las comparaciones (DVH por sitio con IC95: OAR-media / PTV-alta; d2 en
CyC). El MAE sale gratis al cierre de cada época; el DVH exige inferencia, des-padeo, renormalización y
histogramas sobre los 65 pacientes de val, varios minutos por época. No encontré documento que justifique la
elección: probablemente heredada del 2D de la Fase 2 (suposición). Limitaciones: promedia sobre todo el
cuerpo, donde PTV/OAR son una fracción chica; evidencia en el guardrail: A4-2D en el contenedor tenía un
val/mae ~10% peor que el viejo en las mismas épocas (3.21 vs 2.89 en la época 52) y el DVH quedó dentro de
±0.5 pp; diferencias de ~0.1 de MAE entre épocas (p.ej. 1.87 vs 1.95 en el 3D) pueden ser ruido para el
DVH, y el early stopping elige entre ellas. Además se selecciona sobre val, que también entra en la
comparación 2D vs. 3D: sesgo optimista leve y parejo para ambos modelos (test reservado). Por qué se
difirió: cambiar el criterio a mitad de la comparación rompería la regla común 2D/3D, y hoy no cambia
ninguna conclusión. Reabre: que el 3D−2D en DVH caiga en zona gris, o que el mejor checkpoint por MAE sea
sospechoso (curva oscilante, diferencias dentro de ruido). Primer paso, barato y sin entrenar: evaluar DVH
sobre los 3 mejores checkpoints que ya guarda Lightning (análisis de sensibilidad); si dan lo mismo, la
elección por MAE no importa. A futuro: monitor secundario con DVH sobre un subconjunto de val cada ~10
épocas, o criterio combinado MAE + OAR-media + PTV-alta.

**Piso de seed en PTV-alta (Fase 3).** Qué es: el 3D da Pelvis +1.75 / Próstata −1.63 en PTV-alta, con IC que excluyen 0, pero con una sola seed. Medir seed 43 del 2D para conocer la varianza entre corridas. Por qué se difirió: sea real o ruido, la decisión es la misma (el 3D no gana en general). Qué lo reabriría: que una comparación de Fase 4 caiga en zona gris, o querer atribuir esos efectos.

**3D como trade-off de coherencia en CyC (Fase 3).** Qué es: el 3D mejora el SSIM dentro del BODY (+0.03/+0.04) y el MAE-body (1.72 vs 2.32) en CyC, con DVH empatado. Por qué se difirió: la prioridad es el DVH, y el 3D es caro y está al límite de memoria. Qué lo reabriría: que la visualización médica (§1, uso 1) pase a ser el objetivo principal, o que una loss de Fase 4 cambie el ranking en DVH.

**Variantes 3D no probadas (Fase 3).** Qué es: pooling anisótropo en Z, más capacidad, MedNeXt-3D, UpKern k5. Por qué se difirió: firma de piso en OARs; ni el 3D vanilla ni MedNeXt-2D ganaron en DVH. Qué lo reabriría: evidencia de error atribuible a la arquitectura (por ejemplo, el piso de seed muestra que la arquitectura mueve PTV-alta).

**Loss × arquitectura (Fase 3).** Qué es: el veredicto 2D ≥ 3D vale con MAE. Qué lo reabriría: que la Fase 4 adopte una loss no-MAE → una corrida 3D con esa loss.

**PSDM como 2.5D (Fase 3).** Qué es: 3D vs 2D sin PSDM (input A1), para atribuir por qué el 3D no aporta. Por qué se difirió: explica, pero no cambia la decisión operativa. Qué lo reabriría: querer publicar "el 3D no aporta" y atribuirlo.

**A4 sin goals (Fase 3).** Qué es: el efecto de los goals se midió sin PSDM (A0 vs A1); el costo de ~0.9 pp en PTV-alta de CyC no se conoce con PSDM. Por qué se difirió: los goals se mantienen por diseño (canal de intención); la corrida no cambiaría esa decisión. Qué lo reabriría: que PTV-alta de CyC pase a ser el cuello de botella.

**Piso de etiqueta definitivo (Fase 3).** Qué es: medir la variabilidad entre planificadores con pacientes que tengan más de una versión de plan (auditoría de Fase 0). Qué lo reabriría: querer cuantificar el techo alcanzable (Fase 4, publicación).

**Técnicas no coplanares (Fase 3).** Qué es: "el 3D no aporta" es un hallazgo de VMAT coplanar. Con arcos no coplanares (SBRT, RC) deja de valer la separabilidad por corte. Qué lo reabriría: la incorporación de esas técnicas → re-evaluar el 3D desde cero.
