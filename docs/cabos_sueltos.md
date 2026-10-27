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
Reabre: al montar 3D — un modelo con contexto volumétrico podría usar la densidad sin generar esa textura.
Misma familia que el check cráneo-caudal de CyC. Nota menor: para 54/503 pacientes (Brilliance 16, MX
16-slice, 1 ARIA) se usó la curva Big Bore por falta de la propia; conseguir las curvas si se reabre
densidad.
