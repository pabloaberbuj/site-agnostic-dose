# HO#6 — Fase 2: techos de Pelvis por encima de Rx en Ch5

> Prompt para Claude Code. Leer con `contexto_fase_vigente.md` (Tier 2), `CHARTER.md` (§3, §5, §10)
> y la tabla de armonización.
> **Independiente de HO#5.** Higiene de datos en la tabla/canal volumétrico, abierta desde el primer
> reporte de A1a y nunca resuelta. Puede requerir re-pintar canales al cargar (NO regenerar NPZ).

## El problema

En Pelvis, algunos constraints volumétricos de Ch5 tienen techos de **1.3–1.6 × Rx_High** (>1.0 en la
escala del canal, que es fracción de Rx_High). Un "techo" por encima de la propia prescripción pintado
en el canal es, como valor de entrada, ruido: no acota nada físico y mete valores fuera del rango
[0,1] que el resto de los canales respeta.

## Qué hacer

1. **Diagnosticar en la tabla:** qué OARs de Pelvis y qué métricas producen techos >1.0×Rx_High.
   Reportar la lista (OAR, métrica, valor en Gy, valor en fracción, Rx del protocolo).
2. **Entender el origen:** ¿son constraints reales del protocolo (un Vx a una dosis > Rx, legítimo en
   algunos OARs seriales) mal escalados, un error de unidad (Gy vs cGy, o dividido por la Rx
   equivocada), o una entrada espuria de la tabla?
3. **Decidir el tratamiento del canal (reportar propuesta, no aplicar a ciegas):**
   - Si es un constraint real >Rx: ¿se clampea a 1.0, se deja como está, o no se pinta? (Un techo
     >Rx sobre la fracción preservada casi no restringe nada → candidato a no-pintar, como el caso
     cm³ trivial de Thyroid.)
   - Si es error de unidad/escala: corregir en la tabla (bumpea el hash → registrarlo).
4. Confirmar que afecta **solo a Pelvis** y no contamina CyC/Próstata.

## Alcance

Si el fix es en la tabla, **bumpea el hash** — registrarlo y anotar que A1a (hash `7ab16f73fa632a3e`)
quedó corrido con los techos sucios en Pelvis (su resultado de Pelvis se re-lee con la tabla limpia si
hace falta). No regenerar NPZ (los goals se construyen al cargar). Sin reentrenar en este handoff: es
diagnóstico + decisión de tratamiento; el re-run con la tabla limpia se decide con el reporte en mano.

## Handoff de vuelta (§8)

Delta: lista de constraints >Rx + origen + propuesta de tratamiento + hash nuevo si se tocó la tabla.
