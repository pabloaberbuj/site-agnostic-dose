# Aprendizajes Transversales

> **Tier 3 — Append-only.** Memoria institucional curada. Las fases se cierran aquí.
> Cada entrada: línea con el hallazgo, luego puntero a dónde vive la evidencia/detalle.

---

## Heredados del proyecto de próstata

*(Ver CHARTER.md §7 para aplicabilidad a este proyecto)*

- **ML clásico vs. U-Net (próstata, constraints conocidos):** ML clásico ganó para clasificar constraints con features geométricos acotados. No transfiere a DVH completa con OARs variables. [Próstata proyecto, exp0XX]

- **Contaminación vive en dosis, no en máscaras:** Detectar con auditoría de dosis, no volumen/overlap. Lección nodal: 28 pacientes nodales enmascararon "techo de arcos" espurio. [Próstata proyecto, auditoría post-exp0XX]

- **Bugs de copy-paste en datos:** Spacing nativo sobre grilla downsampleada (1.1×–4.2×), cargar_ct cargaba RD. Argumenta preprocesador nuevo. [Próstata proyecto, Fase 0]

- **MAE puro ganó; DVH loss no aportó:** En próstata, MAE simple superó losses basadas en DVH/momentos. Baseline para comparación. [Próstata proyecto, Fase 4]

- **3D no aportó en próstata (contexto específico):** Próstata tiene gradiente axial suave. CyC/pelvis se espera que sí mejoren con 3D (estructura cráneo-caudal). [Próstata proyecto, Fase 3]

- **Higiene documental:** Tier 2 reescrito a verdad de hoy; append-only para cambios históricos. Evita acumulación lineal de correcciones. [Próstata proyecto, docs]

---

## De este proyecto

*(Vacío hasta fin de Fase 0.2)*
