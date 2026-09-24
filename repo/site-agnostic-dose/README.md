# Site-Agnostic Dose Prediction (KBP)

**Entrenamiento de red neuronal site-agnostic única** para predecir distribuciones 3D de dosis en PTVs y OARs a partir de anatomía + información de prescripción.

Ver [CHARTER.md](docs/CHARTER.md) (Tier 1 — estable) para objetivos, fases, decisiones de diseño.

---

## Estructura del proyecto

```
repo/site-agnostic-dose/
├── configs/              # Archivos YAML — un experimento por config
├── src/
│   ├── preprocess/      # Preprocesador nuevo, limpio (input → grillas)
│   ├── datamodules/     # DataModule (splits, carga, augmentation)
│   ├── models/          # Arquitecturas (U-Net, MedNeXt, etc.)
│   ├── losses/          # Loss functions (MAE, DVH loss, SSIM)
│   └── bridge/          # Dosis → grid físico (unet_to_target)
├── scripts/             # train.py, evaluate.py, profile.py, auditorías
├── commissioning/       # (En pausa) stack PDRT → RTPLAN
├── docs/
│   ├── CHARTER.md               # Objetivos, fases, decisiones congeladas
│   ├── contexto_fase_vigente.md # Tier 2 — doc vivo de la fase actual
│   └── aprendizajes_transversales.md  # Tier 3 — append-only curado
├── data/                # Pequeño — SÍ va a git
│   ├── splits/         # JSON de splits (definen reproducibilidad)
│   └── harmonization/  # Tabla TG-263: sitio → (OAR, métrica) → (goal, prioridad)
└── papers/             # Referencias, publicaciones

SiteAgnostic/data/      # Hermana del repo — NO va a git (pesada)
├── dicom_raw/          # DICOM crudos por sitio/paciente
├── dicom_extracted/    # CSV de métricas + estructuras
├── npz_v<N>/           # NPZ preprocesados, VERSIONADOS
├── checkpoints/        # Checkpoints por exp_id
├── predictions/        # Dosis predichas para evaluación
└── results/            # metrics.csv, summary.json, plots
```

## Documentación

- **[CHARTER.md](docs/CHARTER.md)** — Leer primero. Objetivos macro, fases, métricas de éxito, decisiones congeladas, esquema de repo.
- **[contexto_fase_vigente.md](docs/contexto_fase_vigente.md)** — Se reescribe cada fase. Estado actual, bloques, decisiones pendientes.
- **[aprendizajes_transversales.md](docs/aprendizajes_transversales.md)** — Append-only. Lecciones del proyecto de próstata + hallazgos de este proyecto.

---

## Flujo de trabajo (Tier 1 → Tier 2 → Tier 3)

1. **Fase abierta:** Lee [CHARTER.md](docs/CHARTER.md) + [contexto_fase_vigente.md](docs/contexto_fase_vigente.md)
2. **Durante la fase:** Se edita el Tier 2 (contexto_fase_vigente.md)
3. **Al cerrar la fase:** 
   - Congela bloques del Tier 2 → Tier 3 (aprendizajes_transversales.md)
   - Crea nuevo Tier 2 para próxima fase
   - Actualiza CHARTER.md solo si cambios estructurales

Más detalle en CHARTER.md §8.
