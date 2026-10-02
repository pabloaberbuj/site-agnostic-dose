# Insumos para Fase 0.0 (Profiling) — research del proyecto de próstata + DICOM de ejemplo

> Documento de research puntual, no Tier 2 ni Tier 3. Generado para pasar a Claude
> Project (chat) como insumo de arranque de Fase 0.0 (profiling PC) y 0.1 (preprocesador).
> Fuente: `C:\Pablo\ProstateDoseProject\repo` (proyecto de próstata) y DICOM de ejemplo en
> `\\10.100.0.252\centro_de_datos2018\101_Cosas de\PABLO\Dcm export 0.0` (2 pacientes ×
> CyC, Pelvis, Próstata).
>
> Fecha: 2026-09-24.

---

## 1. `profile_step.py` (viejo, próstata)

- **Memoria:** `torch.cuda.memory_allocated()` (VRAM tras cargar el modelo) y
  `torch.cuda.max_memory_allocated()` (pico durante el step) — ambas en MB. No usa
  `torch.profiler` ni nvidia-smi, solo estas dos llamadas nativas.
- **Qué mide:** step completo (forward + backward + `optimizer.step` vía AMP
  `GradScaler`), pero además desglosa cada componente por separado: I/O por paciente
  (carga NPZ + pad), transfer CPU→GPU, forward solo, backward+optim solo, y step
  completo — con 1 warmup + 3 mediciones repetidas de cada uno.
- **U-Net perfilada:** la que arma `DosePredictionModule`
  (`src/models/lightning_module.py`) según el `--config` pasado — es decir, usa el
  modelo real de entrenamiento, no una U-Net de juguete. Es directamente comparable
  con exp002 si se le pasa ese config.
- **Barrido:** NO barre. Corre **un solo config** por invocación
  (`--config configs/expXXX.yaml --n-patients 5`). Para comparar arquitecturas/tamaños
  hay que correrlo múltiples veces a mano.
- Estima tiempo de época extrapolando `(io_avg + step_avg) × n_pacientes_totales` y da
  un veredicto de bottleneck (I/O vs compute) con un umbral simple.

## 2. Config `exp002_unet2d_psdm.yaml`

| Parámetro | Valor |
|---|---|
| Arquitectura | `unet2d`, **depth=4**, **base_features=16** |
| Norm | **GroupNorm(num_groups=8)** confirmado en código (`src/models/unet2d.py` y `unet3d.py`), no BatchNorm — comentario explícito: "los volúmenes de validación van uno a uno" |
| in_channels | **5** = CT + BODY + PSDM_PTV + PSDM_Rectum + PSDM_Bladder |
| out_channels | 1 |
| upsample | bilinear (no transposed conv) |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-5, cosine scheduler, warmup 5 épocas |
| AMP | **Sí** — `precision: "16-mixed"`, `torch.autocast(dtype=float16)` + `GradScaler` |
| batch_size | 1 |
| Corte/parche | **NO es parche 3D ni crop espacial.** Es 2D con contexto axial: `context_slices=3` apila canales de z-1/z/z+1 (bordes replicados) sobre el mismo corte 2D — el target sigue siendo un único corte z. `inplane_size=256` fija el plano completo (no hay recorte in-plane). `z_margin_slices=5` es margen de padding en Z, no crop. |

## 3. DICOM headers (2 pacientes por sitio, RD+RS)

| Sitio-paciente | Grilla dosis (z,y,x) | Spacing (y,x / z) | BBox BODY extent (x,y,z mm) | **Z-extent (cráneo-caudal)** |
|---|---|---|---|---|
| CyC1 | 161×110×251 | 2.5/2.5 mm — 3.0 mm | 625×270×480 | **480.0 mm** |
| CyC2 | 139×106×192 | 2.5/2.5 mm — 3.0 mm | 476×262×414 | **414.0 mm** |
| Pelvis1 | 147×115×194 | 2.5/2.5 mm — 3.0 mm | 482×285×438 | 438.0 mm |
| Pelvis2 | 144×89×164 | 2.5/2.5 mm — 3.0 mm | 407×220×429 | 429.0 mm |
| Ptta1 | 137×100×149 | 2.5/2.5 mm — 3.0 mm | 368×246×408 | 408.0 mm |
| Ptta2 | 134×97×145 | 2.5/2.5 mm — 3.0 mm | 358×239×399 | 399.0 mm |

**Observaciones clave:**

- **Spacing de grilla de dosis es idéntico en los 6 casos**: 2.5×2.5 mm in-plane, 3.0 mm
  en Z. Buena noticia — no hay que armonizar spacing entre sitios en esta muestra
  (aunque con N=100+/sitio hay que confirmar que no varía).
- **CyC no tiene el mayor Z-extent** como se podría asumir — de hecho CyC1/CyC2
  (480/414mm) y Pelvis (438/429mm) son comparables, ambos bastante mayores que
  próstata (408/399mm). El "supuesto de trabajo" del CHARTER (§10, CyC crítico en
  cráneo-caudal) se confirma pero por poco margen sobre pelvis — vale la pena chequear
  con más N antes de asumir que CyC domina el diseño del parche/FOV en Z.
- Grilla de dosis en vóxeles varía bastante por sitio/paciente (134–161 en Z, 89–115
  en Y, 145–251 en X) — consistente con FOV recortado por Eclipse alrededor del
  PTV+margen, no un tamaño fijo. Esto es relevante para el diseño de
  `n_target`/padding del datamodule nuevo (Fase 0.1).
- CyC tiene registros de PTV en 3 niveles (`PTV_High/Intermediate/Low`), Pelvis y
  Próstata en 1-2 niveles (`PTV_High` + a veces `PTV_Sb`/`PTV_LN_Pelvics` como PTV
  secundario) — confirma la convención de nomenclatura ya cerrada en CHARTER §5.

---

## Fuentes

- `C:\Pablo\ProstateDoseProject\repo\scripts\profile_step.py`
- `C:\Pablo\ProstateDoseProject\repo\configs\exp002_unet2d_psdm.yaml`
- `C:\Pablo\ProstateDoseProject\repo\src\models\unet2d.py` / `unet3d.py` (confirmación GroupNorm)
- `C:\Pablo\ProstateDoseProject\repo\src\models\lightning_module.py` (confirmación contexto axial vs. parche)
- DICOM RD/RS de `\\10.100.0.252\centro_de_datos2018\101_Cosas de\PABLO\Dcm export 0.0\{CyC1,CyC2,Pelvis1,Pelvis2,Ptta1,Ptta2}`
