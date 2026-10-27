# HO#10 — A5 (densidad másica): handoff de vuelta (§8)

## Delta
- **Camino de la densidad:** sidecar por paciente `data/density_v1/<sitio>/<pid>.npy` (float16, g/cm³, grilla de dosis nativa) +
  `manifest.json` (curva, hash, escáner por paciente). Generado por `scripts/build_density.py` (503/503, 0 errores);
  `npz_v2b_body`, split y hash `7ab16f73fa632a3e` intactos. Canal 7 del `A5DataModule` (`src/datamodules/a5_datamodule.py`):
  densidad dentro de `mask_BODY`, 0 fuera y en el padding. 8 canales, NO HU.
- **Curva HU→densidad:** "CT Calibration Curve: Big Bore Mass_Density" (Beam Configuration 13.6.30), 14 puntos,
  `data/density/curve_bigbore_massdensity.csv`. Se aplica a la resolución del CT y luego se remuestrea (trilinear) a la grilla de dosis.
  **Excepción registrada:** solo existe la curva de Big Bore; se usó también en 38 Brilliance 16, 15 MX 16-slice y 1 ARIA (y 2 Big Bore a 80/100 kVp).
  Borde en Z: ~62 pacientes CyC con un frame de dosis fuera del CT → se replica el corte CT extremo.
- **Verificación:** `scripts/smoke_a5.py --all` 0 fallas en los 432 (Ch0-6 == A4, Ch7 = 0 fuera de BODY, padding, determinismo);
  cross-check contra HU crudo del DICOM en 6 pacientes: error relativo mediano ~0.6 %.
- **Corrida:** `exp_a5_001_unet2d_density`, todo lo demás congelado (189 ép / patience 28 / seed 42).
  **INTERRUMPIDA** en la época 147 (proceso muerto al cerrarse la sesión; early stopping habría cortado en la 151).
  Mejor ckpt `epoch=123` (val/mae 1.890), sin mejora en 124–146. Evaluado en val con ese ckpt; test intacto.

## A5 vs A4 (val, por sitio; IC95 pareado, >0 = A5 mejor; `results/exp_a5_comparison.json`)
| sitio | SSIM | MAE body | PTV alta | OAR media |
|---|---|---|---|---|
| CyC | 0.571→0.501 (**−0.07** [−0.08,−0.06]) | +0.10 [+0.01,+0.20] | −0.32 [−1.05,+0.42] | +0.11 [+0.01,+0.21] |
| Pelvis | 0.815→0.737 (**−0.08** [−0.09,−0.07]) | +0.02 [−0.03,+0.08] | +0.15 [−0.37,+0.83] | −0.12 [−0.45,+0.17] |
| Próstata | 0.856→0.787 (**−0.07** [−0.08,−0.06]) | +0.01 [−0.02,+0.04] | +0.10 [−0.48,+0.68] | +0.40 [+0.10,+0.73] |

## Lectura
- **No hay evidencia de que la densidad aporte.** PTV alta plana en los 3 sitios (IC con 0). Ganancias chicas en MAE body (CyC) y OAR media
  (CyC, Próstata) con IC que apenas excluyen 0.
- **SSIM cae en los 3 sitios (−0.07/−0.08/−0.07), IC sin 0.** Es lo único claro y va contra A5.
- Hipótesis dirigida (aporte en CyC, heterogéneo): solo marginal en MAE/OAR; no se ve en PTV alta ni en SSIM.
- **Zona gris por ruido:** sin piso de ruido (seed 43 cancelado, ver `cabos_sueltos.md`), corrida interrumpida, un solo seed.
  La caída de SSIM podría tener un componente de varianza de entrenamiento; para decidir "plano vs perjudicial" haría falta seed 43 de A5 (o de A4).
- Propuesta: tratar la densidad como **no aporta** (A4 sigue como representación) salvo que el piso de ruido diga otra cosa; siguiente input, A6 (beam plate).

Artefactos: `results/exp_a5_001_unet2d_density/{metrics.csv,secondary_per_patient.csv,summary.json,record.json}`, `scripts/compare_a5.py`.
