# Auditoría Paso 1 — inventario de headers DICOM (503/503 pacientes)

> Generado por `scripts/audit_dicom_headers.py` sobre RTDOSE (headers) + RTSTRUCT
> (contornos) de los 3 sitios. CT no se lee (no hace falta para este inventario).
> Datos crudos: `results/auditorias/paso1_headers.csv`.

- CyC: 196/196 — Pelvis: 161/161 — Prostata: 146/146. 0 errores de lectura, 0 sin BODY.

## 1. Spacing

In-plane uniforme en los 503: **2.5×2.5 mm**. Z uniforme **dentro** de cada paciente en
los 503, pero **NO uniforme entre pacientes**:

| spacing Z | N pacientes | sitios |
|---|---|---|
| 3.0 mm | 493 | CyC 187, Pelvis 161, Prostata 145 |
| 2.0 mm | 10 | CyC 9, Prostata 1 |

Los 10 casos de 2.0mm no correlacionan con año de plan, técnica, nº de arcos ni ningún
`FlagRevision` de `metadata_planes.csv` — parece elección puntual del planificador al
calcular dosis. `patient_id`: CyC `PT_01e46189c01b5461, PT_2701028e79b3426c,
PT_3206ecc00531646d, PT_3f20b750f6bc403f, PT_98d042ec47757136, PT_9bcdc56462d32b07,
PT_c921d707a84083c3, PT_d139dd94f42ab5dc, PT_f7932e33d14fedfc`; Prostata
`PT_c1ba3262f2e7e342` (ya `reviewed_ok: plan_sum_tratamiento_lejano`).

`DoseGridScaling`: rango 4.87e-5 – 7.21e-5, sin outliers evidentes.

## Decisiones tomadas con este reporte (cierran Paso 1)

- **No hay padding global.** El NPZ guarda la grilla nativa tal cual; el recorte/relleno lo hace
  el datamodule (2D: plano fijo 256×256; 3D: esquema a definir en Fase 3). Esto vuelve moot la
  sección 2 de abajo en su forma original — queda como registro de la distribución observada.
- **El cluster de 4 (Y/X ~241-242) NO es `out_of_convention`.** Su BODY es normal; la grilla de
  RTDOSE ahí cubre toda la imagen (no el paciente). Recortado al BODY, vuelven a tamaño normal.
  No se tocan en `patient_exceptions.yaml`.
- **Los 10 de Z=2.0mm entran al train.** `spacing_z_mm` va en la metadata por paciente.
- **Los 12 de CyC sin `PTV_Mid` van a `patient_exceptions.yaml` como `out_of_convention`**
  (motivo `sin_PTV_Mid_protocolo_1_nivel`) — agregado.
- **0.0 no se re-perfila ahora.** Se re-perfila en Fase 3 sobre los casos reales más grandes
  tras recortar al BODY (ref. `PT_85558912476286d3`, ~7-8M vóxeles), no sobre una caja padeada.

## 2. Grilla de dosis vs. padding provisional (176×128×256, Z×Y×X) — histórico, ver decisión arriba

| sitio | N | Z p50/p95/max | Y p50/p95/max | X p50/p95/max | excede padding |
|---|---|---|---|---|---|
| CyC | 196 | 142/178/239 | 111/129/164 | 206/230/248 | 31 (15.8%) |
| Pelvis | 161 | 149/197/217 | 97/132/242 | 173/214/242 | 23 (14.3%) |
| Prostata | 146 | 144/174/208 | 110/137/241 | 158/193/241 | 17 (11.6%) |
| **total** | 503 | — | — | — | **59 (11.7%)** |

Lista completa de `patient_id` + dims en el CSV. Los 2 `out_of_convention` ya conocidos
(`PT_49f9ceef1c10627d`, `PT_258346f1e40fbf8f`, retroperitoneo) están entre los 59.

**Cluster aparte (4 pacientes, Y y X ~241-242 — muy por encima del resto, que no pasa de
164/248):** `PT_c574673282df609b`, `PT_f0a75c79890616b7`, `PT_f1a76fd793c84778` (Pelvis),
`PT_5d2c6ee9551e7804` (Prostata). Mismo spacing 2.5mm → FOV físico real de ~600mm,
no error de spacing. Candidatos a revisión manual / posible `out_of_convention`
(campo grande, en la línea de los retroperitoneo ya conocidos) — no los marco solo,
falta criterio clínico.

**Candidato de padding que cubre todo excepto ese cluster de 4:** `(240, 176, 256)`
(múltiplo de 16). Con esto, 499/503 (99.2%) entran; solo el cluster de 4 sigue
excediendo. Con el padding actual (176×128×256), son 59/503 (11.7%) los que exceden.

**Atención — esto pisa el profiling de 0.0:** `envelope.md` midió memoria a
176×128×256 (U-Net vanilla entra sin parches, 8.56GB reserved). `(240,176,256)` es
~1.9× más vóxeles — hay que re-perfilar antes de asumir que sigue entrando sin parches.

## 3. PTV_Mid en CyC (protocolo declarado: High/Mid/Low)

12/196 (6.1%) sin `PTV_Mid` resoluble: 7 ausente, 5 presente con volumen 0.
`PTV_Resuelto` de la metadata del extractor es `PTV_High`/`PTV_High-04` en los 12, con
la misma `DosisTotal_Gy` (69.96) que el resto — compatible con un subgrupo que no
recibe boost a Mid/Low (prescripción de 1 nivel), no con un problema de nomenclatura.
`patient_id`: `PT_0325846a2bdff29e, PT_0a2f34e2e0379df0, PT_14b63f3f19bafbce,
PT_1b461e4fa0109c7b, PT_21cb7f818391b0c4, PT_40ffbce41ee8a2a1, PT_46e345b551a2fbc1,
PT_56f3d5797457174f, PT_65ebc54914351e57, PT_6a4269d796a3ab42, PT_7b174492f8ad48eb,
PT_a34f7226d9d2aabc`.

## 4. Volumen de PTV — nota metodológica

El volumen de esta auditoría es una aproximación (suma de áreas de contorno × paso
entre cortes del propio ROI), suficiente para "existe / volumen=0 / ambiguo". El
volumen validado (rasterizado sobre la grilla de dosis nativa, con el test de escala
de vóxel) es de Paso 2.
