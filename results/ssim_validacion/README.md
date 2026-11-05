# Vara de coherencia de Fase 3 (SSIM)

Esta carpeta es la vara de coherencia de Fase 3. Se generó solo sobre **val** (test no se tocó) con `scripts/ssim_validacion.py`;
la lógica del SSIM vive en `src/eval/ssim.py` (la usa también `scripts/eval_a0.py`).

- `per_patient_<modelo>.csv`: una fila por paciente de val, para A0 (ep097), A1 = A1a fill 0 (ep088), 2D = exp_a4_003 (ep113),
  3D = exp_b1_001 (ep076), MedNeXt = exp_b2_001 (ep084). Los checkpoints de A0/A1 son el óptimo de val/mae, los otros son el óptimo de DVH.
- `reporte.txt`: tablas por sitio con IC95 bootstrap (pareado por paciente) de 3D−2D, MedNeXt−2D y A1−A0, la descomposición de CyC en
  luminancia / contraste / estructura y los diagnósticos de error fuera del BODY.
- `verificacion_postproceso_body.txt`: comprobación de que `pred*BODY` no mueve el DVH y de que el harness reproduce esta tabla.

## Definición
SSIM de torchmetrics 1.9 (gauss 11, sigma 1.5, K1=.01, K2=.03, data_range=120, reflect-pad de 5 px sin recortar el borde, cortes con BODY,
media por corte y luego entre cortes), sobre grilla nativa des-padeada (independiente del contenedor). Se descompone como
SSIM = luminancia · contraste · estructura (C3 = C2/2). Las ventanas se calculan sobre la imagen completa; la máscara elige qué píxeles se promedian.

| Columna del harness | Región | Estado |
|---|---|---|
| `ssim_body` | BODY | **vigente** |
| `ssim_body_dosis5` | BODY ∩ dosis real > 5 % de Rx_High (`dose_pct` ya está en % Rx_High) | **vigente** |
| `ssim_frame_obsoleto` | frame completo | **OBSOLETA**, solo referencia: 36–71 % de los píxeles están fuera del BODY y los domina |

## Qué hay que recordar
- Las tablas de acá se calcularon con la predicción **cruda** (sin `pred*BODY`). El harness actual aplica `pred*BODY`; sobre `ssim_body` y
  `ssim_body_dosis5` el efecto es mínimo (ver `verificacion_postproceso_body.txt`); sobre el frame completo es enorme (por eso es obsoleta).
- Las evaluaciones anteriores (`results/exp_*/secondary_per_patient.csv`, columna `ssim`) son de frame completo (obsoleto) y no se
  regeneraron. Los resultados nuevos traen `ssim_body`, `ssim_body_dosis5` y `ssim_frame_obsoleto`.
