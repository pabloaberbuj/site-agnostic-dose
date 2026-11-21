# Prompt Code — Fase 4, loss #1: DVH-loss de momentos (serie/paralelo) + MAE

Primera candidata de loss de Fase 4. Input A4 y U-Net 2D fijos; se varía solo la loss. La vara ya está
calculada (`exp_a4_003` ep113, val) — este experimento se mide contra ella con la misma val y el mismo
harness. **No reabrir eje A ni B.** Test sellado hasta cerrar la fase.

## Configuración fija (igual que la vara)

- Input A4, 7 canales. `npz_v2b_body`, `split_v2.json`, tabla de armonización hash `7ab16f73fa632a3e`.
- U-Net 2D base16/depth4, `a0_module`, contenedor (176,256) crop-to-BODY.
- Dosis predicha × máscara BODY en inferencia/eval.
- AdamW 1e-4/1e-5, cosine `scheduler.max_epochs=189`, patience 28, 16-mixed, seed 42, batch=1 por paciente.
- Dosis normalizada D95(PTV_High)=100%; todos los valores de dosis en % Rx_High.

## Loss

```
loss = masked_mae_body + λ · L_dvh
```

`L_dvh` = **loss de momentos diferenciable (estilo Jhanwar), por estructura, restringida a máscara ∩ BODY**,
en % Rx_High. Para cada estructura S con dosis predicha y real sobre sus vóxeles:

- Momento p-ésimo: `M_p(S) = ( mean_i( d_i^p ) )^(1/p)`. Diferenciable; `p=1` es la media, `p` alto
  aproxima el Dmax (control de cola/hotspot).
- Término por estructura: `Σ_p w_{S,p} · ( M_p(pred) − M_p(real) )²`.
- `L_dvh` = suma sobre estructuras (normalizada por nº de estructuras del paciente para que no dependa del
  conteo de OARs, que es variable por sitio).

**Estructuras que entran:** PTV por nivel (High/Mid/Low según sitio) + cada OAR con goal en la tabla. Las
máscaras ya resueltas vienen del NPZ; el **tipo de constraint sale de la tabla de armonización** al cargar.

**Pesos de momento por tipo de constraint (= split serie/paralelo; no es perilla nueva, mapea al tipo que ya
está en los canales de goals):**

| Tipo de constraint (tabla) | Momentos activos |
|---|---|
| Máximo / serie | `p=10` (aprox. Dmax) |
| Media / paralelo | `p=1` |
| Volumétrico (Dx_cc) | `p=10` — v1, control de cola. *Decisión nivel 2, reabrible.* |
| PTV (por nivel) | `p=1` (cobertura) **y** `p=10` (hotspot) |

Set de momentos de arranque `p ∈ {1, 10}`. Si querés dispersión, `p=2` opcional, pero no lo hagas perilla del
sweep: el sweep es solo λ.

**Crítico:** todo dentro de máscara ∩ BODY. Nada de frame completo (lección del aire, Fase 3).

## Sweep (única perilla libre: λ)

- `λ ∈ {0.1, 0.3, 1.0}` relativo a la escala del MAE-body (MAE y momentos están en la misma escala % Rx, así
  que arranca comparable). Un entreno por λ. Nivel 2: rango sensato, anotado, seguir.
- Criterio del sweep: el λ que **no degrade MAE-body de forma material** y mueva PTV-alta.

## Selección de checkpoint (regla de la fase, igual para todos)

- Guardar el **top-3 por el monitor**. Monitor / early-stop = `val/loss` (la compuesta MAE+λ·DVH).
- Checkpoint elegido = **menor criterio DVH (pooled OAR banda media 40–80, n=65) sobre esos top-3**.
- **Reportar ambos:** el mejor-por-monitor y el mejor-DVH-sobre-top-3. El gap entre ellos es el diagnóstico de
  si esta loss desacopla MAE de DVH (en la vara coincidían; una loss DVH podría separarlos).
- Dejar el elegido anotado en `summary.json` + `seleccion.txt`, como en `scripts/vara_f4_seleccion.py`.

## Evaluación (misma val que la vara, pred × BODY)

Set completo **por sitio con IC95 bootstrap**:

- Criterio DVH: pooled OAR banda media 40–80 + OAR-media + **PTV-alta 80–110** (el punto donde se espera
  movimiento).
- MAE-body, SSIM_BODY y SSIM BODY∩dosis>5%, d2 (xy/meandose/vfrac).

## Regla de corte (pre-registrada — fijarla antes de mirar resultados)

Gana sobre la vara si **las tres**:
1. mejora el pooled OAR banda media;
2. **ningún sitio empeora** con IC95 que excluya 0 (en OAR-media o PTV-alta);
3. **al menos un sitio mejora** con IC95 que excluya 0.

Una mejora aislada en PTV-alta de un solo sitio **no alcanza** sin medir antes el seed-43 del baseline 2D
(piso de ruido). Efectos opuestos en sitios vecinos = señal de varianza entre corridas, no de la loss.

## Salida / handoff

- `results/<exp_id>/`: `metrics.csv`, `summary.json`, `secondary_per_patient.csv`, `d2_coherencia_axial/`,
  subcarpetas por checkpoint + `seleccion.txt`. Un `<exp_id>` por λ (sugerencia: `exp_f4_001_dvhmoment_sp_lambdaXpY`).
- **Sin commit** salvo aviso.
- De vuelta al chat, solo el **delta**: pooled por sitio + OAR-media/PTV-alta con IC + MAE-body + SSIM. No el doc.

## Notas

- Prior honesto: en próstata MAE puro ganó y las DVH-losses no aportaron → lo probable es que no mueva el
  pooled. La apuesta real es **PTV-alta en CyC** (vara 7.49 [6.35, 8.87], contra ~4.7–5.0 en los otros sitios).
- Pelvis es ruidoso (IC ancho) → no creer movimientos ahí sin piso de seed.
- Trigger `cabos_sueltos.md`, **solo si esta loss gana y se adopta**: re-testear el 3D con la misma loss (una
  corrida). No ahora.
