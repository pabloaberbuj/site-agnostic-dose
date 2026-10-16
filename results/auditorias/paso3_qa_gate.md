# Paso 3 — QA gate offline sobre dosis cruda

> Script: `scripts/qa_gate.py`. Lee los NPZ/JSON ya generados por Paso 2 (no vuelve a
> tocar DICOM). D95 recalculado en Gy crudo (denormalizando `dose_pct` con
> `d95_ptv_high_raw_gy`) vs `rx_gy` del protocolo. Banda por sitio
> (`harmonization_schema.yaml:qa_gate.bands`: default [95%,105%], Prostata [92%,105%]
> desde 2026-10-16). Fuera de banda → flag, **no descarta nada** automáticamente.
> Detalle completo por paciente en `paso3_qa_gate.csv`.

## Resultado final (tras las decisiones del usuario, 2026-10-16)

| sitio | nivel | N | flags | flags sin revisar |
|---|---|---|---|---|
| CyC | PTV_High | 196 | 13 | 0 |
| CyC | PTV_Mid | 184 | 10 | 0 |
| CyC | PTV_Low | 196 | 3 | 0 |
| Pelvis | PTV_High | 161 | 0 | 0 |
| Prostata | PTV_High | 146 | 1 | 0 |

0 flags sin revisar en todo el dataset. El único flag de Prostata que queda
(`PT_b1b995a33bfa44ab`, D95=89.6%) ya era `reviewed_ok` de antes de Paso 3
(`D95_bajo_banda_revisado_ok`).

## Decisiones aplicadas

**Banda de Próstata recalibrada a [92%, 105%]** (antes [95%,105%], igual al resto):
8/146 PTV_High caían sistemáticamente en 92.9%–94.6% — franja angosta, no dispersión
al azar; el piso real del protocolo es ~92%, no 95%. Resolvió los 8 flags de una vez,
sin revisión caso por caso.

**9 pacientes CyC revisados individualmente** (`paso3_qa_gate_resuelto.csv`):
- `PT_259b5a8f69cba9bc` (D95 PTV_Mid 89.7%), `PT_5e3a5d473a92cedb` (D95 PTV_Mid 109.6%,
  hot spot — ya tenía `amigdala`/`supra` desestimadas en Paso 2, ahora además sale por
  dosis), `PT_8a9d69ea4ef84f91` (D95 PTV_Low 94.7%) → **`out_of_convention`**.
- `PT_69d8fe738b0ce247`, `PT_7911e792733fc5bc`, `PT_d3337c99c1088b80`,
  `PT_e5061f032d464e41`, `PT_ec857fa843d0ada9`, `PT_fe2c6c68f6985220` → **`reviewed_ok`**
  (D95 95-99% de Rx_High o Rx_Mid, revisado, se mantiene en train).

**17 pacientes de Próstata con `PTV_Low` no declarado en el protocolo** (protocolo de
Próstata es de 1 solo nivel): 16 confirmados idénticos a `PTV_High` (D95 68-70Gy, dentro
de banda) → **`reviewed_ok`** (máscara extra inerte, el protocolo nunca la usa). 1 caso
genuinamente distinto (`PT_fe24a2bdb79bd38a`, D95=50.7Gy, ~72% de Rx_High — mismo
paciente con `CTV_ProstateBed` de Paso 2, probable lecho prostático) → **`out_of_convention`**.

**11 pacientes de Pelvis con `PTV_Low` no declarado:** ya eran los conocidos de campo
extendido (`out_of_convention`, motivo `campo_extendido_Y_mayor_300mm`), sin novedad.

## Paso 3: cerrado

Los `excluded`/`out_of_convention` se muestran en el CSV con su motivo (informativo);
los `reviewed_ok` no se repiten en futuras rondas de QA gate.
