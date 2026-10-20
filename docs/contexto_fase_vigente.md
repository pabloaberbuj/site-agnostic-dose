# Contexto Fase Vigente

> **Tier 2 — Documento vivo.** Se reescribe completamente al fin de cada fase.
> Refleja siempre "qué es verdad ahora", no la traza histórica.
>
> Última actualización: 2026-10-20 (Fase 2, HO#1 cerrado)

---

## Estado actual

- **Fase activa:** 2 — primer site-agnostic (A-eje de input, SOLO VMAT). Plan A0–A6 en CHARTER §4
  y `ARRANQUE_fase2.md`.
- **Subestado:** HO#1 (infra + datamodule A0 + smoke test) **cerrado**. No se entrenó nada todavía.
  Siguiente: loop de train de A0 (config + U-Net 2D fija, MAE, métrica por sitio con IC bootstrap)
  y baseline no-aprendida en paralelo.

## Datos vigentes de Fase 2

- **NPZ:** `data/npz_v2b_body/<sitio>/<patient_id>.npz` (+ sidecar `.json`), 503 pacientes.
  Contenido: `dose_pct` (D95(PTV_High)=100%, grilla de dosis nativa) y `mask_<rol>` booleanas por
  rol canónico, **incluida `mask_BODY`** (v1 no la traía). Metadata (`protocol_id`, `rx_high_gy`,
  `rx_by_level_gy`, espaciados, `out_of_convention`, `harmonization_hash`) en el JSON, NO en el NPZ.
- **Split:** `data/splits/split_v2.json` = `split_v1` menos `PT_5787d1d3258de95d`. **432 pacientes:
  280 train / 65 val / 87 test** (Próstata 138, Pelvis 138, CyC 156).
- **Tabla de armonización:** hash `2a0c2037686c2978` (sha256 de nombre+contenido de todos los
  `.yaml` de `data/harmonization/`, 16 hex). Es el hash que va al record de cada experimento A0.
- `npz_v1_paso2/`, `npz_v2_body/` y `split_v1.json` quedan como histórico; no usar para entrenar.

## Datamodule A0 (`src/datamodules/a0_datamodule.py`)

- Item = paciente completo en cortes axiales: `input (Z,3,256,256)`, `dose (Z,256,256)` float32.
  Solo `batch_size=1` (Z varía; no hay pad en Z).
- Canales: **Ch0** PTV pintado con `Rx_nivel/Rx_High` PRESCRIPTO de la tabla (nunca de la dosis;
  el nivel alto gana el solape) · **Ch1** unión binaria de roles OAR (PRV incluidos) · **Ch2** body.
- Pad in-plane centrado a **256×256** (configurable). El HO decía 512, pero el máximo real es
  248×248; 512 era ~4× vóxeles de ceros. Un paciente que exceda `pad_hw` falla ruidoso.
- Falla ruidoso ante: rol desconocido en el NPZ, NPZ sin `mask_BODY`, paciente `out_of_convention`,
  nivel de PTV sin Rx en el protocolo.
- **Excepción registrada:** 16 pacientes de Próstata traen `PTV_Low` no declarado, idéntica a
  `PTV_High` (motivo `qa_ptv_low_identico_a_ptv_high` en `reviewed_ok`); el datamodule omite esa
  máscara inerte. `PT_5787d1d3258de95d` (PTV_Low distinta, IoU 0.865) pasó a `out_of_convention`
  (decisión 2026-10-20) y está fuera del split.
- `experiment_record()` devuelve el hash de la tabla, `npz_dir`, `splits_file`, `pad_hw`.
- Sin augmentations (A2 es de goals; flip/rotación geométricos no están pedidos para A0).

## Entorno

`requirements.txt` pinneado a las versiones con que corrió Fase 1 (Python 3.13.2, torch
2.12.1+cu130, PL 2.6.5, numpy 2.4.4). `.venv` en el repo (gitignored), independiente del proyecto
viejo. Instalar con `pip --no-cache-dir` (el sandbox bloquea el cache de pip). GPU: RTX A2000 12GB.

## Smoke test (`scripts/smoke_a0.py`)

`python scripts/smoke_a0.py --npz-dir <npz_v2b_body> --splits-file data/splits/split_v2.json [--all]`.
Estado: **0 fallas en los 432 del split** (`--all`); D95(PTV_High)=100.0% (tol ±2%), ratios de Ch0 =
tabla (CyC Mid 0.8491 / Low 0.7783), hash de sidecars = hash de la tabla.

## Pendiente / abierto (Fase 2)

- Loop de train de A0 y configs (`configs/` apunta a `npz_v2b_body` + `split_v2.json`; registrar hash).
- Baseline no-aprendida (DVH promedio por sitio/rol) como control en paralelo.
- Padding en Z si se quiere `batch_size>1`.
- Rama 3D rota (`build_model_3d` ausente): no bloquea Fase 2, la necesita Fase 3.
- Vara de Fase 1 (`dvh_curva_completa`, banda media 40–80%: Rectum 7.30, Bladder 3.34): referencia de
  orden de magnitud, no umbral (cambian cohorte y normalización).
- Ruido de punto flotante en `spacing_z_mm` de ~30 pacientes (3.0000004 en vez de 3.0): inocuo,
  no usar como clave.
