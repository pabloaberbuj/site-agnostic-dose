#!/usr/bin/env python3
"""
Smoke test del datamodule A0 (HO#1, Deliverable 3). No entrena.

Por sitio, 1 paciente (el primero de train por default; --all recorre todo el split):
  1. shapes/dtypes: input (Z,3,pad,pad) float32, dose (Z,pad,pad) float32, sin NaN
  2. integridad de normalización: D95(PTV_High) sobre dosis+máscara cargadas ≈ 100% (tol ±2%)
  3. intención: ratios de Ch0 == Rx_nivel/Rx_High de la TABLA (High=1.0; CyC Mid/Low reales)
  4. roles: ninguno desconocido; OAR consolidado no vacío; body presente
  5. hash de la tabla (y comparado con el del sidecar del NPZ)

Uso (desde la raíz del repo):
    python scripts/smoke_a0.py --npz-dir <npz_v2_*> [--all]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.datamodules.a0_datamodule import (DataError, A0DataModule, A0PatientDataset, CH_BODY, CH_INTENT, CH_OAR, N_CHANNELS, pad_hw_center)

D95_TOL_PCT = 2.0


def d95(dose, mask):
    return float(np.percentile(dose[mask], 5))   # dosis cubierta por el 95% del volumen


def check_patient(ds, pid):
    """Devuelve (lista de fallas, dict de reporte)."""
    fails, rep = [], {'patient_id': pid}
    item = ds[ds.ids.index(pid)]
    x, y = item['input'].numpy(), item['dose'].numpy()
    dose_raw, chans, meta = ds.load_raw(pid)
    z = dose_raw.shape[0]
    rep.update(site=meta['site'], protocol_id=meta['protocol_id'], Z=z, native_hw=tuple(dose_raw.shape[1:]))

    # 1. shapes / dtypes / NaN
    if x.shape != (z, N_CHANNELS, ds.pad_hw, ds.pad_hw) or x.dtype != np.float32:
        fails.append(f'input shape/dtype {x.shape} {x.dtype}')
    if y.shape != (z, ds.pad_hw, ds.pad_hw) or y.dtype != np.float32:
        fails.append(f'dose shape/dtype {y.shape} {y.dtype}')
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        fails.append('NaN/inf')
    if not np.array_equal(pad_hw_center(dose_raw, ds.pad_hw), y):
        fails.append('dose cargada != dose del NPZ (el datamodule no debe tocar el target)')

    # masks crudas del NPZ (independientes de los canales armados)
    with np.load(ds.npz_index[pid]) as d:
        masks = {k[5:]: d[k].astype(bool) for k in d.files if k.startswith('mask_')}

    # 2. D95(PTV_High) de la dosis cargada (grilla de dosis; tolerancia, no igualdad)
    d95_high = d95(dose_raw, masks['PTV_High'])
    rep['D95_PTV_High_%'] = round(d95_high, 3)
    if abs(d95_high - 100.0) > D95_TOL_PCT:
        fails.append(f'D95(PTV_High)={d95_high:.2f}% fuera de 100±{D95_TOL_PCT} (¿problema de vóxel?)')

    # 3. intención vs tabla
    ratios = ds.table.intent_ratios(meta['site'], meta['protocol_id'])
    ch0 = chans[CH_INTENT]
    rep['ratios_tabla'] = {k: round(v, 4) for k, v in ratios.items()}
    levels = sorted((r for r in masks if r in ratios), key=lambda r: -ratios[r])
    painted = np.zeros(ch0.shape, bool)
    rep['ratios_Ch0'] = {}
    for lvl in levels:                       # de mayor a menor Rx; el nivel alto gana el solape
        own = masks[lvl] & ~painted
        painted |= masks[lvl]
        if own.any() and not np.allclose(ch0[own], ratios[lvl], atol=1e-6):
            fails.append(f'Ch0 en {lvl}: {np.unique(ch0[own])} != tabla {ratios[lvl]}')
        rep['ratios_Ch0'][lvl] = round(float(ch0[own].mean()), 4) if own.any() else None
    if not np.all(ch0[masks['PTV_High']] == 1.0):
        fails.append('High != 1.0 en Ch0')
    if (ch0[~painted] != 0).any():
        fails.append('Ch0 no es 0 fuera de los PTV')
    allowed = np.array([0.0] + list(ratios.values()))
    if not np.isclose(np.unique(ch0)[:, None], allowed[None, :], atol=1e-6).any(axis=1).all():
        fails.append(f'Ch0 con valores fuera de la tabla: {np.unique(ch0)}')

    # 4. roles
    oar_roles = [r for r in masks if r in ds.table.oar_roles]
    rep['n_oar_roles'] = len(oar_roles)
    if not chans[CH_OAR].any():
        fails.append('Ch1 (OAR) vacío')
    if not chans[CH_BODY].any():
        fails.append('Ch2 (body) vacío')
    union = np.zeros(ch0.shape, bool)
    for r in oar_roles:
        union |= masks[r]
    if not np.array_equal(union, chans[CH_OAR] > 0):
        fails.append('Ch1 != unión de roles OAR')
    if not np.array_equal(masks['BODY'], chans[CH_BODY] > 0):
        fails.append('Ch2 != mask_BODY')
    rep['vox_frac'] = {'PTV': round(float((ch0 > 0).mean()), 4), 'OAR': round(float(union.mean()), 4),
                       'body': round(float(chans[CH_BODY].mean()), 4)}
    rep['hash_sidecar'] = meta['harmonization_hash']
    return fails, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--splits-file', default='data/splits/split_v1.json')
    ap.add_argument('--harmonization-dir', default='data/harmonization')
    ap.add_argument('--pad-hw', type=int, default=256)
    ap.add_argument('--all', action='store_true', help='recorre TODOS los pacientes del split (train+val+test)')
    a = ap.parse_args()

    dm = A0DataModule(a.npz_dir, a.splits_file, a.harmonization_dir, pad_hw=a.pad_hw, num_workers=0)
    dm.setup()
    print(f'record = {json.dumps(dm.experiment_record())}')
    print(f'HASH TABLA = {dm.table.hash}')

    ds_all = A0PatientDataset([p for ds in (dm.train_ds, dm.val_ds, dm.test_ds) for p in ds.ids],
                              dm.train_ds.npz_index, dm.table, a.pad_hw)
    site_of = {pid: p.parent.name for pid, p in ds_all.npz_index.items()}
    if a.all:
        pids = ds_all.ids
    else:
        first = {}
        for pid in dm.train_ds.ids:
            first.setdefault(site_of[pid], pid)
        pids = [first[s] for s in ('Prostata', 'Pelvis', 'CyC')]

    n_fail, hashes, summary = 0, collections.Counter(), collections.defaultdict(list)
    for pid in pids:
        try:
            fails, rep = check_patient(ds_all, pid)
        except DataError as e:
            fails, rep = [f'DataError: {e}'], {'site': site_of[pid], 'patient_id': pid, 'native_hw': (0, 0),
                                               'hash_sidecar': dm.table.hash}
        hashes[rep['hash_sidecar']] += 1
        summary[rep['site']].append(rep['native_hw'])
        if fails:
            n_fail += 1
        if not a.all or fails:
            print(f"\n[{rep['site']}] {pid}  {'OK' if not fails else 'FALLA'}")
            print('  ' + json.dumps(rep, default=str))
            for f in fails:
                print('  !! ' + f)
    print(f'\npacientes chequeados: {len(pids)}   con fallas: {n_fail}')
    for s, hw in summary.items():
        arr = np.array(hw)
        print(f'  {s}: n={len(hw)} in-plane max={arr.max(0)} (pad={a.pad_hw})')
    status = 'COINCIDE' if set(hashes) == {dm.table.hash} else 'DIFIERE'
    print(f'hash sidecar NPZ {dict(hashes)} vs tabla actual {dm.table.hash}: {status}')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
