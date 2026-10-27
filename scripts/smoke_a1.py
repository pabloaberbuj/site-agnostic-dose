"""
Smoke test de A1 (Fase 2, HO#4): canales de goals (Ch3-5) del A1DataModule.

    .venv/Scripts/python.exe scripts/smoke_a1.py --npz-dir ../../data/npz_v2b_body --splits-file data/splits/split_v2.json [--all] [--fill 0|95pct_rx]

Chequea, por paciente, de forma INDEPENDIENTE del builder (re-deriva lo esperado desde la tabla):
  shapes/NaN · Ch0-2 idénticos a A0 · Ch3/Ch4 uniformes = techo de la tabla (min en solape) · Ch5 con la fracción
  preservada pedida y escalera monótona en distancia al PTV · fuera de OAR = 0 · vóxeles de OAR sin goal = fill ·
  determinismo (dos construcciones idénticas) · roles del NPZ sin goal (se reportan) · goals sin máscara (entradas muertas).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datamodules.a0_datamodule import CH_BODY, CH_INTENT, CH_OAR, build_a0_channels  # noqa: E402
from src.datamodules.a1_datamodule import (A1DataModule, A1PatientDataset, CH_MAX, CH_MEAN, CH_VOL,  # noqa: E402
                                           N_CHANNELS_A1, ptv_distance)

TOL = 1e-6


def check_patient(ds, pid):
    fails, rep = [], {'patient_id': pid}
    dose, chans, meta = ds.load_raw(pid)
    site, z = meta['site'], dose.shape[0]
    rep.update(site=site, protocol_id=meta['protocol_id'], Z=z)
    if chans.shape != (N_CHANNELS_A1,) + dose.shape or chans.dtype != np.float32 or not np.isfinite(chans).all():
        fails.append(f'shape/dtype/NaN: {chans.shape} {chans.dtype}')
        return fails, rep
    with np.load(ds.npz_index[pid]) as d:
        masks = {k[5:]: d[k].astype(bool) for k in d.files if k.startswith('mask_')}

    # Ch0-2 == A0
    a0 = build_a0_channels(masks, ds.table.intent_ratios(site, meta['protocol_id']), ds.table, dose.shape, pid)
    if not np.array_equal(a0, chans[:3]):
        fails.append('Ch0-2 difieren de A0')

    goals = ds.table.oar_goals(site, meta['protocol_id'])
    oar_in = [r for r in masks if r in ds.table.oar_roles]
    union = np.zeros(dose.shape, bool)
    for r in oar_in:
        union |= masks[r]
    rep['roles_sin_goal'] = sorted(r for r in oar_in if r not in goals)
    rep['goals_sin_mascara'] = meta['_goal_info']['goals_sin_mascara']
    rep['trivial_skipped'] = meta['_goal_info']['trivial_skipped']
    fill = ds.fill

    # fuera de OAR = 0 en los 3 canales de goals
    if (chans[3:][:, ~union] != 0).any():
        fails.append('goals != 0 fuera de OAR')

    # esperado Ch3/Ch4: min de techos por vóxel (independiente: loop por rol)
    for ch, name, key in ((CH_MAX, 'maximos', 'maximos'), (CH_MEAN, 'medias', 'medias')):
        exp = np.full(dose.shape, np.inf)
        for r, gl in goals.items():
            if r in masks:
                for g in gl:
                    if key in g['channels']:
                        exp[masks[r]] = np.minimum(exp[masks[r]], g['ceiling'])
        exp = np.where(np.isfinite(exp), exp, np.where(union, fill, 0.0))
        if not np.allclose(chans[ch], exp, atol=TOL):
            fails.append(f'Ch{ch} ({name}) != techo de la tabla (max abs dif {np.abs(chans[ch] - exp).max():.4g})')

    # Ch5: por rol, sin solape con otros roles OAR, chequear escalera
    edt = ptv_distance(masks, ds.table.target_roles, (meta['spacing_z_mm'], meta['spacing_y_mm'], meta['spacing_x_mm']), ds.ptv_ref)
    vox_cm3 = meta['spacing_x_mm'] * meta['spacing_y_mm'] * meta['spacing_z_mm'] / 1000.0
    ch5_stats = {}
    for r, gl in goals.items():
        vg = [g for g in gl if 'volumetrico' in g['channels']]
        if r not in masks or not vg:
            continue
        others = np.zeros(dose.shape, bool)
        for r2 in oar_in:
            if r2 != r:
                others |= masks[r2]
        own = masks[r] & ~others
        if own.sum() < 50:
            continue
        n_role = int(masks[r].sum())
        vals, dist = chans[CH_VOL][own], edt[own]
        pres = ~np.isclose(vals, fill, atol=TOL)    # vóxeles con techo propio (preservados por >=1 goal)
        if pres.any() and (~pres).any() and dist[pres].min() < dist[~pres].max() - 1e-9:
            fails.append(f'Ch5 {r}: lo preservado no es lo más distal al PTV')
        o = np.argsort(-dist[pres], kind='stable')  # distal primero
        if np.any(np.diff(vals[pres][o]) < -TOL):   # distal -> proximal: techo no decreciente
            fails.append(f'Ch5 {r}: escalera no monótona (distal->proximal debe ser no decreciente)')
        for g in vg:
            b = g['budget_frac'] if g['budget_frac'] is not None else g['budget_cm3'] / (n_role * vox_cm3)
            if b >= 1.0:
                continue
            exp_pres = 1.0 - b
            got = float((chans[CH_VOL][masks[r]] <= g['ceiling'] + TOL).mean()) if fill == 0.95 or fill == 0.0 else None
            # fracción de vóxeles del rol con valor <= techo de este goal (>= preservada de este goal; el min con otros goals la sube)
            if got is not None and got + 0.02 < exp_pres - 0.0:
                fails.append(f'Ch5 {r} {g["label"]}: fracción con techo<= {g["ceiling"]:.3f} = {got:.3f} < preservada {exp_pres:.3f}')
        ch5_stats[r] = {'techos': sorted({round(float(v), 4) for v in np.unique(vals)})}
    rep['Ch5_roles'] = {k: v['techos'] for k, v in list(ch5_stats.items())[:4]}

    # vóxeles de OAR sin ningún goal en un canal = fill
    if fill != 0:
        for ch in (CH_MAX, CH_MEAN, CH_VOL):
            inside = chans[ch][union]
            if (inside <= 0).any():
                fails.append(f'Ch{ch}: vóxeles de OAR con valor 0 con fill={fill}')

    # determinismo
    chans2 = ds.load_raw(pid)[1]
    if not np.array_equal(chans, chans2):
        fails.append('no determinista (dos construcciones difieren)')
    rep['frac_vox_OAR_con_goal'] = {c: round(float((chans[i][union] > 0).mean()), 3) for c, i in
                                    (('max', CH_MAX), ('media', CH_MEAN), ('vol', CH_VOL))}
    return fails, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--splits-file', default='data/splits/split_v2.json')
    ap.add_argument('--harmonization-dir', default='data/harmonization')
    ap.add_argument('--fill', default='0')
    ap.add_argument('--ptv-ref', default='union')
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    fill = '95pct_rx' if a.fill == '95pct_rx' else 0

    dm = A1DataModule(a.npz_dir, a.splits_file, a.harmonization_dir, fill_sin_goal=fill, ptv_ref=a.ptv_ref)
    dm.setup()
    print(f'record = {json.dumps(dm.experiment_record(), ensure_ascii=False)}')
    print(f'HASH TABLA = {dm.table.hash}')
    ds_all = A1PatientDataset([p for ds in (dm.train_ds, dm.val_ds, dm.test_ds) for p in ds.ids],
                              dm.train_ds.npz_index, dm.table, dm.pad_hw, fill, 'uniform', a.ptv_ref)
    site_of = {pid: p.parent.name for pid, p in ds_all.npz_index.items()}
    if a.all:
        pids = ds_all.ids
    else:
        first = {}
        for pid in dm.train_ds.ids:
            first.setdefault(site_of[pid], pid)
        pids = [first[s] for s in ('Prostata', 'Pelvis', 'CyC')]

    n_fail = 0
    sin_goal, sin_mask, trivial = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), collections.Counter()
    for pid in pids:
        fails, rep = check_patient(ds_all, pid)
        if 'site' in rep:
            for r in rep['roles_sin_goal']:
                sin_goal[rep['site']][r] += 1
            for r in rep['goals_sin_mascara']:
                sin_mask[rep['site']][r] += 1
            for t in rep['trivial_skipped']:
                trivial[t] += 1
        if fails:
            n_fail += 1
        if not a.all or fails:
            print(f"\n[{rep.get('site')}] {pid}  {'OK' if not fails else 'FALLA'}")
            print('  ' + json.dumps(rep, default=str, ensure_ascii=False))
            for f in fails:
                print('  !! ' + f)
    print(f'\npacientes chequeados: {len(pids)}   con fallas: {n_fail}')
    print('roles OAR en NPZ SIN goal en la tabla (reciben fill):', {s: dict(c) for s, c in sin_goal.items()})
    print('goals de la tabla SIN máscara en el NPZ (entradas muertas):', {s: dict(c) for s, c in sin_mask.items()})
    print('constraints triviales (no pintados), nº de pacientes:', dict(trivial))
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
