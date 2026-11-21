"""
Fase 4 — lectura del harness de calibración: magnitud en %Rx y differencing pareado contra el control MAE.
SOLO LEE CSVs (per_patient_*.csv de scripts/f4_harness_calibracion.py): sin modelo, sin GPU, CPU trivial.

  # magnitud del control solo
  python scripts/f4_harness_compare.py --control results/f4h_mae_control_renorm_none
  # cuantílico vs control (pareado por paciente) + seguridad PTV-alta contra la vara de visualización
  python scripts/f4_harness_compare.py --control results/f4h_mae_control_renorm_none --model results/<id_cuantilico> \
      --vis-control results/exp_a4_003_unet2d_psdm_c176x256_vara_f4 --vis-model results/<eval_a0_del_cuantilico>

Convenciones: magnitud = real - pred en %Rx_High (>0: el real tiene más dosis que lo predicho). Differencing = modelo - control:
  calib/OVH-pct: delta de la fracción / percentil por paciente (calib: hacia 1-tau es lo esperado; OVH: negativo = mejor plan).
  magnitud: delta = pred_control - pred_modelo (>0: el modelo predice MENOS dosis que MAE).
  PTV-alta (80-110, mean|dV| pp): control - modelo, >0 = modelo mejor (misma convención que dvh.paired_diff); seguridad = no degradar.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval import dvh  # noqa: E402  (numpy/pandas puro)


def ci(v):
    r = dvh.bootstrap_ci(np.asarray(v, float))
    return r['mean'], r['ci95'][0], r['ci95'][1], r['n']


def agg(df, col, keys):
    rows = []
    for k, g in df.groupby(keys):
        m, lo, hi, n = ci(g[col].values)
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), 'mean': m, 'ci_lo': lo, 'ci_hi': hi, 'n': n})
    return pd.DataFrame(rows)


def per_patient_primary(df, col):
    """un valor por (sitio, paciente): media sobre los puntos primarios de sus roles"""
    return df[df.primary].groupby(['site', 'patient_id'], as_index=False)[col].mean()


def magnitud(d):
    d = d.assign(mag=d.real - d.pred)
    por_rol = agg(d, 'mag', ['site', 'role', 'point'])
    sitio = agg(per_patient_primary(d, 'mag'), 'mag', ['site'])
    return por_rol, sitio


def paired(ctrl, mod, col, keys, primary_only=False):
    m = ctrl.merge(mod, on=['patient_id', 'site', 'role', 'point'], suffixes=('_c', '_m'))
    m['delta'] = m[f'{col}_m'] - m[f'{col}_c']
    if primary_only:
        m['primary'] = m['primary_c']
        m = per_patient_primary(m, 'delta')
    return agg(m, 'delta', keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True)
    ap.add_argument('--model')
    ap.add_argument('--vis-control', help='dir eval_a0 (metrics.csv) de la vara MAE')
    ap.add_argument('--vis-model', help='dir eval_a0 (metrics.csv) del cuantílico')
    a = ap.parse_args()
    cdir = ROOT / a.control
    cp, co, cv = (pd.read_csv(cdir / f) for f in ('per_patient_dvhpoint.csv', 'per_patient_ovh.csv', 'per_patient_voxel.csv'))

    out = {}
    por_rol, sitio = magnitud(cp)
    por_rol.to_csv(cdir / 'magnitud_dvhpoint_por_rol.csv', index=False)
    sitio.to_csv(cdir / 'magnitud_dvhpoint_primarios_sitio.csv', index=False)
    print('=== magnitud real-pred (%Rx), puntos primarios por sitio — CONTROL ===')
    print(sitio.round(3).to_string(index=False))
    print('--- por (sitio, punto) ---')
    print(agg(cp.assign(mag=cp.real - cp.pred), 'mag', ['site', 'point']).round(3).to_string(index=False))

    if a.model:
        mdir = ROOT / a.model
        mp, mo, mv = (pd.read_csv(mdir / f) for f in ('per_patient_dvhpoint.csv', 'per_patient_ovh.csv', 'per_patient_voxel.csv'))
        tabs = {}
        # primaria: calibración vóxel (modelo - control, pareado por paciente)
        vm = cv.merge(mv, on=['patient_id', 'site'], suffixes=('_c', '_m'))
        rows = []
        for name in ('tejido_sano', 'tejido_sano_dosis5', 'oars_union'):
            vm['delta'] = vm[f'{name}_m'] - vm[f'{name}_c']
            t = agg(vm, 'delta', ['site']); t['variante'] = name; rows.append(t)
        tabs['calib_voxel_delta'] = pd.concat(rows)
        # diagnósticas
        tabs['calib_dvhpoint_delta_primarios_sitio'] = paired(cp, mp, 'real_gt_pred', ['site'], primary_only=True)
        tabs['calib_dvhpoint_delta_por_rol'] = paired(cp, mp, 'real_gt_pred', ['site', 'role', 'point'])
        tabs['ovh_pct_delta_primarios_sitio'] = paired(co.dropna(subset=['ovh_pct']), mo.dropna(subset=['ovh_pct']), 'ovh_pct', ['site'], primary_only=True)
        tabs['ovh_pct_delta_por_rol'] = paired(co.dropna(subset=['ovh_pct']), mo.dropna(subset=['ovh_pct']), 'ovh_pct', ['site', 'role', 'point'])
        tabs['pred_delta_pctRx_primarios_sitio'] = paired(cp.assign(pred=-cp.pred), mp.assign(pred=-mp.pred), 'pred', ['site'], primary_only=True)
        tabs['pred_delta_pctRx_por_rol'] = paired(cp.assign(pred=-cp.pred), mp.assign(pred=-mp.pred), 'pred', ['site', 'role', 'point'])
        # (pred negado: delta = pred_control - pred_modelo, >0 = el modelo predice menos dosis)
        for name, t in tabs.items():
            t.to_csv(mdir / f'vs_control__{name}.csv', index=False)
            if name.endswith('por_rol'):
                continue
            print(f'\n=== {name} (modelo - control) ===')
            print(t.round(3).to_string(index=False))
        if a.vis_control and a.vis_model:
            vc = pd.read_csv(ROOT / a.vis_control / 'metrics.csv')
            vmod = pd.read_csv(ROOT / a.vis_model / 'metrics.csv')
            diff = dvh.paired_diff(vc, vmod)   # control - modelo, >0 = modelo mejor
            print('\n=== SEGURIDAD PTV-alta 80-110 (mean|dV| pp), control - modelo, >0 = modelo mejor ===')
            rows = []
            for site, roles in diff.items():
                if 'PTV_all' in roles:
                    r = roles['PTV_all']['alta_80_110']
                    rows.append({'site': site, 'mean': r['mean'], 'ci_lo': r['ci95'][0], 'ci_hi': r['ci95'][1], 'n': r['n']})
            t = pd.DataFrame(rows)
            t.to_csv(mdir / 'vs_control__seguridad_ptv_alta.csv', index=False)
            print(t.round(3).to_string(index=False))
            print('(aviso: eval_a0 renormaliza D95(PTV pred)=100%; esta lectura de seguridad es la de visualización)')


if __name__ == '__main__':
    main()
