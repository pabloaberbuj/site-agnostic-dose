"""
Fase 3 / eje B: re-selección SIMÉTRICA de checkpoint por DVH (exp_a4_003 y exp_b1_001), no por val/mae. Solo val; no entrena.

Criterio pre-fijado (igual para los dos modelos): mejor ckpt = menor mean|ΔV(D)| en banda media 40-80 %Rx, OAR_all, fila OAR_all
de cada paciente (ya agregada por eval_a0), promediada sobre TODOS los pacientes de val (pooled, 65). Se reportan también los valores por sitio (sensibilidad).
Luego delta pareado 3D−2D (positivo = 3D mejor, IC95 bootstrap, src.eval.dvh.paired_diff) en (a) ckpt de óptimo val/mae y (b) óptimo DVH.

    .venv/Scripts/python.exe scripts/reseleccion_dvh_b1.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval import dvh  # noqa: E402

MODELOS = {
    '2D': ('exp_a4_003_unet2d_psdm_c176x256', {100: 'ckpt_ep100', 102: 'ckpt_ep102', 113: ''}, 113),
    '3D': ('exp_b1_001_unet3d_psdm', {76: '', 82: 'ckpt_ep082', 84: 'ckpt_ep084'}, 76),
}
CELLS = (('OAR_all', 'media_40_80'), ('PTV_all', 'alta_80_110'))


def carga(exp, sub):
    return pd.read_csv(ROOT / 'results' / exp / sub / 'metrics.csv')


def crit(df):
    por_pac = df[df.role == 'OAR_all'][['patient_id', 'site', 'media_40_80']]   # fila agregada por paciente que ya produce eval_a0
    return por_pac.media_40_80.mean(), por_pac.groupby('site').media_40_80.mean().to_dict(), len(por_pac)


def main():
    sel, dfs = {}, {}
    for m, (exp, ckpts, mae_ep) in MODELOS.items():
        print(f'== {m} ({exp}): mean|ΔV| banda media 40-80, OAR, por paciente (menor = mejor)')
        res = {}
        for ep, sub in ckpts.items():
            df = carga(exp, sub)
            dfs[(m, ep)] = df
            c, ps, n = crit(df)
            res[ep] = c
            print(f'  ep{ep:03d}{" (óptimo val/mae)" if ep == mae_ep else ""}: pooled {c:.3f} (n={n}) | ' +
                  ' '.join(f'{s} {v:.2f}' for s, v in sorted(ps.items())))
        sel[m] = (mae_ep, min(res, key=res.get))
        print(f'  -> óptimo DVH: ep{sel[m][1]:03d}')
    for etiqueta, k in (('(a) óptimo val/mae', 0), ('(b) óptimo DVH', 1)):
        e2, e3 = sel['2D'][k], sel['3D'][k]
        print(f'\n{etiqueta}: 2D ep{e2} vs 3D ep{e3}  (2D−3D en paired_diff = ref−new; >0 = 3D mejor)')
        diff = dvh.paired_diff(dfs[('2D', e2)], dfs[('3D', e3)])
        for site, roles in diff.items():
            print('  ' + site.ljust(9) + ' | '.join(
                f'{role[:3]} {band}: {roles[role][band]["mean"]:+.2f} [{roles[role][band]["ci95"][0]:+.2f},{roles[role][band]["ci95"][1]:+.2f}]'
                for role, band in CELLS))


if __name__ == '__main__':
    main()
