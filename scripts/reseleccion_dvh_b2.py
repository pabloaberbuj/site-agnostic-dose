"""
Fase 3 / eje B, B2 (MedNeXt-2D vs U-Net 2D exp_a4_003). Solo val; no entrena. Mismo criterio pre-fijado que reseleccion_dvh_b1.py:
(1) criterio pooled banda media 40-80 OAR sobre los top-3 ckpts de cada modelo (por val/mae);
(2) delta DVH pareado (OAR-media 40-80, PTV-alta 80-110; IC95 bootstrap, por sitio; 2D−MedNeXt > 0 = MedNeXt mejor)
    en (a) ckpt de óptimo val/mae y (b) óptimo DVH; (3) secundarias MAE-body/SSIM por sitio; (4) d2 por sitio.
    .venv/Scripts/python.exe scripts/reseleccion_dvh_b2.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval import dvh  # noqa: E402
from scripts.reseleccion_dvh_b1 import CELLS, carga, crit  # noqa: E402

A4 = 'exp_a4_003_unet2d_psdm_c176x256'
B2 = 'exp_b2_001_mednext2d_psdm'


def sec(exp, sub):
    d = ROOT / 'results' / exp / sub
    return pd.read_csv(d / 'secondary_per_patient.csv')


def main():
    sel, dfs = {}, {}
    # óptimo val/mae por modelo: epoch del ckpt cuyo eval vive en results/<exp>/ (record de eval lo registra)
    for m, exp, mae_ep, ckpts in (('2D', A4, 113, {100: 'ckpt_ep100', 102: 'ckpt_ep102', 113: ''}),
                                  ('MedNeXt', B2, None, None)):
        if ckpts is None:
            import re
            log = (ROOT / 'logs' / f'{B2}_train.log').read_text(encoding='utf-8', errors='ignore')
            tops = sorted(int(p.stem.split('=')[1]) for p in (ROOT / 'checkpoints' / B2).glob('epoch=*.ckpt'))
            bestm = re.findall(r'^best: (.*epoch=(\d+)\.ckpt)', log, re.M)
            mae_ep = int(bestm[-1][1])
            ckpts = {e: f'ckpt_ep{e:03d}' for e in tops}
        print(f'== {m} ({exp}): mean|ΔV| banda media 40-80, OAR, pooled (menor = mejor)')
        res = {}
        for ep, sub in ckpts.items():
            df = carga(exp, sub)
            dfs[(m, ep)] = (df, sub)
            c, ps, n = crit(df)
            res[ep] = c
            print(f'  ep{ep:03d}{" (óptimo val/mae)" if ep == mae_ep else ""}: pooled {c:.3f} (n={n}) | ' +
                  ' '.join(f'{s} {v:.2f}' for s, v in sorted(ps.items())))
        sel[m] = (mae_ep, min(res, key=res.get))
        print(f'  -> óptimo DVH: ep{sel[m][1]:03d}')
    exps = {'2D': A4, 'MedNeXt': B2}
    for etiqueta, k in (('(a) óptimo val/mae', 0), ('(b) óptimo DVH', 1)):
        e2, e3 = sel['2D'][k], sel['MedNeXt'][k]
        print(f'\n{etiqueta}: 2D ep{e2} vs MedNeXt ep{e3}  (paired_diff = 2D−MedNeXt; >0 = MedNeXt mejor)')
        diff = dvh.paired_diff(dfs[('2D', e2)][0], dfs[('MedNeXt', e3)][0])
        for site, roles in diff.items():
            print('  ' + site.ljust(9) + ' | '.join(
                f'{role[:3]} {band}: {roles[role][band]["mean"]:+.2f} [{roles[role][band]["ci95"][0]:+.2f},{roles[role][band]["ci95"][1]:+.2f}]'
                for role, band in CELLS))
        print('  secundarias por sitio (media; MAE-body menor=mejor, SSIM mayor=mejor):')
        for m, e in (('2D', e2), ('MedNeXt', e3)):
            d = sec(exps[m], dfs[(m, e)][1])
            SSIMC = 'ssim_body' if 'ssim_body' in d.columns else 'ssim'   # 'ssim' = frame completo (OBSOLETO, evaluaciones previas a la vara BODY)
            s = d.groupby('site')[['mae_body', SSIMC]].mean()
            print('   ', m.ljust(8), ' | '.join(f'{site} MAE {r.mae_body:.3f} SSIM {r[SSIMC]:.3f}' for site, r in s.iterrows()))
    print('\nd2 (rugosidad/ratio por sitio, ckpt = óptimo val/mae):')
    for m, exp in exps.items():
        t = ROOT / 'results' / exp / 'd2_coherencia_axial' / 'tabla_sitio.csv'
        print(f'  {m}:'); print(pd.read_csv(t).to_string(index=False) if t.exists() else '  (sin d2)')


if __name__ == '__main__':
    main()
