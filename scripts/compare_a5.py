"""HO#10: A5 (A4 + densidad) vs A4, val, por sitio, IC95 bootstrap pareado (A1u y A0 como contexto).
Convención: >0 = A5 mejor (errores: A4-A5; SSIM: A5-A4)."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval import dvh
R = ROOT / 'results'
RUNS = {'A0': ('exp_a0_001_unet2d', ''), 'A1u': ('exp_a1a_001_unet2d_fill0', 'fill_0'),
        'A4': ('exp_a4_001_unet2d_psdm', ''), 'A5': ('exp_a5_001_unet2d_density', '')}
def load(k):
    d, p = RUNS[k]
    return dvh.with_groups(pd.read_csv(R / d / f'{p}metrics.csv')), pd.read_csv(R / d / f'{p}secondary_per_patient.csv')
L = {k: load(k) for k in RUNS}
band = lambda m, role, col: m[m.role == role].set_index('patient_id')[col]
def ci(a, b, sign):
    j = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna()
    return dvh.bootstrap_ci(sign * (j.a - j.b).values)
out, rows = {}, []
for site in ('CyC', 'Pelvis', 'Prostata'):
    for metric, getter in (('SSIM', lambda k: L[k][1][L[k][1].site == site].set_index('patient_id')['ssim']),
                           ('MAE_body', lambda k: L[k][1][L[k][1].site == site].set_index('patient_id')['mae_body']),
                           ('PTV_alta', lambda k: band(L[k][0][L[k][0].site == site], 'PTV_all', 'alta_80_110')),
                           ('OAR_media', lambda k: band(L[k][0][L[k][0].site == site], 'OAR_all', 'media_40_80'))):
        S = {k: getter(k) for k in RUNS}
        better = (lambda a, b: ci(a, b, +1)) if metric == 'SSIM' else (lambda a, b: ci(b, a, +1))
        e = {'mean': {k: float(S[k].mean()) for k in RUNS}, 'A5_vs_A4': better(S['A5'], S['A4']),
             'A5_vs_A0': better(S['A5'], S['A0'])}
        out.setdefault(site, {})[metric] = e
        f = lambda x: f"{x['mean']:+.2f} [{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]"
        rows.append((site, metric, *(f"{e['mean'][k]:.3f}" for k in RUNS), f(e['A5_vs_A4'])))
(R / 'exp_a5_comparison.json').write_text(json.dumps({'convencion': '>0 = A5 mejor; val', 'by_site': out}, indent=1))
print(pd.DataFrame(rows, columns=['sitio', 'metrica', *RUNS, 'A5-A4 (>0 A5 mejor)']).to_string(index=False))
