"""HO#8: A4 (A1-uniform + PSDM PTV_High) vs A1-uniform (A1a fill 0) vs A0, val, por sitio, IC95 bootstrap pareado.
Convención de diff: ref - new (>0 = new mejor, para métricas de error); para SSIM se reporta new - ref (>0 = new mejor).
Piso de ruido: A0 seed 42 vs seed 43."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval import dvh
R = ROOT / 'results'
RUNS = {'A0': ('exp_a0_001_unet2d', ''), 
        'A1u': ('exp_a1a_001_unet2d_fill0', 'fill_0'), 'A4': ('exp_a4_001_unet2d_psdm', '')}
def load(k):
    d, p = RUNS[k]
    m = dvh.with_groups(pd.read_csv(R / d / f'{p}metrics.csv'))
    s = pd.read_csv(R / d / f'{p}secondary_per_patient.csv')
    return m, s
L = {k: load(k) for k in RUNS}
def band(m, role, col):  # por (sitio) -> serie por paciente
    return m[m.role == role].set_index('patient_id')[col]
def ci(a, b, sign):  # sign=+1: a-b ; paired bootstrap
    j = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna()
    return dvh.bootstrap_ci(sign * (j.a - j.b).values)
out, rows = {}, []
for site in ('CyC', 'Pelvis', 'Prostata'):
    for metric, getter in (('SSIM', lambda k: L[k][1][L[k][1].site == site].set_index('patient_id')['ssim']),
                           ('MAE_body', lambda k: L[k][1][L[k][1].site == site].set_index('patient_id')['mae_body']),
                           ('PTV_alta', lambda k: band(L[k][0][L[k][0].site == site], 'PTV_all', 'alta_80_110')),
                           ('OAR_media', lambda k: band(L[k][0][L[k][0].site == site], 'OAR_all', 'media_40_80'))):
        S = {k: getter(k) for k in RUNS}
        better = (lambda a, b: ci(a, b, +1)) if metric == 'SSIM' else (lambda a, b: ci(b, a, +1))  # >0 = primero mejor
        e = {'mean': {k: float(S[k].mean()) for k in RUNS},
             'A4_vs_A1u': better(S['A4'], S['A1u']), 'A4_vs_A0': better(S['A4'], S['A0']),
             'A1u_vs_A0': better(S['A1u'], S['A0'])}
        out.setdefault(site, {})[metric] = e
        f = lambda x: f"{x['mean']:+.2f} [{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]"
        rows.append((site, metric, *(f"{e['mean'][k]:.3f}" for k in ('A0', 'A1u', 'A4')), f(e['A4_vs_A1u']), f(e['A4_vs_A0'])))
(R / 'exp_a4_comparison.json').write_text(json.dumps({'convencion': '>0 = primero mejor (SSIM: new-ref; errores: ref-new); val', 'by_site': out}, indent=1))
print(pd.DataFrame(rows, columns=['sitio', 'metrica', 'A0', 'A1u', 'A4', 'A4-A1u(>0 A4 mejor)', 'A4-A0']).to_string(index=False))
