"""HO#9: 2x2 completo A0 / A1-uniform / A4 / dist-solo, val, por sitio, IC95 bootstrap pareado, contra el piso de ruido.
Convención: >0 = primero mejor (SSIM: new-ref; errores: ref-new). Piso de ruido = A0 seed 42 vs seed 43 (pareado, mismos pacientes);
si seed 43 no tiene metrics.csv se omite y se avisa (la decisión 'dist-solo ≈ A4' queda sin su vara)."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval import dvh
R = ROOT / 'results'
RUNS = {'A0': ('exp_a0_001_unet2d', ''), 'A1u': ('exp_a1a_001_unet2d_fill0', 'fill_0'),
        'A4': ('exp_a4_001_unet2d_psdm', ''), 'dist': ('exp_a4_002_unet2d_distsolo', '')}
NOISE = ('exp_a0_002_unet2d_seed43', '')
def load(d, p):
    if not (R / d / f'{p}metrics.csv').exists():
        return None
    return dvh.with_groups(pd.read_csv(R / d / f'{p}metrics.csv')), pd.read_csv(R / d / f'{p}secondary_per_patient.csv')
L = {k: load(*v) for k, v in RUNS.items()}
missing = [k for k, v in L.items() if v is None]
if missing:
    sys.exit(f'faltan resultados: {missing}')
noise = load(*NOISE)
if noise is None:
    print('AVISO: sin piso de ruido (A0 seed 43 cancelado, CHARTER 11, ver cabos_sueltos.md); "dist-solo ~ A4" se juzga con rango entre celdas + IC95 pareado.\n')
else:
    L['A0s43'] = noise
def band(m, role, col):
    return m[m.role == role].set_index('patient_id')[col]
def sec(k, site, col):
    s = L[k][1]
    return s[s.site == site].set_index('patient_id')[col]
GET = {'SSIM': lambda k, site: sec(k, site, 'ssim'), 'MAE_body': lambda k, site: sec(k, site, 'mae_body'),
       'PTV_alta': lambda k, site: band(L[k][0][L[k][0].site == site], 'PTV_all', 'alta_80_110'),
       'OAR_media': lambda k, site: band(L[k][0][L[k][0].site == site], 'OAR_all', 'media_40_80')}
def better(a, b, metric):  # >0 = a mejor
    j = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna()
    d = (j.a - j.b) if metric == 'SSIM' else (j.b - j.a)
    return dvh.bootstrap_ci(d.values)
f = lambda x: f"{x['mean']:+.2f} [{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]"
out, rows = {}, []
for site in ('CyC', 'Pelvis', 'Prostata'):
    for metric, g in GET.items():
        S = {k: g(k, site) for k in L}
        e = {'mean': {k: float(S[k].mean()) for k in L},
             'dist_vs_A4': better(S['dist'], S['A4'], metric), 'dist_vs_A0': better(S['dist'], S['A0'], metric),
             'A4_vs_A0': better(S['A4'], S['A0'], metric)}
        if 'A0s43' in S:
            e['ruido_A0_s42_vs_s43'] = better(S['A0s43'], S['A0'], metric)
        cells = [e['mean'][k] for k in ('A0', 'A1u', 'A4', 'dist')]
        e['rango_4_celdas'] = max(cells) - min(cells)   # vara cruda sin piso de ruido (decisión 2026-10-25)
        e['dist_vs_A4_frac_rango'] = abs(e['mean']['dist'] - e['mean']['A4']) / e['rango_4_celdas'] if e['rango_4_celdas'] else 0.0
        out.setdefault(site, {})[metric] = e
        rows.append((site, metric, *(f"{e['mean'][k]:.3f}" for k in ('A0', 'A1u', 'A4', 'dist')),
                     f(e['dist_vs_A4']), f(e['dist_vs_A0']), f(e['ruido_A0_s42_vs_s43']) if 'ruido_A0_s42_vs_s43' in e else 'n/d',
                     f"{e['rango_4_celdas']:.3f}", f"{e['dist_vs_A4_frac_rango']:.0%}"))
(R / 'exp_a4_2x2_comparison.json').write_text(json.dumps({'convencion': '>0 = primero mejor; val', 'by_site': out}, indent=1))
print(pd.DataFrame(rows, columns=['sitio', 'metrica', 'A0', 'A1u', 'A4', 'dist', 'dist-A4(>0 dist mejor)', 'dist-A0', 'ruido s43-s42', 'rango 4 celdas', '|dist-A4|/rango']).to_string(index=False))
