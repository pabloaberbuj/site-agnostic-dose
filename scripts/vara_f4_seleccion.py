"""
Vara oficial de Fase 4 = re-eval de exp_a4_003 con el harness vigente (pred*BODY, SSIM_BODY). Solo val; no entrena; test sellado.

Selección de checkpoint (regla de Fase 3, autocontenida): entre los top-3 por val/mae que guarda Lightning, el de menor
mean|ΔV| pooled en banda media 40-80 %Rx de OAR_all (65 pacientes de val). Copia el elegido a la raíz de la carpeta de salida
(metrics.csv + summary.json + secondary_per_patient.csv) y anota la regla en summary.json. Imprime la tabla de la vara y el delta
contra la evaluación legacy (results/exp_a4_003.../, pred cruda, SSIM de frame).

    .venv/Scripts/python.exe scripts/vara_f4_seleccion.py
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval import dvh  # noqa: E402

N = 'exp_a4_003_unet2d_psdm_c176x256'
OUT = ROOT / 'results' / f'{N}_vara_f4'
LEGACY = ROOT / 'results' / N
TABLA = ROOT / 'results' / 'ssim_validacion' / 'per_patient_2D.csv'   # pred cruda, mismo ckpt ep113
VAL_MAE_OPT = 113   # ckpt de óptimo val/mae (best de Lightning)
FILES = ('metrics.csv', 'summary.json', 'secondary_per_patient.csv', 'crop_offsets.csv')


def crit(df):
    por_pac = df[df.role == 'OAR_all'][['patient_id', 'site', 'media_40_80']]
    return por_pac.media_40_80.mean(), por_pac.groupby('site').media_40_80.mean().to_dict()


def main():
    eps = sorted(int(p.name[7:]) for p in OUT.glob('ckpt_ep*') if (p / 'summary.json').exists())
    res = {ep: crit(pd.read_csv(OUT / f'ckpt_ep{ep}' / 'metrics.csv')) for ep in eps}
    print('== criterio pooled OAR banda media 40-80 (menor = mejor), val n=65 ==')
    for ep, (c, ps) in res.items():
        print(f'  ep{ep}{" (óptimo val/mae)" if ep == VAL_MAE_OPT else ""}: pooled {c:.3f} | ' + ' '.join(f'{s} {v:.2f}' for s, v in sorted(ps.items())))
    best = min(res, key=lambda e: res[e][0])
    print(f'  -> checkpoint elegido: ep{best}')
    src = OUT / f'ckpt_ep{best}'
    for f in FILES:
        shutil.copy(src / f, OUT / f)
    sj = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    sj['vara_fase4'] = {'experimento': N, 'ckpt_elegido': f'epoch={best}.ckpt', 'split': 'val',
                        'regla': 'top-3 por val/mae; menor mean|ΔV| pooled OAR_all banda media 40-80 en val (n=65)',
                        'pooled_oar_media_por_ckpt': {str(e): res[e][0] for e in res},
                        'harness': 'pred*BODY; SSIM_BODY y SSIM BODY∩dosis>5% Rx_High (src/eval/ssim.py)'}
    (OUT / 'summary.json').write_text(json.dumps(sj, indent=2), encoding='utf-8')

    new = pd.read_csv(OUT / 'metrics.csv')
    old = pd.read_csv(LEGACY / 'metrics.csv')
    sec = sj['secundarias_por_sitio']
    ci = lambda r: f"{r['mean']:.2f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]"
    cip = lambda r: f"{r['mean']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
    print(f'\n== VARA (ep{best}) por sitio, IC95 bootstrap ==')
    for site in ('CyC', 'Pelvis', 'Prostata'):
        d = sj['dvh_mean_abs_delta_V_pp'][site]
        print(f"  {site:9s} pooled-OAR-media(sitio) {res[best][1][site]:.2f} | OAR-media 40-80 {ci(d['OAR_all']['media_40_80'])} | "
              f"PTV-alta 80-110 {ci(d['PTV_all']['alta_80_110'])}")
        print(f"  {'':9s} MAE-body {ci(sec[site]['mae_body'])} | SSIM_BODY {cip(sec[site]['ssim_body'])} | "
              f"SSIM BODY∩dosis>5% {cip(sec[site]['ssim_body_dosis5'])} | [frame obsoleto {sec[site]['ssim_frame_obsoleto']['mean']:.3f}]")
    print('\n== DELTA vs evaluación legacy de exp_a4_003 ep113 (nuevo − legacy; pred*BODY vs cruda) ==')
    m = new.merge(old, on=['patient_id', 'site', 'role'], suffixes=('_n', '_o'))
    for site in ('CyC', 'Pelvis', 'Prostata'):
        out = []
        for role, band in (('OAR_all', 'media_40_80'), ('PTV_all', 'alta_80_110')):
            g = m[(m.site == site) & (m.role == role)]
            out.append(f'{role} {band} {(g[band + "_n"] - g[band + "_o"]).mean():+.4f} (max|dif| paciente {(g[band + "_n"] - g[band + "_o"]).abs().max():.4f})')
        print(f'  {site:9s} ' + ' | '.join(out))
    if best == VAL_MAE_OPT and TABLA.exists():
        t = pd.read_csv(TABLA).merge(pd.read_csv(OUT / 'secondary_per_patient.csv'), on='patient_id')
        for c, ct in (('ssim_body', 'body_ssim'), ('ssim_body_dosis5', 'body_dosis5_ssim')):
            t['d'] = t[c] - t[ct]
            print(f'  {c} nuevo − tabla ssim_validacion (cruda): media {t.d.mean():+.4f}, max|dif| {t.d.abs().max():.4f}')
        so = pd.read_csv(LEGACY / 'secondary_per_patient.csv').merge(pd.read_csv(OUT / 'secondary_per_patient.csv'), on='patient_id', suffixes=('_o', ''))
        print(f'  mae_body nuevo − legacy: max|dif| {(so.mae_body - so.mae_body_o).abs().max():.2e}')


if __name__ == '__main__':
    main()
