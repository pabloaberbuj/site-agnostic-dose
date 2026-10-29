"""
Guardrail del contenedor (Fase 3): OAR-media y PTV-alta son frame-independientes (DVH sobre grilla nativa) -> el A4-2D en el
contenedor (176,256) con crop-to-BODY debe reproducir al A4-2D viejo (256^2) dentro de ruido. Si driftean, el crop/un-crop
tiene un bug de offsets: PARAR.

    .venv/Scripts/python.exe scripts/guardrail_a4_container.py [--old-dir ...] [--new-dir ...] [--tol 1.0]

Compara por sitio, pareado por paciente (diferencia ref - new, pp de volumen, IC95 bootstrap) en OAR_all banda media (40-80)
y PTV_all banda alta (80-110). Criterio (fijado antes de correr): FALLA (exit 2) si algún |media| > --tol pp. Entre 0.5*tol y tol
con IC que excluye 0 = aviso (zona gris), no corta. SSIM / MAE-body se imprimen pero no entran al criterio (pueden moverse).
Sin piso de ruido seed-a-seed (CHARTER §11): la tolerancia apunta a bugs de offset (que dan drift de varios pp), no a ruido fino.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import dvh  # noqa: E402

CELLS = (('OAR_all', 'media_40_80'), ('PTV_all', 'alta_80_110'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-dir', default='results/exp_a4_001_unet2d_psdm')
    ap.add_argument('--new-dir', default='results/exp_a4_003_unet2d_psdm_c176x256')
    ap.add_argument('--tol', type=float, default=1.0)
    a = ap.parse_args()
    old = pd.read_csv(ROOT / a.old_dir / 'metrics.csv')
    new = pd.read_csv(ROOT / a.new_dir / 'metrics.csv')
    diff = dvh.paired_diff(old, new)   # old - new; >0 = el nuevo mejora
    fail, warn = [], []
    print(f'guardrail contenedor: viejo={a.old_dir} nuevo={a.new_dir} tol={a.tol} pp (old - new, >0 = nuevo mejor)')
    for site, roles in diff.items():
        for role, band in CELLS:
            r = roles[role][band]
            m, (lo, hi) = r['mean'], r['ci95']
            tag = ''
            if abs(m) > a.tol:
                fail.append((site, role, m)); tag = '  <-- FALLA'
            elif abs(m) > 0.5 * a.tol and not (lo <= 0 <= hi):
                warn.append((site, role, m)); tag = '  <-- zona gris'
            print(f'  {site:9s} {role:8s} {band:12s} {m:+.2f} [{lo:+.2f},{hi:+.2f}] n={r["n"]}{tag}')
    for d in (a.old_dir, a.new_dir):
        sj = json.loads((ROOT / d / 'summary.json').read_text(encoding='utf-8'))
        sec = sj.get('secundarias_por_sitio', {})
        print(f'  secundarias {Path(d).name}: ' + ' | '.join(
            f'{s} SSIM {v["ssim"]["mean"]:.3f} MAEbody {v["mae_body"]["mean"]:.2f}' for s, v in sec.items()))
    if fail:
        print(f'GUARDRAIL FALLA en {len(fail)} celda(s): posible bug de offset en crop/un-crop. PARAR.')
        sys.exit(2)
    print('guardrail OK' + (f' ({len(warn)} celda(s) en zona gris, ver arriba)' if warn else ''))


if __name__ == '__main__':
    main()
