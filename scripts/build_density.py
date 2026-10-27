"""Genera <out>/<sitio>/<pid>.npy (float16, g/cm3, grilla de dosis nativa) + manifest.json.
Uso: python scripts/build_density.py --npz-dir <npz_v2b_body> --dicom-root <dicom_raw> --out <density_v1> [--workers 4]"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import density as D


def _work(args):
    pid, site, pdir, npz_path, out = args
    try:
        rho, info = D.density_on_dose_grid(pdir)
        with np.load(npz_path) as d:
            shp = d['dose_pct'].shape
        if tuple(rho.shape) != tuple(shp):
            raise D.DensityError(f'shape densidad {rho.shape} != NPZ {shp}')
        dst = Path(out) / site / f'{pid}.npy'
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, rho.astype(np.float16))
        return pid, dict(info, site=site, status='ok')
    except Exception as e:
        return pid, {'site': site, 'status': 'error', 'error': f'{type(e).__name__}: {e}'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--dicom-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    dirs = {p.name: p for p in Path(a.dicom_root).glob('*/dcm export/*/PT_*') if p.is_dir()}
    jobs = []
    for npz in sorted(Path(a.npz_dir).glob('*/*.npz')):
        if npz.stem not in dirs:
            raise SystemExit(f'{npz.stem}: sin carpeta DICOM en {a.dicom_root}')
        jobs.append((npz.stem, npz.parent.name, str(dirs[npz.stem]), str(npz), a.out))
    if a.limit:
        jobs = jobs[::max(1, len(jobs) // a.limit)][:a.limit]
    t0 = time.time()
    manifest = {}
    with Pool(a.workers) as pool:
        for i, (pid, info) in enumerate(pool.imap_unordered(_work, jobs), 1):
            manifest[pid] = info
            if info['status'] != 'ok':
                print('ERROR', pid, info['error'], flush=True)
            if i % 25 == 0:
                print(f'{i}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump({'curve': 'Big Bore Mass_Density (Beam Configuration 13.6.30)', 'curve_hash': D.curve_hash(),
                   'patients': manifest}, f, indent=1)
    bad = [p for p, i in manifest.items() if i['status'] != 'ok']
    print(f'listo: {len(manifest) - len(bad)} ok, {len(bad)} errores, {time.time() - t0:.0f}s')
