"""
Smoke test de A5 (HO#10): A4 + Ch7 densidad.

    .venv/Scripts/python.exe scripts/smoke_a5.py [--all] [--xcheck N]

Por paciente: shape (8,Z,H,W)/dtype/NaN · Ch0-6 idénticos a A4 · Ch7 == 0 fuera de BODY, en [0.001,~3] dentro (sin NaN) ·
pad_channels: Ch6 (PSDM) pad 1.0 y Ch7 pad 0, Ch0-5 pad 0 · determinismo.
Cross-check (N pacientes): densidad en vóxeles de BODY vs HU leído directo del DICOM (voxel de CT más cercano -> curva);
mediana del error relativo debe ser chica (la diferencia es solo interpolación/resampleo).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datamodules.a4_datamodule import A4DataModule, N_CHANNELS_A4  # noqa: E402
from src.datamodules.a5_datamodule import A5DataModule, N_CHANNELS_A5, CH_DENSITY  # noqa: E402
from src.preprocess import density as D  # noqa: E402
from src.preprocess.dicom_io import find_by_modality  # noqa: E402
from src.preprocess.geometry import build_dose_grid  # noqa: E402

NPZ = ROOT / '../../data/npz_v2b_body'
DENS = ROOT / '../../data/density_v1'
DICOM = ROOT / '../../data/dicom_raw'


def xcheck(pid, site, n=3000, seed=0):
    """Error relativo mediano de densidad vs vecino-más-cercano del CT crudo, en vóxeles de BODY."""
    pdir = next(DICOM.glob(f'*/dcm export/*/{pid}'))
    by = find_by_modality(str(pdir))
    grid = build_dose_grid(pydicom.dcmread(by['RTDOSE'][0], stop_before_pixels=True))
    sl = sorted((pydicom.dcmread(p) for p in by['CT']), key=lambda s: float(s.ImagePositionPatient[2]))
    z = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    x0, y0 = (float(v) for v in sl[0].ImagePositionPatient[:2])
    sp = float(sl[0].PixelSpacing[0])
    curve = D.load_curve()
    rho = np.load(DENS / site / f'{pid}.npy').astype(np.float32)
    body = np.load(NPZ / site / f'{pid}.npz')['mask_BODY']
    idx = np.argwhere(body)
    idx = idx[np.random.default_rng(seed).choice(len(idx), min(n, len(idx)), replace=False)]
    errs = []
    for k, i, j in idx:
        zk = int(np.argmin(np.abs(z - grid.frame_z_mm[k])))
        if abs(z[zk] - grid.frame_z_mm[k]) > 3.8:
            continue
        col = int(round((grid.origin[0] + j * grid.spacing[2] - x0) / sp))
        row = int(round((grid.origin[1] + i * grid.spacing[1] - y0) / sp))
        hu = float(sl[zk].pixel_array[row, col]) * float(sl[zk].RescaleSlope) + float(sl[zk].RescaleIntercept)
        ref = float(np.interp(hu, *curve))
        errs.append(abs(rho[k, i, j] - ref) / max(ref, 0.05))
    return float(np.median(errs)), float(np.mean(np.array(errs) < 0.15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--xcheck', type=int, default=3)
    a = ap.parse_args()
    args = (NPZ, ROOT / 'data/splits/split_v2.json', ROOT / 'data/harmonization')
    dm4 = A4DataModule(*args)
    dm4.setup()
    dm5 = A5DataModule(*args, density_dir=DENS)
    dm5.setup()
    pool = [(dm5.train_ds, dm4.train_ds), (dm5.val_ds, dm4.val_ds), (dm5.test_ds, dm4.test_ds)]
    fails, n, rng_stats, xc = [], 0, [], []
    for d5, d4 in pool:
        sel = d5.ids if a.all else d5.ids[:4]
        for pid in sel:
            n += 1
            dose, c5, meta = d5.load_raw(pid)
            _, c4, _ = d4.load_raw(pid)
            if c5.shape != (N_CHANNELS_A5,) + dose.shape or c5.dtype != np.float32 or not np.isfinite(c5).all():
                fails.append(f'{pid}: shape/dtype/NaN {c5.shape} {c5.dtype}')
                continue
            if not np.array_equal(c5[:N_CHANNELS_A4], c4):
                fails.append(f'{pid}: Ch0-6 difieren de A4')
            body = c5[2] > 0.5
            rho = c5[CH_DENSITY]
            if (rho[~body] != 0).any():
                fails.append(f'{pid}: densidad != 0 fuera de BODY')
            if rho[body].min() < 0.0009 or rho[body].max() > 3.1:
                fails.append(f'{pid}: densidad fuera de rango en BODY [{rho[body].min():.4f},{rho[body].max():.3f}]')
            rng_stats.append((rho[body].mean(), rho[body].max()))
            p = d5.pad_channels(c5)
            if p.shape[0] != N_CHANNELS_A5 or p.shape[-2:] != (d5.pad_hw, d5.pad_hw):
                fails.append(f'{pid}: shape pad {p.shape}')
            if (p[6, :, 0, 0] != 1.0).any() or (p[7, :, 0, 0] != 0.0).any() or (p[:6, :, 0, 0] != 0).any():
                fails.append(f'{pid}: padding de canales incorrecto')
            if not np.array_equal(d5.load_raw(pid)[1], c5):
                fails.append(f'{pid}: no determinista')
            if len(xc) < a.xcheck:
                med, frac = xcheck(pid, meta['site'])
                xc.append((pid, meta['site'], med, frac))
                if med > 0.05:
                    fails.append(f'{pid}: xcheck densidad vs CT crudo, error rel mediano {med:.3f}')
    r = np.array(rng_stats)
    print(f'{n} pacientes; body densidad media {r[:, 0].min():.3f}..{r[:, 0].max():.3f}, max {r[:, 1].max():.2f}')
    for pid, site, med, frac in xc:
        print(f'  xcheck {site} {pid}: error rel mediano {med:.4f}, {frac:.1%} dentro de 15%')
    print(f'{len(fails)} fallas')
    for f in fails[:20]:
        print('  FALLA', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
