"""
Densidad másica por paciente (A5): CT (HU) -> g/cm3 con la curva clínica de Eclipse -> grilla de dosis nativa.

Orden: la curva se aplica en la resolución del CT y recién después se remuestrea la densidad (trilinear) a la
grilla del RTDOSE (CT->dosis, nunca al revés; CHARTER §5). Fuera del FOV del CT = aire (0.001).
La curva es la "CT Calibration Curve: Big Bore Mass_Density" (Beam Configuration 13.6.30), versionada en
data/density/. Solo existe esa; se aplica a todos los pacientes y se registra el escáner de cada uno.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import map_coordinates

from .dicom_io import find_by_modality
from .geometry import build_dose_grid

CURVE_PATH = Path(__file__).resolve().parents[2] / 'data' / 'density' / 'curve_bigbore_massdensity.csv'
AIR_DENSITY = 0.001


class DensityError(Exception):
    pass


def load_curve(path=CURVE_PATH):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    hu = np.array([float(r['hu']) for r in rows])
    rho = np.array([float(r['density_g_cm3']) for r in rows])
    if np.any(np.diff(hu) <= 0):
        raise DensityError('curva HU no estrictamente creciente')
    return hu, rho


def curve_hash(path=CURVE_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def hu_to_density(hu: np.ndarray, curve) -> np.ndarray:
    """Interpolación lineal por tramos; fuera de rango se satura a los extremos (np.interp)."""
    return np.interp(hu, curve[0], curve[1]).astype(np.float32)


def load_ct_hu(patient_dir, by_mod=None):
    """-> (hu (nz,ny,nx) float32 ordenado por z, z_mm (nz,), (x0,y0) mm, (row_sp,col_sp), header)"""
    by_mod = by_mod or find_by_modality(patient_dir)
    paths = by_mod.get('CT')
    if not paths:
        raise DensityError('sin serie CT')
    slices = [pydicom.dcmread(p) for p in paths]
    slices = [s for s in slices if s.Modality == 'CT']
    ori = tuple(float(v) for v in slices[0].ImageOrientationPatient)
    if max(abs(a - b) for a, b in zip(ori, (1, 0, 0, 0, 1, 0))) > 1e-3:
        raise DensityError(f'CT no axial estándar: {ori}')
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    if len(z) > 1 and np.diff(z).min() <= 0:
        raise DensityError('cortes de CT con z repetido (¿más de una serie?)')
    shapes = {(s.Rows, s.Columns) for s in slices}
    pos = {(round(float(s.ImagePositionPatient[0]), 3), round(float(s.ImagePositionPatient[1]), 3)) for s in slices}
    if len(shapes) != 1 or len(pos) != 1:
        raise DensityError(f'CT con grilla in-plane no uniforme: shapes={shapes} pos={len(pos)}')
    hu = np.stack([s.pixel_array.astype(np.float32) * float(getattr(s, 'RescaleSlope', 1)) +
                   float(getattr(s, 'RescaleIntercept', 0)) for s in slices])
    s0 = slices[0]
    header = {'manufacturer': str(s0.get('Manufacturer')), 'model': str(s0.get('ManufacturerModelName')),
              'kvp': str(s0.get('KVP')), 'kernel': str(s0.get('ConvolutionKernel')),
              'ct_spacing_mm': [float(v) for v in s0.PixelSpacing], 'ct_slices': len(z)}
    x0, y0 = pos.pop()
    return hu, z, (x0, y0), tuple(float(v) for v in s0.PixelSpacing), header


def density_on_dose_grid(patient_dir, curve=None):
    """-> (rho (nz,ny,nx) float32 sobre la grilla del RTDOSE, info dict)"""
    curve = curve or load_curve()
    by_mod = find_by_modality(patient_dir)
    rd = by_mod.get('RTDOSE')
    if not rd or len(rd) != 1:
        raise DensityError(f'se esperaba 1 RTDOSE, hay {len(rd) if rd else 0}')
    grid = build_dose_grid(pydicom.dcmread(rd[0], stop_before_pixels=True))
    hu, z_ct, (cx0, cy0), (rsp, csp), header = load_ct_hu(patient_dir, by_mod)
    rho_ct = hu_to_density(hu, curve)

    nz, ny, nx = grid.shape
    _, dry, drx = grid.spacing
    x0, y0, _ = grid.origin
    xs = (x0 + np.arange(nx) * drx - cx0) / csp           # col fraccional en el CT
    ys = (y0 + np.arange(ny) * dry - cy0) / rsp           # fila fraccional en el CT
    zs = np.interp(grid.frame_z_mm, z_ct, np.arange(len(z_ct)), left=-1, right=len(z_ct))  # corte fraccional
    # borde: un frame de dosis a <= 1 corte (+tol) fuera del CT replica el corte CT extremo en vez de caer a aire
    dz_ct = float(np.median(np.diff(z_ct))) if len(z_ct) > 1 else 0.0
    tol = dz_ct + 0.75
    zs = np.where((grid.frame_z_mm < z_ct[0]) & (grid.frame_z_mm >= z_ct[0] - tol), 0.0, zs)
    zs = np.where((grid.frame_z_mm > z_ct[-1]) & (grid.frame_z_mm <= z_ct[-1] + tol), len(z_ct) - 1.0, zs)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing='ij')
    rho = map_coordinates(rho_ct, [zz, yy, xx], order=1, mode='constant', cval=AIR_DENSITY).astype(np.float32)
    info = dict(header, dose_shape=list(grid.shape), z_ct_range=[float(z_ct[0]), float(z_ct[-1])],
                z_dose_range=[float(grid.frame_z_mm[0]), float(grid.frame_z_mm[-1])],
                hu_min=float(hu.min()), hu_max=float(hu.max()))
    return rho, info
