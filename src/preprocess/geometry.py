"""
Geometria de la grilla de dosis (RTDOSE) y mapeo mundo(mm)->voxel.

Asume orientacion axial estandar (ImageOrientationPatient == [1,0,0,0,1,0]): fila
avanza en X, columna avanza en Y, los frames apilan en Z via GridFrameOffsetVector.
Si un paciente no cumple esto, FALLA RUIDOSO (no se adivina una orientacion distinta).
"""
from dataclasses import dataclass
import numpy as np

_AXIAL_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
_ORIENTATION_TOL = 1e-3


@dataclass
class DoseGrid:
    shape: tuple           # (nz, ny, nx)
    spacing: tuple         # (sz, sy, sx) mm
    origin: tuple          # (x0, y0, z0) mm — ImagePositionPatient (frame 0)
    frame_z_mm: np.ndarray  # posicion Z absoluta de cada frame, shape (nz,)


class GeometryError(Exception):
    pass


def build_dose_grid(ds):
    orient = tuple(float(v) for v in ds.ImageOrientationPatient)
    if max(abs(a - b) for a, b in zip(orient, _AXIAL_ORIENTATION)) > _ORIENTATION_TOL:
        raise GeometryError(f'ImageOrientationPatient no es axial estandar: {orient}')

    rows, cols = int(ds.Rows), int(ds.Columns)
    row_sp, col_sp = (float(v) for v in ds.PixelSpacing)   # [spacing filas(Y), spacing columnas(X)]
    offsets = np.array([float(v) for v in ds.GridFrameOffsetVector], dtype=float)
    if len(offsets) < 1:
        raise GeometryError('GridFrameOffsetVector vacio')
    diffs = np.diff(offsets)
    if len(diffs) > 1 and (diffs.max() - diffs.min()) > 1e-2:
        raise GeometryError(f'spacing Z no uniforme dentro del paciente: {diffs.min()}..{diffs.max()}')
    z_spacing = float(diffs.mean()) if len(diffs) else float(getattr(ds, 'SliceThickness', 0.0))

    x0, y0, z0 = (float(v) for v in ds.ImagePositionPatient)
    frame_z = z0 + offsets

    grid = DoseGrid(
        shape=(len(offsets), rows, cols),
        spacing=(z_spacing, row_sp, col_sp),
        origin=(x0, y0, z0),
        frame_z_mm=frame_z,
    )
    return grid


def world_xy_to_pixel(grid: DoseGrid, x_mm, y_mm):
    """(x,y) mm -> (col, row) en coordenadas de pixel (float, sub-pixel)."""
    x0, y0, _ = grid.origin
    _, row_sp, col_sp = grid.spacing
    col = (x_mm - x0) / col_sp
    row = (y_mm - y0) / row_sp
    return col, row


def frame_index_for_z(grid: DoseGrid, z_mm, tol_mm=0.75):
    """Indice de frame cuyo Z esta mas cerca de z_mm. Lanza si no hay ninguno dentro de tol."""
    diffs = np.abs(grid.frame_z_mm - z_mm)
    k = int(np.argmin(diffs))
    if diffs[k] > tol_mm:
        raise GeometryError(f'z={z_mm} no tiene frame de dosis cercano (min dist={diffs[k]:.2f}mm)')
    return k
