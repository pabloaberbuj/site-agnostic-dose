"""
Rasterizacion de contornos RTSTRUCT (mm, mundo) sobre la grilla de dosis nativa.

Metodo: por cada contorno, se mapean sus vertices a pixeles (sub-pixel) con
geometry.world_xy_to_pixel y se redondean al pixel mas cercano (cv2.fillPoly exige
enteros). Varios contornos en la misma slice se componen con regla par-impar (XOR),
que es la convencion DICOM para representar agujeros/regiones separadas de un mismo
ROI en un mismo corte.

Este redondeo es la fuente de error esperable en el volumen rasterizado vs. el volumen
"real" del contorno (poligono continuo); se mide en el test de escala de voxel
(volumen de la mascara vs. volumen por shoelace del contorno).
"""
import numpy as np
import cv2

from .geometry import world_xy_to_pixel, frame_index_for_z, GeometryError


def rasterize_structure(contours, grid):
    """contours: lista de (z_mm, [(x_mm,y_mm), ...]). -> mask bool (nz,ny,nx).

    Contornos fuera del rango Z de la grilla de dosis se DESCARTAN (no fallan): es
    normal que el RTSTRUCT (y sobre todo BODY) cubra mas CT del que Eclipse calculo
    dosis — no hay dosis ahi, esa porcion no puede aportar a D95 ni a ningun calculo
    sobre la grilla nativa de todas formas."""
    nz, ny, nx = grid.shape
    mask = np.zeros((nz, ny, nx), dtype=bool)
    by_frame = {}
    for z_mm, pts in contours:
        try:
            k = frame_index_for_z(grid, z_mm)
        except GeometryError:
            continue
        by_frame.setdefault(k, []).append(pts)

    for k, poly_list in by_frame.items():
        slice_mask = np.zeros((ny, nx), dtype=np.uint8)
        for pts in poly_list:
            px = np.array([world_xy_to_pixel(grid, x, y) for x, y in pts], dtype=np.float64)
            px_int = np.round(px).astype(np.int32)
            layer = np.zeros((ny, nx), dtype=np.uint8)
            cv2.fillPoly(layer, [px_int], 1)
            slice_mask ^= layer   # regla par-impar: contornos superpuestos = agujero
        mask[k] = slice_mask.astype(bool)
    return mask


def voxel_volume_mm3(grid):
    sz, sy, sx = grid.spacing
    return sz * sy * sx


def mask_volume_cm3(mask, grid):
    return mask.sum() * voxel_volume_mm3(grid) / 1000.0


def contour_volume_cm3(contours):
    """Volumen de referencia por shoelace (areas por slice x paso entre slices del propio
    ROI). Es el volumen 'del contorno', independiente de la grilla — se usa como referencia
    en el test de escala de voxel, no se guarda en el NPZ."""
    from collections import defaultdict
    z_areas = defaultdict(float)
    for z_mm, pts in contours:
        z_areas[round(z_mm, 3)] += _polygon_area(pts)
    zvals = sorted(z_areas)
    if not zvals:
        return 0.0
    if len(zvals) > 1:
        dz = sum(zvals[i + 1] - zvals[i] for i in range(len(zvals) - 1)) / (len(zvals) - 1)
    else:
        dz = 1.0
    return sum(z_areas.values()) * dz / 1000.0


def _polygon_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0
