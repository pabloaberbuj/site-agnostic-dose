#!/usr/bin/env python3
"""
Paso 1 de Fase 0 — inventario barato de headers DICOM, para TODOS los pacientes
(no la muestra de 6). Lee RTDOSE (headers, sin pixel data) y RTSTRUCT (contornos,
sin imagen) por paciente; NO carga CT ni la matriz de dosis completa.

Reporta por sitio: dimensiones/spacing de la grilla de RTDOSE, DoseGridScaling,
extent fisico del BODY, y presencia/volumen aproximado de los PTV que el YAML del
sitio espera (en particular el candidato recortado "-04").

El volumen de PTV acá es una SUMA DE AREAS DE CONTORNO (shoelace) x espaciado entre
cortes de ese ROI: sirve para "existe/no existe/volumen=0" y para una primera pasada
de escala de vóxel (Paso 1 ES esa primera pasada). No es el volumen validado de
Paso 2 (eso usa rasterización sobre la grilla de dosis nativa + tests dedicados).

Uso:
    python audit_dicom_headers.py "<export>/20261008_1627_CyC:CyC" \
        "<export>/20261008_1812_PelvisGin:Pelvis" \
        "<export>/20261008_2055_ProstataHipo:Prostata" \
        --harmonization-dir ../data/harmonization \
        --out ../results/auditorias/paso1_headers.csv
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import yaml
import pydicom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess.normalize_structure_name import normalize_ptv_name

PAD_LIMIT = (176, 128, 256)   # (Z, Y, X) — tamaño de padding global provisional (CHARTER/ARRANQUE)
Z_SPACING_TOL_MM = 1e-2


def find_files(patient_dir):
    rd = rs = None
    for fn in os.listdir(patient_dir):
        if fn.startswith('RD.'):
            rd = os.path.join(patient_dir, fn)
        elif fn.startswith('RS.'):
            rs = os.path.join(patient_dir, fn)
    return rd, rs


def read_dose_header(rd_path):
    ds = pydicom.dcmread(rd_path, stop_before_pixels=True)
    rows, cols = int(ds.Rows), int(ds.Columns)
    offsets = [float(x) for x in getattr(ds, 'GridFrameOffsetVector', [])]
    nz = len(offsets) if offsets else int(getattr(ds, 'NumberOfFrames', 1))
    row_sp, col_sp = [float(x) for x in ds.PixelSpacing]
    z_diffs = [round(offsets[i + 1] - offsets[i], 6) for i in range(len(offsets) - 1)]
    z_uniform = (max(z_diffs) - min(z_diffs) <= Z_SPACING_TOL_MM) if len(z_diffs) > 1 else True
    z_spacing = (sum(z_diffs) / len(z_diffs)) if z_diffs else None
    return {
        'rd_nz': nz, 'rd_ny': rows, 'rd_nx': cols,
        'spacing_row_mm': row_sp, 'spacing_col_mm': col_sp, 'spacing_z_mm': z_spacing,
        'z_spacing_uniform': z_uniform,
        'dose_grid_scaling': float(getattr(ds, 'DoseGridScaling', 'nan')),
    }


def polygon_area_xy(points):
    """Shoelace sobre (x,y) de una lista de puntos (x,y,z) coplanares en z."""
    n = len(points)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1, _ = points[i]
        x2, y2, _ = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def read_structures(rs_path):
    """-> dict nombre_crudo -> {'volume_cm3': float, 'bbox': (xmin,xmax,ymin,ymax,zmin,zmax), 'n_contours': int}"""
    ds = pydicom.dcmread(rs_path)
    roi_name = {}
    for item in ds.StructureSetROISequence:
        roi_name[int(item.ROINumber)] = str(item.ROIName)

    out = {}
    for citem in getattr(ds, 'ROIContourSequence', []):
        num = int(citem.ReferencedROINumber)
        name = roi_name.get(num, f'<ROI {num}>')
        contours = getattr(citem, 'ContourSequence', None)
        if not contours:
            out[name] = {'volume_cm3': 0.0, 'bbox': None, 'n_contours': 0}
            continue
        z_areas = defaultdict(float)
        xs, ys, zs = [], [], []
        for c in contours:
            data = [float(v) for v in c.ContourData]
            pts = [(data[i], data[i + 1], data[i + 2]) for i in range(0, len(data), 3)]
            if not pts:
                continue
            z = round(pts[0][2], 3)
            z_areas[z] += polygon_area_xy(pts)
            for x, y, zz in pts:
                xs.append(x); ys.append(y); zs.append(zz)
        if not xs:
            out[name] = {'volume_cm3': 0.0, 'bbox': None, 'n_contours': 0}
            continue
        zvals = sorted(z_areas)
        if len(zvals) > 1:
            dz = sum(zvals[i + 1] - zvals[i] for i in range(len(zvals) - 1)) / (len(zvals) - 1)
        else:
            dz = 1.0   # ROI de un solo corte: aproximación, no afecta el chequeo existe/!=0
        vol_mm3 = sum(z_areas.values()) * dz
        out[name] = {
            'volume_cm3': vol_mm3 / 1000.0,
            'bbox': (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)),
            'n_contours': len(contours),
        }
    return out


_BODY_RE = __import__('re').compile(r'^(BODY|EXTERNAL)([_0-9].*)?$')


def find_body(structures):
    """BODY/EXTERNAL con variantes de sufijo vistas en los datos (BODY1, BODY_P, ...).
    Si hay varias, preferir la que no arranca con 'z' (estructura de optimización)."""
    hits = [(n, i) for n, i in structures.items() if _BODY_RE.match(n.strip().upper())]
    if not hits:
        return None, None
    non_z = [(n, i) for n, i in hits if not n.strip().upper().startswith('Z')]
    return (non_z or hits)[0]


def resolve_level(structures, level):
    """level en {'high','mid','low'}. -> (nombre_elegido|None, cropped|None, motivo|None)"""
    candidates = []
    for name, info in structures.items():
        lvl, cropped = normalize_ptv_name(name)
        if lvl == level:
            candidates.append((name, cropped, info['volume_cm3']))
    for want_crop in (True, False):
        g = [(n, v) for n, c, v in candidates if c == want_crop]
        nonzero = [(n, v) for n, v in g if v > 0]
        if len(nonzero) == 1:
            return nonzero[0][0], want_crop, None
        if len(nonzero) > 1:
            return None, want_crop, 'ambiguo:' + ','.join(n for n, _ in nonzero)
    if candidates:
        return None, None, 'solo_volumen_cero:' + ','.join(n for n, _, _ in candidates)
    return None, None, 'ausente'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='carpeta_export_sitio:Sitio')
    ap.add_argument('--harmonization-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    site_yaml = {'CyC': 'sites_cyc.yaml', 'Pelvis': 'sites_pelvis.yaml', 'Prostata': 'sites_prostata.yaml'}

    rows_out = []
    outliers = []
    for item in a.inputs:
        folder, site = item.rsplit(':', 1)
        with open(os.path.join(a.harmonization_dir, site_yaml[site]), encoding='utf-8') as f:
            site_cfg = yaml.safe_load(f)
        levels = site_cfg['protocols'][0]['levels']

        patient_dirs = sorted(d for d in os.listdir(folder) if d.startswith('PT_'))
        print(f'=== {site}: {len(patient_dirs)} pacientes ===')
        for pid in patient_dirs:
            pdir = os.path.join(folder, pid)
            rd, rs = find_files(pdir)
            row = {'patient_id': pid, 'site': site}
            if rd is None or rs is None:
                row['error'] = f'falta {"RD" if rd is None else "RS"}'
                rows_out.append(row)
                outliers.append((site, pid, row['error']))
                continue
            try:
                row.update(read_dose_header(rd))
            except Exception as e:
                row['error'] = f'RD ilegible: {e}'
                rows_out.append(row)
                outliers.append((site, pid, row['error']))
                continue
            try:
                structures = read_structures(rs)
            except Exception as e:
                row['error'] = f'RS ilegible: {e}'
                rows_out.append(row)
                outliers.append((site, pid, row['error']))
                continue

            body_name, body_info = find_body(structures)
            if body_info and body_info['bbox']:
                xmin, xmax, ymin, ymax, zmin, zmax = body_info['bbox']
                row['body_extent_x_mm'] = round(xmax - xmin, 1)
                row['body_extent_y_mm'] = round(ymax - ymin, 1)
                row['body_extent_z_mm'] = round(zmax - zmin, 1)
            else:
                row['body_extent_x_mm'] = row['body_extent_y_mm'] = row['body_extent_z_mm'] = None
                outliers.append((site, pid, 'sin BODY/EXTERNAL'))

            for level_role in levels:
                level = level_role.split('_')[1].lower()   # 'PTV_High' -> 'high'
                name, cropped, motivo = resolve_level(structures, level)
                row[f'{level_role}_resuelto'] = name
                row[f'{level_role}_cropped'] = cropped
                row[f'{level_role}_volumen_cm3'] = round(structures[name]['volume_cm3'], 1) if name else None
                if motivo:
                    row[f'{level_role}_motivo'] = motivo
                    outliers.append((site, pid, f'{level_role}: {motivo}'))

            grid = (row['rd_nz'], row['rd_ny'], row['rd_nx'])
            if any(g > lim for g, lim in zip(grid, PAD_LIMIT)):
                outliers.append((site, pid, f'grilla {grid} excede padding {PAD_LIMIT}'))
            if not row['z_spacing_uniform']:
                outliers.append((site, pid, 'spacing Z no uniforme'))

            rows_out.append(row)

    fieldnames = sorted({k for r in rows_out for k in r})
    fieldnames = ['patient_id', 'site'] + [f for f in fieldnames if f not in ('patient_id', 'site')]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f'\nOK -> {a.out}  ({len(rows_out)} filas)')
    print(f'Outliers: {len(outliers)}')
    for site, pid, motivo in outliers:
        print(f'  [{site}] {pid}: {motivo}')


if __name__ == '__main__':
    main()
