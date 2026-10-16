#!/usr/bin/env python3
"""
Paso 4 (0.2) — repetir la auditoria de escala de voxel de Paso 1, pero sobre los NPZ
ya generados (PROMPT_code_fase0.md). Paso 1 aproximo el volumen de PTV por contorno
(area de cada corte x paso entre cortes, sin rasterizar); esta ronda usa el volumen
REAL rasterizado sobre la grilla de dosis nativa (el que efectivamente ve el modelo) y
lo compara contra el de Paso 1 para los mismos pacientes. Tambien repite la
distribucion de spacing y dimension de grilla, ahora leida de la metadata de Paso 2
(debe ser identica a la de Paso 1 por construccion; sirve de sanity check).

Lee solo NPZ/JSON (no vuelve a tocar DICOM). Referencia: paso1_headers.csv.

Uso:
    python audit_npz_voxel_scale.py --npz-dir "C:\\Pablo\\Site Agnostic\\data\\npz_v1_paso2" \
        --paso1-csv ../results/auditorias/paso1_headers.csv \
        --out ../results/auditorias/paso4_escala_voxel_npz.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

SITES = ('CyC', 'Pelvis', 'Prostata')
LEVELS = ('PTV_High', 'PTV_Mid', 'PTV_Low')


def load_paso1(path):
    by_pid = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            by_pid[row['patient_id']] = row
    return by_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--paso1-csv', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    paso1 = load_paso1(a.paso1_csv)

    rows = []
    grid_dims = {s: [] for s in SITES}
    spacing_z = {s: [] for s in SITES}
    for site in SITES:
        site_dir = os.path.join(a.npz_dir, site)
        if not os.path.isdir(site_dir):
            continue
        for fn in sorted(f for f in os.listdir(site_dir) if f.endswith('.json')):
            pid = fn[:-len('.json')]
            meta = json.load(open(os.path.join(site_dir, fn), encoding='utf-8'))
            npz = np.load(os.path.join(site_dir, f'{pid}.npz'))

            nz, ny, nx = npz['dose_pct'].shape
            grid_dims[site].append((nz, ny, nx))
            spacing_z[site].append(meta['spacing_z_mm'])

            voxel_cm3 = (meta['spacing_x_mm'] * meta['spacing_y_mm'] * meta['spacing_z_mm']) / 1000.0
            p1 = paso1.get(pid)
            for level in LEVELS:
                mask_key = f'mask_{level}'
                if mask_key not in npz:
                    continue
                vol_npz = float(npz[mask_key].sum()) * voxel_cm3
                vol_p1_str = p1.get(f'{level}_volumen_cm3', '') if p1 else ''
                vol_p1 = float(vol_p1_str) if vol_p1_str not in ('', None) else None
                rel_diff = ((vol_npz - vol_p1) / vol_p1 * 100.0) if vol_p1 and vol_p1 > 0 else ''
                rows.append({'site': site, 'patient_id': pid, 'level': level,
                             'vol_npz_cm3': round(vol_npz, 2),
                             'vol_paso1_contorno_cm3': vol_p1_str,
                             'rel_diff_pct': round(rel_diff, 2) if rel_diff != '' else ''})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['site', 'patient_id', 'level', 'vol_npz_cm3',
                                           'vol_paso1_contorno_cm3', 'rel_diff_pct'])
        w.writeheader()
        w.writerows(rows)

    # ---- reporte (numeros, no prosa) ----
    print(f'{a.out}\n')
    print('=== spacing_z_mm y dimension de grilla (sanity check vs Paso 1) ===')
    for site in SITES:
        zs = spacing_z[site]
        if not zs:
            continue
        n_30 = sum(1 for z in zs if abs(z - 3.0) < 1e-3)
        n_20 = sum(1 for z in zs if abs(z - 2.0) < 1e-3)
        n_other = len(zs) - n_30 - n_20
        dims = np.array(grid_dims[site])
        print(f'  {site}: N={len(zs)}  z=3.0mm:{n_30}  z=2.0mm:{n_20}  otro:{n_other}  | '
              f'grilla (Z,Y,X) p50={tuple(np.percentile(dims, 50, axis=0).astype(int))} '
              f'max={tuple(dims.max(axis=0))}')

    print('\n=== diferencia relativa: volumen rasterizado (NPZ) vs contorno (Paso 1) ===')
    for site in SITES:
        for level in LEVELS:
            vals = [r['rel_diff_pct'] for r in rows
                    if r['site'] == site and r['level'] == level and r['rel_diff_pct'] != '']
            if not vals:
                continue
            p5, p50, p95 = np.percentile(vals, [5, 50, 95])
            outliers = [r for r in rows if r['site'] == site and r['level'] == level
                        and r['rel_diff_pct'] != '' and abs(r['rel_diff_pct']) > 15]
            print(f'  {site:10} {level:10} N={len(vals):>4}  p5={p5:>6.1f}%  p50={p50:>6.1f}%  '
                  f'p95={p95:>6.1f}%  |outliers|>15%: {len(outliers)}')
            for r in outliers:
                print(f'    {r["patient_id"]}: NPZ={r["vol_npz_cm3"]}cm3 '
                      f'Paso1={r["vol_paso1_contorno_cm3"]}cm3 ({r["rel_diff_pct"]}%)')


if __name__ == '__main__':
    main()
