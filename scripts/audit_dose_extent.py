#!/usr/bin/env python3
"""
Paso 4 (0.2) — Auditoria de contaminacion SOBRE LA DOSIS, no sobre nombres de
estructura (PROMPT_code_fase0.md). Indicador: extension craneo-caudal (Z) del volumen
que recibe >= X% de Rx_High, medida directamente sobre la dosis cruda (denormalizada
desde dose_pct con d95_ptv_high_raw_gy). Detecta pacientes con dosis de naturaleza
distinta al sitio (banos nodales, boosts no contemplados, niveles extra) que ningun
chequeo por nombre de estructura atrapa.

Calibracion: los 2 casos conocidos de Pelvis con campo extendido
(PT_49f9ceef1c10627d, PT_258346f1e40fbf8f, ya out_of_convention) deben separarse
claramente del resto de Pelvis en varios umbrales X para confiar en el indicador.

Lee solo NPZ/JSON de Paso 2 (no vuelve a tocar DICOM).

Uso:
    python audit_dose_extent.py --npz-dir "C:\\Pablo\\Site Agnostic\\data\\npz_v1_paso2" \
        --harmonization-dir ../data/harmonization \
        --out ../results/auditorias/paso4_contaminacion_dosis.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess.dose_metrics import denormalize_dose

SITES = ('CyC', 'Pelvis', 'Prostata')
THRESHOLDS_PCT = (10, 25, 50)

KNOWN_CONTAMINATED = {'PT_49f9ceef1c10627d', 'PT_258346f1e40fbf8f'}


def load_exceptions(path):
    d = yaml.safe_load(open(path, encoding='utf-8')) or {}
    by_id = {}
    for pid, v in (d.get('excluded') or {}).items():
        by_id[pid] = ('excluded', v['motivo'])
    for pid, v in (d.get('out_of_convention') or {}).items():
        by_id[pid] = ('out_of_convention', v['motivo'])
    for item in (d.get('reviewed_ok') or []):
        by_id[item['id']] = ('reviewed_ok', item['motivo'])
    return by_id


def z_extent_mm(pct_rx, spacing_z_mm, threshold_pct):
    """Extension craneo-caudal (eje Z, index 0) de los slices con algun voxel >= threshold."""
    slice_has_dose = (pct_rx >= threshold_pct).any(axis=(1, 2))
    idx = np.nonzero(slice_has_dose)[0]
    if idx.size == 0:
        return 0.0
    return float(idx.max() - idx.min() + 1) * spacing_z_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--harmonization-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    exceptions = load_exceptions(os.path.join(a.harmonization_dir, 'patient_exceptions.yaml'))

    rows = []
    for site in SITES:
        site_dir = os.path.join(a.npz_dir, site)
        if not os.path.isdir(site_dir):
            continue
        for fn in sorted(f for f in os.listdir(site_dir) if f.endswith('.json')):
            pid = fn[:-len('.json')]
            meta = json.load(open(os.path.join(site_dir, fn), encoding='utf-8'))
            npz = np.load(os.path.join(site_dir, f'{pid}.npz'))
            raw_dose_gy = denormalize_dose(npz['dose_pct'], meta['d95_ptv_high_raw_gy'])
            pct_rx = raw_dose_gy / meta['rx_high_gy'] * 100.0
            exc = exceptions.get(pid)
            row = {'site': site, 'patient_id': pid,
                   'exception_category': exc[0] if exc else '',
                   'exception_motivo': exc[1] if exc else '',
                   'conocido_contaminado': 'si' if pid in KNOWN_CONTAMINATED else ''}
            for th in THRESHOLDS_PCT:
                row[f'z_extent_mm_ge{th}pct'] = round(
                    z_extent_mm(pct_rx, meta['spacing_z_mm'], th), 1)
            rows.append(row)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fieldnames = ['site', 'patient_id'] + [f'z_extent_mm_ge{th}pct' for th in THRESHOLDS_PCT] + \
                 ['conocido_contaminado', 'exception_category', 'exception_motivo']
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- reporte (numeros, no prosa) ----
    print(f'{a.out}\n')
    for site in SITES:
        site_rows = [r for r in rows if r['site'] == site]
        if not site_rows:
            continue
        print(f'=== {site} (N={len(site_rows)}) ===')
        for th in THRESHOLDS_PCT:
            key = f'z_extent_mm_ge{th}pct'
            vals = [r[key] for r in site_rows]
            p50, p90, p95, p99, pmax = np.percentile(vals, [50, 90, 95, 99, 100])
            known = [r for r in site_rows if r['conocido_contaminado'] == 'si']
            known_str = ', '.join(f'{r["patient_id"]}={r[key]}mm' for r in known)
            print(f'  >={th:>2}% Rx_High: p50={p50:>6.1f}  p90={p90:>6.1f}  p95={p95:>6.1f}  '
                  f'p99={p99:>6.1f}  max={pmax:>6.1f}' +
                  (f'   | conocidos: {known_str}' if known_str else ''))
        print()


if __name__ == '__main__':
    main()
