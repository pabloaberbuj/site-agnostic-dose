#!/usr/bin/env python3
"""
Paso 4 (0.2) — Conteo de versiones de plan por paciente (ARIA), via
metadata_planes_<Sitio>.csv (PROMPT_code_fase0.md). El conteo de no-aprobados ya viene
filtrado por tratamiento (mismo curso, misma Rx, mismo structure set, mismo
TargetVolumeID). Vacio = no calculado (NO es 0). Para pacientes con TargetVolumeID mal
asignado (patient_exceptions.yaml, motivo *cota_inferior_versiones) el conteo es una
cota inferior, no el valor real.

Uso:
    python audit_plan_versions.py --metadata-csv \
        "<export CyC>/metadata_planes.csv:CyC" \
        "<export Pelvis>/metadata_planes.csv:Pelvis" \
        "<export Prostata>/metadata_planes.csv:Prostata" \
        --harmonization-dir ../data/harmonization \
        --out ../results/auditorias/paso4_versiones_plan.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import yaml

COLS = ('NPlanesCurso', 'NAprobados', 'NNoAprobados', 'NIntermedios', 'NPlanSums')


def load_cota_inferior(path):
    d = yaml.safe_load(open(path, encoding='utf-8')) or {}
    return {item['id'] for item in (d.get('reviewed_ok') or [])
            if 'cota_inferior' in item['motivo']}


def parse_int_or_none(s):
    s = (s or '').strip()
    return int(s) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata-csv', nargs='+', required=True, help='archivo.csv:Sitio')
    ap.add_argument('--harmonization-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    cota_inferior = load_cota_inferior(os.path.join(a.harmonization_dir, 'patient_exceptions.yaml'))

    rows = []
    for item in a.metadata_csv:
        path, site = item.rsplit(':', 1)
        with open(path, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f, delimiter=';'):
                row = {'site': site, 'patient_id': r['AnonID'],
                       'cota_inferior': 'si' if r['AnonID'] in cota_inferior else ''}
                for c in COLS:
                    row[c] = parse_int_or_none(r.get(c))
                rows.append(row)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['site', 'patient_id'] + list(COLS) + ['cota_inferior'])
        w.writeheader()
        w.writerows(rows)

    # ---- reporte (numeros, no prosa) ----
    print(f'{a.out}\n')
    sites = sorted(set(r['site'] for r in rows))
    for site in sites:
        site_rows = [r for r in rows if r['site'] == site]
        n = len(site_rows)
        no_calc = sum(1 for r in site_rows if r['NNoAprobados'] is None)
        vals = [r['NNoAprobados'] for r in site_rows if r['NNoAprobados'] is not None]
        print(f'=== {site} (N={n}, sin calcular={no_calc}) ===')
        if vals:
            dist = {}
            for v in vals:
                dist[v] = dist.get(v, 0) + 1
            for k in sorted(dist):
                pct = dist[k] / len(vals) * 100
                print(f'  NNoAprobados={k}: {dist[k]:>4} ({pct:>5.1f}%)')
            p50, p90, p99, pmax = np.percentile(vals, [50, 90, 99, 100])
            print(f'  p50={p50:.0f}  p90={p90:.0f}  p99={p99:.0f}  max={pmax:.0f}')
        cota = [r for r in site_rows if r['cota_inferior'] == 'si']
        if cota:
            print(f'  cota inferior (TargetVolumeID mal asignado): '
                  f'{", ".join(r["patient_id"] + "=" + str(r["NNoAprobados"]) for r in cota)}')
        print()


if __name__ == '__main__':
    main()
