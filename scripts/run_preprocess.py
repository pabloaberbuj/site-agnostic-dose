#!/usr/bin/env python3
"""
Paso 2 — corre el preprocesador sobre todos los pacientes de un sitio y escribe
NPZ versionado (grilla nativa, sin recorte/relleno — eso lo hace el datamodule) +
metadata por paciente.

Nunca pisa una version existente de NPZ (--out-dir debe no existir o estar vacio).
--out-dir debe caer FUERA del repo git (datos pesados, CHARTER §6).

Uso:
    python run_preprocess.py "<export>/20261008_1627_CyC:CyC:sites_cyc.yaml" \
        "<export>/20261008_1812_PelvisGin:Pelvis:sites_pelvis.yaml" \
        "<export>/20261008_2055_ProstataHipo:Prostata:sites_prostata.yaml" \
        --harmonization-dir ../data/harmonization \
        --out-dir "C:\\Pablo\\Site Agnostic\\data\\npz_v1_paso2"
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess.name_resolver import NameResolver
from preprocess.pipeline import process_patient, PipelineError


def is_inside_git_repo(path):
    try:
        r = subprocess.run(['git', '-C', str(path), 'rev-parse', '--is-inside-work-tree'],
                            capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and r.stdout.strip() == 'true'
    except FileNotFoundError:
        return False


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


def harmonization_hash(harmonization_dir):
    h = hashlib.sha256()
    for fn in sorted(os.listdir(harmonization_dir)):
        if fn.endswith('.yaml'):
            with open(os.path.join(harmonization_dir, fn), 'rb') as f:
                h.update(fn.encode() + f.read())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='carpeta_export_sitio:Sitio:archivo_site.yaml')
    ap.add_argument('--harmonization-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--limit', type=int, default=None, help='solo los primeros N por sitio (debug)')
    a = ap.parse_args()

    if is_inside_git_repo(a.out_dir):
        raise SystemExit(f'ABORTADO: --out-dir ({a.out_dir}) cae dentro de un repo git. '
                          'El NPZ es pesado, no puede vivir ahi (CHARTER §6).')

    schema = yaml.safe_load(open(os.path.join(a.harmonization_dir, 'harmonization_schema.yaml'), encoding='utf-8'))
    resolver = NameResolver(schema)
    exceptions = load_exceptions(os.path.join(a.harmonization_dir, 'patient_exceptions.yaml'))
    hhash = harmonization_hash(a.harmonization_dir)

    os.makedirs(a.out_dir, exist_ok=True)

    summary = {}
    for item in a.inputs:
        folder, site, site_yaml_fn = item.rsplit(':', 2)
        site_cfg = yaml.safe_load(open(os.path.join(a.harmonization_dir, site_yaml_fn), encoding='utf-8'))
        site_out = os.path.join(a.out_dir, site)
        os.makedirs(site_out, exist_ok=True)

        patient_dirs = sorted(d for d in os.listdir(folder) if d.startswith('PT_'))
        if a.limit:
            patient_dirs = patient_dirs[:a.limit]

        ok, failed, excluded_skipped = [], [], []
        for pid in patient_dirs:
            exc = exceptions.get(pid)
            if exc and exc[0] == 'excluded':
                excluded_skipped.append((pid, exc[1]))
                continue
            try:
                res = process_patient(os.path.join(folder, pid), pid, site_cfg, resolver,
                                       exception_entry=exc if exc and exc[0] == 'out_of_convention' else None)
            except PipelineError as e:
                failed.append((pid, str(e)))
                continue
            except Exception:
                failed.append((pid, traceback.format_exc(limit=3)))
                continue

            res.metadata['site'] = site
            res.metadata['harmonization_hash'] = hhash
            npz_path = os.path.join(site_out, f'{pid}.npz')
            arrays = {'dose_pct': res.dose_pct.astype(np.float32)}
            for role, mask in res.masks.items():
                arrays[f'mask_{role}'] = mask
            np.savez_compressed(npz_path, **arrays)
            with open(os.path.join(site_out, f'{pid}.json'), 'w', encoding='utf-8') as f:
                json.dump(res.metadata, f, ensure_ascii=False, indent=2)
            ok.append(pid)

        summary[site] = {'ok': len(ok), 'failed': failed, 'excluded_skipped': len(excluded_skipped),
                          'n_total': len(patient_dirs)}
        print(f'{site}: OK={len(ok)}  FAILED={len(failed)}  excluded_skipped={len(excluded_skipped)}  '
              f'de {len(patient_dirs)}')
        for pid, reason in failed:
            print(f'  FAILED {pid}: {reason.splitlines()[-1] if reason else reason}')

    report_path = os.path.join(a.out_dir, '_preprocess_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\nharmonization_hash={hhash}')
    print(f'reporte -> {report_path}')


if __name__ == '__main__':
    main()
