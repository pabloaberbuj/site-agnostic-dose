#!/usr/bin/env python3
"""
Paso 3 — QA gate offline sobre dosis absoluta cruda (CHARTER / PROMPT_code_fase0.md).

Nunca toca el modelo: lee los NPZ/JSON ya generados por Paso 2 (no vuelve a tocar
DICOM). Para cada paciente y cada nivel de PTV REALMENTE presente, recalcula D95 en Gy
crudo (denormalizando dose_pct con d95_ptv_high_raw_gy) y lo compara contra el
`rx_gy` del protocolo. Banda [floor_pct, ceil_pct] por sitio, con fallback a 'default'
(ver harmonization_schema.yaml:qa_gate). Fuera de banda -> flag, NUNCA descarta.

No vuelve a calcular D95(PTV_High): ya esta en la metadata exactamente como aca lo
recalcularia (sirve igual de sanity-check, que D95 recalculado matchee el guardado).

Uso:
    python qa_gate.py --npz-dir "C:\\Pablo\\Site Agnostic\\data\\npz_v1_paso2" \
        --harmonization-dir ../data/harmonization \
        --out results/auditorias/paso3_qa_gate.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess.dose_metrics import d95_gy, denormalize_dose, DoseMetricsError

SITES = ('CyC', 'Pelvis', 'Prostata')


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--harmonization-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    schema = yaml.safe_load(open(os.path.join(a.harmonization_dir, 'harmonization_schema.yaml'),
                                  encoding='utf-8'))
    bands = schema['qa_gate']['bands']
    exceptions = load_exceptions(os.path.join(a.harmonization_dir, 'patient_exceptions.yaml'))

    def band_for(site):
        b = bands.get(site, bands['default'])
        return b['floor_pct'], b['ceil_pct']

    rows = []
    for site in SITES:
        site_dir = os.path.join(a.npz_dir, site)
        if not os.path.isdir(site_dir):
            continue
        json_files = sorted(f for f in os.listdir(site_dir) if f.endswith('.json'))
        for fn in json_files:
            pid = fn[:-len('.json')]
            meta = json.load(open(os.path.join(site_dir, fn), encoding='utf-8'))
            npz = np.load(os.path.join(site_dir, f'{pid}.npz'))
            d95_high_gy = meta['d95_ptv_high_raw_gy']
            raw_dose_gy = denormalize_dose(npz['dose_pct'], d95_high_gy)
            exc = exceptions.get(pid)
            floor_pct, ceil_pct = band_for(site)

            for level in meta['ptv_levels_present']:
                mask_key = f'mask_{level}'
                if mask_key not in npz:
                    continue
                try:
                    d95 = d95_gy(raw_dose_gy, npz[mask_key])
                except DoseMetricsError as e:
                    rows.append({'site': site, 'patient_id': pid, 'level': level,
                                 'd95_gy': '', 'rx_gy': meta['rx_by_level_gy'].get(level, ''),
                                 'pct_rx': '', 'flag': 'ERROR', 'detalle': str(e),
                                 'exception_category': exc[0] if exc else '',
                                 'exception_motivo': exc[1] if exc else ''})
                    continue
                # CHARTER / "Cuando frenar": nivel de PTV resuelto que el protocolo del sitio
                # NO declara (ej. PTV_Low en Pelvis/Prostata, protocolos de 1 solo nivel) —
                # "mas niveles de PTV que el protocolo" es una de las condiciones explicitas
                # de parar y preguntar. Se reporta el D95 igual (dato real), pero SIN inventar
                # un Rx ni un %: eso lo decide el usuario.
                if level not in meta['rx_by_level_gy']:
                    rows.append({'site': site, 'patient_id': pid, 'level': level,
                                 'd95_gy': round(d95, 4), 'rx_gy': '', 'pct_rx': '',
                                 'flag': 'NIVEL_NO_DECLARADO_EN_PROTOCOLO', 'detalle': '',
                                 'exception_category': exc[0] if exc else '',
                                 'exception_motivo': exc[1] if exc else ''})
                    continue
                rx = meta['rx_by_level_gy'][level]
                pct_rx = d95 / rx * 100.0
                flag = not (floor_pct <= pct_rx <= ceil_pct)
                rows.append({'site': site, 'patient_id': pid, 'level': level,
                             'd95_gy': round(d95, 4), 'rx_gy': rx, 'pct_rx': round(pct_rx, 2),
                             'flag': 'FUERA_DE_BANDA' if flag else 'ok', 'detalle': '',
                             'exception_category': exc[0] if exc else '',
                             'exception_motivo': exc[1] if exc else ''})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['site', 'patient_id', 'level', 'd95_gy', 'rx_gy',
                                           'pct_rx', 'flag', 'detalle', 'exception_category',
                                           'exception_motivo'])
        w.writeheader()
        w.writerows(rows)

    # ---- reporte (numeros, no prosa) ----
    bands_str = ', '.join(f'{s}=[{b["floor_pct"]}%,{b["ceil_pct"]}%]' for s, b in bands.items())
    print(f'bandas: {bands_str}  ({a.out})\n')
    print(f'{"sitio":10} {"nivel":10} {"N":>5} {"flags":>6} {"flags_sin_revisar":>18} '
          f'{"p5":>8} {"p50":>8} {"p95":>8}')
    for site in SITES:
        site_rows = [r for r in rows if r['site'] == site]
        levels = sorted(set(r['level'] for r in site_rows))
        for level in levels:
            lvl_rows = [r for r in site_rows if r['level'] == level]
            n = len(lvl_rows)
            flagged = [r for r in lvl_rows if r['flag'] == 'FUERA_DE_BANDA']
            flagged_unreviewed = [r for r in flagged if not r['exception_category']]
            pcts = [r['pct_rx'] for r in lvl_rows if isinstance(r['pct_rx'], (int, float))]
            p5, p50, p95 = (np.percentile(pcts, [5, 50, 95]) if pcts else (float('nan'),) * 3)
            print(f'{site:10} {level:10} {n:>5} {len(flagged):>6} {len(flagged_unreviewed):>18} '
                  f'{p5:>8.1f} {p50:>8.1f} {p95:>8.1f}')
    errors = [r for r in rows if r['flag'] == 'ERROR']
    if errors:
        print(f'\n{len(errors)} ERROR (no D95 calculable):')
        for r in errors:
            print(f'  [{r["site"]}] {r["patient_id"]} {r["level"]}: {r["detalle"]}')

    no_declarado = [r for r in rows if r['flag'] == 'NIVEL_NO_DECLARADO_EN_PROTOCOLO']
    if no_declarado:
        print(f'\n{len(no_declarado)} NIVEL_NO_DECLARADO_EN_PROTOCOLO (mas niveles de PTV que '
              f'el protocolo del sitio -- CHARTER "cuando frenar", no resuelto automaticamente):')
        by_site_level = {}
        for r in no_declarado:
            by_site_level.setdefault((r['site'], r['level']), []).append(r)
        for (site, level), rs in sorted(by_site_level.items()):
            sin_revisar = [r for r in rs if not r['exception_category']]
            print(f'  [{site}] {level}: {len(rs)} pacientes ({len(sin_revisar)} sin excepcion '
                  f'registrada)')
            for r in sin_revisar:
                print(f'    {r["patient_id"]}: D95={r["d95_gy"]} Gy')


if __name__ == '__main__':
    main()
