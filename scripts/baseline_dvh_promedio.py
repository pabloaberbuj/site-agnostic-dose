#!/usr/bin/env python3
"""
Paso 5 — baseline no-aprendida: DVH promedio poblacional por (sitio, rol de
estructura) (CHARTER §2). Control real para el proyecto: ¿la red aprende algo mas que
"la forma tipica del protocolo"? Funciona con cualquier cantidad de OARs (los roles se
descubren de los NPZ, una curva por rol, NO un modelo por paciente).

Calculada SOLO con pacientes de TRAIN del split (nunca con val/test: usar el promedio
poblacional de train como prediccion sobre val/test y compararlo contra la red
entrenada es justamente el punto del control).

Dosis en eje %Rx_High (CHARTER: "la banda % Rx de la metrica DVH se define como % de
Rx_High") — dose_pct del NPZ ya esta normalizado a D95(PTV_High)=100%, que coincide
con %Rx_High dentro de la banda QA (92-105%); se usa tal cual, es el eje nativo del NPZ
y el que ve el modelo.

DVH(d) = fraccion del volumen del rol con dosis >= d, promediada sin ponderar por
paciente (cada paciente de train pesa igual, estandar en poblacion DVH).

Uso:
    python baseline_dvh_promedio.py --npz-dir "C:\\Pablo\\Site Agnostic\\data\\npz_v1_paso2" \
        --split ../data/splits/split_v1.json \
        --out-dir ../results/baseline
"""
import argparse
import json
import os

import numpy as np

SITES = ('CyC', 'Pelvis', 'Prostata')
DOSE_GRID = np.arange(0, 115.5, 1.0)  # % Rx_High


def dvh_fractions(dose_pct, mask, dose_grid):
    """-> array (len(dose_grid),): fraccion del volumen de `mask` con dosis >= cada d."""
    vals = np.sort(dose_pct[mask])
    n = vals.size
    idx = np.searchsorted(vals, dose_grid, side='left')
    return (n - idx) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    split = json.load(open(a.split, encoding='utf-8'))
    train_ids = set(split['train'])

    os.makedirs(a.out_dir, exist_ok=True)
    report = {}
    for site in SITES:
        site_dir = os.path.join(a.npz_dir, site)
        if not os.path.isdir(site_dir):
            continue
        pids = sorted(f[:-len('.npz')] for f in os.listdir(site_dir) if f.endswith('.npz')
                       and f[:-len('.npz')] in train_ids)

        # curvas por rol: acumular (suma de fracciones, n) para promediar al final
        acc = {}  # role -> (sum_fracciones (array), n_pacientes)
        for pid in pids:
            npz = np.load(os.path.join(site_dir, f'{pid}.npz'))
            dose_pct = npz['dose_pct']
            for key in npz.files:
                if not key.startswith('mask_'):
                    continue
                role = key[len('mask_'):]
                mask = npz[key]
                if not mask.any():
                    continue
                frac = dvh_fractions(dose_pct, mask, DOSE_GRID)
                if role not in acc:
                    acc[role] = (np.zeros_like(DOSE_GRID, dtype=np.float64), 0)
                s, n = acc[role]
                acc[role] = (s + frac, n + 1)

        site_out = {'n_train': len(pids), 'dose_grid_pct_rx_high': DOSE_GRID.tolist(), 'roles': {}}
        for role, (s, n) in sorted(acc.items()):
            site_out['roles'][role] = {'n_patients': n, 'volume_fraction_mean': (s / n).tolist()}

        out_path = os.path.join(a.out_dir, f'baseline_dvh_{site}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(site_out, f, indent=2)
        report[site] = site_out

    # ---- reporte (numeros, no prosa) ----
    print(f'dose_grid: 0-115% Rx_High, paso 1%  ({a.out_dir})\n')
    for site, site_out in report.items():
        print(f'=== {site} (N_train={site_out["n_train"]}, {len(site_out["roles"])} roles) ===')
        for role, r in sorted(site_out['roles'].items()):
            vf = np.array(r['volume_fraction_mean'])
            # V(d) a dosis de referencia (50% y 80% Rx_High), interpolado
            v50 = float(np.interp(50.0, DOSE_GRID, vf))
            v80 = float(np.interp(80.0, DOSE_GRID, vf))
            cov = r['n_patients'] / site_out['n_train'] * 100
            print(f'  {role:28} n={r["n_patients"]:>4} ({cov:>5.1f}%)  V50%={v50*100:>5.1f}%  '
                  f'V80%={v80*100:>5.1f}%')
        print()


if __name__ == '__main__':
    main()
