#!/usr/bin/env python3
"""
Paso 5 — splits train/val/test, estratificados por sitio, 65/15/20 aprox (CHARTER §2).

Elegibles: pacientes con NPZ generado (Paso 2) que NO son `out_of_convention`
(conjunto reservado, nunca entra a ningun split) ni `excluded` (no tienen NPZ de
entrada, ni hace falta filtrarlos aparte). `reviewed_ok` SI es elegible (ya revisado,
entra normal).

Guarda SOLO patient_id (CHARTER: "con patient_id solamente"), un JSON por split con
estructura {"train": [...], "val": [...], "test": [...]} en repo/data/splits/
(va a git, es liviano).

Uso:
    python make_splits.py --npz-dir "C:\\Pablo\\Site Agnostic\\data\\npz_v1_paso2" \
        --harmonization-dir ../data/harmonization \
        --out ../data/splits/split_v1.json \
        --seed 42
"""
import argparse
import json
import os
import random

import yaml

SITES = ('CyC', 'Pelvis', 'Prostata')
TRAIN_FRAC, VAL_FRAC = 0.65, 0.15  # test = resto (~0.20)


def load_out_of_convention(path):
    d = yaml.safe_load(open(path, encoding='utf-8')) or {}
    return set((d.get('out_of_convention') or {}).keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--harmonization-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    ooc = load_out_of_convention(os.path.join(a.harmonization_dir, 'patient_exceptions.yaml'))
    rng = random.Random(a.seed)

    splits = {'train': [], 'val': [], 'test': []}
    summary = {}
    for site in SITES:
        site_dir = os.path.join(a.npz_dir, site)
        if not os.path.isdir(site_dir):
            continue
        all_pids = sorted(f[:-len('.json')] for f in os.listdir(site_dir) if f.endswith('.json'))
        eligible = sorted(pid for pid in all_pids if pid not in ooc)
        rng.shuffle(eligible)

        n = len(eligible)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        train = eligible[:n_train]
        val = eligible[n_train:n_train + n_val]
        test = eligible[n_train + n_val:]

        splits['train'] += train
        splits['val'] += val
        splits['test'] += test
        summary[site] = {'total_npz': len(all_pids), 'out_of_convention': len(all_pids) - n,
                          'elegibles': n, 'train': len(train), 'val': len(val), 'test': len(test)}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump({'seed': a.seed, 'train_frac': TRAIN_FRAC, 'val_frac': VAL_FRAC,
                   **splits}, f, indent=2)

    # ---- reporte (numeros, no prosa) ----
    print(f'seed={a.seed}  ({a.out})\n')
    print(f'{"sitio":10} {"npz":>5} {"ooc":>5} {"elegibles":>10} {"train":>7} {"val":>6} {"test":>6}')
    tot = {'total_npz': 0, 'out_of_convention': 0, 'elegibles': 0, 'train': 0, 'val': 0, 'test': 0}
    for site, s in summary.items():
        print(f'{site:10} {s["total_npz"]:>5} {s["out_of_convention"]:>5} {s["elegibles"]:>10} '
              f'{s["train"]:>7} {s["val"]:>6} {s["test"]:>6}')
        for k in tot:
            tot[k] += s[k]
    print(f'{"total":10} {tot["total_npz"]:>5} {tot["out_of_convention"]:>5} {tot["elegibles"]:>10} '
          f'{tot["train"]:>7} {tot["val"]:>6} {tot["test"]:>6}')


if __name__ == '__main__':
    main()
