#!/usr/bin/env python3
"""
Inventario de nombres de estructura RAW por paciente, los 3 sitios — una sola vez.

Motivo: evitar rescanear DICOM cada vez que surge una decision de armonizacion nueva
("que pacientes tienen la estructura X?"). Esto es solo lectura de RTSTRUCT (barato,
sin dosis); el volumen rasterizado (mas caro: requiere grilla de dosis) se sigue
calculando aparte, bajo demanda, solo cuando una decision puntual depende de el.

Salida: results/auditorias/paso2_estructuras_por_paciente.csv (site,patient_id,raw_name)
formato largo (una fila por estructura) para poder grepear o filtrar con pandas/csv.

Uso:
    python inventory_structures.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess.dicom_io import find_by_modality, load_structures

EXPORT = r"C:\Pablo\Site Agnostic\data\dicom_raw\00_Corrida1_CyC_Pelvis_Ptta\dcm export"
SITES = {
    'CyC': os.path.join(EXPORT, '20261008_1627_CyC'),
    'Pelvis': os.path.join(EXPORT, '20261008_1812_PelvisGin'),
    'Prostata': os.path.join(EXPORT, '20261008_2055_ProstataHipo'),
}
OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'auditorias',
                    'paso2_estructuras_por_paciente.csv')


def main():
    rows = []
    errors = []
    for site, folder in SITES.items():
        pids = sorted(d for d in os.listdir(folder) if d.startswith('PT_'))
        for pid in pids:
            pdir = os.path.join(folder, pid)
            try:
                by_mod = find_by_modality(pdir)
                _, raw = load_structures(pdir, by_mod)
            except Exception as e:
                errors.append((site, pid, str(e)))
                continue
            for name in raw:
                rows.append((site, pid, name))
        print(f'{site}: {len(pids)} pacientes procesados')

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['site', 'patient_id', 'raw_name'])
        w.writerows(rows)

    print(f'\n{len(rows)} filas -> {OUT}')
    if errors:
        print(f'{len(errors)} pacientes con error de carga (no incluidos):')
        for site, pid, e in errors:
            print(f'  [{site}] {pid}: {e}')


if __name__ == '__main__':
    main()
