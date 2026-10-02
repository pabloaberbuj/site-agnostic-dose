#!/usr/bin/env python3
"""
Construye/actualiza la tabla de de-identificacion HC -> patient_id por sitio, a partir
de los CSV de pacientes (HC, Apellido, Nombre, Curso, Plan, Estado, Fecha, Equipo, Fx,
Dosis/fx (cGy), Dosis total (cGy), Modalidad, Haz).

La tabla de mapeo (HC, Apellido, Nombre incluidos) se escribe SOLO fuera del repo
(--out-dir debe apuntar a la carpeta de datos hermana). El script rechaza escribir
si --out-dir cae dentro de un repo git, para no arrastrar datos identificables a git
por error.

patient_id es estable entre corridas: si ya existe un mapeo previo en --out-dir, las
asignaciones existentes no se tocan; los HC nuevos se agregan al final (orden HC asc).

Uso:
    python build_deidentification.py CyC.csv:CyC PelvisGin.csv:Pelvis \
        ProstataHipo.csv:Prostata --out-dir "C:\\Pablo\\Site Agnostic\\data\\deidentification"
"""
import csv
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

FIELDS_IN = ["HC", "Apellido", "Nombre", "Curso", "Plan", "Estado", "Fecha",
             "Equipo", "Fx", "Dosis/fx (cGy)", "Dosis total (cGy)", "Modalidad", "Haz"]


def is_inside_git_repo(path: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except FileNotFoundError:
        return False


def load_existing_map(map_path: Path):
    """HC -> patient_id, a partir de un mapeo previo (si existe)."""
    if not map_path.exists():
        return {}
    existing = {}
    with open(map_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            existing[r["HC"]] = r["patient_id"]
    return existing


def next_counter(existing_ids, site):
    prefix = f"{site}_"
    nums = [int(pid[len(prefix):]) for pid in existing_ids if pid.startswith(prefix)]
    return (max(nums) + 1) if nums else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="archivo.csv:Sitio")
    ap.add_argument("--out-dir", required=True,
                     help="Carpeta FUERA del repo donde vive la tabla de mapeo (no va a git)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_inside_git_repo(out_dir):
        raise SystemExit(
            f"ABORTADO: --out-dir ({out_dir}) cae dentro de un repo git. "
            "La tabla de mapeo (HC/Apellido/Nombre) no puede vivir ahi."
        )

    report = []
    for item in a.inputs:
        csv_path, site = item.rsplit(":", 1)
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        map_path = out_dir / f"map_{site}.csv"
        existing = load_existing_map(map_path)
        counter = next_counter(set(existing.values()), site)

        by_hc = defaultdict(list)
        for r in rows:
            by_hc[r["HC"]].append(r)

        for hc in sorted(by_hc):
            if hc not in existing:
                existing[hc] = f"{site}_{counter:03d}"
                counter += 1

        out_rows = []
        for hc in sorted(by_hc):
            pid = existing[hc]
            for r in by_hc[hc]:
                row = {k: r.get(k, "") for k in FIELDS_IN}
                row["patient_id"] = pid
                row["site"] = site
                out_rows.append(row)

        with open(map_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["patient_id", "site"] + FIELDS_IN)
            w.writeheader()
            w.writerows(out_rows)

        multi_course = sorted(hc for hc, recs in by_hc.items() if len(recs) > 1)
        report.append({
            "site": site,
            "n_patients": len(by_hc),
            "n_rows": len(rows),
            "n_patients_multi_course": len(multi_course),
            "map_path": str(map_path),
        })

    print(f"{'sitio':<12}{'pacientes':>10}{'filas':>8}{'multi-curso':>13}")
    for r in report:
        print(f"{r['site']:<12}{r['n_patients']:>10}{r['n_rows']:>8}{r['n_patients_multi_course']:>13}")
    print()
    for r in report:
        print(f"-> {r['map_path']}")


if __name__ == "__main__":
    main()
