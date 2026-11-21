"""
Vigilante de la época 40 (pre-registrado): exp_f4_001_dvhmoment_sp_lambda0p3 (loss cuadrática). Espera a que exista val/mae de la época 40.
  val/mae(ep40) <= GATE  -> la cadena cuadrática sigue (0.3 -> 0.1 -> 1.0). No hace nada más. (L1 NO se lanza solo.)
  val/mae(ep40) >  GATE  -> cierra la cadena cuadrática (tarea sa_f4 + procesos) y lanza el sweep L1 (scripts/run_f4_l1.sh, tarea sa_f4_l1).
GATE = 6.2 = 1.4 x val/mae de la vara exp_a4_003 en la época 40 (4.44). Fijado antes de mirar. Registra en logs/f4_gate.log y results/<exp>/gate_epoca40.txt.
"""
import glob
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = 'exp_f4_001_dvhmoment_sp_lambda0p3'
EPOCH, GATE, VARA_EP40 = 40, 6.2, 4.44
LOG = ROOT / 'logs' / 'f4_gate.log'


def log(m):
    line = f'[{datetime.now():%F %T}] {m}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def val_mae_ep():
    for f in sorted(glob.glob(str(ROOT / 'lightning_logs' / EXP / 'version_*' / 'metrics.csv'))):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        r = d[(d['epoch'] == EPOCH) & d['val/mae'].notna()]
        if len(r):
            return float(r['val/mae'].iloc[-1])
    return None


def ps(cmd):
    return subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True).stdout.strip()


def main():
    log(f'vigilante activo: gate val/mae(ep{EPOCH}) <= {GATE} (vara {VARA_EP40}) en {EXP}')
    while (v := val_mae_ep()) is None:
        time.sleep(60)
    ok = v <= GATE
    txt = (f'val/mae ep{EPOCH} = {v:.3f} | gate <= {GATE} (1.4x vara {VARA_EP40}) -> '
           + ('PASA: sigue la cadena cuadrática (L1 no se lanza solo)' if ok else 'NO PASA: se cierra la cuadrática y se lanza el sweep L1'))
    log(txt)
    (ROOT / 'results' / EXP).mkdir(parents=True, exist_ok=True)
    (ROOT / 'results' / EXP / 'gate_epoca40.txt').write_text(txt + '\n', encoding='utf-8')
    if ok:
        return
    log(ps("schtasks /end /tn sa_f4"))
    # primero el bash de la cadena (tiene el bucle de reintentos), después el entrenamiento y sus workers
    log(ps("Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'run_f4[.]sh' } | % { Stop-Process -Id $_.ProcessId -Force; 'bash '+$_.ProcessId }"))
    log(ps("Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'exp_f4_001|multiprocessing' -and $_.CommandLine -match 'site-agnostic-dose' } | % { Stop-Process -Id $_.ProcessId -Force; 'py '+$_.ProcessId }"))
    time.sleep(15)
    log(ps(f'powershell -NoProfile -File "{ROOT / "scripts" / "launch_task.ps1"}" -Name sa_f4_l1 -Cmd "bash scripts/run_f4_l1.sh >> logs/run_f4_l1.out 2>&1"'))


if __name__ == '__main__':
    main()
