"""
Vigilante de la época 40 (pre-registrado): exp_f4_001_dvhmoment_sp_lambda0p3 (loss cuadrática). Espera a que exista val/mae de la época 40.
  val/mae(ep40) <= GATE  -> la cadena cuadrática sigue (0.3 -> 0.1 -> 1.0). No hace nada más. (L1 NO se lanza solo.)
  val/mae(ep40) >  GATE  -> cierra la cadena cuadrática (tarea sa_f4 + procesos) y lanza el sweep L1 (scripts/run_f4_l1.sh, tarea sa_f4_l1).
GATE = 6.2 = 1.4 x val/mae de la vara exp_a4_003 en la época 40 (4.44). Fijado antes de mirar. Registra en logs/f4_gate.log y results/<exp>/gate_epoca40.txt.
"""
import argparse
import glob
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = 'exp_f4_001_dvhmoment_sp_lambda0p3'
EPOCH, GATE, VARA_EP40 = 40, 6.2, 4.44
FAIL_ACTION = 'launch_l1'   # 'launch_l1' (gate de la cuadrática) | 'stop' (gate de L1: solo cierra la cadena, no lanza nada)
CHAIN, TASK = 'run_f4', 'sa_f4'
LOG = ROOT / 'logs' / 'f4_gate.log'


def log(m):
    line = f'[{datetime.now():%F %T}] {m}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def val_mae_ep(exp):
    for f in sorted(glob.glob(str(ROOT / 'lightning_logs' / exp / 'version_*' / 'metrics.csv'))):
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=EXP)
    ap.add_argument('--chain', default=CHAIN, help='nombre del run_*.sh de la cadena, sin .sh')
    ap.add_argument('--task', default=TASK, help='tarea programada de la cadena')
    ap.add_argument('--fail-action', default=FAIL_ACTION, choices=['launch_l1', 'stop', 'warmstart'])
    ap.add_argument('--gate', type=float, default=GATE)
    a = ap.parse_args()
    exp, gate = a.exp, a.gate
    log(f'vigilante activo: gate val/mae(ep{EPOCH}) <= {gate} (vara {VARA_EP40}) en {exp} | si falla: {a.fail_action}')
    while (v := val_mae_ep(exp)) is None:
        time.sleep(60)
    ok = v <= gate
    if ok:
        res = 'PASA: sigue la cadena (no se lanza nada más)'
    elif a.fail_action == 'launch_l1':
        res = 'NO PASA: se cierra la cadena y se lanza el sweep L1'
    elif a.fail_action == 'warmstart':
        res = 'NO PASA: from-scratch NO CONCLUIDO por convergencia lenta (no es resultado clinico); se cierra la cadena y se lanza el warm-start (docs/PROMPT_code_f4_dvhloss_warmstart.md)'
    else:
        res = 'NO PASA: se cierra la cadena (no se lanza nada más)'
    txt = f'val/mae ep{EPOCH} = {v:.3f} | gate <= {gate} (1.4x vara {VARA_EP40}) -> {res}'
    log(txt)
    (ROOT / 'results' / exp).mkdir(parents=True, exist_ok=True)
    (ROOT / 'results' / exp / 'gate_epoca40.txt').write_text(txt + chr(10), encoding='utf-8')
    if ok:
        return
    log(ps(f'schtasks /end /tn {a.task}'))
    # primero el bash de la cadena (tiene el bucle de reintentos), después el entrenamiento y sus workers
    log(ps("Get-CimInstance Win32_Process | ? { $_.CommandLine -match '" + a.chain + "[.]sh' } | % { Stop-Process -Id $_.ProcessId -Force; 'bash '+$_.ProcessId }"))
    log(ps("Get-CimInstance Win32_Process | ? { $_.CommandLine -match '" + exp.rsplit('_lambda', 1)[0] + "|multiprocessing' -and $_.CommandLine -match 'site-agnostic-dose' } | % { Stop-Process -Id $_.ProcessId -Force; 'py '+$_.ProcessId }"))
    if a.fail_action == 'launch_l1':
        time.sleep(15)
        log(ps(f'powershell -NoProfile -File "{ROOT / "scripts" / "launch_task.ps1"}" -Name sa_f4_l1 -Cmd "TRAIN_W=2 bash scripts/run_f4_l1.sh >> logs/run_f4_l1.out 2>&1"'))
    if a.fail_action == 'warmstart':
        time.sleep(15)
        log(ps(f'powershell -NoProfile -File "{ROOT / "scripts" / "launch_task.ps1"}" -Name sa_f4_ws -Cmd "TRAIN_W=2 bash scripts/run_f4_ws.sh >> logs/run_f4_ws.out 2>&1"'))


if __name__ == '__main__':
    main()
