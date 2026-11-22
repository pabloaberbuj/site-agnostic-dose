"""
Supervisor del sweep L1 de Fase 4 (tarea programada `sa_f4_sup`, cada 10 min). Cubre los cortes de la PC (reinicios duros, kernel-power 41):
  1. Si la cadena (scripts/run_f4_l1.sh) está viva -> no hace nada.
  2. Si terminó (los 3 .eval.done) o el gate de L1 dio "NO PASA" -> no hace nada (la cadena cuadrática NO se supervisa: se cerró a propósito).
  3. Si está caída: valida los checkpoints del experimento en curso (un corte durante el guardado deja zips ilegibles; ver
     checkpoints/<exp>/_corruptos_*): aparta los corruptos, y deja last.ckpt = el último epoch=*.ckpt sano. Relanza la cadena (--resume) con 2 workers.
  4. Relanza también el vigilante de la época 40 si cayó y todavía no decidió.
Tope: 5 relanzamientos por ventana de 6 h (si no, algo anda mal: se detiene y lo deja en el log). Pausa manual: crear logs/f4_supervisor.pause.
Log: logs/f4_supervisor.log
"""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / 'logs' / 'f4_supervisor.log'
STATE = ROOT / 'logs' / 'f4_supervisor_state.json'
PAUSE = ROOT / 'logs' / 'f4_supervisor.pause'
TAGS = ('0p3', '0p1', '1p0')                      # orden de la cadena run_f4_l1.sh
NAME = 'exp_f4_002_dvhmoment_l1_sp_lambda{}'
GATE_EXP = NAME.format('0p3')
MAX_RELAUNCH, WINDOW_S = 5, 6 * 3600


def log(m):
    line = f'[{datetime.now():%F %T}] {m}'
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def ps(cmd):
    return subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True, creationflags=0x08000000).stdout.strip()


def alive(pattern):
    return int(ps(f"@(Get-CimInstance Win32_Process | ? {{ $_.CommandLine -match '{pattern}' -and $_.ProcessId -ne {os_pid()} }}).Count") or 0) > 0


def os_pid():
    import os
    return os.getpid()


def ckpt_ok(path):
    try:
        c = torch.load(path, map_location='cpu', weights_only=False)
        return 'state_dict' in c and 'epoch' in c
    except Exception:
        return False


def current_exp():
    """Primer λ de la cadena sin .done (el que está/estaría entrenando); None si todos terminaron el train."""
    for t in TAGS:
        if not (ROOT / 'logs' / f'{NAME.format(t)}.done').exists():
            return NAME.format(t)
    return None


def repair_ckpts(exp):
    d = ROOT / 'checkpoints' / exp
    if not d.exists():
        return
    bad_dir = d / f'_corruptos_{datetime.now():%Y%m%d_%H%M}'
    cks = sorted(d.glob('epoch=*.ckpt'), key=lambda p: int(p.stem.split('=')[1]))
    good = [p for p in cks if ckpt_ok(p)]
    moved = []
    for p in cks:
        if p not in good:
            bad_dir.mkdir(exist_ok=True)
            shutil.move(str(p), str(bad_dir / p.name))
            moved.append(p.name)
    last = d / 'last.ckpt'
    last_ok = last.exists() and ckpt_ok(last)
    if not last_ok:
        if last.exists():
            bad_dir.mkdir(exist_ok=True)
            shutil.move(str(last), str(bad_dir / 'last.ckpt'))
            moved.append('last.ckpt')
        if good:
            shutil.copy(good[-1], last)
            log(f'[{exp}] last.ckpt corrupto/ausente -> restaurado desde {good[-1].name}')
        else:
            log(f'[{exp}] sin checkpoints sanos: el train arranca desde cero')
    if moved:
        log(f'[{exp}] checkpoints apartados: {moved}')


def can_relaunch():
    now = time.time()
    st = json.loads(STATE.read_text()) if STATE.exists() else {'t': []}
    st['t'] = [x for x in st['t'] if now - x < WINDOW_S]
    ok = len(st['t']) < MAX_RELAUNCH
    if ok:
        st['t'].append(now)
    STATE.write_text(json.dumps(st))
    return ok


def launch(name, cmd):
    log(f'lanzando {name}: {cmd}')
    out = ps(f'powershell -NoProfile -File "{ROOT / "scripts" / "launch_task.ps1"}" -Name {name} -Cmd "{cmd}"')
    log(out.replace('\n', ' | '))


def main_ws():
    """Warm-start (exp_f4_003): si la cadena run_f4_ws.sh está caída y no terminó, repara ckpts y relanza (--resume)."""
    ws = 'exp_f4_003_dvhmoment_sp_L1_warmstart'
    if (ROOT / 'logs' / f'{ws}.eval.done').exists() or alive('run_f4_ws[.]sh'):
        return
    if not can_relaunch():
        log('ABORTO (ws): tope de relanzamientos'); PAUSE.write_text('tope de relanzamientos'); return
    log('cadena warm-start caída; reparo checkpoints y relanzo')
    repair_ckpts(ws)
    launch('sa_f4_ws', 'TRAIN_W=2 bash scripts/run_f4_ws.sh >> logs/run_f4_ws.out 2>&1')


def main():
    if PAUSE.exists():
        return
    gate_file = ROOT / 'results' / GATE_EXP / 'gate_epoca40.txt'
    gate_txt = gate_file.read_text(encoding='utf-8') if gate_file.exists() else ''
    if 'NO PASA' in gate_txt:
        return main_ws()   # from-scratch cerrado a propósito; se supervisa el warm-start
    if all((ROOT / 'logs' / f'{NAME.format(t)}.eval.done').exists() for t in TAGS):
        return
    if not alive('run_f4_l1[.]sh'):
        # Un reinicio de la PC o un cierre manual: la cadena está caída y no terminó.
        if not can_relaunch():
            log(f'ABORTO: {MAX_RELAUNCH} relanzamientos en {WINDOW_S // 3600} h; revisar a mano (crear/borrar logs/f4_supervisor.pause)')
            PAUSE.write_text('tope de relanzamientos')
            return
        exp = current_exp()
        log(f'cadena L1 caída (exp en curso: {exp}); reparo checkpoints y relanzo')
        if exp:
            repair_ckpts(exp)
        launch('sa_f4_l1', 'TRAIN_W=2 bash scripts/run_f4_l1.sh >> logs/run_f4_l1.out 2>&1')
    if not gate_file.exists() and not alive('f4_gate[.]py'):
        launch('sa_f4_l1_gate', f'.venv/Scripts/python.exe scripts/f4_gate.py --exp {GATE_EXP} --chain run_f4_l1 --task sa_f4_l1 --fail-action warmstart >> logs/f4_gate_l1.out 2>&1')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # el supervisor nunca debe morir en silencio
        log(f'ERROR del supervisor: {type(e).__name__}: {e}')
        sys.exit(1)
