#!/usr/bin/env bash
# Harness de calibracion completo sobre los ckpts elegidos (elegido por calib + mejor por monitor) de cada tau_s. GPU libre.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for p in "exp_f4q_001_pinball_tau0p35 0.35 96" "exp_f4q_002_pinball_tau0p30 0.30 100" "exp_f4q_002_pinball_tau0p30 0.30 110" \
         "exp_f4q_003_pinball_tau0p40 0.40 100" "exp_f4q_003_pinball_tau0p40 0.40 112"; do
  set -- $p; HID=f4h_${1#exp_}_ep$3
  [ -f results/$HID/summary.json ] && continue
  $PY scripts/f4_harness_calibracion.py --harness-id $HID --tau $2 --config configs/$1.yaml --ckpt checkpoints/$1/epoch=$(printf %03d $3).ckpt --renorm none > logs/$HID.log 2>&1 || echo "[$HID] fallo"
done
date > logs/f4q_harness.done
