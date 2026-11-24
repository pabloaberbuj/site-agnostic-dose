#!/usr/bin/env bash
# Seleccion por calibracion de los 3 tau_s (GPU libre). Salida: results/<exp>/seleccion_calib.{json,txt}, logs/f4q_sel_<exp>.log
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for p in "exp_f4q_001_pinball_tau0p35 0.35" "exp_f4q_002_pinball_tau0p30 0.30" "exp_f4q_003_pinball_tau0p40 0.40"; do
  set -- $p
  [ -f results/$1/seleccion_calib.json ] && continue
  $PY scripts/f4q_seleccion.py $1 $2 > logs/f4q_sel_$1.log 2>&1 || echo "[$1] seleccion fallo"
done
date > logs/f4q_sel.done
