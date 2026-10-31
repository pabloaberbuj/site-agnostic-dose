#!/usr/bin/env bash
# Fase 3 / eje B, corrida completa y desacoplada (Scheduled Task, ver scripts/launch_task.ps1):
#   0. espera GPU libre
#   1. CANARIO: 1 época de vanilla-3D (ckpt, contenedor 176x256) sobre la cohorte real, midiendo VRAM. Sin reintento: un OOM acá
#      es la señal (reabre parches) -> PARA, no lanza nada más.
#   2. A4-2D en el contenedor nuevo -> eval val
#   3. GUARDRAIL vs A4-2D viejo (OAR-media / PTV-alta frame-independientes). Si driftea -> bug de offset -> PARA.
#   4. vanilla-3D (189/28, ckpt) -> eval val
# Reanudable y con reintento: cada train usa --resume (last.ckpt) y se reintenta ante fallas (OOM por GPU compartida, cuelgue).
# Los pasos terminados dejan logs/<name>.done y se saltean al relanzar la tarea.   Uso: bash scripts/run_b_chain.sh
set -u
cd "$(dirname "$0")/.."
source scripts/_lib_run.sh
PY=.venv/Scripts/python.exe
# fragmentación del allocator (medido: baja la reserva de ~10.4 a ~8.6 GiB en el paso de vanilla-3D)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
mkdir -p logs
echo "[chain] inicio $(date '+%F %T')  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# workers: 2 (medido: 3D 0.31 it/s con 0, 0.54 con 1, 0.66 con 2). Si el watchdog de RAM corta (exit 4, p.ej. vuelve la VM), el reintento baja a 0.
TRAIN_W="${TRAIN_W:-2}"
train_once () {
  $PY scripts/train_a0.py --config "$1" --resume --min-free-gib 1.0 --cap-gib 11.0 --workers "$TRAIN_W" 2>&1 | tr '\r' '\n' >> "$2"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 4 ] && [ "$TRAIN_W" -gt 0 ]; then TRAIN_W=0; echo "[workers] corte por RAM: los reintentos siguen con 0 workers"; fi
  return "$rc"
}
eval_once ()  { $PY scripts/eval_a0.py net --config "$1" --ckpt "$2" --split val --out-dir "$3" > "$4" 2>&1; }

run_exp () {   # $1 = nombre del experimento (configs/<name>.yaml)
  local name=$1 cfg="configs/$1.yaml" log="logs/$1_train.log"
  [ -f "logs/${name}.done" ] && { echo "[${name}] ya completo, salteo"; return 0; }
  wait_gpu_free || return 1
  retry_cmd 30 train_once "$cfg" "$log" || { echo "[${name}] train no terminó"; return 1; }
  local ckpt; ckpt=$(grep -a '^best:' "$log" | tail -1 | sed -E 's/^best: (.*\.ckpt) .*/\1/')
  echo "[${name}] best ckpt: ${ckpt}"
  [ -n "$ckpt" ] || { echo "[${name}] sin best ckpt"; return 1; }
  retry_cmd 3 eval_once "$cfg" "$ckpt" "results/${name}" "logs/${name}_eval.log" || { echo "[${name}] eval falló"; return 1; }
  date > "logs/${name}.done"
}

# --- 1. canario (sin reintento) -------------------------------------------------------------------------------
if [ ! -f logs/canary_b1_1ep.done ]; then
  wait_gpu_free || exit 1
  rm -rf checkpoints/exp_b1_canary_1ep results/exp_b1_canary_1ep
  $PY scripts/train_a0.py --config configs/_canary_b1_1ep.yaml --max-epochs 1 --report-vram --min-free-gib 1.0 --cap-gib 11.0 2>&1 | tr '\r' '\n' > logs/canary_b1_1ep.log
  rc=${PIPESTATUS[0]}
  grep -a "\[vram\]\|ABORT-RAM" logs/canary_b1_1ep.log
  if [ "$rc" -eq 4 ]; then echo "CANARIO ABORTADO POR RAM del host (exit 4), NO es un OOM de GPU: no se reabren parches. Ver logs/canary_b1_1ep.log"; exit 4; fi
  if [ "$rc" -ne 0 ] || ! grep -aq "\[vram\] epoch=0 tras val" logs/canary_b1_1ep.log; then
    echo "CANARIO FALLÓ (exit ${rc}): vanilla-3D no sobrevive 1 época en (176,256)+ckpt -> PARAR, se reabren parches. No se lanza nada más."
    tail -5 logs/canary_b1_1ep.log; exit 3
  fi
  date > logs/canary_b1_1ep.done
  echo "[canario] OK: sobrevivió 1 época completa"
fi
[ "${CANARY_ONLY:-0}" = "1" ] && { echo "[chain] CANARY_ONLY: fin tras el canario"; exit 0; }

# --- 2-3. A4-2D en contenedor nuevo + guardrail ---------------------------------------------------------------
run_exp exp_a4_003_unet2d_psdm_c176x256 || exit 1
$PY scripts/guardrail_a4_container.py > logs/guardrail_a4_container.log 2>&1
rc=$?
cat logs/guardrail_a4_container.log
if [ $rc -ne 0 ]; then echo "GUARDRAIL FALLÓ: no se lanza vanilla-3D (ver logs/guardrail_a4_container.log)"; exit 2; fi

# --- 4. vanilla-3D --------------------------------------------------------------------------------------------
run_exp exp_b1_001_unet3d_psdm || exit 1
echo "[chain] completo $(date '+%F %T')"
