#!/usr/bin/env bash
# Fase 3 / eje B: benchmark de num_workers para vanilla-3D (tiempo/época), con todas las protecciones puestas.
# 80 pasos reales por variante (sin val/ckpt), mismo seed => mismos pacientes entre variantes. Una variante por proceso.
# Protecciones (se corre sin nadie cerca para reiniciar la PC): tope de VRAM 11 GiB (evita el derrame a RAM que cuelga Windows),
# watchdog de RAM libre (aborta < MIN_FREE_GIB), timeout duro por variante, espera de GPU/RAM libres, y para al primer fallo
# (no prueba una variante más pesada si la anterior abortó).   Uso: bash scripts/run_b_bench.sh
set -u
cd "$(dirname "$0")/.."
source scripts/_lib_run.sh
PY=.venv/Scripts/python.exe
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
MIN_FREE_GIB="${MIN_FREE_GIB:-3.0}"
STEPS="${STEPS:-80}"
VARIANTS="${VARIANTS:-0 1 2}"
mkdir -p logs
echo "[bench] inicio $(date '+%F %T') variantes=[$VARIANTS] pasos=$STEPS guard RAM=${MIN_FREE_GIB} GiB"
wait_gpu_free || exit 1
for w in $VARIANTS; do
  wait_ram_free || { echo "[bench] sin RAM libre, paro"; exit 1; }
  log="logs/bench_b1_w${w}.log"
  echo "[bench] --- workers=$w $(date '+%T') ---"
  timeout 1200 $PY scripts/train_a0.py --config configs/exp_b1_001_unet3d_psdm.yaml --workers "$w" --bench-steps "$STEPS" \
      --min-free-gib "$MIN_FREE_GIB" --cap-gib 11.0 2>&1 | tr '\r' '\n' > "$log"
  rc=${PIPESTATUS[0]}
  grep -a "\[bench\]\|ABORT-RAM\|OutOfMemory" "$log" | tail -4
  if [ "$rc" -ne 0 ]; then echo "[bench] workers=$w FALLÓ (exit $rc): no pruebo variantes más pesadas"; exit "$rc"; fi
  sleep 20
done
echo "[bench] completo $(date '+%F %T')"
