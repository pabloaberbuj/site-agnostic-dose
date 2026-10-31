#!/usr/bin/env bash
# Utilidades para corridas largas desacopladas (source desde run_*.sh).
#   wait_gpu_free : espera a que la GPU esté libre (otra gente / VM la puede estar usando); aborta si no se libera.
#   retry_cmd N cmd... : reintenta hasta N veces ante exit != 0 (OOM, cuelgue, reinicio); los trains usan --resume.

GPU_FREE_MAX_MIB="${GPU_FREE_MAX_MIB:-1500}"    # uso máximo tolerado de otros procesos para arrancar
GPU_WAIT_MIN="${GPU_WAIT_MIN:-120}"             # minutos máximos de espera

gpu_used_mib () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r'; }

wait_gpu_free () {
  local waited=0 used
  while true; do
    used=$(gpu_used_mib)
    if [ -n "$used" ] && [ "$used" -le "$GPU_FREE_MAX_MIB" ]; then
      echo "[gpu] libre (${used} MiB en uso, tope ${GPU_FREE_MAX_MIB}) $(date '+%F %T')"; return 0
    fi
    if [ "$waited" -ge "$GPU_WAIT_MIN" ]; then
      echo "[gpu] NO se liberó en ${GPU_WAIT_MIN} min (${used:-?} MiB en uso): abortando sin lanzar $(date '+%F %T')"; return 1
    fi
    echo "[gpu] ocupada (${used:-?} MiB), espero 1 min..."; sleep 60; waited=$((waited + 1))
  done
}

# RAM libre del host en MiB (Windows); espera a que haya >= RAM_FREE_MIN_MIB antes de reanudar tras un aborto por RAM (exit 4)
RAM_FREE_MIN_MIB="${RAM_FREE_MIN_MIB:-2500}"
RAM_WAIT_MIN="${RAM_WAIT_MIN:-180}"
ram_free_mib () { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" 2>/dev/null | tr -d ' \r'; }
wait_ram_free () {
  local waited=0 f
  while true; do
    f=$(ram_free_mib)
    if [ -n "$f" ] && [ "$f" -ge "$RAM_FREE_MIN_MIB" ]; then echo "[ram] libre ${f} MiB (mín ${RAM_FREE_MIN_MIB}) $(date '+%F %T')"; return 0; fi
    if [ "$waited" -ge "$RAM_WAIT_MIN" ]; then echo "[ram] NO se liberó en ${RAM_WAIT_MIN} min (${f:-?} MiB): abortando $(date '+%F %T')"; return 1; fi
    echo "[ram] baja (${f:-?} MiB libres), espero 1 min..."; sleep 60; waited=$((waited + 1))
  done
}

# retry_cmd N cmd args...: stdout/stderr van al caller (el caller redirige a su log)
retry_cmd () {
  local max=$1; shift
  local i rc
  for i in $(seq 1 "$max"); do
    "$@"; rc=$?
    [ "$rc" -eq 0 ] && return 0
    echo "[retry] intento ${i}/${max} falló (exit ${rc}) $(date '+%F %T'); espero 90 s y reanudo"
    sleep 90
    wait_gpu_free || return 1
    wait_ram_free || return 1
  done
  echo "[retry] agotados ${max} intentos"; return 1
}
