"""
Sonda de RAM/VRAM de vanilla-3D con pacientes REALES de train, ordenados por Z creciente (Fase 3, eje B).
Un paso completo (forward + backward + AdamW) por paciente, en el proceso principal (sin workers), con un watchdog que
aborta el proceso (exit 4) si la RAM libre del host baja de --min-free-gib: el corte lo hace el script, no Windows.
Responde: ¿el pico de RAM/VRAM/tiempo crece con Z? ¿qué RAM libre queda durante el paso?

    .venv/Scripts/python.exe scripts/probe_b1_real.py [--min-free-gib 1.5] [--n-top 5]
"""
import argparse
import csv
import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from src.datamodules.a0_datamodule import CH_BODY  # noqa: E402
from src.models.a0_module import A0Module, masked_mae  # noqa: E402
from train_a0 import build_dm  # noqa: E402

GIB = 2 ** 30


class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('sullAvailExtendedVirtual', ctypes.c_ulonglong)]


class _Pmc(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t)]


def free_gib():
    s = _MemStatus(); s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / GIB


def rss_gib():
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Pmc), wintypes.DWORD]
    p = _Pmc(); p.cb = ctypes.sizeof(p)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.WorkingSetSize / GIB, p.PeakWorkingSetSize / GIB


def watchdog(min_free, state):
    lo = 1e9
    while True:
        f = free_gib()
        lo = min(lo, f)
        state['min_free'] = lo
        if f < min_free:
            print(f'[ABORT] RAM libre {f:.2f} GiB < {min_free} GiB (fase: {state.get("fase")}) -> exit 4', flush=True)
            os._exit(4)
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs/exp_b1_001_unet3d_psdm.yaml'))
    ap.add_argument('--bbox-csv', default=None, help='csv con split,pid,Z (default: calcula Z del dataset)')
    ap.add_argument('--min-free-gib', type=float, default=1.5)
    ap.add_argument('--n-top', type=int, default=5, help='cantidad de pacientes de Z máximo')
    ap.add_argument('--cap-gib', type=float, default=11.0)
    a = ap.parse_args()

    state = {'fase': 'init'}
    threading.Thread(target=watchdog, args=(a.min_free_gib, state), daemon=True).start()
    print(f'[inicio] RAM libre {free_gib():.2f} GiB, RSS {rss_gib()[0]:.2f} GiB, tope abort {a.min_free_gib} GiB', flush=True)

    cfg = OmegaConf.load(a.config)
    cfg.data.num_workers = 0
    dm = build_dm(cfg)
    dm.setup()
    ds = dm.train_ds
    state['fase'] = 'seleccion'
    zs = {}
    if a.bbox_csv:
        zs = {r['pid']: int(r['Z']) for r in csv.DictReader(open(a.bbox_csv))}
    order = sorted(range(len(ds)), key=lambda i: zs[ds.ids[i]] if zs else ds.load_raw(ds.ids[i])[0].shape[0])
    z_of = lambda i: zs[ds.ids[i]] if zs else ds.load_raw(ds.ids[i])[0].shape[0]
    n = len(order)
    sel = [order[0], order[n // 4], order[n // 2], order[(9 * n) // 10]] + order[-a.n_top:]
    sel = list(dict.fromkeys(sel))
    print(f'[seleccion] {len(sel)} pacientes, Z = {[z_of(i) for i in sel]}', flush=True)

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gib * GIB / total))
    torch.set_float32_matmul_precision('high')
    mod = A0Module(cfg).cuda().train()
    opt = mod.configure_optimizers()['optimizer']
    amp = torch.float16 if '16' in str(cfg.training.precision) and 'bf' not in str(cfg.training.precision) else torch.bfloat16
    print(f'[modelo] cargado, RAM libre {free_gib():.2f} GiB, RSS {rss_gib()[0]:.2f} GiB', flush=True)
    print('Z    site       t_carga  t_paso  RSS(GiB) pico_RSS  RAMlibre_min  VRAM_alloc  VRAM_res', flush=True)

    for i in sel:
        state['fase'] = f'idx {i}'
        state['min_free'] = 1e9
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        item = ds[i]
        t_load = time.time() - t0
        x = item['input'].unsqueeze(0).cuda(non_blocking=True)
        y = item['dose'].unsqueeze(0).cuda(non_blocking=True)
        t0 = time.time()
        try:
            with torch.autocast('cuda', dtype=amp):
                pred = mod(x)
            loss = masked_mae(pred.float(), y, x[:, :, CH_BODY])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), cfg.training.gradient_clip_val)
            opt.step()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f'{x.shape[1]:<4d} {item["site"]:<10s} OOM CUDA (pico alloc {torch.cuda.max_memory_allocated() / GIB:.2f} GiB)',
                  flush=True)
            break
        t_step = time.time() - t0
        rss, peak = rss_gib()
        print(f'{x.shape[1]:<4d} {item["site"]:<10s} {t_load:6.1f}s {t_step:6.1f}s {rss:8.2f} {peak:8.2f} {state["min_free"]:12.2f} '
              f'{torch.cuda.max_memory_allocated() / GIB:10.2f} {torch.cuda.max_memory_reserved() / GIB:9.2f}', flush=True)
        del x, y, pred, loss, item
    print(f'[fin] RAM libre {free_gib():.2f} GiB. (t_paso incluye cudnn warmup en el 1.º)', flush=True)


if __name__ == '__main__':
    main()
