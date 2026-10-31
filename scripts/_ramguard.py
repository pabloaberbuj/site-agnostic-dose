"""Watchdog de RAM del host (Windows): aborta el proceso (exit 4) si la RAM física libre baja de un mínimo.
Convierte un cuelgue del equipo por presión de RAM en un aborto limpio con evidencia en el log."""
import ctypes
import os
import threading
import time
from ctypes import wintypes

GIB = 2 ** 30


class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('sullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_gib():
    s = _MemStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / GIB


def _rss_gib():
    class P(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('pf', wintypes.DWORD), ('peak', ctypes.c_size_t), ('ws', ctypes.c_size_t),
                    ('a', ctypes.c_size_t), ('b', ctypes.c_size_t), ('c', ctypes.c_size_t), ('d', ctypes.c_size_t),
                    ('priv', ctypes.c_size_t), ('peakpriv', ctypes.c_size_t)]
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(P), wintypes.DWORD]
    p = P(); p.cb = ctypes.sizeof(p)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.ws / GIB, p.priv / GIB


def _top_procs():
    import subprocess
    try:
        return subprocess.run(['powershell', '-NoProfile', '-Command',
                               'Get-Process | Sort WS -desc | Select -First 6 | ForEach-Object { "{0} {1:N2}" -f $_.Name, ($_.WS/1GB) }'],
                              capture_output=True, text=True, timeout=10).stdout.strip().replace('\n', ' | ')
    except Exception as e:  # noqa: BLE001
        return f'(top falló: {e})'


def start(min_free_gib, log_every_s=3):
    """Hilo daemon: cada 0.5 s mide; imprime la RAM libre cada log_every_s y aborta bajo min_free_gib."""
    def run():
        last, lo = 0.0, 1e9
        while True:
            f = free_gib()
            lo = min(lo, f)
            if f < min_free_gib:
                ws, priv = _rss_gib()
                print(f'[ABORT-RAM] RAM libre {f:.2f} GiB < {min_free_gib} GiB (este proceso: WS {ws:.2f}, privada {priv:.2f} GiB) '
                      f'-> exit 4. Top: {_top_procs()}', flush=True)
                os._exit(4)
            if time.time() - last >= log_every_s:
                print(f'[ram] libre {f:.2f} GiB (mín {lo:.2f}) proceso WS/priv {_rss_gib()[0]:.2f}/{_rss_gib()[1]:.2f} GiB', flush=True)
                last = time.time()
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()
