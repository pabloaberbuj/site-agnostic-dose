"""
Seleccion de checkpoint de la loss cuantilica (Fase 4). Entre los top-3 por val/loss: calibracion voxel tejido sano (BODY - union PTV,
pred = clip>=0 * BODY, convencion del harness) en val, pooled (media sobre pacientes) y POR SITIO con IC95 bootstrap.
Elegido = pooled mas cercana a (1 - tau_s). Reporta tambien el mejor-por-monitor, y marca sitios con |calib - (1-tau_s)| > 0.10 o IC que no contiene el objetivo.
Escribe results/<exp>/seleccion_calib.{json,txt}; ultimas lineas: CHOSEN=<ckpt> / CHOSEN_MON=<ckpt>.
    .venv/Scripts/python.exe scripts/f4q_seleccion.py <exp_id> <tau_s>
GPU: correr solo con la GPU libre (despues de los trains).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import f4_harness_calibracion as H  # noqa: E402
import common as C  # noqa: E402

FLAG_ABS = 0.10


def top3(exp):
    ck = torch.load(ROOT / 'checkpoints' / exp / 'last.ckpt', map_location='cpu', weights_only=False)
    st = next(v for k, v in ck['callbacks'].items() if k.startswith('ModelCheckpoint'))
    return {int(Path(p).stem.split('=')[1]): float(s) for p, s in st['best_k_models'].items()}


def calib_ckpt(config, ckpt):
    src = H.ModelSource(config, ckpt, 'val')
    rows = []
    for pid in src.ids:
        site = next(s for s in C.SITES if (C.NPZ_DIR / s / f'{pid}.npz').exists())
        dose, masks, _, _ = H.load_npz(site, pid)
        body = masks['BODY']
        pred = np.clip(src.predict(pid, None), 0.0, None) * body
        ptv = np.zeros_like(body)
        for l in C.PTV_LEVELS[site]:
            if l in masks:
                ptv |= masks[l]
        sel = body & ~ptv
        rows.append((site, float((dose[sel] > pred[sel]).mean())))
    del src
    torch.cuda.empty_cache()
    return rows


def main(exp, tau_s):
    target = 1 - tau_s
    config = f'configs/{exp}.yaml'
    mon = top3(exp)
    res, lines = {}, [f'== {exp}: calib voxel tejido sano, objetivo (1-tau_s) = {target:.2f}; val/loss = monitor ==']
    for ep in sorted(mon):
        rows = calib_ckpt(config, f'checkpoints/{exp}/epoch={ep:03d}.ckpt')
        v = np.array([r[1] for r in rows])
        site = {}
        for s in C.SITES:
            x = [r[1] for r in rows if r[0] == s]
            if x:
                m, lo, hi, n = H.bootstrap_mean(x)
                site[s] = {'mean': m, 'ci': [lo, hi], 'n': n,
                           'flag': bool(abs(m - target) > FLAG_ABS or not (lo <= target <= hi))}
        m, lo, hi, n = H.bootstrap_mean(v)
        res[ep] = {'val_loss': mon[ep], 'pooled': {'mean': m, 'ci': [lo, hi], 'n': n}, 'sitios': site,
                   'dist_pooled': abs(m - target)}
        lines.append(f'  ep{ep}: val/loss {mon[ep]:.4f} | pooled {m:.3f} [{lo:.3f},{hi:.3f}] | ' +
                     ' '.join(f"{s} {d['mean']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}]{'*' if d['flag'] else ''}" for s, d in site.items()))
    best_cal = min(res, key=lambda e: res[e]['dist_pooled'])
    best_mon = min(mon, key=mon.get)
    lines.append(f'  -> elegido (pooled mas cercana a {target:.2f}): ep{best_cal}; mejor-por-monitor: ep{best_mon}'
                 + ('  (coinciden)' if best_cal == best_mon else '  (SE SEPARAN)'))
    lines.append('  (* = sitio a >0.10 del objetivo o con IC que no lo contiene: heterogeneidad entre sitios, revisar)')
    out = ROOT / 'results' / exp
    out.mkdir(parents=True, exist_ok=True)
    (out / 'seleccion_calib.json').write_text(json.dumps({'tau_s': tau_s, 'target': target, 'elegido': best_cal, 'mejor_monitor': best_mon,
                                                          'por_ckpt': {str(k): v for k, v in res.items()}}, indent=2), encoding='utf-8')
    (out / 'seleccion_calib.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'CHOSEN=checkpoints/{exp}/epoch={best_cal:03d}.ckpt')
    print(f'CHOSEN_MON=checkpoints/{exp}/epoch={best_mon:03d}.ckpt')


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]))
