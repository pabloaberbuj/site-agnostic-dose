"""
Selección de checkpoint de Fase 4 (regla común): entre los top-3 por el monitor (val/loss) que guarda Lightning, el elegido es el de
menor criterio DVH = mean|ΔV| pooled OAR_all banda media 40-80 %Rx (val, n=65). Reporta AMBOS: mejor-por-monitor y mejor-DVH.
Requiere results/<exp>/ckpt_ep<N>/ (eval_a0.py net por cada top-3). Copia el elegido a la raíz de results/<exp>/ y escribe
summary.json (+ bloque `seleccion_fase4`) y seleccion.txt. Imprime el chosen ckpt en la última línea (`CHOSEN=<path>`).

    .venv/Scripts/python.exe scripts/f4_seleccion.py <exp_id>
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
FILES = ('metrics.csv', 'summary.json', 'secondary_per_patient.csv', 'crop_offsets.csv')


def best_k(exp):
    """{epoch: val/loss} de los top-3 guardados (de ModelCheckpoint.best_k_models en last.ckpt)."""
    ck = torch.load(ROOT / 'checkpoints' / exp / 'last.ckpt', map_location='cpu', weights_only=False)
    st = next(v for k, v in ck['callbacks'].items() if k.startswith('ModelCheckpoint'))
    out = {}
    for path, score in st['best_k_models'].items():
        out[int(Path(path).stem.split('=')[1])] = float(score)
    return out


def crit(df):
    por_pac = df[df.role == 'OAR_all'][['patient_id', 'site', 'media_40_80']]
    return por_pac.media_40_80.mean(), por_pac.groupby('site').media_40_80.mean().to_dict()


def main(exp):
    out = ROOT / 'results' / exp
    mon = best_k(exp)
    res = {ep: crit(pd.read_csv(out / f'ckpt_ep{ep:03d}' / 'metrics.csv')) for ep in sorted(mon)}
    best_mon = min(mon, key=mon.get)
    best_dvh = min(res, key=lambda e: res[e][0])
    lines = ['== criterio pooled OAR banda media 40-80 (menor = mejor), val n=65; monitor = val/loss ==']
    for ep in sorted(mon):
        tags = (' [mejor-monitor]' if ep == best_mon else '') + (' [mejor-DVH]' if ep == best_dvh else '')
        lines.append(f'  ep{ep}: val/loss {mon[ep]:.4f} | pooled {res[ep][0]:.3f} | ' +
                     ' '.join(f'{s} {v:.2f}' for s, v in sorted(res[ep][1].items())) + tags)
    lines.append(f'  -> elegido (mejor-DVH sobre top-3): ep{best_dvh}; mejor-por-monitor: ep{best_mon}; '
                 f'gap pooled (monitor - DVH) = {res[best_mon][0] - res[best_dvh][0]:+.3f}'
                 + ('  (coinciden)' if best_mon == best_dvh else '  (SE SEPARAN: la loss desacopla MAE/loss de DVH)'))
    src = out / f'ckpt_ep{best_dvh:03d}'
    for f in FILES:
        shutil.copy(src / f, out / f)
    sj = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    sj['seleccion_fase4'] = {'experimento': exp, 'ckpt_elegido': f'epoch={best_dvh:03d}.ckpt', 'ckpt_mejor_monitor': f'epoch={best_mon:03d}.ckpt',
                             'regla': 'top-3 por val/loss (MAE+λ·DVH); elegido = menor mean|ΔV| pooled OAR_all banda media 40-80 en val (n=65)',
                             'val_loss_por_ckpt': {str(e): mon[e] for e in mon},
                             'pooled_oar_media_por_ckpt': {str(e): res[e][0] for e in res},
                             'pooled_por_sitio_por_ckpt': {str(e): res[e][1] for e in res}}
    (out / 'summary.json').write_text(json.dumps(sj, indent=2), encoding='utf-8')
    (out / 'seleccion.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'CHOSEN=checkpoints/{exp}/epoch={best_dvh:03d}.ckpt')


if __name__ == '__main__':
    main(sys.argv[1])
