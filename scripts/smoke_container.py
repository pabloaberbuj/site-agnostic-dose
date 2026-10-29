"""
Smoke del contenedor (176,256) + crop-to-BODY (Fase 3): ida y vuelta exacta, estructuras dentro del box, paridad legacy.

    .venv/Scripts/python.exe scripts/smoke_container.py [--config configs/exp_a4_003_unet2d_psdm_c176x256.yaml] [--n-dvh 12]

Sobre los 432 del split (train+val+test):
  1. el bbox del BODY entra en el contenedor y TODA máscara de estructura queda dentro del bbox;
  2. dosis y canales: apply -> undo reproduce el original bit a bit dentro del box (0 fuera);
  3. el item que sale del dataset tiene shape (Z, C, Hc, Wc) / (Z, Hc, Wc);
  4. paridad con el camino legacy: Frame((256,256), crop_body=False) == pad_hw_center;
  5. sobre --n-dvh pacientes: dosis verdadera -> apply -> undo -> DVH == DVH directo (el crop no cambia el DVH nativo).
Reporta Z-máx por split. Falla (exit 1) ante cualquier discrepancia.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from train_a0 import build_dm  # noqa: E402
from eval_a0 import masks_from_npz  # noqa: E402
from src.datamodules.a0_datamodule import CH_BODY, Frame, pad_hw_center  # noqa: E402
from src.eval import dvh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs/exp_a4_003_unet2d_psdm_c176x256.yaml'))
    ap.add_argument('--n-dvh', type=int, default=12)
    a = ap.parse_args()
    cfg = OmegaConf.load(a.config)
    dm = build_dm(cfg)
    dm.setup()
    frame = dm.pad_hw
    assert isinstance(frame, Frame) and frame.crop_body, 'el config no define container.crop_body'
    fails = []
    zmax, hw_max = {}, [0, 0]
    for split in ('train', 'val', 'test'):
        ds = getattr(dm, f'{split}_ds')
        zmax[split] = 0
        for k, pid in enumerate(ds.ids):
            dose, chans, meta = ds.load_raw(pid)
            Z, H, W = dose.shape
            zmax[split] = max(zmax[split], Z)
            box = ds.frame_box(chans)
            y0, y1, x0, x1 = box
            hw_max = [max(hw_max[0], y1 - y0), max(hw_max[1], x1 - x0)]
            masks = masks_from_npz(ds.npz_index[pid])
            for role, m in masks.items():
                if m.any() and (m[:, :y0].any() or m[:, y1:].any() or m[:, :, :x0].any() or m[:, :, x1:].any()):
                    fails.append(f'{pid}: {role} fuera del box {box}')
            if not np.array_equal(frame.undo(frame.apply(dose, box), box, (H, W))[:, y0:y1, x0:x1], dose[:, y0:y1, x0:x1]):
                fails.append(f'{pid}: dosis no hace ida y vuelta')
            padded = ds.pad_channels(chans)
            if not np.array_equal(frame.undo(padded, box, (H, W))[:, :, y0:y1, x0:x1], chans[:, :, y0:y1, x0:x1]):
                fails.append(f'{pid}: canales no hacen ida y vuelta')
            item = ds[k] if k < 3 else None
            if item is not None and (tuple(item['input'].shape) != (Z, chans.shape[0], *frame.hw)
                                     or tuple(item['dose'].shape) != (Z, *frame.hw)):
                fails.append(f'{pid}: shape del item {tuple(item["input"].shape)} / {tuple(item["dose"].shape)}')
            if k < 3:  # paridad legacy
                legacy = Frame((256, 256), False)
                if not np.array_equal(legacy.apply(dose, legacy.box(chans[CH_BODY] > 0)), pad_hw_center(dose, 256)):
                    fails.append(f'{pid}: Frame legacy != pad_hw_center')
        print(f'{split}: n={len(ds.ids)} Zmax={zmax[split]}')
    print(f'bbox BODY máx (H,W) sobre el split = {tuple(hw_max)}; contenedor = {frame.hw}')

    ds = dm.val_ds   # DVH: el crop/undo no altera el DVH nativo
    for pid in ds.ids[:a.n_dvh]:
        dose, chans, meta = ds.load_raw(pid)
        box = ds.frame_box(chans)
        pred = frame.undo(frame.apply(dose, box), box, dose.shape[1:])
        masks = masks_from_npz(ds.npz_index[pid])
        base = dvh.patient_rows(pid, meta['site'], masks, dose, dose_pred=dose)
        viaf = dvh.patient_rows(pid, meta['site'], masks, dose, dose_pred=pred)
        for r0, r1 in zip(base, viaf):
            for key, v in r0.items():
                if isinstance(v, float) and not np.isclose(v, r1[key], equal_nan=True):
                    fails.append(f'{pid} {r0["role"]}: DVH {key} {v} != {r1[key]}')
    print(f'DVH ida y vuelta verificado en {min(a.n_dvh, len(ds.ids))} pacientes de val')
    if fails:
        print(f'FALLAS ({len(fails)}):')
        for f in fails[:30]:
            print(' ', f)
        sys.exit(1)
    print('0 fallas')


if __name__ == '__main__':
    main()
