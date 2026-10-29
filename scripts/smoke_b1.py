"""
Smoke test de B1 (Fase 3, eje B): A0Module rama unet3d con el datamodule A4 real.

    .venv/Scripts/python.exe scripts/smoke_b1.py [--config configs/exp_b1_001_unet3d_psdm.yaml]

Toma el paciente de menor y de mayor Z del train, corre forward+backward bajo la precisión del config (autocast fp16),
y reporta shape, NaN en loss/grads y pico de VRAM. El de mayor Z es el caso límite de memoria (sin ckpt).
"""
import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from train_a0 import build_dm  # noqa: E402
from src.models.a0_module import A0Module, masked_mae  # noqa: E402
from src.datamodules.a0_datamodule import CH_BODY  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs/exp_b1_001_unet3d_psdm.yaml'))
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    dm = build_dm(cfg)
    dm.setup('fit')
    ds = dm.train_ds
    zs = {pid: ds.load_raw(pid)[0].shape[0] for pid in ds.ids}
    picks = [min(zs, key=zs.get), max(zs, key=zs.get)]
    mod = A0Module(cfg).cuda().train()
    amp = torch.float16 if '16' in str(cfg.training.precision) and 'bf' not in str(cfg.training.precision) else torch.bfloat16
    for pid in picks:
        i = ds.ids.index(pid)
        b = ds[i]
        x, y = b['input'][None].cuda(), b['dose'][None].cuda()
        mod.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=amp):
            pred = mod(x)
        loss = masked_mae(pred.float(), y, x[:, :, CH_BODY])
        loss.backward()
        bad = sum(int((~torch.isfinite(p.grad)).sum()) for p in mod.parameters() if p.grad is not None)
        print(f'{pid} site={b["site"]} Z={zs[pid]} x={tuple(x.shape)} pred={tuple(pred.shape)} loss={loss.item():.4f} '
              f'nonfinite_grads={bad} peakVRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GiB')


if __name__ == '__main__':
    main()
