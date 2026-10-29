"""
Smoke de VRAM sintético de B1 (Fase 3, eje B): A0Module rama unet3d con un tensor aleatorio (1, Z, C, H, W).
No toca datos reales. Un paso completo (forward + backward + AdamW.step) bajo la precisión del config,
con tope de memoria del proceso y guard de OOM (reporta y sale 0; no tira abajo la GPU).

    .venv/Scripts/python.exe scripts/smoke_b1_vram.py --hw 176 256 [--z 239] [--no-ckpt] [--cap-gib 11]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.a0_module import A0Module, masked_mae  # noqa: E402
from src.datamodules.a0_datamodule import CH_BODY  # noqa: E402


def smi_used_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs/exp_b1_001_unet3d_psdm.yaml'))
    ap.add_argument('--hw', type=int, nargs=2, required=True, metavar=('H', 'W'))
    ap.add_argument('--z', type=int, default=239)
    ap.add_argument('--no-ckpt', action='store_true')
    ap.add_argument('--cap-gib', type=float, default=11.0, help='tope duro de memoria del proceso (GiB)')
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    cfg.model.grad_checkpointing = not a.no_ckpt
    H, W = a.hw
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gib * 2**30 / total))
    torch.cuda.reset_peak_memory_stats()
    tag = f'Z={a.z} HxW={H}x{W} ({H * W} px/corte) ckpt={cfg.model.grad_checkpointing} cap={a.cap_gib}GiB'
    amp = torch.float16 if '16' in str(cfg.training.precision) and 'bf' not in str(cfg.training.precision) else torch.bfloat16
    try:
        mod = A0Module(cfg).cuda().train()
        opt = mod.configure_optimizers()['optimizer']
        x = torch.rand(1, a.z, cfg.model.in_channels, H, W, device='cuda')
        x[:, :, CH_BODY] = 1.0
        y = torch.rand(1, a.z, H, W, device='cuda')
        with torch.autocast('cuda', dtype=amp):
            pred = mod(x)
        loss = masked_mae(pred.float(), y, x[:, :, CH_BODY])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), cfg.training.gradient_clip_val)
        opt.step()
        torch.cuda.synchronize()
        print(f'OK   {tag} | pico alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, '
              f'reservado {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB, nvidia-smi {smi_used_mib() / 1024:.2f} GiB '
              f'de {total / 2**30:.2f} | loss {loss.item():.4f}')
    except torch.cuda.OutOfMemoryError:
        print(f'OOM  {tag} | pico alloc antes de fallar {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')


if __name__ == '__main__':
    main()
