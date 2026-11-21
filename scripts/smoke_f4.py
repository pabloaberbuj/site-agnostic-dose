"""Smoke de la loss de momentos F4: estructuras por sitio, sanidad (pred=target -> 0), gradiente finito, escala vs MAE."""
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datamodules.f4_datamodule import structure_specs
from src.datamodules.a0_datamodule import frame_from_cfg
from src.datamodules.f4_datamodule import F4DataModule
from src.losses.dvh_moments import dvh_moment_loss

cfg = OmegaConf.load(ROOT / 'configs/exp_f4_001_dvhmoment_sp_lambda0p3.yaml'); d = cfg.data
dm = F4DataModule(ROOT / d.npz_dir, ROOT / d.splits_file, ROOT / d.harmonization_dir, pad_hw=frame_from_cfg(d),
                  fill_sin_goal=d.fill_sin_goal, goal_mode=d.goal_mode)
dm.setup()
seen = {}
for ds in (dm.train_ds, dm.val_ds):
    for pid in ds.ids:
        s = ds.npz_index[pid].parent.name
        if seen.get(s, 0) >= 2: continue
        seen[s] = seen.get(s, 0) + 1
        b = ds[ds.ids.index(pid)]
        x, y, bits, sp = b['input'][None], b['dose'][None], b['struct_bits'][None], b['struct_p'][None]
        dev = 'cuda'
        y, bits, sp = y.to(dev), bits.to(dev), sp.to(dev)
        z, _ = dvh_moment_loss(y.clone(), y, bits, sp)
        pred = (y * 1.05 + 2).clone().requires_grad_(True)
        l, n = dvh_moment_loss(pred, y, bits, sp); l.backward()
        body = x[:, :, 2].to(dev)
        mae = ((pred - y).abs() * body).sum() / body.sum()
        print(f"{s:9s} {pid} Z={y.shape[1]} S={n} p-activos={sp[0].sum().item()} bitsmax={int(bits.max()).bit_length()} | L(pred=y)={z.item():.2e} | "
              f"L(1.05y+2)={l.item():.3f} MAE={mae.item():.3f} grad finito={bool(torch.isfinite(pred.grad).all())}")
