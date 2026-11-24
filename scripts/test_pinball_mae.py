"""Paso 0 de la loss cuantilica: con tau=0.5 en todas las regiones pinball_body debe reproducir masked_mae_body (A0Module.masked_mae).
Corre sobre pacientes reales de val (uno por sitio) con pred sintetica. Ademas: asimetria (tau_s<0.5 penaliza mas sobrepredecir en sano, PTV intacto)."""
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datamodules.a0_datamodule import frame_from_cfg, CH_BODY
from src.datamodules.f4q_datamodule import F4QDataModule
from src.losses.pinball import pinball_body
from src.models.a0_module import masked_mae

cfg = OmegaConf.load(ROOT / 'configs/exp_f4q_001_pinball_tau0p35.yaml'); d = cfg.data
dm = F4QDataModule(ROOT / d.npz_dir, ROOT / d.splits_file, ROOT / d.harmonization_dir, pad_hw=frame_from_cfg(d),
                   fill_sin_goal=d.fill_sin_goal, goal_mode=d.goal_mode); dm.setup()
seen, ok = set(), True
torch.manual_seed(0)
for pid in dm.val_ds.ids:
    b = dm.val_ds[dm.val_ds.ids.index(pid)]
    if b['site'] in seen: continue
    seen.add(b['site'])
    x, y, ptv = b['input'][None].cuda(), b['dose'][None].cuda(), b['ptv_mask'][None].cuda()
    body = x[:, :, CH_BODY]
    pred = y + torch.randn_like(y) * 5 + 2
    mae = masked_mae(pred, y, body); pb = pinball_body(pred, y, body, ptv, 0.5)
    diff = abs(mae.item() - pb.item())
    ptv_ok = bool(((ptv > 0) & (body == 0)).sum() == 0)
    over = pinball_body(y + 5, y, body, ptv, 0.35); under = pinball_body(y - 5, y, body, ptv, 0.35)
    print(f"{b['site']:9s} {pid} MAE={mae.item():.6f} pinball(0.5)={pb.item():.6f} |diff|={diff:.2e} | PTV frac BODY={((ptv*body).sum()/body.sum()).item():.3f} "
          f"ptv_in_body={ptv_ok} | sobrepred(+5)={over.item():.3f} subpred(-5)={under.item():.3f}")
    ok &= diff < 1e-5 and ptv_ok and over.item() > under.item()
print('PASO 0:', 'OK' if ok else 'FALLA')
sys.exit(0 if ok else 1)
