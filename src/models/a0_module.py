"""
LightningModule de A0 (Fase 2): U-Net 2D sobre los canales de A0, MAE puro enmascarado por body.

Recibe lo que entrega `A0DataModule`: input (B=1, Z, 3, H, W), dose (B, Z, H, W). Ch2 del input ES el
body, así que la máscara de la loss/MAE sale del propio input (no hay `body_mask` aparte).
Sampling por paciente: el volumen completo del paciente por paso (batch_size=1).
"""
import torch
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint

from src.datamodules.a0_datamodule import CH_BODY
from src.models.unet2d import build_model


def masked_mae(pred, target, body):
    return ((pred - target).abs() * body).sum() / body.sum().clamp(min=1.0)


class A0Module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters({'cfg': dict(cfg) if hasattr(cfg, 'keys') else cfg})
        self.cfg = cfg
        self.model = build_model(cfg)
        self.chunk = int(cfg.model.get('slice_chunk', 32))

    def forward(self, x):
        """x: (B, Z, C, H, W) -> (B, Z, H, W). Cortes 2D independientes, en chunks para acotar VRAM.

        arch == 'unet3d': volumen completo, se permuta a (B, C, Z, H, W) y las convs 3D cruzan Z.
        """
        B, Z, C, H, W = x.shape
        if self.cfg.model.arch == "unet3d":
            return self.model(x.permute(0, 2, 1, 3, 4)).squeeze(1)   # (B, 1, Z, H, W) -> (B, Z, H, W)
        x2 = x.reshape(B * Z, C, H, W)
        outs = []
        for i in range(0, B * Z, self.chunk):
            xi = x2[i:i + self.chunk]
            if self.training and torch.is_grad_enabled():
                outs.append(checkpoint(self.model, xi, use_reentrant=False))
            else:
                outs.append(self.model(xi))
        return torch.cat(outs, 0).view(B, Z, H, W)

    def _step(self, batch, stage):
        x, y = batch['input'], batch['dose']
        pred = self(x)
        body = x[:, :, CH_BODY]
        mae = masked_mae(pred.float(), y, body)
        self.log(f'{stage}/mae', mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        for site in set(batch['site']):
            if len(set(batch['site'])) == 1:   # batch_size=1: un sitio por paso
                self.log(f'{stage}/mae_{site}', mae, on_step=False, on_epoch=True, batch_size=x.size(0))
        return mae

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.optimizer.lr,
                                weight_decay=self.cfg.optimizer.weight_decay)
        # T_max fijo al horizonte completo, independiente de trainer.max_epochs (CHARTER §5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.scheduler.max_epochs)
        return {'optimizer': opt, 'lr_scheduler': sched}
