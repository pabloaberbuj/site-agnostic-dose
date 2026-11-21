"""
LightningModule de Fase 4: A0Module + loss = masked_mae_body + λ·L_dvh (momentos, ∩BODY). Mismo modelo/forward (state_dict
idéntico al de A0Module: eval_a0.py carga estos ckpts con A0Module). Loguea val/loss (monitor), val/mae y val/dvh.
"""
from src.datamodules.a0_datamodule import CH_BODY
from src.losses.dvh_moments import dvh_moment_loss
from src.models.a0_module import A0Module, masked_mae


class F4Module(A0Module):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lam = float(cfg.loss.lambda_dvh)
        self.form = str(cfg.loss.get('moment_form', 'sq'))

    def _step(self, batch, stage):
        x, y = batch['input'], batch['dose']
        pred = self(x)
        mae = masked_mae(pred.float(), y, x[:, :, CH_BODY])
        dvh, _ = dvh_moment_loss(pred, y, batch['struct_bits'], batch['struct_p'], self.form)
        loss = mae + self.lam * dvh
        kw = dict(on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log(f'{stage}/loss', loss, prog_bar=True, **kw)
        self.log(f'{stage}/mae', mae, prog_bar=True, **kw)
        self.log(f'{stage}/dvh', dvh, **kw)
        return loss
