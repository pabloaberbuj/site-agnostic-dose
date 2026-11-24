"""
LightningModule de la loss cuantílica: A0Module + loss = pinball_body (PTV tau=0.5, tejido sano tau_s). state_dict idéntico a A0Module
(eval_a0 / harness cargan estos ckpts con A0Module). Loguea val/loss (monitor), val/mae y val/calib_sano
(= fracción de vóxeles de tejido sano con real>pred, media por paciente; objetivo 1-tau_s).
"""
from src.datamodules.a0_datamodule import CH_BODY
from src.losses.pinball import pinball_body
from src.models.a0_module import A0Module, masked_mae


class F4QModule(A0Module):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tau_s = float(cfg.loss.tau_s)
        self.tau_ptv = float(cfg.loss.get('tau_ptv', 0.5))

    def _step(self, batch, stage):
        x, y, ptv = batch['input'], batch['dose'], batch['ptv_mask']
        pred = self(x).float()
        body = x[:, :, CH_BODY]
        loss = pinball_body(pred, y, body, ptv, self.tau_s, self.tau_ptv)
        mae = masked_mae(pred, y, body)
        kw = dict(on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log(f'{stage}/loss', loss, prog_bar=True, **kw)
        self.log(f'{stage}/mae', mae, prog_bar=True, **kw)
        if stage == 'val':
            sano = (body * (1 - ptv)) > 0
            if sano.any():
                self.log('val/calib_sano', (y[sano] > pred[sano]).float().mean(), prog_bar=True, **kw)
        return loss
