"""
MedNeXt-2D (v1, vendorizado en src/models/mednext_v1, Apache-2.0) para el eje B de la Fase 3.

Mismo path 2D que la U-Net (A0Module: aplanado B*Z, chunks con checkpointing, masked_mae_body). Regresión: salida de 1 canal
lineal (OutBlock = ConvTranspose 1x1, sin activación), sin deep supervision.

Variantes (exp_r por etapa, block_counts) = las oficiales S/B/M/L de create_mednext_v1.py, pero con `dim='2d'` (el factory
oficial no expone dim) y `n_channels` configurable. Oficial: n_channels=32.
"""
import torch.nn as nn

from src.models.mednext_v1.MedNextV1 import MedNeXt

VARIANTS = {
    'S': dict(exp_r=2, block_counts=[2] * 9),
    'B': dict(exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2], block_counts=[2] * 9),
    'M': dict(exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2], block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3]),
    'L': dict(exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3], block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3]),
}


def build_mednext2d(cfg) -> nn.Module:
    m = cfg.model
    v = VARIANTS[m.get('mednext_variant', 'S')]
    return MedNeXt(
        in_channels=m.in_channels,
        n_channels=int(m.get('mednext_n_channels', 32)),
        n_classes=m.out_channels,
        exp_r=v['exp_r'],
        kernel_size=int(m.get('mednext_kernel_size', 3)),
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        checkpoint_style=None,      # el checkpointing lo hace A0Module (slice_chunk)
        block_counts=v['block_counts'],
        dim='2d',
    )
