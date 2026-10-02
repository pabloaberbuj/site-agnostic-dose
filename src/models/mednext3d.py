"""
MedNeXt-style 3D block (kernel 5) — SOLO para profiling de envelope (Fase 0.0).

NO es la arquitectura final de Fase 3 (eso se decide con el código público de Xiong,
ver CHARTER.md §4). Es un stand-in representativo del costo de memoria/cómputo de un
bloque MedNeXt real (Roy et al. 2023): conv depthwise kernel grande + expansión
pointwise (ratio r) + proyección pointwise + residual. Si se perfilara solo la U-Net
vanilla, el número de Fase 3 saldría optimista (ver SPEC_profiling_0.0.md §2C) — este
módulo existe para que el barrido de envelope no mienta sobre MedNeXt.

Diverge del paper original en el upsampling: acá se usa trilinear + conv (consistente
con la decisión congelada del resto del repo, revisable en B-eje) en vez de transposed
conv depthwise. No afecta el orden de magnitud de memoria del bloque completo (el
downsample/encoder es donde MedNeXt es mucho más pesado que U-Net vanilla).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedNeXtBlock3D(nn.Module):
    """Depthwise conv (kernel k) -> GroupNorm -> pointwise expand -> GELU -> pointwise reduce.
    Residual si in_ch == out_ch."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, exp_ratio: int = 4):
        super().__init__()
        self.residual = in_ch == out_ch
        pad = kernel_size // 2
        self.dwconv = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, padding=pad,
                                 groups=in_ch, bias=False)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_ch)
        hidden = in_ch * exp_ratio
        self.pw_expand = nn.Conv3d(in_ch, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw_reduce = nn.Conv3d(hidden, out_ch, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_reduce(x)
        if self.residual:
            x = x + identity
        return x


class DownMedNeXt(nn.Module):
    """Downsample: maxpool + MedNeXtBlock (cambia canales)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, exp_ratio: int = 4):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = MedNeXtBlock3D(in_ch, out_ch, kernel_size=kernel_size, exp_ratio=exp_ratio)

    def forward(self, x):
        return self.block(self.pool(x))


class UpMedNeXt(nn.Module):
    """Trilinear upsample + conv 1x1x1 (reduce) + MedNeXtBlock sobre concat con skip."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 kernel_size: int = 5, exp_ratio: int = 4):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.block = MedNeXtBlock3D(out_ch + skip_ch, out_ch, kernel_size=kernel_size,
                                     exp_ratio=exp_ratio)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2,
                          diff_d // 2, diff_d - diff_d // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class MedNeXt3D(nn.Module):
    """
    Encoder/decoder con la misma topología de profundidad que UNet3D, pero con
    bloques MedNeXt (kernel=5, exp_ratio=4 por defecto — configuración M/L del paper).

    Entrada: (B, in_channels, D, H, W)
    Salida:  (B, out_channels, D, H, W)
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1,
                 base_features: int = 16, depth: int = 4,
                 kernel_size: int = 5, exp_ratio: int = 4,
                 grad_checkpointing: bool = False):
        super().__init__()
        assert depth >= 2
        self.depth = depth
        self.grad_checkpointing = grad_checkpointing

        self.stem = nn.Conv3d(in_channels, base_features, kernel_size=1)
        self.inc = MedNeXtBlock3D(base_features, base_features,
                                   kernel_size=kernel_size, exp_ratio=exp_ratio)

        self.downs = nn.ModuleList()
        ch = base_features
        for _ in range(depth):
            self.downs.append(DownMedNeXt(ch, ch * 2, kernel_size=kernel_size, exp_ratio=exp_ratio))
            ch *= 2

        self.ups = nn.ModuleList()
        for _ in range(depth):
            in_ch = ch
            skip_ch = ch // 2
            out_ch_up = ch // 2
            self.ups.append(UpMedNeXt(in_ch, skip_ch, out_ch_up,
                                       kernel_size=kernel_size, exp_ratio=exp_ratio))
            ch = out_ch_up

        self.outc = nn.Conv3d(ch, out_channels, kernel_size=1)

    def _run_block(self, block, *inputs):
        if self.grad_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(block, *inputs, use_reentrant=False)
        return block(*inputs)

    def forward(self, x):
        x = self.stem(x)
        x = self._run_block(self.inc, x)
        skips = [x]
        for down in self.downs:
            x = self._run_block(down, x)
            skips.append(x)
        x = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            x = self._run_block(up, x, skip)
        return self.outc(x)
