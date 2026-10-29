"""
DataModule de A5 (Fase 2): A4 (7 canales: A1-uniform + PSDM PTV_High) + Ch7 = densidad másica (g/cm3). NO HU.

La densidad sale de `scripts/build_density.py` (sidecar `<density_dir>/<sitio>/<pid>.npy`, float16, ya en la grilla
de dosis nativa; curva Big Bore Mass_Density, ver src/preprocess/density.py). Se enmascara con `mask_BODY`: fuera del
cuerpo = 0 (couch / inmovilizadores no entran como "tejido"), igual que el padding in-plane (0). Dentro del cuerpo el
aire queda en ~0.001 y el agua en ~1.0. Sin normalización adicional (rango útil 0.001–~2.9).
"""
import json

import numpy as np

from src.datamodules.a0_datamodule import DataError
from src.datamodules.a4_datamodule import (A4DataModule, A4PatientDataset, N_CHANNELS_A4, pad_with_psdm_last)
from src.config.harmonization import BODY_ROLE

CH_DENSITY = N_CHANNELS_A4        # 7
N_CHANNELS_A5 = N_CHANNELS_A4 + 1
DENSITY_PAD = 0.0


def density_channel(density_dir, npz_path, pid: str, dose_shape) -> np.ndarray:
    """(1,Z,H,W) float32: densidad másica dentro del BODY, 0 fuera."""
    f = density_dir / npz_path.parent.name / f'{pid}.npy'
    if not f.exists():
        raise DataError(f'{pid}: sin densidad en {f} (correr scripts/build_density.py)')
    rho = np.load(f).astype(np.float32)
    if rho.shape != tuple(dose_shape):
        raise DataError(f'{pid}: densidad {rho.shape} != dosis {tuple(dose_shape)}')
    with np.load(npz_path) as d:
        key = f'mask_{BODY_ROLE}'
        if key not in d.files:
            raise DataError(f'{pid}: NPZ sin {key}')
        body = d[key].astype(bool)
    return np.where(body, rho, np.float32(DENSITY_PAD))[None]


class A5PatientDataset(A4PatientDataset):
    def __init__(self, *args, density_dir, **kw):
        super().__init__(*args, **kw)
        self.density_dir = density_dir

    def load_raw(self, pid: str):
        """(dose, channels (8,Z,H,W) float32, meta) sin pad."""
        dose, chans, meta = super().load_raw(pid)
        dens = density_channel(self.density_dir, self.npz_index[pid], pid, dose.shape)
        return dose, np.concatenate([chans, dens], axis=0), meta

    def pad_channels(self, chans: np.ndarray) -> np.ndarray:
        """Ch0-6 como A4 (PSDM con pad 1.0), Ch7 densidad con pad 0."""
        box = self.frame_box(chans)
        base = pad_with_psdm_last(chans[:-1], self.frame, box)
        return np.concatenate([base, self.frame.apply(chans[-1:], box)], axis=0)


class A5DataModule(A4DataModule):
    def __init__(self, *args, density_dir, **kw):
        super().__init__(*args, **kw)
        from pathlib import Path
        self.density_dir = Path(density_dir)

    def _make_ds(self, ids, index):
        return A5PatientDataset(ids, index, self.table, self.pad_hw, self.fill_sin_goal, self.goal_mode, self.ptv_ref,
                                density_dir=self.density_dir)

    def experiment_record(self) -> dict:
        rec = super().experiment_record()
        man = json.load(open(self.density_dir / 'manifest.json', encoding='utf-8'))
        rec.update(n_channels=N_CHANNELS_A5, density_dir=str(self.density_dir), density_curve=man['curve'],
                   density_curve_hash=man['curve_hash'], density_pad=DENSITY_PAD, density_masked_by='BODY')
        return rec
