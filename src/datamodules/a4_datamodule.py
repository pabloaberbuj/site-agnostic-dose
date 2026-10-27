"""
DataModule de A4 (Fase 2): A1-uniform (6 canales) + Ch6 = PSDM del PTV (Physical Signed Distance Map).

PSDM portado verbatim de próstata (ProstateDoseProject/data/preprocess.py::calcular_psdm):
  EDT(fuera) - EDT(dentro), espaciado nativo en mm -> /10 (cm) -> /PSDM_NORM_CM (15.0). Positivo afuera, negativo
  adentro, 0 en el borde; sin clip. Un canal por estructura en próstata (PTV, Rectum, Bladder); acá solo el del PTV.

Multi-PTV (nuevo vs. próstata): el PSDM es el de `PTV_High` (default de A1b: la unión cruda mis-ordenaba CyC).
Padding in-plane del canal = 1.0 ("lejos de la estructura", como el PSDM_PAD de próstata), no 0 (= borde del PTV).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.datamodules.a0_datamodule import A0DataModule, A0PatientDataset, DataError, pad_hw_center
from src.datamodules.a1_datamodule import A1DataModule, A1PatientDataset, N_CHANNELS_A1

CH_PSDM = N_CHANNELS_A1          # 6
N_CHANNELS_A4 = N_CHANNELS_A1 + 1
PSDM_NORM_CM = 15.0
PSDM_PAD = 1.0
PSDM_ROLE = 'PTV_High'


def calcular_psdm(mask: np.ndarray, spacing_zyx) -> np.ndarray:
    """Idéntico a calcular_psdm de próstata (mismo orden de operaciones). mask bool (Z,H,W) -> float32 normalizado."""
    m = mask.astype(np.uint8)
    dist_fuera = distance_transform_edt(1 - m, sampling=spacing_zyx)
    dist_dentro = distance_transform_edt(m, sampling=spacing_zyx)
    return (((dist_fuera - dist_dentro) / 10.0) / PSDM_NORM_CM).astype(np.float32)


def psdm_channel(npz_path, pid: str, meta: dict) -> np.ndarray:
    """(1,Z,H,W) float32: PSDM de PTV_High. Compartido por A4 (A1+PSDM) y dist-solo (A0+PSDM)."""
    with np.load(npz_path) as d:
        key = f'mask_{PSDM_ROLE}'
        if key not in d.files:
            raise DataError(f'{pid}: sin {PSDM_ROLE} para el PSDM (roles={[k[5:] for k in d.files if k.startswith("mask_")]})')
        ptv = d[key].astype(bool)
    if not ptv.any():
        raise DataError(f'{pid}: {PSDM_ROLE} vacío')
    spacing = (meta['spacing_z_mm'], meta['spacing_y_mm'], meta['spacing_x_mm'])
    return calcular_psdm(ptv, spacing)[None]


def pad_with_psdm_last(chans: np.ndarray, pad_hw: int) -> np.ndarray:
    """Pad centrado: canales previos con 0, el ÚLTIMO canal (PSDM) con PSDM_PAD."""
    base = pad_hw_center(chans[:-1], pad_hw)
    psdm = pad_hw_center(chans[-1:] - PSDM_PAD, pad_hw) + PSDM_PAD
    return np.concatenate([base, psdm], axis=0)


class A4PatientDataset(A1PatientDataset):
    def load_raw(self, pid: str):
        """(dose, channels (7,Z,H,W) float32, meta) sin pad."""
        dose, chans, meta = super().load_raw(pid)
        return dose, np.concatenate([chans, psdm_channel(self.npz_index[pid], pid, meta)], axis=0), meta

    def pad_channels(self, chans: np.ndarray) -> np.ndarray:
        return pad_with_psdm_last(chans, self.pad_hw)


class A4SoloPatientDataset(A0PatientDataset):
    """HO#9 dist-solo: A0 (3 canales) + Ch3 = PSDM(PTV_High), idéntico al de A4. Sin goals."""
    def load_raw(self, pid: str):
        dose, chans, meta = super().load_raw(pid)
        return dose, np.concatenate([chans, psdm_channel(self.npz_index[pid], pid, meta)], axis=0), meta

    def pad_channels(self, chans: np.ndarray) -> np.ndarray:
        return pad_with_psdm_last(chans, self.pad_hw)


class A4DataModule(A1DataModule):
    def _make_ds(self, ids, index):
        return A4PatientDataset(ids, index, self.table, self.pad_hw, self.fill_sin_goal, self.goal_mode, self.ptv_ref)

    def experiment_record(self) -> dict:
        rec = super().experiment_record()
        rec.update(n_channels=N_CHANNELS_A4, psdm_role=PSDM_ROLE, psdm_norm_cm=PSDM_NORM_CM, psdm_pad=PSDM_PAD)
        return rec


class A4SoloDataModule(A0DataModule):
    def _make_ds(self, ids, index):
        return A4SoloPatientDataset(ids, index, self.table, self.pad_hw)

    def experiment_record(self) -> dict:
        rec = super().experiment_record()
        rec.update(n_channels=4, psdm_role=PSDM_ROLE, psdm_norm_cm=PSDM_NORM_CM, psdm_pad=PSDM_PAD, goals=False)
        return rec
