"""
DataModule de A0 (Fase 2): geometría + intención, SIN canales de goals.

Lee NPZ v2+ (npz_v2_*: dosis normalizada + máscaras por rol canónico + mask_BODY, sidecar JSON con
metadata). NO re-normaliza ni re-resuelve roles; solo arma los 3 canales de entrada al cargar:

  Ch0  intención/PTV: cada nivel de PTV pintado con Rx_nivel/Rx_High PRESCRIPTO (tabla vía
       protocol_id; High=1.0, fondo=0). Nunca se deriva de la dosis (leakage).
  Ch1  OAR consolidado: unión binaria de todos los roles OAR (PRV incluidos).
  Ch2  body: máscara binaria del contorno externo.

Sampling por paciente: cada item es el volumen completo de UN paciente, ya en cortes axiales listos
para la U-Net 2D: input (Z, 3, H, W), dose (Z, H, W). Con batch_size=1 nunca se mezclan cortes de
pacientes distintos. Padding in-plane centrado a (pad_hw, pad_hw) sobre la grilla de dosis nativa
(sin resamplear); un paciente más grande que pad_hw falla ruidoso, no se recorta.

La tabla es input versionado: `experiment_record()` devuelve su hash para el config/record del run.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.config.harmonization import BODY_ROLE, HarmonizationTable

CH_INTENT, CH_OAR, CH_BODY = 0, 1, 2
N_CHANNELS = 3


class DataError(Exception):
    pass


def pad_hw_center(a: np.ndarray, pad_hw: int) -> np.ndarray:
    """Pad con ceros de los dos últimos ejes a (pad_hw, pad_hw), centrado."""
    h, w = a.shape[-2:]
    if h > pad_hw or w > pad_hw:
        raise DataError(f'grilla in-plane {h}x{w} excede pad_hw={pad_hw}')
    ph, pw = pad_hw - h, pad_hw - w
    width = [(0, 0)] * (a.ndim - 2) + [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)]
    return np.pad(a, width)


def build_a0_channels(masks: dict, ratios: dict, table: HarmonizationTable, shape, patient_id: str = None) -> np.ndarray:
    """masks: {rol: bool (Z,H,W)} (incluye BODY). Devuelve (3, Z, H, W) float32, sin pad."""
    unknown = sorted(set(masks) - table.known_roles)
    if unknown:
        raise DataError(f'roles desconocidos en el NPZ (no están en la tabla): {unknown}')
    if BODY_ROLE not in masks:
        raise DataError('NPZ sin mask_BODY (¿es un npz_v1? A0 necesita npz_v2+)')

    out = np.zeros((N_CHANNELS,) + tuple(shape), dtype=np.float32)
    for role, m in masks.items():
        if role in table.target_roles:
            if role not in ratios and role == 'PTV_Low' and patient_id in table.inert_ptv_low:
                continue   # nivel no declarado, revisado como idéntico a PTV_High (ver harmonization.py)
            if role not in ratios:
                raise DataError(f'{role} presente en el NPZ pero sin Rx en el protocolo de la tabla')
            # np.maximum: en solape gana el nivel más alto (High=1.0)
            np.maximum(out[CH_INTENT], np.where(m, np.float32(ratios[role]), np.float32(0)), out=out[CH_INTENT])
        elif role in table.oar_roles:
            out[CH_OAR][m] = 1.0
        else:  # BODY
            out[CH_BODY][m] = 1.0
    return out


class A0PatientDataset(Dataset):
    def __init__(self, patient_ids, npz_index: dict, table: HarmonizationTable, pad_hw: int = 256):
        self.ids = list(patient_ids)
        self.npz_index = npz_index
        self.table = table
        self.pad_hw = pad_hw

    def __len__(self):
        return len(self.ids)

    def load_raw(self, pid: str):
        """(dose (Z,H,W) float32, channels (3,Z,H,W) float32, meta) sin pad."""
        path = self.npz_index[pid]
        meta = json.load(open(path.with_suffix('.json'), encoding='utf-8'))
        if meta.get('out_of_convention') is not None:
            raise DataError(f'{pid} es out_of_convention ({meta["out_of_convention"]}): conjunto reservado, no entra a train/val/test')
        with np.load(path) as d:
            dose = np.asarray(d['dose_pct'], dtype=np.float32)
            masks = {k[len('mask_'):]: d[k].astype(bool) for k in d.files if k.startswith('mask_')}
        ratios = self.table.intent_ratios(meta['site'], meta['protocol_id'])
        chans = build_a0_channels(masks, ratios, self.table, dose.shape, pid)
        return dose, chans, meta

    def pad_channels(self, chans: np.ndarray) -> np.ndarray:
        """(C,Z,H,W) -> (C,Z,pad,pad). Único punto de padding de canales (train y eval); las subclases fijan el valor por canal."""
        return pad_hw_center(chans, self.pad_hw)

    def __getitem__(self, idx: int) -> dict:
        pid = self.ids[idx]
        dose, chans, meta = self.load_raw(pid)
        x = self.pad_channels(chans).transpose(1, 0, 2, 3)   # (Z, 3, H, W)
        y = pad_hw_center(dose, self.pad_hw)                           # (Z, H, W)
        return {
            'input': torch.from_numpy(np.ascontiguousarray(x)),
            'dose': torch.from_numpy(np.ascontiguousarray(y)),
            'patient_id': pid,
            'site': meta['site'],
            'protocol_id': meta['protocol_id'],
            'rx_high_gy': float(meta['rx_high_gy']),
        }


class A0DataModule(pl.LightningDataModule):
    def __init__(self, npz_dir, splits_file, harmonization_dir, batch_size: int = 1,
                 num_workers: int = 0, pin_memory: bool = True, pad_hw: int = 256):
        super().__init__()
        self.npz_dir = Path(npz_dir)
        self.splits_file = Path(splits_file)
        self.table = HarmonizationTable(harmonization_dir)
        self.batch_size, self.num_workers, self.pin_memory, self.pad_hw = batch_size, num_workers, pin_memory, pad_hw
        if batch_size > 1:
            # el volumen de cada paciente tiene Z distinto: sin pad en Z no se pueden apilar
            raise NotImplementedError('A0: batch_size>1 requiere pad en Z (no implementado); usar 1')

    def setup(self, stage: Optional[str] = None):
        splits = json.load(open(self.splits_file))
        index = {p.stem: p for p in self.npz_dir.glob('*/*.npz')}
        missing = [p for s in ('train', 'val', 'test') for p in splits[s] if p not in index]
        if missing:
            raise DataError(f'{len(missing)} pacientes del split sin NPZ en {self.npz_dir}: {missing[:5]}')
        mk = lambda s: self._make_ds(splits[s], index)
        self.train_ds, self.val_ds, self.test_ds = mk('train'), mk('val'), mk('test')

    def _make_ds(self, ids, index):
        return A0PatientDataset(ids, index, self.table, self.pad_hw)

    def experiment_record(self) -> dict:
        """Lo que el config/record del experimento debe registrar (CHARTER §5: hash de la tabla)."""
        return {'harmonization_hash': self.table.hash, 'npz_dir': str(self.npz_dir),
                'splits_file': str(self.splits_file), 'pad_hw': self.pad_hw}

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)

    def test_dataloader(self):
        return self._loader(self.test_ds, False)
