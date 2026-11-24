"""
DataModule de la loss cuantílica (Fase 4): A4 idéntico a la vara + item 'ptv_mask' (Z,Hc,Wc) float32 = unión de PTVs con Rx ∩ BODY,
recortada al mismo marco que dose. El input/target NO cambia.
"""
import numpy as np
import torch

from src.datamodules.a4_datamodule import A4DataModule, A4PatientDataset


class F4QPatientDataset(A4PatientDataset):
    def __getitem__(self, idx: int) -> dict:
        pid = self.ids[idx]
        dose, chans, meta = self.load_raw(pid)
        box = self.frame_box(chans)
        x = self.pad_channels(chans).transpose(1, 0, 2, 3)
        y = self.frame.apply(dose, box)
        with np.load(self.npz_index[pid]) as d:
            masks = {k[len('mask_'):]: d[k].astype(bool) for k in d.files if k.startswith('mask_')}
        ratios = self.table.intent_ratios(meta['site'], meta['protocol_id'])
        ptv = np.zeros(masks['BODY'].shape, bool)
        for r in self.table.target_roles:
            if r in masks and r in ratios:
                ptv |= masks[r]
        ptv &= masks['BODY']
        ptv = self.frame.apply(ptv.astype(np.float32), box)
        return {
            'input': torch.from_numpy(np.ascontiguousarray(x)),
            'dose': torch.from_numpy(np.ascontiguousarray(y)),
            'ptv_mask': torch.from_numpy(np.ascontiguousarray(ptv)),
            'patient_id': pid, 'site': meta['site'], 'protocol_id': meta['protocol_id'],
            'rx_high_gy': float(meta['rx_high_gy']),
        }


class F4QDataModule(A4DataModule):
    def _make_ds(self, ids, index):
        return F4QPatientDataset(ids, index, self.table, self.pad_hw, self.fill_sin_goal, self.goal_mode, self.ptv_ref)
