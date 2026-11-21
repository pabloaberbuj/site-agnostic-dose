"""
DataModule de Fase 4 (loss de momentos): A4 (7 canales, idéntico a la vara) + por paciente el bitmask de estructuras ∩ BODY.

Estructuras que entran: cada PTV por nivel (rol de target con Rx en el protocolo) + cada OAR con goal en la tabla de
armonización. El tipo de constraint sale de la tabla (`oar_goals(...)[rol][i]['channels']`):
    maximos / volumetrico -> p=10 ; medias -> p=1 ; PTV -> p=1 y p=10.
Un OAR con varios goals acumula sus momentos (unión). Estructuras vacías tras ∩ BODY se omiten. El input/target NO cambia.

Item extra: 'struct_bits' (Z,Hc,Wc) int64 (bit k = estructura k), 'struct_p' (S,len(MOMENTS_P)) bool.
"""
import numpy as np
import torch

from src.datamodules.a0_datamodule import DataError
from src.datamodules.a4_datamodule import A4DataModule, A4PatientDataset
from src.losses.dvh_moments import MAX_STRUCTS, MOMENTS_P

CH_TO_P = {'maximos': 10, 'volumetrico': 10, 'medias': 1}
PTV_P = (1, 10)


def structure_specs(table, site, protocol_id, masks: dict) -> list:
    """[(rol, set_de_p)] en orden determinista (PTV primero, luego OAR alfabético), solo no vacías ∩ BODY."""
    body = masks['BODY']
    ratios = table.intent_ratios(site, protocol_id)
    specs = []
    for r in table.target_roles:
        if r in masks and r in ratios and (masks[r] & body).any():
            specs.append((r, set(PTV_P)))
    goals = table.oar_goals(site, protocol_id)
    for role in sorted(goals):
        if role in masks and (masks[role] & body).any():
            specs.append((role, {CH_TO_P[c] for g in goals[role] for c in g['channels']}))
    if len(specs) > MAX_STRUCTS:
        raise DataError(f'{len(specs)} estructuras > {MAX_STRUCTS} bits')
    return specs


class F4PatientDataset(A4PatientDataset):
    def __getitem__(self, idx: int) -> dict:
        pid = self.ids[idx]
        dose, chans, meta = self.load_raw(pid)
        box = self.frame_box(chans)
        x = self.pad_channels(chans).transpose(1, 0, 2, 3)
        y = self.frame.apply(dose, box)
        with np.load(self.npz_index[pid]) as d:
            masks = {k[len('mask_'):]: d[k].astype(bool) for k in d.files if k.startswith('mask_')}
        specs = structure_specs(self.table, meta['site'], meta['protocol_id'], masks)
        body = masks['BODY']
        bits = np.zeros(body.shape, dtype=np.int64)
        sp = np.zeros((len(specs), len(MOMENTS_P)), dtype=bool)
        for k, (role, ps) in enumerate(specs):
            bits |= (masks[role] & body).astype(np.int64) << k
            for j, p in enumerate(MOMENTS_P):
                sp[k, j] = p in ps
        bits = self.frame.apply(bits, box)
        return {
            'input': torch.from_numpy(np.ascontiguousarray(x)),
            'dose': torch.from_numpy(np.ascontiguousarray(y)),
            'struct_bits': torch.from_numpy(np.ascontiguousarray(bits)),
            'struct_p': torch.from_numpy(sp),
            'patient_id': pid, 'site': meta['site'], 'protocol_id': meta['protocol_id'],
            'rx_high_gy': float(meta['rx_high_gy']),
        }


class F4DataModule(A4DataModule):
    def _make_ds(self, ids, index):
        return F4PatientDataset(ids, index, self.table, self.pad_hw, self.fill_sin_goal, self.goal_mode, self.ptv_ref)

    def experiment_record(self) -> dict:
        rec = super().experiment_record()
        rec.update(f4_moments_p=list(MOMENTS_P), f4_ptv_p=list(PTV_P), f4_oar_p=CH_TO_P)
        return rec
