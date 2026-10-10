"""
Orquestacion por paciente: DICOM -> (dosis normalizada, mascaras por rol canonico,
metadata). NO escribe NPZ (eso es cosa de scripts/run_preprocess.py, para poder testear
`process_patient` con datos sinteticos sin tocar disco).

Falla ruidoso (PipelineError) ante: inconsistencia de FrameOfReferenceUID, estructura
desconocida, PTV requerido no resuelto (salvo excepcion registrada), BODY ausente.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .dicom_io import find_by_modality, load_ct_frame_uid, load_dose, load_structures
from .rasterize import rasterize_structure, mask_volume_cm3, contour_volume_cm3
from .dose_metrics import d95_gy, normalize_dose
from .normalize_structure_name import pick_ptv

_BODY_NAMES = ('BODY', 'EXTERNAL')


class PipelineError(Exception):
    pass


@dataclass
class PatientResult:
    dose_pct: np.ndarray
    masks: dict                    # role -> bool array (nz,ny,nx)
    body_bbox_voxels: dict         # {i_min,i_max,j_min,j_max,k_min,k_max} (k=Z,i=Y,j=X... ver nota)
    metadata: dict
    warnings: list = field(default_factory=list)


def _find_body_raw_name(raw_structures):
    hits = [n for n in raw_structures if n.strip().upper().startswith(_BODY_NAMES)]
    non_z = [n for n in hits if not n.strip().upper().startswith('Z')]
    chosen = (non_z or hits)
    return chosen[0] if chosen else None


def _bbox_from_mask(mask):
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise PipelineError('BODY con contorno pero mascara rasterizada vacia')
    kmin, imin, jmin = idx.min(axis=0)
    kmax, imax, jmax = idx.max(axis=0)
    return {'z_min': int(kmin), 'z_max': int(kmax),
            'y_min': int(imin), 'y_max': int(imax),
            'x_min': int(jmin), 'x_max': int(jmax)}


def process_patient(patient_dir, patient_id, site_cfg, resolver, exception_entry=None):
    """site_cfg: dict cargado de sites_<site>.yaml. exception_entry: (categoria, motivo) o None."""
    by_mod = find_by_modality(patient_dir)
    ct_frame = load_ct_frame_uid(patient_dir, by_mod)
    dose_gy, grid, dose_frame = load_dose(patient_dir, by_mod)
    struct_frame, raw_structures = load_structures(patient_dir, by_mod)

    frames = {ct_frame, dose_frame, struct_frame}
    if len(frames) > 1:
        raise PipelineError(f'FrameOfReferenceUID inconsistente: CT={ct_frame} RD={dose_frame} RS={struct_frame}')

    body_raw = _find_body_raw_name(raw_structures)
    if body_raw is None:
        raise PipelineError('no se encontro estructura BODY/EXTERNAL')
    # puede haber mas de un BODY* (ej. 'Body' + 'BODY1' recontoreado): ninguno de los
    # candidatos BODY/EXTERNAL debe pasar por el resolver de nombres (no son OAR/PTV).
    body_like = {n for n in raw_structures if n.strip().upper().startswith(_BODY_NAMES)}

    resolved = {name: resolver.resolve(name) for name in raw_structures if name not in body_like}
    unknown = sorted(n for n, r in resolved.items() if r.outcome == 'unknown')
    if unknown:
        raise PipelineError(f'estructuras desconocidas (no canonico/alias/ignore/exclude): {unknown}')

    aliases_applied = sorted(f'{raw} -> {r.role}' for raw, r in resolved.items()
                              if r.outcome == 'alias' and r.role)

    volumes_by_raw = {}
    for raw_name, r in resolved.items():
        if r.outcome in ('canonical', 'alias'):
            m = rasterize_structure(raw_structures[raw_name], grid)
            volumes_by_raw[raw_name] = (m, mask_volume_cm3(m, grid))

    ptv_report = {}
    masks = {}
    for level in ('high', 'mid', 'low'):
        role = 'PTV_' + level.capitalize()
        candidates = [(raw, vol) for raw, (m, vol) in volumes_by_raw.items() if resolved[raw].role == role]
        if not candidates:
            ptv_report[role] = 'ausente'
            continue
        chosen, motivo = pick_ptv(candidates, level=level)
        if chosen is None:
            ptv_report[role] = motivo
        else:
            ptv_report[role] = 'ok'
            masks[role] = volumes_by_raw[chosen][0]

    declared_levels = site_cfg['protocols'][0]['levels']
    for role in declared_levels:
        if role not in masks and exception_entry is None:
            raise PipelineError(f'{role} no resuelto ({ptv_report.get(role)}) y sin excepcion en patient_exceptions.yaml')
    if 'PTV_High' not in masks:
        raise PipelineError(f'PTV_High no resuelto ({ptv_report.get("PTV_High")}) — requerido siempre')

    oar_raw_by_role = defaultdict(list)
    for raw_name, r in resolved.items():
        if r.outcome in ('canonical', 'alias') and r.role and not r.role.startswith('PTV_'):
            oar_raw_by_role[r.role].append(raw_name)
    for role, raws in oar_raw_by_role.items():
        nonzero = [raw for raw in raws if volumes_by_raw[raw][1] > 0]
        if len(nonzero) > 1:
            raise PipelineError(f'rol OAR {role} ambiguo: {nonzero} todos con volumen>0')
        masks[role] = volumes_by_raw[(nonzero or raws)[0]][0]

    body_mask = rasterize_structure(raw_structures[body_raw], grid)
    bbox = _bbox_from_mask(body_mask)

    d95_high_gy = d95_gy(dose_gy, masks['PTV_High'])
    dose_pct = normalize_dose(dose_gy, d95_high_gy)

    rx_by_level = site_cfg['protocols'][0]['rx_gy']
    metadata = {
        'patient_id': patient_id,
        'protocol_id': site_cfg['protocols'][0]['id'],
        'rx_high_gy': rx_by_level['PTV_High'],
        'rx_by_level_gy': rx_by_level,
        'ptv_levels_present': sorted(r for r in masks if r.startswith('PTV_')),
        'spacing_z_mm': grid.spacing[0], 'spacing_y_mm': grid.spacing[1], 'spacing_x_mm': grid.spacing[2],
        'body_bbox_voxels': bbox,
        'd95_ptv_high_raw_gy': d95_high_gy,
        'out_of_convention': exception_entry[1] if exception_entry else None,
        'qa_flags': [],
        'aliases_applied': aliases_applied,
    }
    return PatientResult(dose_pct=dose_pct, masks=masks, body_bbox_voxels=bbox, metadata=metadata)
