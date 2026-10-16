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
from .name_resolver import Resolved
from .normalize_structure_name import pick_ptv

_BODY_NAMES = ('BODY', 'EXTERNAL')

# Anulacion de nombre crudo por paciente puntual: casos donde el nombre resuelve BIEN
# segun la regla general, pero ESE paciente especifico tiene una segunda geometria con
# volumen real que es la correcta y la de siempre debe descartarse. No se puede resolver
# via alias/exclude_names (romperia a todos los demas pacientes que usan ese mismo
# nombre normalmente) — es, a proposito, la unica forma de "excepcion por nombre" en el
# pipeline; para "excepcion por paciente completo" usar patient_exceptions.yaml.
_PATIENT_RAW_NAME_OVERRIDES = {
    # PT_c83d88c040c3a900 (Pelvis): dos contornos reales con volumen>0 resolviendo a
    # PTV_High ('PTV_High' 1583cc, 'PTV_High1' 1711cc) — no es duplicado numerado tipico
    # (la regla de duplicado no aplica a PTV de todas formas). Decision del usuario
    # 2026-10-14: el PTV real es 'PTV_High1' (alias agregado en
    # normalize_structure_name.py); 'PTV_High' llano se descarta SOLO para este
    # paciente.
    'PT_c83d88c040c3a900': {'PTV_High'},
    # PT_0e31af2ee9f039e5 (CyC): tiene 'Oral_Cavity' Y 'Cavity_Oral' (alias normal de
    # Oral_Cavity, ver MANUAL_OAR_ALIASES) como dos contornos separados con volumen>0.
    # Decision del usuario 2026-10-15: son identicos, retener 'Oral_Cavity'. NO se puede
    # sacar el alias 'cavity_oral' globalmente: otros 5 pacientes (ej.
    # PT_016b58467d741df0) dependen de el como UNICA fuente de Oral_Cavity (no tienen
    # el nombre llano) y se romperian.
    'PT_0e31af2ee9f039e5': {'Cavity_Oral'},
}


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
    """site_cfg: dict cargado de sites_<site>.yaml.
    exception_entry: (categoria, motivo) de patient_exceptions.yaml, o None. Con CUALQUIER
    categoria (out_of_convention o reviewed_ok) relaja: estructuras desconocidas (se ignoran,
    no fallan) y nivel de PTV declarado no resuelto. Solo category=='out_of_convention' se
    refleja en metadata (reviewed_ok entra al train normalmente)."""
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
    for raw in _PATIENT_RAW_NAME_OVERRIDES.get(patient_id, ()):
        if raw in resolved:
            resolved[raw] = Resolved('exclude', note='anulacion por paciente (ver _PATIENT_RAW_NAME_OVERRIDES)')

    unknown = sorted(n for n, r in resolved.items() if r.outcome == 'unknown')
    warnings = []
    if unknown:
        # Paciente ya revisado/flaggeado (patient_exceptions.yaml, cualquier categoria): un
        # humano ya puso el ojo encima, no tiene sentido re-bloquear por ruido de nombres no
        # relacionado. Sin excepcion registrada, sigue siendo falla ruidosa (CHARTER §5).
        if exception_entry is None:
            raise PipelineError(f'estructuras desconocidas (no canonico/alias/ignore/exclude): {unknown}')
        warnings.append(f'estructuras desconocidas ignoradas (paciente con excepcion registrada): {unknown}')
        resolved = {n: r for n, r in resolved.items() if r.outcome != 'unknown'}

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
            # Desempate: si exactamente uno de los candidatos tiene una marca que lo hace
            # mas explicito/confiable que el resto, gana. 'prv' = geometria del serial
            # (ej. 'SpinalCanal' + 'SpinalCanal_PRV' -> ambos alias a SpinalCord_PRV05, el
            # PRV explicito gana sobre el organo llano). 'corregido' = recontoreo corregido
            # reemplaza al original (ej. 'Rectum' + 'RectumCorregido', decision del usuario
            # 2026-10-14). Si ninguna marca desempata sola, sigue siendo ambiguedad real.
            for marca in ('prv', 'corregido'):
                marcados = [raw for raw in nonzero if marca in raw.lower()]
                if len(marcados) == 1:
                    nonzero = marcados
                    break
            else:
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
        'out_of_convention': exception_entry[1] if exception_entry and exception_entry[0] == 'out_of_convention' else None,
        'qa_flags': [],
        'aliases_applied': aliases_applied,
    }
    return PatientResult(dose_pct=dose_pct, masks=masks, body_bbox_voxels=bbox, metadata=metadata, warnings=warnings)
