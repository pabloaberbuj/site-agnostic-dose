"""
Carga de DICOM por paciente: CT (solo header, para chequear FrameOfReferenceUID),
RTDOSE (dosis en Gy + geometria de grilla) y RTSTRUCT (contornos crudos por ROI).

Identifica la modalidad leyendo el tag `Modality`, NUNCA por el nombre del archivo
(precedente de bug en el proyecto de prostata: una funcion cargaba la serie RD como
si fuera CT).
"""
import os
import pydicom
import numpy as np

from .geometry import build_dose_grid


class DicomLoadError(Exception):
    pass


def _iter_dicom_files(patient_dir):
    for fn in os.listdir(patient_dir):
        p = os.path.join(patient_dir, fn)
        if os.path.isfile(p) and fn.lower().endswith('.dcm'):
            yield p


def find_by_modality(patient_dir):
    """-> {'CT': [paths...], 'RTDOSE': [path], 'RTSTRUCT': [path], 'RTPLAN': [path]}"""
    by_mod = {}
    for p in _iter_dicom_files(patient_dir):
        ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=['Modality'])
        by_mod.setdefault(ds.Modality, []).append(p)
    return by_mod


def load_ct_frame_uid(patient_dir, by_mod=None):
    by_mod = by_mod or find_by_modality(patient_dir)
    ct_paths = by_mod.get('CT')
    if not ct_paths:
        raise DicomLoadError('no hay serie CT (Modality=CT)')
    ds = pydicom.dcmread(ct_paths[0], stop_before_pixels=True,
                          specific_tags=['FrameOfReferenceUID'])
    return str(ds.FrameOfReferenceUID)


def load_dose(patient_dir, by_mod=None):
    by_mod = by_mod or find_by_modality(patient_dir)
    rd = by_mod.get('RTDOSE')
    if not rd or len(rd) != 1:
        raise DicomLoadError(f'se esperaba 1 RTDOSE, hay {len(rd) if rd else 0}')
    ds = pydicom.dcmread(rd[0])
    if ds.Modality != 'RTDOSE':
        raise DicomLoadError(f'Modality={ds.Modality}, no RTDOSE (chequeo anti-bug carga-por-nombre)')
    grid = build_dose_grid(ds)
    scaling = float(ds.DoseGridScaling)
    dose_gy = ds.pixel_array.astype(np.float64) * scaling
    frame_uid = str(ds.FrameOfReferenceUID)
    return dose_gy, grid, frame_uid


def max_y_field_size_mm(patient_dir, by_mod=None):
    """Tamano maximo del campo en Y (cabeza-pies) sobre todos los beams/control points
    del RTPLAN, mirando el jaw Y (tipo 'Y' o 'ASYMY'). Proxy para distinguir campo
    pelviano estandar de campo extendido/para-aortico (ver patient_exceptions.yaml,
    decision del usuario sobre los casos de Kidney en Pelvis, Paso 2)."""
    by_mod = by_mod or find_by_modality(patient_dir)
    rp = by_mod.get('RTPLAN')
    if not rp:
        raise DicomLoadError('no hay RTPLAN (Modality=RTPLAN)')
    ds = pydicom.dcmread(rp[0])
    if ds.Modality != 'RTPLAN':
        raise DicomLoadError(f'Modality={ds.Modality}, no RTPLAN')

    max_size = 0.0
    for beam in getattr(ds, 'BeamSequence', []):
        for cp in getattr(beam, 'ControlPointSequence', []):
            for bld in getattr(cp, 'BeamLimitingDevicePositionSequence', []) or []:
                if bld.RTBeamLimitingDeviceType in ('Y', 'ASYMY'):
                    y1, y2 = (float(v) for v in bld.LeafJawPositions)
                    max_size = max(max_size, abs(y2 - y1))
    if max_size == 0.0:
        raise DicomLoadError('no se encontro jaw Y/ASYMY en ningun beam del RTPLAN')
    return max_size


def load_structures(patient_dir, by_mod=None):
    """-> (frame_uid, {raw_name: [(z_mm, [(x,y),...]), ...]})
    Un nombre puede tener contornos en varias slices; varios contornos en la misma
    slice (holes / regiones separadas) se preservan como entradas separadas con igual z."""
    by_mod = by_mod or find_by_modality(patient_dir)
    rs = by_mod.get('RTSTRUCT')
    if not rs or len(rs) != 1:
        raise DicomLoadError(f'se esperaba 1 RTSTRUCT, hay {len(rs) if rs else 0}')
    ds = pydicom.dcmread(rs[0])
    if ds.Modality != 'RTSTRUCT':
        raise DicomLoadError(f'Modality={ds.Modality}, no RTSTRUCT')

    roi_name = {int(item.ROINumber): str(item.ROIName) for item in ds.StructureSetROISequence}
    structures = {}
    for citem in getattr(ds, 'ROIContourSequence', []):
        num = int(citem.ReferencedROINumber)
        name = roi_name.get(num)
        if name is None:
            raise DicomLoadError(f'ROIContourSequence referencia ROINumber {num} sin StructureSetROISequence')
        contours = []
        for c in getattr(citem, 'ContourSequence', []):
            geo = getattr(c, 'ContourGeometricType', 'CLOSED_PLANAR')
            if geo != 'CLOSED_PLANAR':
                continue   # puntos/lineas sueltas: no son una region, se ignoran para rasterizar
            data = [float(v) for v in c.ContourData]
            pts = [(data[i], data[i + 1]) for i in range(0, len(data), 3)]
            z = data[2]
            if len(pts) >= 3:
                contours.append((z, pts))
        structures[name] = contours
    frame_uid = str(ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID) \
        if getattr(ds, 'ReferencedFrameOfReferenceSequence', None) else str(ds.FrameOfReferenceUID)
    return frame_uid, structures
