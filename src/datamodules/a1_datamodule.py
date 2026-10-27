"""
DataModule de A1 (Fase 2): A0 + 3 canales de goals de OAR, construidos al cargar desde la tabla.

  Ch0 intención PTV · Ch1 OAR consolidado · Ch2 body   (idénticos a A0, se reusan)
  Ch3 máximos      techo Dmax (+ D2%-tipo con channels: [.., maximos]), uniforme sobre el OAR
  Ch4 medias       Dmean, uniforme sobre el OAR
  Ch5 volumétrico  techo pintado SOLO en la fracción preservada (distal al PTV por EDT)

Todo en fracción de Rx_High (1.0 = Rx_High, como Ch0). Reglas (HO#4):
  - Varios goals del mismo tipo/solape en un canal: mínimo (techo más estricto).
  - Volumétrico: escalera-min. Cada vóxel toma el techo más estricto entre los V/Dx cuya fracción
    preservada (1-budget, por ranking de distancia al PTV, distal primero) lo incluye. El resto -> fill.
  - Límites en cm³: budget = cm³/vol real del OAR; si budget >= 1 (constraint trivial) no se pinta.
  - Empates de distancia (p.ej. overlap con PTV, d=0): orden de índice del vóxel (argsort estable) -> determinista.
  - fill_sin_goal: valor de todo vóxel de OAR sin goal de ese tipo en ese canal (incluye la fracción
    sacrificada). Fuera de OARs siempre 0.
  - Contradicción PTV<->OAR en overlap: se pintan ambas, sin safeguard (la resuelve el target / A3).
"""
import json
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.datamodules.a0_datamodule import A0DataModule, A0PatientDataset, DataError, build_a0_channels

CH_MAX, CH_MEAN, CH_VOL = 3, 4, 5
N_CHANNELS_A1 = 6
FILL_VALUES = {0: 0.0, '0': 0.0, '95pct_rx': 0.95}
GOAL_MODES = ('uniform', 'distance_modulated')


def fill_value(fill) -> float:
    if fill not in FILL_VALUES:
        raise DataError(f'fill_sin_goal {fill!r} inválido (opciones: {list(FILL_VALUES)})')
    return FILL_VALUES[fill]


def check_goal_mode(goal_mode):
    if goal_mode not in GOAL_MODES:
        raise DataError(f'goal_mode {goal_mode!r} inválido (opciones: {GOAL_MODES})')


# A1b (distance_modulated): w = max_nivel r_nivel * exp(-d_nivel/LAMBDA_MM) en [0,1] (cerca de un PTV de dosis alta -> 1);
# Ch3/Ch4 = techo * (W_MIN + (1-W_MIN)*w). W_MIN>0: un vóxel con goal nunca cae a 0 (0 = "sin goal" con fill 0).
# Ch5 conserva el valor del techo y ordena la fracción preservada por w (distal = w menor) en vez de la EDT a la unión.
LAMBDA_MM = 20.0
W_MIN = 0.5


def proximity_weight(masks: dict, ratios: dict, target_roles, spacing_zyx, lam: float = LAMBDA_MM) -> np.ndarray:
    """w (Z,H,W) float32 en [0,1]: max sobre niveles de PTV de ratio_nivel*exp(-d_nivel/lam). Distancia en mm, espaciado nativo."""
    shape = next(iter(masks.values())).shape
    w = np.zeros(shape, np.float32)
    roles = [r for r in target_roles if r in masks and r in ratios]   # PTV_Low inerte (sin Rx) no aporta
    if not roles:
        raise DataError(f'sin PTV con Rx para la modulación (roles en NPZ={sorted(masks)})')
    for r in roles:
        d = distance_transform_edt(~masks[r], sampling=spacing_zyx)
        np.maximum(w, (ratios[r] * np.exp(-d / lam)).astype(np.float32), out=w)
    return w


def ptv_distance(masks: dict, target_roles, spacing_zyx, ptv_ref: str = 'union') -> np.ndarray:
    """Distancia (mm) de cada vóxel al PTV. `union` = unión de PTVs; `high` = solo PTV_High (fallback CyC)."""
    roles = ['PTV_High'] if ptv_ref == 'high' else [r for r in target_roles if r in masks]
    if not roles or any(r not in masks for r in roles):
        raise DataError(f'sin PTV para la distancia (ptv_ref={ptv_ref}, roles en NPZ={sorted(masks)})')
    ptv = np.zeros(next(iter(masks.values())).shape, bool)
    for r in roles:
        ptv |= masks[r]
    return distance_transform_edt(~ptv, sampling=spacing_zyx)


def build_goal_channels(masks: dict, goals: dict, table, spacing_zyx, fill: float, ptv_ref: str = 'union',
                        edt: Optional[np.ndarray] = None, goal_mode: str = 'uniform', ratios: Optional[dict] = None):
    """Devuelve ((3, Z, H, W) float32 [max, medias, vol], info). Sin pad. goal_mode=distance_modulated requiere `ratios`."""
    modulated = goal_mode == 'distance_modulated'
    if modulated:
        if ratios is None:
            raise DataError('distance_modulated requiere ratios (Rx_nivel/Rx_High)')
        w = proximity_weight(masks, ratios, table.target_roles, spacing_zyx)
        mod = np.float32(W_MIN) + np.float32(1.0 - W_MIN) * w
        edt = None   # el orden de Ch5 sale de w, no de la EDT a la unión
    shape = next(iter(masks.values())).shape
    vox_cm3 = float(np.prod(spacing_zyx)) / 1000.0
    val = np.full((3,) + tuple(shape), np.inf, dtype=np.float32)
    info = {'trivial_skipped': [], 'goals_sin_mascara': []}
    union_oar = np.zeros(shape, bool)
    for r, m in masks.items():
        if r in table.oar_roles:
            union_oar |= m

    need_edt = (not modulated) and any('volumetrico' in g['channels'] for role, gl in goals.items() if role in masks for g in gl)
    if need_edt and edt is None:
        edt = ptv_distance(masks, table.target_roles, spacing_zyx, ptv_ref)

    for role, gl in goals.items():
        if role not in masks:
            info['goals_sin_mascara'].append(role)   # p.ej. PenileBulb en Pelvis (entrada muerta)
            continue
        m = masks[role]
        for g in gl:
            if 'maximos' in g['channels']:
                val[0][m] = np.minimum(val[0][m], g['ceiling'])
            if 'medias' in g['channels']:
                val[1][m] = np.minimum(val[1][m], g['ceiling'])
        vol_goals = [g for g in gl if 'volumetrico' in g['channels']]
        idx = np.flatnonzero(m)
        n = idx.size
        if not vol_goals or n == 0:
            continue
        key = w.ravel()[idx] if modulated else -edt.ravel()[idx]   # distal primero (w menor / EDT mayor); empates por índice
        order = np.argsort(key, kind='stable')
        ladder = np.full(n, np.inf, dtype=np.float32)
        for g in vol_goals:
            b = g['budget_frac'] if g['budget_frac'] is not None else g['budget_cm3'] / (n * vox_cm3)
            if b >= 1.0:
                info['trivial_skipped'].append(g['label'])
                continue
            k = int(round((1.0 - b) * n))
            ladder[:k] = np.minimum(ladder[:k], g['ceiling'])
        flat = val[2].reshape(-1)
        tgt = idx[order]
        flat[tgt] = np.minimum(flat[tgt], ladder)

    if modulated:
        val[:2] *= mod   # inf * mod = inf: los vóxeles sin goal siguen yendo a fill
    out = np.where(np.isfinite(val), val, np.where(union_oar, np.float32(fill), np.float32(0))).astype(np.float32)
    return out, info


class A1PatientDataset(A0PatientDataset):
    def __init__(self, patient_ids, npz_index, table, pad_hw=256, fill_sin_goal=0, goal_mode='uniform', ptv_ref='union'):
        super().__init__(patient_ids, npz_index, table, pad_hw)
        check_goal_mode(goal_mode)
        if ptv_ref not in ('union', 'high'):
            raise DataError(f'ptv_ref {ptv_ref!r} inválido')
        self.fill_sin_goal, self.goal_mode, self.ptv_ref = fill_sin_goal, goal_mode, ptv_ref
        self.fill = fill_value(fill_sin_goal)

    def load_raw(self, pid: str):
        """(dose (Z,H,W), channels (6,Z,H,W) float32, meta) sin pad. meta['_goal_info'] = diagnóstico."""
        path = self.npz_index[pid]
        meta = json.load(open(path.with_suffix('.json'), encoding='utf-8'))
        if meta.get('out_of_convention') is not None:
            raise DataError(f'{pid} es out_of_convention ({meta["out_of_convention"]}): conjunto reservado, no entra a train/val/test')
        with np.load(path) as d:
            dose = np.asarray(d['dose_pct'], dtype=np.float32)
            masks = {k[len('mask_'):]: d[k].astype(bool) for k in d.files if k.startswith('mask_')}
        ratios = self.table.intent_ratios(meta['site'], meta['protocol_id'])
        base = build_a0_channels(masks, ratios, self.table, dose.shape, pid)
        goals = self.table.oar_goals(meta['site'], meta['protocol_id'])
        spacing = (meta['spacing_z_mm'], meta['spacing_y_mm'], meta['spacing_x_mm'])
        gch, info = build_goal_channels(masks, goals, self.table, spacing, self.fill, self.ptv_ref,
                                        goal_mode=self.goal_mode, ratios=ratios)
        meta['_goal_info'] = info
        return dose, np.concatenate([base, gch], axis=0), meta


class A1DataModule(A0DataModule):
    def __init__(self, *args, fill_sin_goal=0, goal_mode='uniform', ptv_ref='union', **kw):
        super().__init__(*args, **kw)
        check_goal_mode(goal_mode)
        fill_value(fill_sin_goal)
        self.fill_sin_goal, self.goal_mode, self.ptv_ref = fill_sin_goal, goal_mode, ptv_ref

    def _make_ds(self, ids, index):
        return A1PatientDataset(ids, index, self.table, self.pad_hw, self.fill_sin_goal, self.goal_mode, self.ptv_ref)

    def experiment_record(self) -> dict:
        rec = super().experiment_record()
        if self.goal_mode == 'distance_modulated':
            rec.update(mod_lambda_mm=LAMBDA_MM, mod_w_min=W_MIN)
        rec.update(fill_sin_goal=self.fill_sin_goal, goal_mode=self.goal_mode, ptv_ref=self.ptv_ref,
                   n_channels=N_CHANNELS_A1, conditional_diferidos_a_A3=self.table.conditional_goals())
        return rec
