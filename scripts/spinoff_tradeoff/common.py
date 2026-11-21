"""
Spin-off trade-off, Etapa 1 — utilidades compartidas (sin entrenar nada, sin tocar configs/splits/tabla).

Convención: residual = real - predicho. Dosis en % de Rx_High (el NPZ ya viene con D95(PTV_High)=100%).
Curvas V(D) con las MISMAS funciones del harness dvh_curva_completa (src/eval/dvh.py): curve_vd / renorm_pred.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from src.eval import dvh  # noqa: E402

OUT = ROOT / 'results' / 'spinoff_tradeoff' / 'etapa1'
CACHE = OUT / 'cache'
TABLAS = OUT / 'tablas'
GRAFICOS = OUT / 'graficos'
NPZ_DIR = ROOT / '../../data/npz_v2b_body'
SPLIT_FILE = ROOT / 'data/splits/split_v2.json'
HARM_DIR = ROOT / 'data/harmonization'
CONFIG = ROOT / 'configs/exp_a4_003_unet2d_psdm_c176x256.yaml'
CKPT = ROOT / 'checkpoints/exp_a4_003_unet2d_psdm_c176x256/epoch=113.ckpt'
HASH_ESPERADO = '7ab16f73fa632a3e'

SITES = ('Prostata', 'Pelvis', 'CyC')
MANUALES = ('Prostata', 'Pelvis')
PTV_LEVELS = {'Prostata': ['PTV_High'], 'Pelvis': ['PTV_High'], 'CyC': ['PTV_High', 'PTV_Mid', 'PTV_Low']}
SITE_YAML = {'Prostata': 'sites_prostata.yaml', 'Pelvis': 'sites_pelvis.yaml', 'CyC': 'sites_cyc.yaml'}
PRESENCIA_MIN = 0.90

D_BINS = dvh.D_BINS                       # 0..110 paso 1 (%Rx_High)
# bandas de dosis (%Rx_High). Semiabiertas [lo,hi) salvo la última, que incluye hi (=100).
BANDS_COMPACT = {'baja_5_40': (5, 40), 'media_40_80': (40, 80), 'alta_80_100': (80, 100)}
BANDS_FINE = {'f5_20': (5, 20), 'f20_40': (20, 40), 'f40_60': (40, 60), 'f60_80': (60, 80), 'f80_100': (80, 100)}
SHELLS_MM = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, None)]
R_GRID = np.arange(0.0, 150.1, 2.5)       # mm, OVH
HOT_PCT = 107.0
D0035_CC = 0.035

UNITS = {'dV': 'pp', 'dose': 'pct', 'cm3': 'cm3', 'ratio': 'ratio'}


def band_idx(lo, hi, last):
    """Índices de D_BINS para la banda [lo,hi) (o [lo,hi] si last)."""
    d = D_BINS
    return np.flatnonzero((d >= lo) & ((d <= hi) if last else (d < hi)))


def band_means(curve, fine=False):
    bands = BANDS_FINE if fine else BANDS_COMPACT
    keys = list(bands)
    return {k: float(np.mean(curve[band_idx(*bands[k], last=(i == len(keys) - 1))])) for i, k in enumerate(keys)}


def load_split():
    s = json.load(open(SPLIT_FILE))
    return {pid: sp for sp in ('train', 'val', 'test') for pid in s[sp]}


def site_yaml(site):
    return yaml.safe_load(open(HARM_DIR / SITE_YAML[site], encoding='utf-8'))


def role_has_dmax_goal(site, role):
    return any(g['metric']['type'] == 'Dmax' for g in site_yaml(site).get('oars', {}).get(role, []))


def pair_base(role):
    """'Parotid_L' -> ('Parotid', 'L'); sin lateralidad -> (role, None)."""
    if role.endswith('_L') or role.endswith('_R'):
        return role[:-2], role[-1]
    return role, None


def dose_at_vol_pct(vals, v_pct):
    """D(v%): dosis cubierta por el v% del volumen (percentil 100-v%), igual que d95 del harness."""
    return float(np.percentile(vals, 100.0 - v_pct))


def dose_at_cc(vals, voxel_cc, cc=D0035_CC):
    """D0.035cm3: dosis que recibe el 'cc' más caliente de la estructura (si V<cc, el máximo)."""
    v_cc = vals.size * voxel_cc
    frac = min(cc / v_cc, 1.0)
    return float(np.percentile(vals, 100.0 * (1.0 - frac)))


def edt_outside(mask, spacing_zyx):
    """Distancia (mm) de cada vóxel a la máscara (0 dentro)."""
    return distance_transform_edt(~mask, sampling=spacing_zyx).astype(np.float32)


def load_eligible():
    return json.load(open(OUT / 'eligible.json', encoding='utf-8'))


def load_cache(sites=SITES):
    out = []
    for p in sorted(CACHE.glob('PT_*.pkl')):
        with open(p, 'rb') as f:
            d = pickle.load(f)
        if d['site'] in sites:
            out.append(d)
    return out


def block_of(var):
    return var.split('|')[0]


def unit_of(var):
    n = var.split('|')[-1]
    if n.startswith('dV'):
        return 'pp'
    if n == 'V107_cm3':
        return 'cm3'
    if n == 'homog':
        return 'ratio'
    return 'pct'


# signo de "calidad" (>0: que el residual real-pred sea positivo es MEJOR; <0: es PEOR; 0: ambiguo, no orienta)
def quality_sign(var):
    b, n = block_of(var), var.split('|')[-1]
    if b == 'OAR':
        return -1          # más dosis/volumen en el real que en el predicho = OAR peor
    if b == 'PTV':
        if n in ('D98', 'D95rel'):
            return +1
        if n in ('D2', 'homog'):
            return -1
        return 0           # D50: ambiguo
    return -1              # HOT y FALL: más dosis fuera del PTV = peor
