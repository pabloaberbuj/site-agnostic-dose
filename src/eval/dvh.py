"""
Harness compartido `dvh_curva_completa` (CHARTER §2): red (HO#2) y baseline no-aprendida (HO#2b).

Métrica principal: mean|ΔV(D)| (pp de volumen) por banda de dosis y por estructura, con el eje de dosis
en % de Rx_High (el NPZ ya está normalizado D95(PTV_High)=100%). Bandas como en Fase 1: baja 0-40,
media 40-80 (la crítica para OARs), alta 80-110 (donde vive el error del PTV).
DVH sobre la grilla de dosis NATIVA del NPZ (sin resamplear). Reporte por sitio con IC95 bootstrap
resampleando pacientes dentro de cada sitio.

Convención de predicción de la red: se clipea a 0 y se renormaliza a D95(PTV_High pred)=100% antes del
DVH (mide forma, no sesgo de escala global), igual que Fase 1. La baseline no tiene dosis 3D: se
pasa su curva V(D) directamente.
"""
import numpy as np
import pandas as pd

D_BINS = np.arange(0, 111, 1.0)
BANDS = {'baja_0_40': slice(0, 41), 'media_40_80': slice(41, 81), 'alta_80_110': slice(81, 111)}
PTV_ROLES = ('PTV_High', 'PTV_Mid', 'PTV_Low')
BODY = 'BODY'
D95_TOL_PP = 0.5


def curve_vd(dose, mask, d_bins=D_BINS):
    """V(D) = % del volumen de `mask` con dosis >= D, para cada D en d_bins."""
    vals = dose[mask]
    if vals.size == 0:
        return np.full(len(d_bins), np.nan)
    vals = np.sort(vals)
    return 100.0 * (vals.size - np.searchsorted(vals, d_bins, side='left')) / vals.size


def d95(dose, mask):
    vals = dose[mask]
    return float(np.percentile(vals, 5)) if vals.size else float('nan')


def renorm_pred(pred, ptv_high_mask):
    """clip a 0 y renormaliza a D95(PTV_High pred)=100%. Devuelve (pred_renorm, factor)."""
    pred = np.clip(pred, 0.0, None)
    d = d95(pred, ptv_high_mask)
    f = 100.0 / d if d > 0 else float('nan')
    return pred * f, f


def band_metrics(v_true, v_pred):
    delta = np.asarray(v_pred) - np.asarray(v_true)
    out = {'global': float(np.mean(np.abs(delta)))}
    for name, sl in BANDS.items():
        out[name] = float(np.mean(np.abs(delta[sl])))
    return out


def patient_rows(pid, site, masks, dose_true, dose_pred=None, curves_pred=None):
    """
    Una fila por estructura presente. `masks`: {rol: bool (Z,H,W)} sin pad (volumen nativo).
    Pasar `dose_pred` (red; ya renormalizada) o `curves_pred` {rol: V(D) en D_BINS} (baseline).
    """
    rows = []
    for role, m in masks.items():
        if not m.any():
            continue
        v_true = curve_vd(dose_true, m)
        if dose_pred is not None:
            v_pred = curve_vd(dose_pred, m)
        elif role in (curves_pred or {}):
            v_pred = curves_pred[role]
        else:
            continue   # baseline sin curva para este (sitio, rol)
        rows.append({'patient_id': pid, 'site': site, 'role': role, 'n_voxels': int(m.sum()),
                     'v0_true_ok': bool(np.isclose(v_true[0], 100.0, atol=1e-6)),
                     'monotona_true_ok': bool(np.all(np.diff(v_true) <= 1e-9)),
                     **band_metrics(v_true, v_pred)})
    return rows


METRIC_COLS = ['global'] + list(BANDS)


def with_groups(df):
    """Agrega filas agregadas por paciente: 'PTV_all' (promedio de niveles PTV) y 'OAR_all' (promedio de OARs).
    Idempotente: si df ya trae las filas agregadas, las recalcula en vez de duplicarlas."""
    df = df[~df['role'].isin(('PTV_all', 'OAR_all'))]
    parts = [df]
    for name, sel in (('PTV_all', df['role'].isin(PTV_ROLES)),
                      ('OAR_all', ~df['role'].isin(PTV_ROLES + (BODY,)))):
        g = df[sel].groupby(['patient_id', 'site'], as_index=False)[METRIC_COLS].mean()
        g['role'], g['n_voxels'] = name, 0
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def bootstrap_ci(values, n_boot=1000, seed=42):
    """IC95 percentil de la media, resampleando pacientes (values: 1 valor por paciente)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {'mean': float('nan'), 'ci95': [float('nan')] * 2, 'n': 0, 'std': float('nan'), 'degenerate': True}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    # degenerate: sin varianza entre pacientes (o n<2) -> el IC colapsa a un punto; es honesto pero trivial
    # (p.ej. estructura que no recibe dosis en esa banda), NO una estimación precisa.
    return {'mean': float(v.mean()), 'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            'n': int(v.size), 'std': std, 'degenerate': bool(std == 0.0)}


def summarize(df, n_boot=1000, seed=42):
    """{site: {role: {banda: {mean, ci95, n}}}} con IC95 bootstrap por sitio (df ya con_groups)."""
    out = {}
    for (site, role), g in df.groupby(['site', 'role']):
        out.setdefault(site, {})[role] = {c: bootstrap_ci(g[c].values, n_boot, seed) for c in METRIC_COLS}
    return out


def paired_diff(df_ref, df_new, n_boot=1000, seed=42):
    """
    Diferencia pareada por paciente: ref - new (positivo = `new` mejor que `ref`), con IC95 bootstrap
    resampleando pacientes dentro de cada (sitio, rol). Se usa pacientes presentes en ambos.
    """
    m = df_ref.merge(df_new, on=['patient_id', 'site', 'role'], suffixes=('_ref', '_new'))
    out = {}
    for (site, role), g in m.groupby(['site', 'role']):
        out.setdefault(site, {})[role] = {
            c: bootstrap_ci((g[f'{c}_ref'] - g[f'{c}_new']).values, n_boot, seed) for c in METRIC_COLS}
    return out
