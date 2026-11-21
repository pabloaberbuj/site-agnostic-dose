"""
Fase 4 — harness de evaluación del track de optimización (loss cuantílica). SOLO EVALUACIÓN.
Ver docs/PROMPT_code_f4_harness_calibracion.md.

Convención (confirmada): predictor del cuantil tau de la dosis (dosis baja = mejor plan).
    calibración = P(dosis_real > dosis_pred)  ->  esperado (1 - tau) en un predictor tau bien calibrado; 0.50 en MAE.
    OVH-percentil = percentil del predicho en la distribución de los valores REALES de los k vecinos anatómicos
                    (mismo sitio, OVH similar); esperado ~ 100*tau para el cuantílico, ~50 para MAE.

Métricas (sobre val; test sellado; por sitio; IC95 bootstrap sobre pacientes):
  1a. calibración DVH-point por (sitio, rol, punto): Dmean (todos los OARs elegibles) y Dmax=D0.035cc (roles con goal Dmax en la tabla).
  1b. calibración vóxel-wise en tejido sano (BODY - PTV; variante dosis>5%; variante unión de OARs), fracción por paciente.
  2.  OVH-percentil por (sitio, rol, punto): vecinos = k-NN en espacio OVH(PTV_High) dentro de train+val (sin test, sin el propio paciente).

Fuente de predicción: --ckpt (+ --config) o --pred-dir (<patient_id>.npy en grilla nativa, % Rx_High, ya sin pad/crop).
Pred = clip>=0 * BODY. --renorm none|ptv95 (ptv95 = D95(PTV_High pred)=100, como el harness de visualización).

Uso (raíz del repo):
  .venv/Scripts/python.exe scripts/f4_harness_calibracion.py --harness-id f4h_mae_control --tau 0.5 \
      --config configs/exp_a4_003_unet2d_psdm_c176x256.yaml --ckpt checkpoints/exp_a4_003_unet2d_psdm_c176x256/epoch=113.ckpt
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'spinoff_tradeoff'))

import common as C  # noqa: E402  (edt_outside, dose_at_cc, R_GRID, load_split, role_has_dmax_goal, site_yaml)
from src.eval import dvh  # noqa: E402

POOL_DIR = ROOT / 'results' / 'f4_harness_pool'          # cache solo-real (independiente de la predicción)
ELIG = ROOT / 'results/spinoff_tradeoff/etapa1/eligible.json'
POINTS = ('Dmean', 'Dmax')
DOSE_FLOOR = 5.0   # %Rx_High para la variante vóxel "dosis relevante"


# ----------------------------------------------------------------------------- utilidades
def load_npz(site, pid):
    p = C.NPZ_DIR / site / f'{pid}.npz'
    with np.load(p) as z:
        dose = np.asarray(z['dose_pct'], dtype=np.float32)
        masks = {k[5:]: z[k].astype(bool) for k in z.files if k.startswith('mask_')}
    meta = json.load(open(p.with_suffix('.json'), encoding='utf-8'))
    sp = (meta['spacing_z_mm'], meta['spacing_y_mm'], meta['spacing_x_mm'])
    return dose, masks, meta, sp


def role_points(dose, m, vox_cc, with_dmax):
    vals = dose[m]
    out = {'Dmean': float(vals.mean())}
    if with_dmax:
        out['Dmax'] = C.dose_at_cc(vals, vox_cc)
    return out


def ovh_curve(dist_high, m):
    d = dist_high[m]
    return np.array([(d <= r).mean() for r in C.R_GRID], dtype=np.float32)


def site_roles(elig, site):
    """{rol: tiene_dmax} de los OARs elegibles (presencia >=90%, con goal) del sitio."""
    return {r: bool(elig[site]['con_dmax'][r]) for r in elig[site]['elegibles']}


def pool_record(site, pid, roles):
    """Real puro: Dmean/Dmax por rol + OVH respecto de PTV_High. Cacheado por paciente."""
    f = POOL_DIR / f'{pid}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    dose, masks, meta, sp = load_npz(site, pid)
    vox_cc = sp[0] * sp[1] * sp[2] / 1000.0
    dist_high = C.edt_outside(masks['PTV_High'], sp)
    rec = {'pid': pid, 'site': site, 'roles': {}}
    for r, has_dmax in roles.items():
        if r in masks and masks[r].any():
            rec['roles'][r] = {'pts': role_points(dose, masks[r], vox_cc, has_dmax), 'ovh': ovh_curve(dist_high, masks[r])}
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(rec, open(f, 'wb'))
    return rec


def build_pool(elig, split):
    """Pool de referencia por sitio: train+val (test sellado)."""
    pool = {s: [] for s in C.SITES}
    for pid, sp in split.items():
        if sp == 'test':
            continue
        for s in C.SITES:
            if (C.NPZ_DIR / s / f'{pid}.npz').exists():
                pool[s].append(pool_record(s, pid, site_roles(elig, s)))
    return pool


def bootstrap_mean(v, n_boot=1000, seed=42):
    r = dvh.bootstrap_ci(np.asarray(v, float), n_boot, seed)
    return r['mean'], r['ci95'][0], r['ci95'][1], r['n']


# ----------------------------------------------------------------------------- predicciones
class ModelSource:
    def __init__(self, config, ckpt, split):
        import torch
        from omegaconf import OmegaConf
        import eval_a0
        from src.models.a0_module import A0Module
        self.torch, self.eval_a0 = torch, eval_a0
        cfg = OmegaConf.load(ROOT / config)
        self.dm, self.ds = eval_a0.get_ds(split, cfg.data)
        assert self.dm.table.hash == C.HASH_ESPERADO, f'hash armonización {self.dm.table.hash} != {C.HASH_ESPERADO}'
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = A0Module.load_from_checkpoint(str(ROOT / ckpt), cfg=cfg, strict=True, map_location='cpu',
                                                   weights_only=False).to(self.dev).eval()
        self.ids = list(self.ds.ids)

    def predict(self, pid, dose_shape):
        torch = self.torch
        dose, chans, meta = self.ds.load_raw(pid)
        Z, H, W = dose.shape
        x = torch.from_numpy(np.ascontiguousarray(self.ds.pad_channels(chans).transpose(1, 0, 2, 3)))[None].to(self.dev)
        with torch.no_grad():
            pred = self.model(x)[0].cpu().numpy()
        return self.ds.frame.undo(pred, self.ds.frame_box(chans), (H, W))


class DirSource:
    def __init__(self, pred_dir, split):
        self.dir = Path(pred_dir)
        self.ids = [pid for pid, sp in C.load_split().items() if sp == split]

    def predict(self, pid, dose_shape):
        return np.load(self.dir / f'{pid}.npy')


# ----------------------------------------------------------------------------- evaluación por paciente
def ovh_percentile(pool_site, role, pt, pid, ovh, pred_val, k):
    cands = [(float(np.abs(rec['roles'][role]['ovh'] - ovh).mean()), rec['roles'][role]['pts'][pt])
             for rec in pool_site if rec['pid'] != pid and role in rec['roles'] and pt in rec['roles'][role]['pts']]
    if len(cands) < k:
        return np.nan, len(cands)
    cands.sort(key=lambda t: t[0])
    v = np.array([c[1] for c in cands[:k]])
    return 100.0 * ((v < pred_val).mean() + 0.5 * (v == pred_val).mean()), k


def eval_patient(pid, pred_raw, elig, pool_site, k, renorm):
    site = next(s for s in C.SITES if (C.NPZ_DIR / s / f'{pid}.npz').exists())
    dose, masks, meta, sp = load_npz(site, pid)
    vox_cc = sp[0] * sp[1] * sp[2] / 1000.0
    body = masks['BODY']
    pred = np.clip(pred_raw, 0.0, None) * body
    factor = 1.0
    if renorm == 'ptv95':
        pred, factor = dvh.renorm_pred(pred, masks['PTV_High'])
    dist_high = C.edt_outside(masks['PTV_High'], sp)
    levels = [l for l in C.PTV_LEVELS[site] if l in masks and masks[l].any()]
    ptv_union = np.zeros_like(body)
    for l in levels:
        ptv_union |= masks[l]
    pts_rows, ovh_rows = [], []
    for r, has_dmax in site_roles(elig, site).items():
        if r not in masks or not masks[r].any():
            continue
        real_p = role_points(dose, masks[r], vox_cc, has_dmax)
        pred_p = role_points(pred, masks[r], vox_cc, has_dmax)
        ovh = ovh_curve(dist_high, masks[r])
        for pt in real_p:
            primary = (pt == 'Dmax') if has_dmax else (pt == 'Dmean')
            pts_rows.append({'patient_id': pid, 'site': site, 'role': r, 'point': pt, 'primary': primary,
                             'real': real_p[pt], 'pred': pred_p[pt], 'real_gt_pred': float(real_p[pt] > pred_p[pt])})
            pct, nn = ovh_percentile(pool_site[site], r, pt, pid, ovh, pred_p[pt], k)
            ovh_rows.append({'patient_id': pid, 'site': site, 'role': r, 'point': pt, 'primary': primary,
                             'pred': pred_p[pt], 'ovh_pct': pct, 'k': nn})
    # vóxel-wise
    healthy = body & ~ptv_union
    oars = np.zeros_like(body)
    for r in site_roles(elig, site):
        if r in masks:
            oars |= masks[r]
    oars &= ~ptv_union
    vox = {}
    for name, sel in (('tejido_sano', healthy), ('tejido_sano_dosis5', healthy & (np.maximum(dose, pred) > DOSE_FLOOR)),
                      ('oars_union', oars)):
        n = int(sel.sum())
        vox[name] = float((dose[sel] > pred[sel]).mean()) if n else np.nan
        vox[name + '_n'] = n
    vox_row = {'patient_id': pid, 'site': site, 'factor_renorm': factor, **vox}
    return pts_rows, ovh_rows, vox_row


# ----------------------------------------------------------------------------- agregación
def aggregate(df, value, group, extra=None):
    rows = []
    for key, g in df.groupby(group):
        m, lo, hi, n = bootstrap_mean(g[value].values)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group, key)), 'mean': m, 'ci_lo': lo, 'ci_hi': hi, 'n': n, **(extra or {})})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--harness-id', required=True)
    ap.add_argument('--tau', type=float, default=0.5, help='cuantil de entrenamiento (0.5 = MAE/mediana)')
    ap.add_argument('--split', default='val', choices=['val'])   # test sellado hasta cerrar Fase 4
    ap.add_argument('--config', default='configs/exp_a4_003_unet2d_psdm_c176x256.yaml')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--pred-dir', default=None)
    ap.add_argument('--k', type=int, default=10, help='vecinos OVH (nivel 2)')
    ap.add_argument('--k-sens', type=int, nargs='*', default=[5, 20], help='sensibilidad de k (solo percentil OVH)')
    ap.add_argument('--renorm', default='none', choices=['none', 'ptv95'])
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    assert (a.ckpt is None) != (a.pred_dir is None), 'pasar --ckpt o --pred-dir'

    elig = json.load(open(ELIG, encoding='utf-8'))
    split = C.load_split()
    print('pool de referencia (train+val)...', flush=True)
    pool = build_pool(elig, split)
    print({s: len(v) for s, v in pool.items()}, flush=True)

    src = ModelSource(a.config, a.ckpt, a.split) if a.ckpt else DirSource(a.pred_dir, a.split)
    ids = src.ids[:a.limit] if a.limit else src.ids
    pts, ovh, vox = [], [], []
    for i, pid in enumerate(ids):
        pred = src.predict(pid, None)
        p, o, v = eval_patient(pid, pred, elig, pool, a.k, a.renorm)
        pts += p
        ovh += o
        vox.append(v)
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(ids)}', flush=True)

    # sensibilidad de k: recomputar percentiles
    sens_rows = []
    if a.k_sens:
        pool_by_pid = {s: pool[s] for s in C.SITES}
        for k in a.k_sens:
            for row in ovh:
                dose, masks, meta, sp = load_npz(row['site'], row['patient_id'])
                dist_high = C.edt_outside(masks['PTV_High'], sp)
                ovh_c = ovh_curve(dist_high, masks[row['role']])
                pct, _ = ovh_percentile(pool_by_pid[row['site']], row['role'], row['point'], row['patient_id'], ovh_c, row['pred'], k)
                sens_rows.append({**row, 'ovh_pct': pct, 'k': k})

    df_pts, df_ovh, df_vox = pd.DataFrame(pts), pd.DataFrame(ovh), pd.DataFrame(vox)
    out = ROOT / 'results' / a.harness_id
    out.mkdir(parents=True, exist_ok=True)
    df_pts.to_csv(out / 'per_patient_dvhpoint.csv', index=False)
    df_ovh.to_csv(out / 'per_patient_ovh.csv', index=False)
    df_vox.to_csv(out / 'per_patient_voxel.csv', index=False)

    cal_pt = aggregate(df_pts, 'real_gt_pred', ['site', 'role', 'point'])
    cal_pt['primary'] = cal_pt.apply(lambda r: bool(df_pts[(df_pts.site == r.site) & (df_pts.role == r.role) &
                                                           (df_pts.point == r.point)].primary.iloc[0]), axis=1)
    cal_pt['metric'] = 'calib_dvhpoint'
    cal_pt['target'] = 1 - a.tau
    cal_site = aggregate(df_pts[df_pts.primary].groupby(['site', 'patient_id'], as_index=False).real_gt_pred.mean(), 'real_gt_pred', ['site'])   # 1 valor por paciente
    cal_site['metric'], cal_site['target'] = 'calib_dvhpoint_primarios_sitio', 1 - a.tau
    vox_rows = []
    for name in ('tejido_sano', 'tejido_sano_dosis5', 'oars_union'):
        t = aggregate(df_vox, name, ['site'])
        t['metric'], t['variant'], t['target'] = 'calib_voxel', name, 1 - a.tau
        vox_rows.append(t)
    ovh_pt = aggregate(df_ovh.dropna(subset=['ovh_pct']), 'ovh_pct', ['site', 'role', 'point'])
    ovh_pt['metric'], ovh_pt['k'], ovh_pt['target'] = 'ovh_percentil', a.k, 100 * a.tau
    ovh_site = aggregate(df_ovh[df_ovh.primary].dropna(subset=['ovh_pct']).groupby(['site', 'patient_id'], as_index=False).ovh_pct.mean(), 'ovh_pct', ['site'])
    ovh_site['metric'], ovh_site['k'], ovh_site['target'] = 'ovh_percentil_primarios_sitio', a.k, 100 * a.tau
    sens = []
    if sens_rows:
        ds = pd.DataFrame(sens_rows)
        ds = ds[ds.primary].dropna(subset=['ovh_pct'])
        for k, g in ds.groupby('k'):
            g = g.groupby(['site', 'patient_id'], as_index=False).ovh_pct.mean()
            t = aggregate(g, 'ovh_pct', ['site'])
            t['metric'], t['k'], t['target'] = 'ovh_percentil_primarios_sitio_sens_k', k, 100 * a.tau
            sens.append(t)
    metrics = pd.concat([cal_pt, cal_site, *vox_rows, ovh_pt, ovh_site, *sens], ignore_index=True)
    metrics.to_csv(out / 'metrics.csv', index=False)

    def tol_ok(r, tol):
        return bool(r['ci_lo'] - tol <= r['target'] <= r['ci_hi'] + tol)
    sanity = {}
    if abs(a.tau - 0.5) < 1e-9:
        for _, r in cal_site.iterrows():
            sanity[f'calib_dvhpoint_{r.site}'] = {'mean': r['mean'], 'ci': [r.ci_lo, r.ci_hi], 'ok_0.50': tol_ok(r, 0.0)}
        for name in ('tejido_sano', 'tejido_sano_dosis5'):
            for _, r in aggregate(df_vox, name, ['site']).iterrows():
                r['target'] = 0.5
                sanity[f'calib_voxel_{name}_{r.site}'] = {'mean': r['mean'], 'ci': [r.ci_lo, r.ci_hi], 'ok_0.50': tol_ok(r, 0.0)}
        for _, r in ovh_site.iterrows():
            sanity[f'ovh_pct_{r.site}'] = {'mean': r['mean'], 'ci': [r.ci_lo, r.ci_hi], 'ok_p50': tol_ok(r, 0.0)}
    summary = {'harness_id': a.harness_id, 'split': a.split, 'tau': a.tau, 'convencion': 'calib = P(real > pred), esperado 1-tau; OVH-pct esperado 100*tau',
               'fuente': {'ckpt': a.ckpt, 'config': a.config, 'pred_dir': a.pred_dir}, 'renorm': a.renorm,
               'k_ovh': a.k, 'k_sens': a.k_sens, 'pool_referencia': 'train+val (test sellado), sin el propio paciente',
               'n_val_por_sitio': df_vox.groupby('site').size().to_dict(),
               'pool_n_por_sitio': {s: len(v) for s, v in pool.items()},
               'puntos': {'Dmean': 'todos los OARs elegibles (presencia>=90%, con goal)', 'Dmax': 'D0.035cc, solo roles con goal Dmax en la tabla; primario = Dmax si hay goal Dmax, si no Dmean'},
               'voxel_variantes': {'tejido_sano': 'BODY - union PTV', 'tejido_sano_dosis5': 'idem y max(real,pred)>5%Rx',
                                   'oars_union': 'union de OARs elegibles - PTV'},
               'ic': 'bootstrap percentil 1000, seed 42, sobre pacientes',
               'calib_dvhpoint_por_sitio_primarios': cal_site.to_dict('records'),
               'calib_voxel': pd.concat(vox_rows).to_dict('records'),
               'ovh_percentil_por_sitio_primarios': ovh_site.to_dict('records'),
               'sanity_mae': sanity or None}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')

    print(f'\n=== {a.harness_id} (tau={a.tau}, renorm={a.renorm}, k={a.k}) ===')
    print('calib DVH-point (primarios) esperado', 1 - a.tau)
    print(cal_site[['site', 'mean', 'ci_lo', 'ci_hi', 'n']].round(3).to_string(index=False))
    for t in vox_rows:
        print('calib voxel', t.variant.iloc[0]); print(t[['site', 'mean', 'ci_lo', 'ci_hi', 'n']].round(3).to_string(index=False))
    print('OVH-percentil (primarios) esperado', 100 * a.tau)
    print(ovh_site[['site', 'mean', 'ci_lo', 'ci_hi', 'n']].round(1).to_string(index=False))


if __name__ == '__main__':
    main()
