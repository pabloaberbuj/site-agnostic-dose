"""
Etapa 1 / Pasos A+B+C(descriptores): inferencia A4 (ep113) sobre los 432, renormalización a D95(PTV_High)=100%,
vector de resúmenes real/predicho por paciente y descriptores anatómicos (solo geometría). Cache por paciente
(reanudable): results/spinoff_tradeoff/etapa1/cache/<patient_id>.pkl

Uso (desde la raíz del repo):  .venv/Scripts/python.exe scripts/spinoff_tradeoff/s1_extract.py
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403,E402
import common as C  # noqa: E402


def compute_eligible():
    """OARs con rol+goal en sites_<sitio>.yaml y presentes (máscara no vacía) en >=90% de los pacientes del sitio."""
    p = C.OUT / 'eligible.json'
    if p.exists():
        return json.load(open(p, encoding='utf-8'))
    split = C.load_split()
    res = {}
    for site in C.SITES:
        goals = C.site_yaml(site).get('oars', {})
        con_goal = [r for r, g in goals.items() if g]
        pids = sorted(pid for pid in split if (C.NPZ_DIR / site / f'{pid}.npz').exists())
        cnt = {r: 0 for r in con_goal}
        for pid in pids:
            with np.load(C.NPZ_DIR / site / f'{pid}.npz') as z:
                for r in con_goal:
                    k = f'mask_{r}'
                    if k in z.files and z[k].any():
                        cnt[r] += 1
        n = len(pids)
        pres = {r: cnt[r] / n for r in con_goal}
        res[site] = {'n_pacientes': n, 'presencia': pres,
                     'elegibles': [r for r in con_goal if pres[r] >= C.PRESENCIA_MIN],
                     'descartados_por_presencia': {r: round(pres[r], 3) for r in con_goal if pres[r] < C.PRESENCIA_MIN},
                     'con_dmax': {r: C.role_has_dmax_goal(site, r) for r in con_goal if pres[r] >= C.PRESENCIA_MIN}}
    C.OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p, 'w', encoding='utf-8'), indent=2)
    return res


def assign_slots(site, elig, masks, dist_high):
    """{slot: rol}. Bilaterales (_L/_R, ambos elegibles) -> <base>_prox / <base>_dist por distancia mínima al PTV_High.
    Si falta uno de los dos en el paciente, ambos slots quedan sin asignar (NaN)."""
    slots, done = {}, set()
    for r in elig:
        base, side = C.pair_base(r)
        if side is None or f'{base}_L' not in elig or f'{base}_R' not in elig:
            if r in masks and masks[r].any():
                slots[r] = r
            continue
        if base in done:
            continue
        done.add(base)
        l, rr = f'{base}_L', f'{base}_R'
        if not (l in masks and masks[l].any() and rr in masks and masks[rr].any()):
            continue
        dl, dr = float(dist_high[masks[l]].min()), float(dist_high[masks[rr]].min())
        prox, dist = (l, rr) if dl <= dr else (rr, l)
        slots[f'{base}_prox'], slots[f'{base}_dist'] = prox, dist
    return slots


def all_slot_names(elig):
    out, done = [], set()
    for r in elig:
        base, side = C.pair_base(r)
        if side is None or f'{base}_L' not in elig or f'{base}_R' not in elig:
            out.append(r)
        elif base not in done:
            done.add(base)
            out += [f'{base}_prox', f'{base}_dist']
    return out


def summarize(dose, masks, ptv_union, body, meta, slots, elig_dmax, ratios, vox_cc, dist_union, ptv_z):
    """Variables por bloque (valores crudos; el residual se arma como real - pred)."""
    v = {}
    for slot, role in slots.items():
        m = masks[role]
        curve = C.dvh.curve_vd(dose, m)
        for k, x in C.band_means(curve, fine=False).items():
            v[f'OAR|{slot}|dV_{k}'] = x
        for k, x in C.band_means(curve, fine=True).items():
            v[f'OAR|{slot}|dVf_{k}'] = x
        vals = dose[m]
        v[f'OAR|{slot}|dDmean'] = float(vals.mean())
        if elig_dmax.get(role):
            v[f'OAR|{slot}|dDmax'] = C.dose_at_cc(vals, vox_cc)
    for lvl in meta['levels']:
        vals = dose[masks[lvl]]
        d98, d2, d50 = (C.dose_at_vol_pct(vals, 98), C.dose_at_vol_pct(vals, 2), C.dose_at_vol_pct(vals, 50))
        v[f'PTV|{lvl}|D98'], v[f'PTV|{lvl}|D2'], v[f'PTV|{lvl}|D50'] = d98, d2, d50
        v[f'PTV|{lvl}|homog'] = (d2 - d98) / d50 if d50 > 0 else np.nan
        if lvl != 'PTV_High':
            v[f'PTV|{lvl}|D95rel'] = C.dose_at_vol_pct(vals, 95) / ratios[lvl]   # % de la Rx del propio nivel
    bvals = dose[body]
    v['HOT|D0035_BODY'] = C.dose_at_cc(bvals, vox_cc)
    v['HOT|V107_cm3'] = float(((dose > C.HOT_PCT) & body & ~ptv_union).sum() * vox_cc)
    for lo, hi in C.SHELLS_MM:
        sel = body & ~ptv_union & (dist_union > lo) & ((dist_union <= hi) if hi else True)
        name = f'FALL|shell_{lo}_{hi if hi else "inf"}'
        v[name] = float(dose[sel].mean()) if sel.sum() >= 50 else np.nan   # <50 vóxeles: no confiable
    outz = body.copy()
    outz[ptv_z[0]:ptv_z[1] + 1] = False
    v['FALL|body_outZ'] = float(dose[outz].mean()) if outz.sum() >= 50 else np.nan
    v['FALL|body_all'] = float(bvals.mean())
    return v


def geometry(site, masks, slots, ptv_union, ptv_high, body, meta, spacing, vox_cc, dist_high, dist_union, ptv_z, levels):
    g = {'vol_oar_cc': {}, 'frac_in_z': {}, 'ovh_high': {}, 'ovh_union': {}, 'mind_high': {}, 'vol_ptv_cc': {}}
    for slot, role in slots.items():
        m = masks[role]
        g['vol_oar_cc'][slot] = float(m.sum() * vox_cc)
        g['frac_in_z'][slot] = float(m[ptv_z[0]:ptv_z[1] + 1].sum() / m.sum())
        g['mind_high'][slot] = float(dist_high[m].min())
        for key, dm in (('ovh_high', dist_high), ('ovh_union', dist_union)):
            d = dm[m]
            g[key][slot] = np.array([(d <= r).mean() for r in C.R_GRID], dtype=np.float32)
    for lvl in levels:
        g['vol_ptv_cc'][lvl] = float(masks[lvl].sum() * vox_cc)
    ba = body[ptv_z[0]:ptv_z[1] + 1].sum(axis=(1, 2)).mean() * spacing[1] * spacing[2] / 100.0   # cm2
    g['body_area_cm2'] = float(ba)
    if site == 'Prostata':   # candidata A: las 7 features de d3_sonda_vecinos.py, recalculadas desde las máscaras
        rec, bla = masks.get('Rectum'), masks.get('Bladder')
        vp, vr, vb = (float(ptv_high.sum() * vox_cc), float(rec.sum() * vox_cc), float(bla.sum() * vox_cc))
        sr, sb = float((ptv_high & rec).sum() * vox_cc), float((ptv_high & bla).sum() * vox_cc)
        g['f7'] = {'VolPTV_cc': vp, 'VolRectum_cc': vr, 'VolBladder_cc': vb, 'Solap_PTV_Rectum_cc': sr,
                   'Solap_PTV_Bladder_cc': sb, 'overlap_rel_recto': sr / vr, 'overlap_rel_vejiga': sb / vb}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='solo para humo: procesar N pacientes por split')
    a = ap.parse_args()
    import torch
    from omegaconf import OmegaConf
    import eval_a0
    from src.datamodules.a0_datamodule import Frame  # noqa: F401
    from src.eval import ssim as ssim_mod
    from src.models.a0_module import A0Module

    C.CACHE.mkdir(parents=True, exist_ok=True)
    elig = compute_eligible()
    cfg = OmegaConf.load(C.CONFIG)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = None
    t0 = time.time()
    n_done = 0
    for split in ('val', 'test', 'train'):
        dm, ds = eval_a0.get_ds(split, cfg.data)
        assert dm.table.hash == C.HASH_ESPERADO, f'hash armonización {dm.table.hash} != {C.HASH_ESPERADO}'
        ids = ds.ids[:a.limit] if a.limit else ds.ids
        for pid in ids:
            out = C.CACHE / f'{pid}.pkl'
            if out.exists():
                continue
            if model is None:
                model = A0Module.load_from_checkpoint(str(C.CKPT), cfg=cfg, strict=True, map_location='cpu', weights_only=False)
                model.to(dev).eval()
            dose, chans, meta = ds.load_raw(pid)
            site = meta['site']
            Z, H, W = dose.shape
            x = torch.from_numpy(np.ascontiguousarray(ds.pad_channels(chans).transpose(1, 0, 2, 3)))[None].to(dev)
            with torch.no_grad():
                pred = model(x)[0].cpu().numpy()
            box = ds.frame_box(chans)
            pred = ds.frame.undo(pred, box, (H, W))
            masks = eval_a0.masks_from_npz(ds.npz_index[pid])
            body = masks['BODY']
            pred = ssim_mod.postproc_body(pred, body)
            levels = [l for l in C.PTV_LEVELS[site] if l in masks and masks[l].any()]
            predn, scale = C.dvh.renorm_pred(pred, masks['PTV_High'])   # clip a 0 + D95(PTV_High)=100%
            sp = (meta['spacing_z_mm'], meta['spacing_y_mm'], meta['spacing_x_mm'])
            vox_cc = sp[0] * sp[1] * sp[2] / 1000.0
            ratios = ds.table.intent_ratios(site, meta['protocol_id'])
            ptv_union = np.zeros_like(body)
            for l in levels:
                ptv_union |= masks[l]
            ptv_zs = np.flatnonzero(ptv_union.any(axis=(1, 2)))
            ptv_z = (int(ptv_zs[0]), int(ptv_zs[-1]))
            dist_union = C.edt_outside(ptv_union, sp)
            dist_high = dist_union if levels == ['PTV_High'] else C.edt_outside(masks['PTV_High'], sp)
            slots = assign_slots(site, elig[site]['elegibles'], masks, dist_high)
            m_ = {'levels': levels}
            vr = summarize(dose, masks, ptv_union, body, m_, slots, elig[site]['con_dmax'], ratios, vox_cc, dist_union, ptv_z)
            vp = summarize(predn, masks, ptv_union, body, m_, slots, elig[site]['con_dmax'], ratios, vox_cc, dist_union, ptv_z)
            geo = geometry(site, masks, slots, ptv_union, masks['PTV_High'], body, meta, sp, vox_cc, dist_high, dist_union, ptv_z, levels)
            curves = {'real': {}, 'pred': {}}
            for slot, role in slots.items():
                curves['real'][slot] = C.dvh.curve_vd(dose, masks[role]).astype(np.float32)
                curves['pred'][slot] = C.dvh.curve_vd(predn, masks[role]).astype(np.float32)
            # covariable del plan real: D95(PTV_High) en Gy / Rx del protocolo (QA gate de Fase 0)
            d95_cov = float(meta['d95_ptv_high_raw_gy']) / float(meta['rx_high_gy'])
            rec = {'pid': pid, 'site': site, 'split': split, 'scale': float(scale), 'd95_over_rx': d95_cov,
                   'spacing': sp, 'levels': levels, 'slots': slots, 'real': vr, 'pred': vp, 'geom': geo,
                   'curves': curves, 'out_of_body': {r: int((m & ~body).sum()) for r, m in masks.items()
                                                      if r != 'BODY' and (m & ~body).any()}}
            with open(out, 'wb') as f:
                pickle.dump(rec, f)
            n_done += 1
            if n_done % 10 == 0:
                print(f'{n_done} nuevos ({split}) {time.time() - t0:.0f}s', flush=True)
    print('listo', len(list(C.CACHE.glob('PT_*.pkl'))), 'pacientes en cache', flush=True)


if __name__ == '__main__':
    main()
