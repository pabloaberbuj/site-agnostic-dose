"""
Etapa 1b / extracción: residual en la normalización ORIGINAL del plan (real = dosis del NPZ x D95_abs(PTV_High)/Rx_High).
Reusa de la Etapa 1: slots, predicho renormalizado (vars cacheadas), métrica B. Sin GPU.

  --phase real : recalcula el vector REAL reescalado para los 432 (solo NPZ). Sanity del reescalado vs QA gate de Fase 0.
  --phase pred : inferencia A4 en CPU SOLO para val+test (152) para el único dato del predicho que no estaba cacheado:
                 hotspot fuera de PTV (D0.035cm3 de BODY - union PTV). Verifica que reproduce el factor de escala cacheado.
Cache: results/spinoff_tradeoff/etapa1b/cache/<pid>.pkl
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import s1_extract as S1  # noqa: E402

OUT_B = C.ROOT / 'results' / 'spinoff_tradeoff' / 'etapa1b'
CACHE_B = OUT_B / 'cache'


def geom_for(rec, masks):
    levels = rec['levels']
    body = masks['BODY']
    ptv_union = np.zeros_like(body)
    for l in levels:
        ptv_union |= masks[l]
    zs = np.flatnonzero(ptv_union.any(axis=(1, 2)))
    sp = rec['spacing']
    return body, ptv_union, (int(zs[0]), int(zs[-1])), C.edt_outside(ptv_union, sp), sp[0] * sp[1] * sp[2] / 1000.0


def summarize_b(dose, masks, rec, elig, ratios, geo):
    body, ptv_union, ptv_z, dist_union, vox_cc = geo
    v = S1.summarize(dose, masks, ptv_union, body, {'levels': rec['levels']}, rec['slots'], elig[rec['site']]['con_dmax'],
                     ratios, vox_cc, dist_union, ptv_z)
    v.pop('HOT|D0035_BODY', None)                                   # Etapa 1: caía dentro del PTV (corr 0.92-0.94 con PTV)
    out = body & ~ptv_union                                         # hotspot FUERA de los PTV
    v['HOT|D0035_out'] = C.dose_at_cc(dose[out], vox_cc)
    v['PTV|PTV_High|D95'] = C.dose_at_vol_pct(dose[masks['PTV_High']], 95)
    return v


def phase_real(a):
    from src.config.harmonization import HarmonizationTable
    table = HarmonizationTable(C.HARM_DIR)
    assert table.hash == C.HASH_ESPERADO
    elig = C.load_eligible()
    qa = pd.read_csv(C.ROOT / 'results/auditorias/paso3_qa_gate.csv')
    qa = qa[qa.level == 'PTV_High'].set_index('patient_id')
    CACHE_B.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, p1 in enumerate(sorted(C.CACHE.glob('PT_*.pkl'))):
        rec = pickle.load(open(p1, 'rb'))
        pid, site = rec['pid'], rec['site']
        out = CACHE_B / f'{pid}.pkl'
        npz = C.NPZ_DIR / site / f'{pid}.npz'
        meta = json.load(open(npz.with_suffix('.json'), encoding='utf-8'))
        k = float(meta['d95_ptv_high_raw_gy']) / float(meta['rx_high_gy'])     # D95_abs / Rx_High
        with np.load(npz) as z:
            dose_n = np.asarray(z['dose_pct'], dtype=np.float32)
            masks = {kk[5:]: z[kk].astype(bool) for kk in z.files if kk.startswith('mask_')}
        dose = dose_n * np.float32(k)
        d95_resc = C.dvh.d95(dose, masks['PTV_High'])
        d95_norm = C.dvh.d95(dose_n, masks['PTV_High'])
        qa_pct = float(qa.loc[pid, 'pct_rx']) if pid in qa.index else np.nan
        rows.append({'patient_id': pid, 'site': site, 'split': rec['split'], 'k_d95_over_rx': k, 'D95_reescalado_pct': d95_resc,
                     'D95_normalizado_pct': d95_norm, 'qa_gate_pct_rx': qa_pct, 'dif_vs_qa_pp': d95_resc - qa_pct,
                     'dif_norm_vs_100_pp': d95_norm - 100.0})
        if out.exists():
            continue
        ratios = table.intent_ratios(site, meta['protocol_id'])
        geo = geom_for(rec, masks)
        real = summarize_b(dose, masks, rec, elig, ratios, geo)
        curves = {s: C.dvh.curve_vd(dose, masks[r]).astype(np.float32) for s, r in rec['slots'].items()}
        pickle.dump({'pid': pid, 'site': site, 'split': rec['split'], 'k': k, 'real': real, 'curves_real': curves}, open(out, 'wb'))
        if (i + 1) % 25 == 0:
            print(f'{i + 1} {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    df['discrepancia_gt_0.5pp'] = df.dif_vs_qa_pp.abs() > 0.5
    OUT_B.mkdir(parents=True, exist_ok=True)
    (OUT_B / 'tablas').mkdir(exist_ok=True)
    df.to_csv(OUT_B / 'tablas' / 'sanity_reescalado.csv', index=False)
    print('discrepantes >0.5pp vs QA gate:', int(df['discrepancia_gt_0.5pp'].sum()), ' |D95_norm-100|>0.5:', int((df.dif_norm_vs_100_pp.abs() > 0.5).sum()))


def phase_pred(a):
    import torch
    from omegaconf import OmegaConf
    import eval_a0
    from src.eval import ssim as ssim_mod
    from src.models.a0_module import A0Module
    from src.config.harmonization import HarmonizationTable
    torch.set_num_threads(a.threads)
    elig = C.load_eligible()
    cfg = OmegaConf.load(C.CONFIG)
    model = A0Module.load_from_checkpoint(str(C.CKPT), cfg=cfg, strict=True, map_location='cpu', weights_only=False).eval()
    t0, n = time.time(), 0
    for split in ('val', 'test'):
        dm, ds = eval_a0.get_ds(split, cfg.data)
        for pid in ds.ids[:a.limit] if a.limit else ds.ids:
            f = CACHE_B / f'{pid}.pkl'
            d = pickle.load(open(f, 'rb'))
            if 'pred' in d:
                continue
            rec = pickle.load(open(C.CACHE / f'{pid}.pkl', 'rb'))
            dose, chans, meta = ds.load_raw(pid)
            Z, H, W = dose.shape
            x = torch.from_numpy(np.ascontiguousarray(ds.pad_channels(chans).transpose(1, 0, 2, 3)))[None]
            with torch.no_grad():
                pred = model(x)[0].numpy()
            box = ds.frame_box(chans)
            pred = ds.frame.undo(pred, box, (H, W))
            masks = eval_a0.masks_from_npz(ds.npz_index[pid])
            pred = ssim_mod.postproc_body(pred, masks['BODY'])
            predn, scale = C.dvh.renorm_pred(pred, masks['PTV_High'])
            ratios = ds.table.intent_ratios(rec['site'], meta['protocol_id'])
            geo = geom_for(rec, masks)
            pv = summarize_b(predn, masks, rec, elig, ratios, geo)
            # verificación: reproduce lo cacheado de la Etapa 1 (GPU) -> mismo predicho
            chk = max(abs(pv[k] - rec['pred'][k]) for k in ('FALL|body_all', 'PTV|PTV_High|D50') if k in rec['pred'])
            d['pred'] = pv
            d['pred_check'] = {'scale_cpu': float(scale), 'scale_gpu_etapa1': rec['scale'], 'max_abs_dif_vars': float(chk)}
            pickle.dump(d, open(f, 'wb'))
            n += 1
            print(f'{pid} {split} {time.time() - t0:.0f}s escala cpu/gpu {scale:.4f}/{rec["scale"]:.4f} dif_vars {chk:.3g}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['real', 'pred'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    {'real': phase_real, 'pred': phase_pred}[a.phase](a)
