"""
Evalúa A0 (red) y/o la baseline no-aprendida con el harness compartido `src/eval/dvh.py`.

    # red
    .venv/Scripts/python.exe scripts/eval_a0.py net --config configs/exp_a0_001_unet2d.yaml --ckpt <path> --split val
    # baseline (curvas de scripts/baseline_dvh_promedio.py, ajustadas SOLO en train)
    .venv/Scripts/python.exe scripts/eval_a0.py baseline --baseline-dir results/baseline_v2b --split val
    # comparación pareada (necesita ambos metrics.csv)
    .venv/Scripts/python.exe scripts/eval_a0.py compare --net-dir results/exp_a0_001_unet2d --baseline-dir-out results/baseline_v2b_eval

Por defecto `--split val` (test reservado al cierre del eje A; usar `--split test` es decisión explícita).
Salidas por directorio: metrics.csv (paciente x estructura), summary.json (por sitio, IC95 bootstrap).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datamodules.a0_datamodule import frame_from_cfg  # noqa: E402
from src.eval import dvh  # noqa: E402
from src.eval import ssim as ssim_mod  # noqa: E402

SEC_COLS = ('mae_body', 'ssim_body', 'ssim_body_dosis5', ssim_mod.COL_FRAME_OBSOLETA)
DEF_NPZ = ROOT / '../../data/npz_v2b_body'
DEF_SPLIT = ROOT / 'data/splits/split_v2.json'
DEF_HARM = ROOT / 'data/harmonization'
NOTA_SEC = ('dose_score OpenKBP removido del reporte: con possible-dose-mask = body era idéntico al MAE body (no era una '
            'segunda señal). MAE body incluye vóxeles de soporte/inmovilizadores en fracción variable por paciente '
            '(mask_BODY = External de Eclipse): no sobre-interpretar comparaciones entre pacientes/sitios. '
            'SSIM por corte; MS-SSIM omitido (CyC <160 px). '
            'SSIM: ssim_body y ssim_body_dosis5 (src/eval/ssim.py, vara de coherencia de Fase 3) con pred*BODY; '
            'ssim_frame_obsoleto = frame completo, OBSOLETO: lo dominan los píxeles fuera del BODY.')


def get_ds(split, cfg_data=None):
    from src.datamodules.a0_datamodule import A0DataModule
    d = cfg_data
    args = (ROOT / d.npz_dir if d else DEF_NPZ, ROOT / d.splits_file if d else DEF_SPLIT,
            ROOT / d.harmonization_dir if d else DEF_HARM)
    if d and d.get('datamodule', 'a0') == 'a5':
        from src.datamodules.a5_datamodule import A5DataModule
        dm = A5DataModule(*args, pad_hw=frame_from_cfg(d), fill_sin_goal=d.fill_sin_goal, goal_mode=d.goal_mode,
                          ptv_ref=d.get('ptv_ref', 'union'), density_dir=ROOT / d.density_dir)
    elif d and d.get('datamodule', 'a0') in ('a1', 'a4'):
        from src.datamodules.a1_datamodule import A1DataModule
        if d.datamodule == 'a4':
            from src.datamodules.a4_datamodule import A4DataModule as A1DataModule
        dm = A1DataModule(*args, pad_hw=frame_from_cfg(d), fill_sin_goal=d.fill_sin_goal, goal_mode=d.goal_mode,
                          ptv_ref=d.get('ptv_ref', 'union'))
    elif d and d.get('datamodule', 'a0') == 'a4solo':
        from src.datamodules.a4_datamodule import A4SoloDataModule
        dm = A4SoloDataModule(*args, pad_hw=frame_from_cfg(d))
    else:
        dm = A0DataModule(*args, pad_hw=frame_from_cfg(d) if d else 256)
    dm.setup()
    return dm, getattr(dm, f'{split}_ds')


def masks_from_npz(npz_path):
    with np.load(npz_path) as z:
        return {k[len('mask_'):]: z[k].astype(bool) for k in z.files if k.startswith('mask_')}


def write_outputs(out_dir, df, extra, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = dvh.with_groups(df)
    df.to_csv(out_dir / 'metrics.csv', index=False)
    summary = {'label': label, **extra, 'banda_media_oars_y_alta_ptv': 'ver OAR_all.media_40_80 y PTV_all.alta_80_110',
               'dvh_mean_abs_delta_V_pp': dvh.summarize(df)}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print_headline(summary['dvh_mean_abs_delta_V_pp'], label)


def print_headline(s, label):
    print(f'\n=== {label}: mean|dV| pp  [IC95]  (OAR_all banda media, PTV_all banda alta) ===')
    for site, roles in s.items():
        o, p = roles.get('OAR_all'), roles.get('PTV_all')
        fmt = lambda r, b: f"{r[b]['mean']:.2f} [{r[b]['ci95'][0]:.2f},{r[b]['ci95'][1]:.2f}] n={r[b]['n']}" if r else 'n/a'
        print(f'{site:9s} OAR media {fmt(o, "media_40_80")} | PTV alta {fmt(p, "alta_80_110")}')


def run_net(a):
    from omegaconf import OmegaConf
    from src.models.a0_module import A0Module
    cfg = OmegaConf.load(a.config)
    dm, ds = get_ds(a.split, cfg.data)
    model = A0Module.load_from_checkpoint(a.ckpt, cfg=cfg, strict=True, map_location='cpu', weights_only=False)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev).eval()
    rows, sec, fallas, offsets, fuera_del_body = [], [], [], [], []
    for i, pid in enumerate(ds.ids):
        dose, chans, meta = ds.load_raw(pid)
        Z, H, W = dose.shape
        x = torch.from_numpy(np.ascontiguousarray(ds.pad_channels(chans).transpose(1, 0, 2, 3)))[None].to(dev)
        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()
        box = ds.frame_box(chans)
        pred = ds.frame.undo(pred, box, (H, W))   # saca pad y repone el recorte -> grilla nativa (fuera del box = 0)
        masks = masks_from_npz(ds.npz_index[pid])
        y0, y1, x0, x1 = box
        for role, m in masks.items():   # el crop no puede dejar fuera de la caja ninguna estructura (pred=0 ahí)
            if m.any() and (m[:, :y0].any() or m[:, y1:].any() or m[:, :, :x0].any() or m[:, :, x1:].any()):
                raise RuntimeError(f'{pid}: la máscara {role} sale del box de recorte {box}')
        offsets.append({'patient_id': pid, 'site': meta['site'], 'H': H, 'W': W, 'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1})
        body = masks['BODY']
        fb = {r: int((m & ~body).sum()) for r, m in masks.items() if r != 'BODY' and (m & ~body).any()}
        if fb:
            fuera_del_body.append((pid, fb))
        if not a.no_body_mask:
            pred = ssim_mod.postproc_body(pred, body)   # post-proceso estándar: pred * BODY (BODY es input, sin leakage)
        mae_body = float(np.abs(pred - dose)[body].mean())
        predn, f = dvh.renorm_pred(pred, masks['PTV_High'])
        r = dvh.patient_rows(pid, meta['site'], masks, dose, dose_pred=predn)
        for row in r:
            if not (row['v0_true_ok'] and row['monotona_true_ok']):
                fallas.append((pid, row['role']))
        rows += r
        rs = ssim_mod.region_stats(pred, dose, body, dev)
        sec.append({'patient_id': pid, 'site': meta['site'], 'mae_body': mae_body,
                    'ssim_body': rs['body_ssim'], 'ssim_body_dosis5': rs['body_dosis5_ssim'],
                    ssim_mod.COL_FRAME_OBSOLETA: rs['frame_ssim'],   # OBSOLETA: solo referencia (la domina lo de fuera del BODY)
                    'factor_renorm': f})
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(ds.ids)}')
    out = ROOT / a.out_dir
    sec_df = pd.DataFrame(sec)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(offsets).to_csv(out / 'crop_offsets.csv', index=False)
    sec_df.to_csv(out.mkdir(parents=True, exist_ok=True) or out / 'secondary_per_patient.csv', index=False)
    secondary = {s: {c: dvh.bootstrap_ci(g[c].values) for c in SEC_COLS}
                 for s, g in sec_df.groupby('site')}
    record = json.loads((ROOT / 'results' / cfg.experiment.name / 'record.json').read_text()) \
        if (ROOT / 'results' / cfg.experiment.name / 'record.json').exists() else dm.experiment_record()
    extra = {'split_evaluado': a.split, 'ckpt': str(a.ckpt), 'sanity_fallas': fallas,
             'secundarias_por_sitio': secondary,
             'nota_secundarias': NOTA_SEC,
             'post_proceso_pred_por_body': not a.no_body_mask,
             'estructuras_fuera_del_body': fuera_del_body,   # si no está vacío, pred*BODY podría mover el DVH de esas estructuras
             'record': {k: record[k] for k in ('exp_id', 'harmonization_hash', 'npz_dir', 'splits_file', 'pad_hw', 'epoch_scale') if k in record}}
    write_outputs(out, pd.DataFrame(rows), extra, f'A0 net ({a.split})')


def run_baseline(a):
    dm, ds = get_ds(a.split)
    bdir = ROOT / a.baseline_dir
    curves = {}
    for site in ('Prostata', 'Pelvis', 'CyC'):
        j = json.load(open(bdir / f'baseline_dvh_{site}.json', encoding='utf-8'))
        grid = np.array(j['dose_grid_pct_rx_high'])
        curves[site] = {r: 100.0 * np.interp(dvh.D_BINS, grid, np.array(v['volume_fraction_mean']))
                        for r, v in j['roles'].items()}
    rows = []
    for pid in ds.ids:
        meta = json.load(open(ds.npz_index[pid].with_suffix('.json'), encoding='utf-8'))
        with np.load(ds.npz_index[pid]) as z:
            dose = np.asarray(z['dose_pct'], dtype=np.float32)
        masks = masks_from_npz(ds.npz_index[pid])
        rows += dvh.patient_rows(pid, meta['site'], masks, dose, curves_pred=curves[meta['site']])
    extra = {'split_evaluado': a.split, 'baseline_dir': str(bdir), 'fit': 'solo train',
             'record': {'harmonization_hash': dm.table.hash, 'npz_dir': str(dm.npz_dir),
                        'splits_file': str(dm.splits_file)}}
    write_outputs(ROOT / a.out_dir, pd.DataFrame(rows), extra, f'baseline no-aprendida ({a.split})')


def run_compare(a):
    net = pd.read_csv(ROOT / a.net_dir / 'metrics.csv')
    base = pd.read_csv(ROOT / a.baseline_eval_dir / 'metrics.csv')
    diff = dvh.paired_diff(base, net)   # positivo = red mejor que baseline
    out = ROOT / a.net_dir / 'comparison_vs_baseline.json'
    out.write_text(json.dumps({'convencion': 'baseline - red en pp de volumen; >0 = la red mejora; IC95 bootstrap pareado por paciente',
                               'diff': diff}, indent=2), encoding='utf-8')
    print('\n=== baseline - red (pp, >0 red mejor) ===')
    for site, roles in diff.items():
        for role in ('OAR_all', 'PTV_all'):
            if role in roles:
                b = 'media_40_80' if role == 'OAR_all' else 'alta_80_110'
                r = roles[role][b]
                print(f'{site:9s} {role:8s} {b:12s} {r["mean"]:+.2f} [{r["ci95"][0]:+.2f},{r["ci95"][1]:+.2f}] n={r["n"]}')


def run_resummary(a):
    """Regenera summary.json desde metrics.csv (sin inferencia): IC con std/degenerate y secundarias sin dose_score."""
    for d in a.dirs:
        out = ROOT / d
        df = dvh.with_groups(pd.read_csv(out / 'metrics.csv'))
        df.to_csv(out / 'metrics.csv', index=False)
        sj = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        sj['dvh_mean_abs_delta_V_pp'] = dvh.summarize(df)
        if 'secundarias_por_sitio' in sj:
            sec = pd.read_csv(out / 'secondary_per_patient.csv').drop(columns=['dose_score_openkbp'], errors='ignore')
            sec.to_csv(out / 'secondary_per_patient.csv', index=False)
            cols = [c for c in ('mae_body', 'ssim', *SEC_COLS) if c in sec.columns]
            sj['secundarias_por_sitio'] = {s: {c: dvh.bootstrap_ci(g[c].values) for c in dict.fromkeys(cols)}
                                           for s, g in sec.groupby('site')}
            sj['nota_secundarias'] = NOTA_SEC
        sj['nota_ic'] = ('cells con degenerate=true (std entre pacientes = 0) tienen IC colapsado a un punto: la estructura '
                         'no recibe dosis en esa banda; no leer como estimación precisa. Los agregados OAR_all/PTV_all mandan.')
        (out / 'summary.json').write_text(json.dumps(sj, indent=2), encoding='utf-8')
        print('resummary', d)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    n = sp.add_parser('net')
    n.add_argument('--config', required=True)
    n.add_argument('--ckpt', required=True)
    n.add_argument('--split', default='val', choices=['val', 'test'])
    n.add_argument('--out-dir', default='results/exp_a0_001_unet2d')
    n.add_argument('--no-body-mask', action='store_true',
                   help='NO aplicar pred*BODY (solo para reproducir/verificar evaluaciones previas)')
    b = sp.add_parser('baseline')
    b.add_argument('--baseline-dir', default='results/baseline_v2b')
    b.add_argument('--split', default='val', choices=['val', 'test'])
    b.add_argument('--out-dir', default='results/baseline_v2b_eval')
    c = sp.add_parser('compare')
    c.add_argument('--net-dir', default='results/exp_a0_001_unet2d')
    c.add_argument('--baseline-eval-dir', default='results/baseline_v2b_eval')
    r = sp.add_parser('resummary')
    r.add_argument('dirs', nargs='+')
    a = ap.parse_args()
    {'net': run_net, 'baseline': run_baseline, 'compare': run_compare, 'resummary': run_resummary}[a.cmd](a)


if __name__ == '__main__':
    main()
