"""
D2 — Coherencia axial (cráneo-caudal) del baseline A4-2D. Vara 2D contra la que se mide el 3D (foco CyC).

Portado de prostate-dose-prediction/scripts/diagnostico_piso/d2_coherencia_axial.py. Diagnóstico puro: NO entrena,
NO toca configs/splits/datamodule (solo los lee). El modelo 2D procesa cada corte independiente; la dosis real es
un campo continuo en Z. Se compara la "rugosidad en Z" (= media de |Δ entre cortes consecutivos|) real vs pred sobre
3 perfiles por OAR, calculados sobre la dosis NATIVA des-padeada (frame-independiente: no se rehace al cambiar de contenedor):

  xy       D(z) en un pixel XY fijo (centroide 3D del OAR, mismo (row,col) en todos los cortes)
  meandose dosis media del OAR por corte
  vfrac    %vol del OAR en el corte con dosis en banda media (40,80] %Rx

Diferencias con el original: OARs por sitio desde la tabla de armonización (roles `oars` de sites_*.yaml, no
Rectum/Bladder fijos); renorm D95(PTV_High)=100% de este proyecto (src.eval.dvh.renorm_pred); inferencia con el
mismo camino que eval_a0.py (frame.undo a grilla nativa). Se corren los 3 sitios.

Salida: results/<exp>/d2_coherencia_axial/{por_oar_paciente.csv, por_paciente_sitio.csv, d2_coherencia_axial.json,
tabla_sitio.csv, plots/}. Default split val (test reservado al cierre del eje A).

Uso:
    .venv/Scripts/python.exe scripts/d2_coherencia_axial.py \
        --config configs/exp_a4_001_unet2d_psdm.yaml --ckpt checkpoints/exp_a4_001_unet2d_psdm/epoch=105.ckpt
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from src.config.harmonization import HarmonizationTable, SITE_FILES  # noqa: E402
from src.eval import dvh  # noqa: E402
from eval_a0 import get_ds, masks_from_npz  # noqa: E402

MIN_PIXELS_SLICE = 10          # área mínima del OAR en un corte para entrar al perfil
BANDA_MEDIA = (40.0, 80.0)     # %Rx_High, igual que dvh.BANDS['media_40_80'] (lo, hi]
METRICAS = ('xy', 'meandose', 'vfrac')
SITES = tuple(SITE_FILES)
RUG_RATIO_ALTO, RUG_RATIO_BAJO = 1.3, 0.7


def centroide_3d(mask):
    zz, rr, cc = np.nonzero(mask)
    return (int(round(rr.mean())), int(round(cc.mean()))) if len(zz) else None


def rugosidad(perfil):
    v = perfil[~np.isnan(perfil)]
    return float(np.mean(np.abs(np.diff(v)))) if len(v) >= 2 else float('nan')


def rho_seguro(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float('nan')
    return float(stats.spearmanr(a, b).statistic)


def perfiles_oar(mask, dose_real, dose_pred):
    z_ok = [z for z in range(mask.shape[0]) if mask[z].sum() >= MIN_PIXELS_SLICE]
    if len(z_ok) < 3:
        return None
    rc, cc = centroide_3d(mask)
    lo, hi = BANDA_MEDIA

    def perfil(d):
        return {'xy': np.array([d[z, rc, cc] for z in z_ok]),
                'meandose': np.array([d[z][mask[z]].mean() for z in z_ok]),
                'vfrac': np.array([100.0 * np.mean((d[z][mask[z]] > lo) & (d[z][mask[z]] <= hi)) for z in z_ok])}
    pr, pp = perfil(dose_real), perfil(dose_pred)
    out = {'z_validos': z_ok, 'n_slices': len(z_ok), 'perfil_real': pr, 'perfil_pred': pp}
    for m in METRICAS:
        out[f'rug_{m}_real'], out[f'rug_{m}_pred'] = rugosidad(pr[m]), rugosidad(pp[m])
        out[f'rho_{m}'] = rho_seguro(pr[m], pp[m])
    return out


def predecir(model, ds, pid, dev):
    """Misma ruta que eval_a0.run_net: pad -> red -> undo a grilla nativa (fuera del box = 0)."""
    dose, chans, meta = ds.load_raw(pid)
    Z, H, W = dose.shape
    x = torch.from_numpy(np.ascontiguousarray(ds.pad_channels(chans).transpose(1, 0, 2, 3)))[None].to(dev)
    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()
    return dose, ds.frame.undo(pred, ds.frame_box(chans), (H, W)), meta


def veredicto(p, ratio):
    if p < 0.05 and ratio > RUG_RATIO_ALTO:
        return 'pred MÁS RUGOSA que la real'
    if p < 0.05 and ratio < RUG_RATIO_BAJO:
        return 'pred MÁS SUAVE que la real'
    if p < 0.05:
        return f'pred algo más {"suave" if ratio < 1 else "rugosa"} (p<0.05) pero ratio dentro de [{RUG_RATIO_BAJO}, {RUG_RATIO_ALTO}]'
    return 'sin diferencia significativa'


def agregar(df_oar):
    """Por sitio y métrica. Unidad estadística = paciente (promedio de sus OARs) para Wilcoxon/ratio;
    rho = media sobre perfiles (paciente, OAR). Las filas por OAR quedan en tabla_oar."""
    filas, filas_oar = [], []
    for site, g in df_oar.groupby('site'):
        for m in METRICAS:
            pac = g.groupby('patient_id')[[f'rug_{m}_real', f'rug_{m}_pred']].mean().dropna()
            real, pred = pac[f'rug_{m}_real'], pac[f'rug_{m}_pred']
            p = float(stats.wilcoxon(pred, real).pvalue) if len(pac) > 3 and (pred != real).any() else float('nan')
            ratio = float(pred.mean() / real.mean()) if real.mean() > 1e-9 else float('nan')
            rho = g[f'rho_{m}'].dropna()
            filas.append({'site': site, 'metrica': m, 'n_pacientes': len(pac), 'n_perfiles_oar': int(len(rho)),
                          'rug_real': float(real.mean()), 'rug_pred': float(pred.mean()), 'ratio_pred_real': ratio,
                          'ratio_mediana_por_paciente': float((pred / real).median()),
                          'wilcoxon_p': p, 'rho_spearman_media': float(rho.mean()) if len(rho) else float('nan'),
                          'rho_spearman_mediana': float(rho.median()) if len(rho) else float('nan'),
                          'veredicto': veredicto(p, ratio) if not np.isnan(p) else 'n insuficiente'})
            for role, go in g.groupby('role'):
                a, b = go[f'rug_{m}_real'].dropna(), go[f'rug_{m}_pred'].loc[go[f'rug_{m}_real'].dropna().index]
                filas_oar.append({'site': site, 'role': role, 'metrica': m, 'n': len(a),
                                  'rug_real': float(a.mean()), 'rug_pred': float(b.mean()),
                                  'ratio_pred_real': float(b.mean() / a.mean()) if a.mean() > 1e-9 else float('nan'),
                                  'rho_media': float(go[f'rho_{m}'].mean())})
    return pd.DataFrame(filas), pd.DataFrame(filas_oar)


def plot_ejemplos(crudos, df_oar, site, metrica, plots_dir):
    """Un OAR-paciente bajo/mediano/alto en ratio de rugosidad, para el sitio."""
    g = df_oar[df_oar.site == site].copy()
    g['ratio'] = g[f'rug_{metrica}_pred'] / g[f'rug_{metrica}_real']
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=['ratio']).sort_values('ratio')
    if len(g) < 3:
        return
    sel = {'bajo (pred más suave)': g.iloc[0], 'mediano': g.iloc[len(g) // 2], 'alto (pred más rugosa)': g.iloc[-1]}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (et, r) in zip(axes, sel.items()):
        c = crudos[(r.patient_id, r.role)]
        ax.plot(c['z_validos'], c['perfil_real'][metrica], 'o-', color='black', ms=3, label='Real')
        ax.plot(c['z_validos'], c['perfil_pred'][metrica], 'o--', color='firebrick', ms=3, label='Pred (renorm)')
        ax.set_title(f'{et}\n{r.role} {r.patient_id[-6:]} (ratio={r.ratio:.2f})', fontsize=9)
        ax.set_xlabel('Corte Z (índice nativo)')
        ax.set_ylabel(metrica)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'D2 — perfil en Z ({metrica}) real vs pred — {site}', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(str(plots_dir / f'd2_zprofile_{metrica}_{site.lower()}.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--out-dir', default=None, help='default: results/<exp>/d2_coherencia_axial')
    ap.add_argument('--device', default=None, help='default: cuda si hay (usar cpu si la GPU está tomada por el 3D)')
    a = ap.parse_args()

    from omegaconf import OmegaConf
    from src.models.a0_module import A0Module
    cfg = OmegaConf.load(a.config)
    out = ROOT / (a.out_dir or f'results/{cfg.experiment.name}/d2_coherencia_axial')
    plots_dir = out / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    dm, ds = get_ds(a.split, cfg.data)
    table = HarmonizationTable(ROOT / cfg.data.harmonization_dir)
    oars_sitio = {s: set(table._sites[s].get('oars', {})) for s in SITES}
    model = A0Module.load_from_checkpoint(a.ckpt, cfg=cfg, strict=True, map_location='cpu', weights_only=False)
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()
    print(f'{cfg.experiment.name} | split={a.split} n={len(ds.ids)} | dev={dev} | ckpt={a.ckpt}')

    filas, crudos = [], {}
    for i, pid in enumerate(ds.ids):
        dose_real, pred_raw, meta = predecir(model, ds, pid, dev)
        masks = masks_from_npz(ds.npz_index[pid])
        dose_pred, f = dvh.renorm_pred(pred_raw, masks['PTV_High'])
        site = meta['site']
        for role in sorted(oars_sitio[site] & set(masks)):
            if not masks[role].any():
                continue
            r = perfiles_oar(masks[role], dose_real, dose_pred)
            if r is None:
                continue
            crudos[(pid, role)] = r
            filas.append({'patient_id': pid, 'site': site, 'role': role, 'n_slices': r['n_slices'],
                          'factor_renorm': f, **{k: v for k, v in r.items() if k.startswith(('rug_', 'rho_'))}})
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(ds.ids)}')

    df_oar = pd.DataFrame(filas)
    df_oar.to_csv(out / 'por_oar_paciente.csv', index=False)
    tabla, tabla_oar = agregar(df_oar)
    tabla.to_csv(out / 'tabla_sitio.csv', index=False)
    tabla_oar.to_csv(out / 'tabla_oar.csv', index=False)
    for site in df_oar.site.unique():
        for m in ('meandose', 'vfrac'):
            plot_ejemplos(crudos, df_oar, site, m, plots_dir)

    summary = {'exp': cfg.experiment.name, 'ckpt': str(a.ckpt), 'split': a.split, 'n': len(ds.ids),
               'harmonization_hash': table.hash, 'banda_media_pct_rx': list(BANDA_MEDIA),
               'min_pixels_slice': MIN_PIXELS_SLICE,
               'renorm': 'clip>=0 y D95(PTV_High pred)=100% (dvh.renorm_pred), grilla nativa des-padeada',
               'rugosidad': 'mean(|diff(perfil)|) a lo largo de Z; unidad estadística = paciente (media de sus OARs)',
               'ratio': 'media(rug_pred)/media(rug_real) sobre pacientes; >1.3 / <0.7 con Wilcoxon p<0.05 = distinto',
               'oars_por_sitio': {s: sorted(v) for s, v in oars_sitio.items()},
               'tabla_sitio': tabla.to_dict('records')}
    (out / 'd2_coherencia_axial.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')

    pd.set_option('display.width', 200)
    print('\n=== D2 — vara 2D de coherencia axial (por sitio) ===')
    print(tabla[['site', 'metrica', 'n_pacientes', 'rug_real', 'rug_pred', 'ratio_pred_real', 'wilcoxon_p',
                 'rho_spearman_media', 'veredicto']].round(3).to_string(index=False))
    print(f'\nGuardado en {out}')


if __name__ == '__main__':
    main()
