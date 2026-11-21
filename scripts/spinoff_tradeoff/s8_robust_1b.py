"""
Etapa 1b / sensibilidad por valores extremos de dD95 (no prevista en el prompt; ver decisiones en el resumen).
dD95(PTV_High) tiene colas pesadas (CyC hasta -26 pp: casos 'cavum con PTV solapado al PRV de quiasma', ya revisados en el QA gate).
Dos versiones robustas, mismas pruebas que s6:
  (a) rangos: Spearman (correlación sobre rangos; R² LOO de rangos de dD95)
  (b) sin extremos: se sacan del pool los pacientes con z robusto de dD95 < -3.5 (z = (x - mediana)/(1.4826*MAD), MAD sobre el pool)
Agrega 'robustez' a summary.json y una sección a resumen.md.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import tools as T  # noqa: E402
import s3_residual as S3  # noqa: E402
import s6_analyze_1b as A  # noqa: E402

Z_CUT = -3.5


def rz(x):
    m = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - m))
    return (x - m) / (mad if mad > 0 else 1.0)


def direct(y, Rdf, blocks_scores, kind):
    """kind 'pearson' | 'spearman'. Misma familia que s6.direct_test."""
    labs, xs, tipo = [], [], []
    for c in Rdf.columns:
        if c.startswith('OAR|') and (c.endswith('|dV_media_40_80') or c.endswith('|dDmean')):
            labs.append(c.replace('OAR|', ''))
            xs.append(-Rdf[c].values)
            tipo.append('OAR_slot')
    for b in ('OAR', 'HOT', 'FALL'):
        if b in blocks_scores:
            labs.append(f'score_{b}')
            xs.append(blocks_scores[b])
            tipo.append('bloque')
    Xm = np.column_stack(xs)
    if kind == 'spearman':
        Xm = np.column_stack([rankdata(Xm[:, j]) for j in range(Xm.shape[1])])
        y = rankdata(y)
    r, p, lo, hi = T.corr_many(y, Xm, n_perm=10000, n_boot=2000, seed=5)
    padj = T.holm(p)
    df = pd.DataFrame({'prueba': labs, 'tipo': tipo, 'r_calidad': r, 'ic95_lo': lo, 'ic95_hi': hi, 'p_perm': p, 'p_holm': padj, 'n': len(y)})
    df['signo'] = np.where(df.r_calidad < 0, 'antagonico', 'sinergico')
    df['significativo'] = (df.p_holm < 0.05) & ((df.ic95_lo > 0) | (df.ic95_hi < 0))
    return df


def r2_pool(recs_all, site, elig, pids, y, rank):
    d = np.load(C.CACHE / f'dist_{site}.npz')
    pid_all = list(d['pids'])
    by = {r['pid']: r for r in recs_all}
    desc = M2_desc(recs_all, site, elig, pid_all)
    ix = [pid_all.index(p) for p in pids]
    Xd = desc[ix]
    Xd = Xd[:, Xd.std(0) > 1e-9]
    yy = rankdata(y) if rank else y
    r2, pred = T.loo_r2(Xd, yy)
    return {'n': len(yy), 'r2_loo': float(r2), 'r2_ic95': T.r2_boot_ci(yy, pred), 'r2_p_perm': T.loo_r2_perm_p(Xd, yy, r2)}


_DESC = {}


def M2_desc(recs_all, site, elig, pid_all):
    import s2_metric as M2
    if site not in _DESC:
        by = {r['pid']: r for r in recs_all}
        desc = M2.descriptors([by[p] for p in pid_all], site, 'B', elig)
        X = desc.values.astype(float)
        _DESC[site] = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    return _DESC[site]


def summarize_direct(df):
    oar = df[(df.tipo == 'OAR_slot') | (df.prueba == 'score_OAR')]
    b = df[df.prueba == 'score_OAR'].iloc[0]
    return {'n_pruebas': int(len(df)), 'n_sig_holm': int(df.significativo.sum()),
            'antagonicas_sig_OAR': oar[oar.significativo & (oar.signo == 'antagonico')].prueba.tolist(),
            'sinergicas_sig_OAR': oar[oar.significativo & (oar.signo == 'sinergico')].prueba.tolist(),
            'score_OAR': {'r': float(b.r_calidad), 'ic95': [float(b.ic95_lo), float(b.ic95_hi)], 'p_holm': float(b.p_holm)}}


def main():
    elig = C.load_eligible()
    work = json.load(open(C.OUT / 'metrica_trabajo.json', encoding='utf-8'))
    recs_all = C.load_cache()
    out = {}
    lines = ['', '## Sensibilidad por extremos de ΔD95 (no prevista en el prompt)', '',
             'ΔD95 tiene colas pesadas; en CyC los extremos son los casos ya revisados "cavum con PTV solapado al PRV de quiasma" (QA gate `reviewed_ok`). '
             'Se repite la prueba directa y el R² anatómico (a) con rangos y (b) sin los pacientes con z robusto < −3.5.', '',
             '| Sitio | Versión | n | ΔD95 vs score OAR (r calidad [IC95], p Holm) | Pruebas sig. (de n) | OAR sig. antagónicas | R² anat. ΔD95 (todos) |', '|---|---|---|---|---|---|---|']
    for site in C.SITES:
        recs1 = sorted([r for r in recs_all if r['site'] == site], key=lambda r: r['pid'])
        recs = A.assemble(site, recs1)
        oos = [r for r in recs if r['split'] in ('val', 'test')]
        R_raw = T.residual_matrix(oos)
        R, _ = T.clean_matrix(R_raw)
        Rc = R - R.mean()
        y_all = Rc[A.D95].values
        z = rz(np.array([r['real'][A.D95] - 100.0 for r in recs]))
        zo = rz(y_all)
        res0 = pickle.load(open(A.CACHE_B / f'res1b_{site}.pkl', 'rb'))
        out[site] = {}
        variants = {}
        # (a) rangos, mismas muestras
        variants['rangos (Spearman)'] = (Rc, y_all, res0['blocks']['scores'], [r['pid'] for r in recs], True)
        # (b) sin extremos
        keep_o = zo >= Z_CUT
        keep_all = z >= Z_CUT
        Rb = Rc[keep_o]
        Rb = Rb - Rb.mean()
        bl = S3.block_coupling(list(Rb.columns), Rb.values - Rb.values.mean(0), seed=3)
        variants[f'sin extremos (z<{Z_CUT})'] = (Rb, Rb[A.D95].values, bl['scores'], [r['pid'] for r, k in zip(recs, keep_all) if k], False)
        for name, (Rv, yv, bs, pool, rank) in variants.items():
            dt = direct(yv, Rv, bs, 'spearman' if rank else 'pearson') if not rank else direct(yv, Rv, bs, 'spearman')
            yall = np.array([next(r for r in recs if r['pid'] == p)['real'][A.D95] - 100.0 for p in pool])
            an = r2_pool(recs_all, site, elig, pool, yall, rank)
            sm = summarize_direct(dt)
            out[site][name] = {'n_val_test': int(len(yv)), 'n_pool_todos': len(pool), 'excluidos_val_test': int((~keep_o).sum()) if not rank else 0,
                               'excluidos_todos': int((~keep_all).sum()) if not rank else 0, 'directa': sm, 'anatomia_todos': an}
            dt.assign(sitio=site, version=name).to_csv(C.ROOT / 'results/spinoff_tradeoff/etapa1b/tablas' / f'robustez_directa_{site}_{"rangos" if rank else "sin_extremos"}.csv', index=False)
            s = sm['score_OAR']
            lines.append(f'| {site} | {name} | {len(yv)} (pool {len(pool)}) | {s["r"]:+.2f} [{s["ic95"][0]:+.2f},{s["ic95"][1]:+.2f}], {s["p_holm"]:.3g} | {sm["n_sig_holm"]} de {sm["n_pruebas"]} | '
                         f'{", ".join(sm["antagonicas_sig_OAR"]) or "—"} | {an["r2_loo"]:.2f} [{an["r2_ic95"][0]:.2f},{an["r2_ic95"][1]:.2f}] |')
        print(site, {k: (v['directa']['n_sig_holm'], round(v['anatomia_todos']['r2_loo'], 2)) for k, v in out[site].items()}, flush=True)
    sj = json.load(open(A.OUT_B / 'summary.json', encoding='utf-8'))
    sj['robustez_extremos_dD95'] = out
    sj['decisiones'].append('Sensibilidad no prevista: dD95 con colas pesadas (CyC hasta -26 pp: cavum con PTV solapado al PRV de quiasma, reviewed_ok); se repiten prueba directa y R² con rangos y sin pacientes con z robusto < -3.5.')
    json.dump(sj, open(A.OUT_B / 'summary.json', 'w', encoding='utf-8'), indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    rs = (A.OUT_B / 'resumen.md').read_text(encoding='utf-8')
    head, tail = rs.split('## Decisiones no previstas')
    (A.OUT_B / 'resumen.md').write_text(head.rstrip() + '\n' + '\n'.join(lines) + '\n\n## Decisiones no previstas' + tail + f'- Sensibilidad por extremos de dD95 (ver sección arriba): corte z robusto < {Z_CUT} fijado antes de correr la sensibilidad (tras ver la distribución de dD95).\n', encoding='utf-8')


if __name__ == '__main__':
    main()
