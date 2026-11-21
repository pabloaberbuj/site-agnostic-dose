"""
Etapa 1 / Paso D: análisis del residual (real - A4 renormalizado), por sitio, SOLO val+test (fuera de muestra), vector compacto.
Sin métricas agregadas de desempeño sobre test: solo estructura del residual.
Guarda results/spinoff_tradeoff/etapa1/cache/resD_<sitio>.pkl y tablas en tablas/.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import tools as T  # noqa: E402
import s2_metric as M2  # noqa: E402

BLOCKS = ['OAR', 'PTV', 'HOT', 'FALL']
N_PERM_PA = 1000
BONF_PAIRS = 3          # pares OAR vs {PTV, HOT, FALL}
ALPHA = 0.05


def corr_loadings(scores, X):
    """corr(score_k, var_j): (p, K). Es el 'loading' estructural comparable entre análisis con distinto escalado/n."""
    Sz = (scores - scores.mean(0)) / scores.std(0, ddof=1)
    sx = X.std(0, ddof=1)
    Xz = (X - X.mean(0)) / np.where(sx < 1e-12, 1.0, sx)   # variable constante -> corr 0 (no NaN)
    return Xz.T @ Sz / (X.shape[0] - 1)


def best_cos(Lref, Lcmp, topk=None):
    """Para cada columna de Lref, mejor |cos| contra las primeras `topk` columnas de Lcmp. (valores, índices)"""
    topk = topk or Lcmp.shape[1]
    out, idx = [], []
    for k in range(Lref.shape[1]):
        c = [T.cos_abs(Lref[:, k], Lcmp[:, j]) for j in range(min(topk, Lcmp.shape[1]))]
        out.append(max(c))
        idx.append(int(np.argmax(c)))
    return np.array(out), idx


def _t(msg):
    print(f'  [{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def analyze_matrix(Xdf, label, n_perm=N_PERM_PA, n_boot=500, seed=0):
    """Estructura del residual/desviación: PCA estandarizado, análisis paralelo, estabilidad, sensibilidad por unidades,
    acoplamiento entre bloques. Xdf: DataFrame (n pacientes x p variables), ya limpio e imputado."""
    cols = list(Xdf.columns)
    X = Xdf.values.astype(float)
    n, p = X.shape
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1)
    ev, scores, Vt, Z = T.pca_eigs(Xc, standardize=True)
    _t(f'{label}: PCA n={n} p={p}')
    pa = T.parallel_analysis(Xc, n_perm=n_perm, seed=seed)
    _t(f'{label}: PA listo, retenidos={pa["n_ret"]}')
    nret = pa['n_ret']
    out = {'label': label, 'cols': cols, 'pids': list(Xdf.index), 'n': n, 'p': p, 'sd': sd, 'eig': ev[:pa['eig'].size],
           'pa': pa, 'n_ret': nret, 'var_expl': (ev / ev.sum())[:pa['eig'].size]}
    K = max(nret, 3)
    out['scores'] = scores[:, :K]
    out['load'] = corr_loadings(scores[:, :K], Xc)
    if nret:
        stab = T.bootstrap_loading_stability(Xc, nret, n_boot=n_boot, seed=seed + 1)
        out['stab'] = {'median': stab['median'].tolist(), 'p5': stab['p5'].tolist()}
    else:
        out['stab'] = {'median': [], 'p5': []}
    _t('estabilidad lista')
    # sensibilidad: sin estandarizar por variable, solo por grupo de unidades
    sc = T.unit_scale(cols, Xc)
    pa_s = T.parallel_analysis(Xc, n_perm=max(200, n_perm // 4), seed=seed + 2, scale=sc)
    _, scs, _, _ = T.pca_eigs(Xc, scale=sc)
    Ls = corr_loadings(scs[:, :max(pa_s['n_ret'], 3)], Xc)
    cs, _ = best_cos(out['load'][:, :max(nret, 1)], Ls) if nret else (np.array([]), [])
    out['sens'] = {'n_ret_unidades': int(pa_s['n_ret']), 'cos_vs_principal': cs.tolist()}
    _t('sensibilidad lista')
    out['blocks'] = block_coupling(cols, Xc, seed=seed + 3)
    _t('acoplamiento listo')
    out['comp_block_q'] = component_block_pattern(cols, out['load'][:, :max(nret, 1)] if nret else out['load'][:, :0])
    return out


def block_members(cols):
    return {b: [j for j, c in enumerate(cols) if C.block_of(c) == b] for b in BLOCKS if any(C.block_of(c) == b for c in cols)}


def block_coupling(cols, Xc, seed=0):
    mem = block_members(cols)
    sd = Xc.std(0, ddof=1)
    Z = Xc / sd
    sc, align = {}, {}
    for b, ix in mem.items():
        q = [C.quality_sign(cols[j]) for j in ix]
        if len(ix) == 1:
            sc[b], align[b] = Z[:, ix[0]] * (q[0] if q[0] != 0 else 1), np.nan
        else:
            sc[b], align[b], _ = T.block_score(Z[:, ix], q)
    names = list(sc)
    corr = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            corr[f'{a}~{b}'] = T.corr_perm_boot(sc[a], sc[b], n_perm=10000, n_boot=2000, seed=seed)
    return {'scores': sc, 'align': align, 'corr': corr, 'n_vars': {b: len(ix) for b, ix in mem.items()}}


def component_block_pattern(cols, L):
    """Por componente retenido y bloque: media de q_j * corr(score,var_j) sobre variables con q != 0 (signo de calidad del componente)."""
    mem = block_members(cols)
    out = []
    for k in range(L.shape[1]):
        d = {}
        for b, ix in mem.items():
            v = [C.quality_sign(cols[j]) * L[j, k] for j in ix if C.quality_sign(cols[j]) != 0]
            d[b] = float(np.mean(v)) if v else float('nan')
        out.append(d)
    return out


def classify_coupling(blocks):
    """Resumen del acoplamiento OAR vs {PTV, HOT, FALL} con Bonferroni (3 pares), en calidad: r<0 antagónico, r>0 sinérgico."""
    res = {}
    for pair, v in blocks['corr'].items():
        a, b = pair.split('~')
        if 'OAR' not in (a, b):
            continue
        other = b if a == 'OAR' else a
        sig = v['p_perm'] < ALPHA / BONF_PAIRS and (v['ci95'][0] > 0 or v['ci95'][1] < 0)
        res[other] = {'r': v['r'], 'ci95': v['ci95'], 'p_perm': v['p_perm'], 'significativo': bool(sig),
                      'signo': 'antagonico' if v['r'] < 0 else 'sinergico' if v['r'] > 0 else 'nulo'}
    return res


def optimism(Rtr, Roos, cols, n_boot=1000, seed=9):
    """Razón RMS(train)/RMS(fuera de muestra) por variable (sin centrar: incluye el sesgo), resumida por mediana; IC bootstrap de pacientes."""
    rng = np.random.default_rng(seed)
    a, b = Rtr[cols].values, Roos[cols].values

    def med(a_, b_):
        ra = np.sqrt(np.nanmean(a_ ** 2, axis=0))
        rb = np.sqrt(np.nanmean(b_ ** 2, axis=0))
        ok = rb > 1e-9
        return float(np.median(ra[ok] / rb[ok])), ra[ok] / rb[ok]
    m, per = med(a, b)
    bs = [med(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])[0] for _ in range(n_boot)]
    per_block = {}
    okc = [c for c, ok in zip(cols, np.sqrt(np.nanmean(b ** 2, axis=0)) > 1e-9) if ok]
    for bl in BLOCKS:
        ix = [i for i, c in enumerate(okc) if C.block_of(c) == bl]
        if ix:
            per_block[bl] = float(np.median(per[ix]))
    return {'mediana_ratio': m, 'ic95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            'por_bloque': per_block, 'n_train': int(len(a)), 'n_oos': int(len(b)), 'n_vars': int(len(per))}


def anatomy(res, recs_oos, site, work, elig, k_nn=3):
    """§5.5: R² LOO anidado desde descriptores de la métrica de trabajo; correlación con el promedio de scores de vecinos; D95 abs."""
    metric = work[site]['metrica_trabajo']
    pids = res['pids']
    by = {r['pid']: r for r in recs_oos}
    recs = [by[p] for p in pids]
    desc = M2.descriptors(recs, site, metric, elig)
    Xd = desc.values.astype(float)
    mu = np.nanmean(Xd, axis=0)
    Xd = np.where(np.isnan(Xd), mu, Xd)
    keep = Xd.std(0) > 1e-9
    Xd = Xd[:, keep]
    # vecinos dentro del pool fuera de muestra (el score solo existe ahí), distancia de la métrica de trabajo
    allD = np.load(C.CACHE / f'dist_{site}.npz')
    pid_all = list(allD['pids'])
    D = allD[metric]
    ix = [pid_all.index(p) for p in pids]
    Dsub = D[np.ix_(ix, ix)]
    nn = T.knn_from_dist(Dsub, k_nn)
    cov = np.array([by[p]['d95_over_rx'] for p in pids])
    out = []
    for k in range(res['n_ret']):
        _t(f'anatomia comp {k + 1}/{res["n_ret"]}')
        y = res['scores'][:, k]
        r2, pred = T.loo_r2(Xd, y)
        navg = np.array([y[nn[i]].mean() for i in range(len(y))])
        rn = T.corr_perm_boot(y, navg, n_perm=2000, n_boot=2000, seed=11)
        rc = T.corr_perm_boot(y, cov, n_perm=2000, n_boot=2000, seed=12)
        from scipy.stats import spearmanr
        out.append({'comp': k + 1, 'r2_loo': float(r2), 'r2_ic95': T.r2_boot_ci(y, pred), 'r2_p_perm': T.loo_r2_perm_p(Xd, y, r2),
                    'n_descriptores': int(Xd.shape[1]), 'metrica': metric,
                    'corr_vecinos': rn, 'corr_d95_abs': rc, 'spearman_d95_abs': float(spearmanr(y, cov)[0])})
    return out


def run_site(site, recs_site, work, elig):
    recs_oos = [r for r in recs_site if r['split'] in ('val', 'test')]
    recs_tr = [r for r in recs_site if r['split'] == 'train']
    R_oos_raw = T.residual_matrix(recs_oos)
    R_oos, drop = T.clean_matrix(R_oos_raw)
    R_tr = T.residual_matrix(recs_tr)
    meta = pd.DataFrame([{'pid': r['pid'], 'split': r['split'], 'scale': r['scale'], 'd95_over_rx': r['d95_over_rx']} for r in recs_site])
    # sesgo y magnitud (val+test)
    bias = pd.DataFrame({'variable': R_oos.columns, 'bloque': [C.block_of(c) for c in R_oos.columns],
                         'sesgo_media': R_oos.mean().values, 'desvio': R_oos.std(ddof=1).values, 'n': int(len(R_oos))})
    rng = np.random.default_rng(21)
    v = R_oos_raw[R_oos.columns].values
    mb = np.array([np.nanmean(v[rng.integers(0, len(v), len(v))], axis=0) for _ in range(500)])
    bias['sesgo_ic95_lo'], bias['sesgo_ic95_hi'] = np.percentile(mb, 2.5, axis=0), np.percentile(mb, 97.5, axis=0)
    bias['sesgo_sobre_desvio'] = bias['sesgo_media'] / bias['desvio']
    bias.insert(0, 'sitio', site)
    bias.to_csv(C.TABLAS / f'sesgo_desvio_{site}.csv', index=False)
    res = analyze_matrix(R_oos - R_oos.mean(), f'D_{site}')
    res['site'], res['dropped'], res['bias_summary'] = site, drop, {'n_vars_sesgo_significativo': int(((bias.sesgo_ic95_lo > 0) | (bias.sesgo_ic95_hi < 0)).sum())}
    res['coupling'] = classify_coupling(res['blocks'])
    res['optimism'] = optimism(R_tr, R_oos_raw, list(R_oos.columns))
    res['anatomy'] = anatomy(res, recs_oos, site, work, elig) if res['n_ret'] else []
    res['meta_scale'] = {'escala_mediana_oos': float(meta[meta.split != 'train'].scale.median()),
                         'escala_p5_p95_oos': np.percentile(meta[meta.split != 'train'].scale, [5, 95]).tolist()}
    # lo que necesita la réplica: loadings de los retenidos (corr con variables) y columnas
    pickle.dump(res, open(C.CACHE / f'resD_{site}.pkl', 'wb'))
    return res


def main():
    elig = C.load_eligible()
    work = json.load(open(C.OUT / 'metrica_trabajo.json', encoding='utf-8'))
    recs_all = C.load_cache()
    C.TABLAS.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or list(C.SITES)
    for site in only:
        recs = sorted([r for r in recs_all if r['site'] == site], key=lambda r: r['pid'])
        res = run_site(site, recs, work, elig)
        print(site, 'n_oos', res['n'], 'p', res['p'], 'retenidos', res['n_ret'], 'acople', {k: (round(v['r'], 2), v['signo'], v['significativo']) for k, v in res['coupling'].items()},
              'optimismo', round(res['optimism']['mediana_ratio'], 2), flush=True)


if __name__ == '__main__':
    main()
