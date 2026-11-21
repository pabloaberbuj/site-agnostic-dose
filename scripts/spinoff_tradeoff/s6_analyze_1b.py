"""
Etapa 1b / análisis: residual real(normalización original del plan) - A4(renormalizado a D95(PTV_High)=100%).
Por sitio, val+test. Nula por BLOQUES para el análisis paralelo; prueba directa de compromiso dD95(PTV_High) vs OARs;
carga por bloque de cada componente; ¿es anatómica la concesión de cobertura?; réplica sin modelo; comparación con Etapa 1.
Salidas: results/spinoff_tradeoff/etapa1b/{cache/res1b_<sitio>.pkl, tablas/*.csv}
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
import s3_residual as S3  # noqa: E402

OUT_B = C.ROOT / 'results' / 'spinoff_tradeoff' / 'etapa1b'
CACHE_B = OUT_B / 'cache'
TAB_B = OUT_B / 'tablas'
D95 = 'PTV|PTV_High|D95'
BLOCKS = S3.BLOCKS
N_PA = 1000
MIX_THR = 0.15          # carga media orientada a calidad por bloque para llamar "mezcla de signos opuestos"
R2_ANAT = 0.2


def _t(msg):
    print(f'  [{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def assemble(site, recs1):
    out = []
    for r in recs1:
        b = pickle.load(open(CACHE_B / f'{r["pid"]}.pkl', 'rb'))
        if 'pred' in b:
            pred = b['pred']
        else:   # train: predicho cacheado en la Etapa 1; sin hotspot fuera de PTV (no se usa fuera de muestra)
            pred = {k: v for k, v in r['pred'].items() if k != 'HOT|D0035_BODY'}
            pred['PTV|PTV_High|D95'] = 100.0
            pred['HOT|D0035_out'] = np.nan
        out.append({'pid': r['pid'], 'site': site, 'split': r['split'], 'real': b['real'], 'pred': pred, 'r1': r})
    return out


def groups_of(cols):
    mem = S3.block_members(cols)
    return [np.array(ix) for ix in mem.values()]


def analyze_1b(Xdf, label, n_perm=N_PA, n_boot=500, seed=0):
    cols = list(Xdf.columns)
    X = Xdf.values.astype(float)
    n, p = X.shape
    Xc = X - X.mean(0)
    ev, scores, Vt, Z = T.pca_eigs(Xc, standardize=True)
    groups = groups_of(cols)
    pa_b = T.parallel_analysis_blocks(Xc, groups, n_perm=n_perm, seed=seed)
    pa_c = T.parallel_analysis(Xc, n_perm=max(300, n_perm // 3), seed=seed + 1)   # nula de la Etapa 1 (columnas), solo para comparar
    nret = pa_b['n_ret']
    _t(f'{label}: n={n} p={p} retenidos nula-bloques={nret} (nula-columnas={pa_c["n_ret"]})')
    K = max(nret, 3)
    out = {'label': label, 'cols': cols, 'pids': list(Xdf.index), 'n': n, 'p': p, 'eig': ev[:pa_b['eig'].size], 'pa': pa_b,
           'n_ret_columnas': pa_c['n_ret'], 'pa_cols': pa_c, 'n_ret': nret, 'var_expl': (ev / ev.sum())[:pa_b['eig'].size],
           'scores': scores[:, :K], 'load': S3.corr_loadings(scores[:, :K], Xc)}
    if nret:
        st = T.bootstrap_loading_stability(Xc, nret, n_boot=n_boot, seed=seed + 2)
        out['stab'] = {'median': st['median'].tolist(), 'p5': st['p5'].tolist()}
    else:
        out['stab'] = {'median': [], 'p5': []}
    out['blocks'] = S3.block_coupling(cols, Xc, seed=seed + 3)    # permutar el score de un bloque = permutar pacientes por bloque entero
    out['coupling'] = S3.classify_coupling(out['blocks'])
    # carga por bloque de cada componente, orientada a calidad
    mem = S3.block_members(cols)
    Lr = out['load'][:, :max(nret, 1)] if nret else out['load'][:, :0]
    rows = []
    for k in range(Lr.shape[1]):
        d = {'comp': k + 1}
        for b, ix in mem.items():
            qs = [(C.quality_sign(cols[j]), Lr[j, k]) for j in ix if C.quality_sign(cols[j]) != 0]
            d[f'suma_{b}'] = float(sum(q * l for q, l in qs)) if qs else np.nan
            d[f'media_{b}'] = float(np.mean([q * l for q, l in qs])) if qs else np.nan
        m = [d[f'media_{b}'] for b in mem if not np.isnan(d[f'media_{b}'])]
        d['mezcla_signos_opuestos'] = bool(any(x > MIX_THR for x in m) and any(x < -MIX_THR for x in m))
        d['carga_dD95'] = float(Lr[cols.index(D95), k]) if D95 in cols else np.nan
        rows.append(d)
    out['comp_block'] = rows
    return out


def direct_test(Rdf, blocks):
    """dD95(PTV_High) vs dV media y dDmean de cada slot OAR + score de calidad de OAR/HOT/FALL. Signo en calidad
    (cobertura: + = más cobertura que lo predicho; OAR: + = menos dosis). r<0 = antagónico (menos cobertura <-> OAR mejor)."""
    y = Rdf[D95].values
    labs, cols_x, kinds = [], [], []
    for c in Rdf.columns:
        if c.startswith('OAR|') and (c.endswith('|dV_media_40_80') or c.endswith('|dDmean')):
            labs.append(c.replace('OAR|', ''))
            cols_x.append(-Rdf[c].values)      # calidad OAR = -residual
            kinds.append('OAR_slot')
    for b in ('OAR', 'HOT', 'FALL'):
        if b in blocks['scores']:
            labs.append(f'score_{b}')
            cols_x.append(blocks['scores'][b])
            kinds.append('bloque')
    Xm = np.column_stack(cols_x)
    r, p, lo, hi = T.corr_many(y, Xm)
    padj = T.holm(p)
    df = pd.DataFrame({'prueba': labs, 'tipo': kinds, 'r_calidad': r, 'ic95_lo': lo, 'ic95_hi': hi, 'p_perm': p, 'p_holm': padj, 'n': len(y)})
    df['signo'] = np.where(df.r_calidad < 0, 'antagonico', 'sinergico')
    df['significativo'] = (df.p_holm < 0.05) & ((df.ic95_lo > 0) | (df.ic95_hi < 0))
    return df


def anatomy_dD95(recs_all, site, work, elig, pool_pids):
    """R² LOO anidado de dD95 desde descriptores B; corr con promedio de vecinos (k=3, métrica B). Pool = todos o val+test."""
    from scipy.stats import spearmanr
    by = {r['pid']: r for r in recs_all}
    d = np.load(C.CACHE / f'dist_{site}.npz')
    pid_all = list(d['pids'])
    desc = M2.descriptors([by[p]['r1'] for p in pid_all], site, 'B', elig)
    Xd_all = desc.values.astype(float)
    Xd_all = np.where(np.isnan(Xd_all), np.nanmean(Xd_all, axis=0), Xd_all)
    out = {}
    for name, pool in pool_pids.items():
        ix = [pid_all.index(p) for p in pool]
        Xd = Xd_all[ix]
        Xd = Xd[:, Xd.std(0) > 1e-9]
        y = np.array([by[p]['real'][D95] - 100.0 for p in pool])
        r2, pred = T.loo_r2(Xd, y)
        D = d['B'][np.ix_(ix, ix)]
        nn = T.knn_from_dist(D, 3)
        navg = np.array([y[nn[i]].mean() for i in range(len(y))])
        out[name] = {'n': len(y), 'n_descriptores': int(Xd.shape[1]), 'dD95_media': float(y.mean()), 'dD95_sd': float(y.std(ddof=1)),
                     'r2_loo': float(r2), 'r2_ic95': T.r2_boot_ci(y, pred), 'r2_p_perm': T.loo_r2_perm_p(Xd, y, r2),
                     'corr_vecinos': T.corr_perm_boot(y, navg, n_perm=5000, n_boot=2000, seed=13),
                     'spearman_vecinos': float(spearmanr(y, navg)[0])}
    return out


def run_site(site, recs1, work, elig, stage1):
    recs = assemble(site, recs1)
    oos = [r for r in recs if r['split'] in ('val', 'test')]
    R_raw = T.residual_matrix(oos)
    R, drop = T.clean_matrix(R_raw)
    assert D95 in R.columns, 'dD95 descartada por limpieza'
    rng = np.random.default_rng(21)
    v = R_raw[R.columns].values
    mb = np.array([np.nanmean(v[rng.integers(0, len(v), len(v))], axis=0) for _ in range(500)])
    bias = pd.DataFrame({'sitio': site, 'variable': R.columns, 'bloque': [C.block_of(c) for c in R.columns], 'sesgo_media': R.mean().values,
                         'desvio': R.std(ddof=1).values, 'sesgo_ic95_lo': np.percentile(mb, 2.5, axis=0), 'sesgo_ic95_hi': np.percentile(mb, 97.5, axis=0),
                         'n': len(R)})
    bias.to_csv(TAB_B / f'sesgo_desvio_1b_{site}.csv', index=False)
    res = analyze_1b(R - R.mean(), f'D1b_{site}')
    res.update({'site': site, 'dropped': drop})
    res['directa'] = direct_test(R - R.mean(), res['blocks'])
    # score de cada componente retenido vs dD95
    res['comp_vs_dD95'] = [T.corr_perm_boot(res['scores'][:, k], R[D95].values, n_perm=5000, n_boot=2000, seed=17) for k in range(res['n_ret'])]
    pools = {'todos_los_elegibles': [r['pid'] for r in recs], 'val+test': [r['pid'] for r in oos]}
    res['anatomia_dD95'] = anatomy_dD95(recs, site, work, elig, pools)
    # comparación con la Etapa 1 (variables comunes)
    s1 = stage1[site]
    common = [c for c in res['cols'] if c in s1['cols']]
    ia, ib = [res['cols'].index(c) for c in common], [s1['cols'].index(c) for c in common]
    La = res['load'][ia][:, :max(res['n_ret'], 1)]
    Lb = s1['load'][ib][:, :max(s1['n_ret'], 1)]
    cm = np.array([[T.cos_abs(La[:, i], Lb[:, j]) for j in range(Lb.shape[1])] for i in range(La.shape[1])])
    # eje de derrame (Pelvis): componente de la Etapa 1 con mayor carga media |.| sobre el bloque FALL
    fall_ix = [i for i, c in enumerate(s1['cols']) if C.block_of(c) == 'FALL']
    spill = int(np.argmax([np.mean(np.abs(s1['load'][fall_ix, k])) for k in range(max(s1['n_ret'], 1))])) if fall_ix and s1['n_ret'] else None
    res['vs_etapa1'] = {'n_vars_comunes': len(common), 'cos': cm.tolist(), 'comp_1b': list(range(1, La.shape[1] + 1)), 'comp_etapa1': list(range(1, Lb.shape[1] + 1)),
                        'eje_derrame_etapa1': None if spill is None else spill + 1}
    pickle.dump(res, open(CACHE_B / f'res1b_{site}.pkl', 'wb'))
    return res, recs


def replica(site, recs, work, resD):
    d = np.load(C.CACHE / f'dist_{site}.npz')
    assert list(d['pids']) == [r['pid'] for r in recs]
    D = d['B']
    Vf, Vc = T.tidy_vars(recs, 'real', fine=True), T.tidy_vars(recs, 'real', fine=False)
    out = {}
    for k in (3, 5):
        nn = T.knn_from_dist(D, k)

        def dev(V):
            A = V.values.astype(float)
            o = np.empty_like(A)
            for i in range(A.shape[0]):
                o[i] = A[i] - np.nanmean(A[nn[i]], axis=0)
            return pd.DataFrame(o, index=V.index, columns=V.columns)
        devf, dropf = T.clean_matrix(dev(Vf))
        devc = dev(Vc)
        res = analyze_1b(devf - devf.mean(), f'E1b_{site}_k{k}', n_perm=500, n_boot=300)
        cols = [c for c in resD['cols'] if c in devc.columns]
        ix_d = [resD['cols'].index(c) for c in cols]
        Xc = devc[cols].fillna(devc[cols].mean()).values
        LE = S3.corr_loadings(res['scores'], Xc - Xc.mean(0))
        LD = resD['load'][ix_d, :max(resD['n_ret'], 1)]
        cos_top, idx_top = S3.best_cos(LD, LE, topk=3)
        res['vs_D'] = {'n_vars_comunes': len(cols), 'cos_mejor_top3': cos_top.tolist(), 'comp_E_top3': idx_top}
        res['n_pacientes'] = len(recs)
        out[f'k{k}'] = res
    return out


def read_site(site, res, stage1_ok=True):
    """Lectura predeclarada (tabla del §4 del prompt 1b)."""
    tags = []
    dt = res['directa']
    oar = dt[(dt.tipo == 'OAR_slot') | (dt.prueba == 'score_OAR')]
    anti = oar[oar.significativo & (oar.signo == 'antagonico')]
    sine = oar[oar.significativo & (oar.signo == 'sinergico')]
    an_all = res['anatomia_dD95']['todos_los_elegibles']
    an_vt = res['anatomia_dD95']['val+test']
    if len(anti):
        if an_all['r2_loo'] >= R2_ANAT:
            tags.append(f'dD95 antagonico con OARs ({len(anti)} pruebas sig., Holm) y R² anatomico {an_all["r2_loo"]:.2f}>=0.2: concesion de cobertura = compromiso real, '
                        'en parte predecible desde la anatomia -> candidato a salida extra del modelo principal / modelo clasico')
        else:
            tags.append(f'dD95 antagonico con OARs ({len(anti)} pruebas sig., Holm) y R² anatomico {an_all["r2_loo"]:.2f}<0.2: compromiso real pero de criterio del planificador '
                        '-> candidato a coordenada de preferencia')
    elif len(sine):
        tags.append(f'dD95 SINERGICO con OARs ({len(sine)} pruebas sig.): mas cobertura y OAR mejor a la vez (no es compromiso)')
    else:
        tags.append('dD95 sin acoplamiento significativo con OARs: la cobertura varia por politica o costumbre, no como compromiso')
    comps = res['comp_block']
    if res['n_ret'] and D95 in res['cols']:
        c1 = abs(res['comp_vs_dD95'][0]['r'])
        if c1 > 0.9:
            tags.append(f'El componente 1 carga casi solo en dD95 (|r|={c1:.2f}): reportar los siguientes como resultado de interes')
    s1 = res['vs_etapa1']
    if site == 'Pelvis' and s1['eje_derrame_etapa1'] is not None:
        j = s1['eje_derrame_etapa1'] - 1
        cm = np.array(s1['cos'])
        i = int(np.argmax(cm[:, j])) if cm.size else None
        if i is not None:
            cb = comps[i]
            sin = (cb.get('media_OAR', np.nan) * cb.get('media_FALL', np.nan)) > 0
            cdd = res['comp_vs_dD95'][i]
            sin_acopl = not (cdd['p_perm'] < 0.05 and (cdd['ci95'][0] > 0 or cdd['ci95'][1] < 0))
            tags.append(f'Eje de derrame de Pelvis (Etapa 1 comp {j + 1}): coseno max {cm[i, j]:.2f} con comp {i + 1} de 1b; sinergico={sin}; sin acoplamiento con dD95={sin_acopl}; '
                        + ('SOBREVIVE (eje de calidad del plan independiente de la normalizacion)' if cm[i, j] > 0.7 and sin and sin_acopl else 'no cumple el patron completo'))
    return tags
