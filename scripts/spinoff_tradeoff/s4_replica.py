"""
Etapa 1 / Paso E: réplica sin modelo. Solo DVH/dosis REALES, todos los elegibles del sitio, vector fino (OAR en 5 bandas).
Desviación de cada paciente respecto del promedio de sus k vecinos (métrica de trabajo, k=3 y 5, vecinos dentro del sitio).
PCA + análisis paralelo + acoplamiento entre bloques (mismo código que el §5); coseno del subespacio contra el del §5,
llevando el vector fino al compacto vía corr(score, variables compactas).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import tools as T  # noqa: E402
import s3_residual as S3  # noqa: E402


def deviation(V, nn):
    """V DataFrame (n x p); desviación = V_i - nanmean(V[vecinos de i])."""
    A = V.values.astype(float)
    out = np.empty_like(A)
    for i in range(A.shape[0]):
        out[i] = A[i] - np.nanmean(A[nn[i]], axis=0)
    return pd.DataFrame(out, index=V.index, columns=V.columns)


def run_site(site, recs, work):
    metric = work[site]['metrica_trabajo']
    d = np.load(C.CACHE / f'dist_{site}.npz')
    assert list(d['pids']) == [r['pid'] for r in recs]
    D = d[metric]
    resD = pickle.load(open(C.CACHE / f'resD_{site}.pkl', 'rb'))
    Vf, Vc = T.tidy_vars(recs, 'real', fine=True), T.tidy_vars(recs, 'real', fine=False)
    out = {'site': site, 'metrica': metric, 'n': len(recs)}
    for k in (3, 5):
        nn = T.knn_from_dist(D, k)
        devf, dropf = T.clean_matrix(deviation(Vf, nn))
        devc = deviation(Vc, nn)
        res = S3.analyze_matrix(devf - devf.mean(), f'E_{site}_k{k}', n_perm=500, n_boot=300)
        res['dropped'] = dropf
        res['coupling'] = S3.classify_coupling(res['blocks'])
        # subespacio: corr(score_E, variables compactas) vs corr(score_D, variables compactas), mismas variables
        cols = [c for c in resD['cols'] if c in devc.columns]
        ix_d = [resD['cols'].index(c) for c in cols]
        Xc = devc[cols].fillna(devc[cols].mean()).values
        LE = S3.corr_loadings(res['scores'], Xc - Xc.mean(0))
        LD = resD['load'][ix_d, :max(resD['n_ret'], 1)]
        cos_top, idx_top = S3.best_cos(LD, LE, topk=3)
        cos_ret, idx_ret = S3.best_cos(LD, LE, topk=res['n_ret']) if res['n_ret'] else (np.full(LD.shape[1], np.nan), [None] * LD.shape[1])
        res['vs_D'] = {'n_vars_comunes': len(cols), 'cos_mejor_top3': cos_top.tolist(), 'comp_E_top3': idx_top,
                       'cos_mejor_retenidos': [None if np.isnan(x) else float(x) for x in cos_ret], 'comp_E_retenidos': idx_ret,
                       'comp_D_evaluados': list(range(1, LD.shape[1] + 1)) if resD['n_ret'] else []}
        res['loadE_compacto'] = LE
        res['cols_compactas'] = cols
        out[f'k{k}'] = res
    pickle.dump(out, open(C.CACHE / f'resE_{site}.pkl', 'wb'))
    return out


def main():
    work = json.load(open(C.OUT / 'metrica_trabajo.json', encoding='utf-8'))
    recs_all = C.load_cache()
    for site in (sys.argv[1:] or C.SITES):
        recs = sorted([r for r in recs_all if r['site'] == site], key=lambda r: r['pid'])
        o = run_site(site, recs, work)
        for k in ('k3', 'k5'):
            r = o[k]
            print(site, k, 'n', r['n'], 'p', r['p'], 'retenidos', r['n_ret'], 'cos vs D (top3)', np.round(r['vs_D']['cos_mejor_top3'], 2),
                  'acople', {a: (round(v['r'], 2), v['signo'], v['significativo']) for a, v in r['coupling'].items()}, flush=True)


if __name__ == '__main__':
    main()
