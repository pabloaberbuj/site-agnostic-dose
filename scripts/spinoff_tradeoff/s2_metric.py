"""
Etapa 1 / Paso C: descriptores anatómicos (solo geometría) y elección de la métrica de similitud.

  A (solo Próstata): 7 features de d3_sonda_vecinos.py, estandarizadas, euclídea.
  B (OVH por rol):  por slot de OAR, curva OVH(r) respecto de PTV_High y de la unión de PTV; distancia = área entre curvas (mm)
                    promediada sobre los slots presentes en ambos pacientes (peso igual por slot y por referencia). Escalares
                    (vol OAR, vol PTV por nivel, fracción del OAR dentro del rango Z de los PTV, área media BODY en ese rango)
                    estandarizados, euclídea/sqrt(p). Combinación: d = sqrt((d_curvas/med)^2 + (d_escalares/med)^2), med = mediana
                    de las distancias entre pares del sitio (peso igual entre los dos grupos).
  C (cota): euclídea entre DVH predichos por A4, vector compacto (3 bandas por slot), estandarizado.
Criterio (igual que D3): k=3 vecinos dentro del sitio, spread (desvío, ddof=0) del DVH REAL en grupos de 4, por slot y banda,
contra k=3 vecinos al azar del mismo sitio (100 repeticiones).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import tools as T  # noqa: E402
from s1_extract import all_slot_names  # noqa: E402

K = 3
N_RAND = 100
OVH_R_DESC = [2.5, 5.0, 10.0, 20.0, 40.0]   # radios (mm) en los que se muestrea el OVH para los descriptores de la regresión


def offdiag_median(D):
    n = D.shape[0]
    return float(np.median(D[~np.eye(n, dtype=bool)]))


def zscore_impute(M):
    M = np.asarray(M, float)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (M - mu) / sd
    return np.where(np.isnan(Z), 0.0, Z)


def euclid(Z):
    d = Z[:, None, :] - Z[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def scalars_B(recs, slots, levels):
    rows = []
    for r in recs:
        g = r['geom']
        row = {}
        for s in slots:
            row[f'vol|{s}'] = g['vol_oar_cc'].get(s, np.nan)
            row[f'fz|{s}'] = g['frac_in_z'].get(s, np.nan)
        for l in levels:
            row[f'volptv|{l}'] = g['vol_ptv_cc'].get(l, np.nan)
        row['body_area'] = g['body_area_cm2']
        rows.append(row)
    return pd.DataFrame(rows)


def ovh_arrays(recs, slots, key):
    A = np.full((len(recs), len(slots), len(C.R_GRID)), np.nan, dtype=np.float32)
    for i, r in enumerate(recs):
        for j, s in enumerate(slots):
            if s in r['geom'][key]:
                A[i, j] = r['geom'][key][s]
    return A


def dist_B(recs, site, elig):
    slots = all_slot_names(elig[site]['elegibles'])
    levels = C.PTV_LEVELS[site]
    refs = ['ovh_high'] if len(levels) == 1 else ['ovh_high', 'ovh_union']
    n = len(recs)
    num, cnt = np.zeros((n, n)), np.zeros((n, n))
    dr = C.R_GRID[1] - C.R_GRID[0]
    for key in refs:
        A = ovh_arrays(recs, slots, key)
        for j in range(len(slots)):
            a = A[:, j, :]
            ok = ~np.isnan(a[:, 0])
            idx = np.flatnonzero(ok)
            if idx.size < 2:
                continue
            diff = np.abs(a[idx][:, None, :] - a[idx][None, :, :])
            area = (diff.sum(-1) - 0.5 * (diff[..., 0] + diff[..., -1])) * dr   # trapecio, mm
            num[np.ix_(idx, idx)] += area
            cnt[np.ix_(idx, idx)] += 1
    dc = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    dc = np.where(np.isnan(dc), np.nanmedian(dc), dc)
    S = scalars_B(recs, slots, levels)
    Z = zscore_impute(S.values)
    ds = euclid(Z) / np.sqrt(Z.shape[1])
    d = np.sqrt((dc / offdiag_median(dc)) ** 2 + (ds / offdiag_median(ds)) ** 2)
    np.fill_diagonal(d, 0.0)
    return d, slots


def feats_A(recs):
    keys = list(recs[0]['geom']['f7'])
    return pd.DataFrame([[r['geom']['f7'][k] for k in keys] for r in recs], columns=keys)


def dist_A(recs):
    Z = zscore_impute(feats_A(recs).values)
    return euclid(Z)


def pred_compact_vec(recs, slots):
    M = np.full((len(recs), len(slots) * 3), np.nan)
    for i, r in enumerate(recs):
        for j, s in enumerate(slots):
            if s in r['curves']['pred']:
                M[i, 3 * j:3 * j + 3] = list(C.band_means(r['curves']['pred'][s]).values())
    return M


def dist_C(recs, slots):
    return euclid(zscore_impute(pred_compact_vec(recs, slots)))


def descriptors(recs, site, metric, elig):
    """Descriptores anatómicos de la métrica de trabajo (para el R² de §5.5). Solo geometría."""
    if metric == 'A':
        return feats_A(recs)
    slots = all_slot_names(elig[site]['elegibles'])
    S = scalars_B(recs, slots, C.PTV_LEVELS[site])
    cols = {}
    ridx = [int(np.argmin(np.abs(C.R_GRID - r))) for r in OVH_R_DESC]
    for key in (['ovh_high'] if len(C.PTV_LEVELS[site]) == 1 else ['ovh_high', 'ovh_union']):
        A = ovh_arrays(recs, slots, key)
        for j, s in enumerate(slots):
            for r, ri in zip(OVH_R_DESC, ridx):
                cols[f'{key}|{s}|r{r}'] = A[:, j, ri]
    return pd.concat([S, pd.DataFrame(cols)], axis=1)


def curve_matrix(recs, slots, which='real'):
    """(n, n_slots, 111) con NaN si el slot no existe."""
    M = np.full((len(recs), len(slots), len(C.D_BINS)), np.nan, dtype=np.float32)
    for i, r in enumerate(recs):
        for j, s in enumerate(slots):
            if s in r['curves'][which]:
                M[i, j] = r['curves'][which][s]
    return M


def group_spread(CM, G):
    """CM (n,S,111); G (n,k+1) índices del grupo. -> (n,S,3 bandas): media sobre bins de la banda del desvío (ddof=0) del grupo."""
    grp = CM[G]                       # (n,k+1,S,111)
    sd = grp.std(axis=1)              # NaN si algún miembro no tiene el slot
    keys = list(C.BANDS_COMPACT)
    out = np.stack([sd[..., C.band_idx(*C.BANDS_COMPACT[k], last=(i == len(keys) - 1))].mean(-1) for i, k in enumerate(keys)], axis=-1)
    return out


def boot_ci(v, n_boot=1000, seed=5):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, (n_boot, v.size))].mean(1)
    return float(v.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], int(v.size)


def main():
    elig = C.load_eligible()
    recs_all = C.load_cache()
    C.TABLAS.mkdir(parents=True, exist_ok=True)
    rows, agg, work = [], [], {}
    for site in C.SITES:
        recs = sorted([r for r in recs_all if r['site'] == site], key=lambda r: r['pid'])
        n = len(recs)
        slots = all_slot_names(elig[site]['elegibles'])
        CM = curve_matrix(recs, slots, 'real')
        dB, _ = dist_B(recs, site, elig)
        cands = {'B': dB, 'C': dist_C(recs, slots)}
        if site == 'Prostata':
            cands['A'] = dist_A(recs)
        pids = [r['pid'] for r in recs]
        np.savez(C.CACHE / f'dist_{site}.npz', pids=np.array(pids), **cands)
        rng = np.random.default_rng(7)
        res = {}
        for cname, D in cands.items():
            nn = T.knn_from_dist(D, K)
            G = np.array([[i, *nn[i]] for i in range(n)])
            res[cname] = group_spread(CM, G)
        rnd = np.zeros_like(next(iter(res.values())))
        cnt = np.zeros_like(rnd)
        for _ in range(N_RAND):
            G = np.array([[i, *rng.choice(np.delete(np.arange(n), i), K, replace=False)] for i in range(n)])
            s = group_spread(CM, G)
            rnd += np.nan_to_num(s)
            cnt += ~np.isnan(s)
        rnd = np.where(cnt > 0, rnd / np.maximum(cnt, 1), np.nan)
        # el azar usa la misma máscara de validez por paciente/slot que el grupo de vecinos de la candidata más completa: se compara
        # sobre celdas (paciente, slot) válidas en TODAS las candidatas para que sean comparables
        valid = np.all([~np.isnan(v) for v in res.values()], axis=0) & ~np.isnan(rnd)
        bands = list(C.BANDS_COMPACT)
        for cname, S in res.items():
            for j, sl in enumerate(slots):
                for b, bn in enumerate(bands):
                    v = valid[:, j, b]
                    if v.sum() == 0:
                        continue
                    rows.append({'sitio': site, 'candidata': cname, 'slot': sl, 'banda': bn, 'spread_pp': float(S[:, j, b][v].mean()),
                                 'azar_pp': float(rnd[:, j, b][v].mean()), 'n': int(v.sum())})
            for b, bn in enumerate(bands):
                per_pat = np.nanmean(np.where(valid[:, :, b], S[:, :, b], np.nan), axis=1)
                per_rnd = np.nanmean(np.where(valid[:, :, b], rnd[:, :, b], np.nan), axis=1)
                m, ci, nn_ = boot_ci(per_pat)
                mr, cir, _ = boot_ci(per_rnd)
                agg.append({'sitio': site, 'candidata': cname, 'banda': bn, 'spread_pp': m, 'ic95': ci, 'azar_pp': mr,
                            'ic95_azar': cir, 'ratio_vs_azar': m / mr, 'n_pacientes': nn_})
        df_a = pd.DataFrame([a for a in agg if a['sitio'] == site])
        score = df_a[df_a.candidata.isin(['A', 'B'])].groupby('candidata')['ratio_vs_azar'].mean()
        work[site] = {'metrica_trabajo': str(score.idxmin()), 'ratio_medio_vs_azar': {k: float(v) for k, v in score.items()},
                      'nota': 'ratio = spread_vecinos/spread_azar promediado sobre las 3 bandas; C se reporta pero no se usa'}
        print(site, n, work[site], flush=True)
    pd.DataFrame(rows).to_csv(C.TABLAS / 'metrica_spread_por_slot.csv', index=False)
    dfa = pd.DataFrame(agg)
    dfa['ic95'] = dfa['ic95'].apply(lambda x: f'[{x[0]:.2f},{x[1]:.2f}]')
    dfa['ic95_azar'] = dfa['ic95_azar'].apply(lambda x: f'[{x[0]:.2f},{x[1]:.2f}]')
    dfa.to_csv(C.TABLAS / 'metrica_eleccion.csv', index=False)
    json.dump(work, open(C.OUT / 'metrica_trabajo.json', 'w', encoding='utf-8'), indent=2)
    print(dfa.to_string())


if __name__ == '__main__':
    main()
