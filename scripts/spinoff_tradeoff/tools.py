"""Herramientas estadísticas del spin-off (numpy/scipy puro: el venv no tiene sklearn)."""
import numpy as np

import common as C


# ----------------------------------------------------------------------------- tablas
def tidy_vars(recs, which='real', fine=False):
    """DataFrame (paciente x variable) de 'real' o 'pred'. fine=False: vector compacto (sin dVf_); True: fino (sin dV_)."""
    import pandas as pd
    rows = {}
    for r in recs:
        d = r[which]
        rows[r['pid']] = {k: v for k, v in d.items() if (('|dV_' not in k) if fine else ('|dVf_' not in k))}
    return pd.DataFrame.from_dict(rows, orient='index')


def residual_matrix(recs, fine=False):
    import pandas as pd
    a, b = tidy_vars(recs, 'real', fine), tidy_vars(recs, 'pred', fine)
    cols = [c for c in a.columns if c in b.columns]
    return (a[cols] - b.reindex(a.index)[cols]).astype(float)


def clean_matrix(df, max_nan=0.30):
    """Quita columnas con >30% NaN o sin varianza; imputa NaN con la media (=0 tras centrar). Devuelve (df, descartadas)."""
    drop = {}
    for c in df.columns:
        if df[c].isna().mean() > max_nan:
            drop[c] = 'nan>30%'
        elif np.nanstd(df[c].values) < 1e-9:
            drop[c] = 'sin_varianza'
    out = df.drop(columns=list(drop))
    return out.fillna(out.mean()), drop


# ----------------------------------------------------------------------------- PCA
def pca_eigs(X, standardize=True, scale=None):
    """X (n,p) ya centrada. Devuelve (autovalores, scores U*S, loadings Vt) con la normalización pedida."""
    if scale is not None:
        Xs = X / scale
    elif standardize:
        sd = X.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)   # columna constante (p.ej. remuestreo bootstrap): sin escalar, evita NaN en el SVD
        Xs = X / sd
    else:
        Xs = X
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    return S ** 2 / (X.shape[0] - 1), U * S, Vt, Xs


def unit_scale(cols, X):
    """Escala por variable para la sensibilidad 'sin estandarizar dentro de bloque de unidades homogéneas':
    cada variable se divide por el desvío RMS de su grupo de unidades (pp / % dosis / cm3 / razón); se conserva la
    varianza relativa dentro de cada grupo."""
    sd = X.std(axis=0, ddof=1)
    units = np.array([C.unit_of(c) for c in cols])
    sc = np.empty(len(cols))
    for u in set(units):
        m = units == u
        sc[m] = np.sqrt((sd[m] ** 2).mean())
    return sc


def parallel_analysis(X, n_perm=1000, seed=0, pct=95, standardize=True, scale=None):
    """Horn: permuta cada columna por separado; retiene componentes con autovalor > pct-ésimo de la nula (en secuencia)."""
    rng = np.random.default_rng(seed)
    ev, *_ = pca_eigs(X, standardize, scale)
    n, p = X.shape
    k = min(n, p)
    nulls = np.empty((n_perm, k))
    for b in range(n_perm):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(p)])
        e, *_ = pca_eigs(Xp, standardize, scale)
        nulls[b] = e[:k]
    thr = np.percentile(nulls, pct, axis=0)
    ret = 0
    for i in range(k):
        if ev[i] > thr[i]:
            ret += 1
        else:
            break
    return {'eig': ev[:k], 'null_p95': thr, 'null_median': np.median(nulls, axis=0), 'n_ret': ret}


def cos_abs(a, b):
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bootstrap_loading_stability(X, n_ret, n_boot=500, seed=1, standardize=True, scale=None, topk=None):
    """Para cada componente retenido: mediana (y p5) de la similitud coseno |cos| entre el loading original y el
    componente de mejor coincidencia (entre los primeros `topk`) de cada remuestreo de pacientes."""
    rng = np.random.default_rng(seed)
    _, _, Vt, _ = pca_eigs(X, standardize, scale)
    n = X.shape[0]
    topk = topk or max(n_ret + 3, 6)
    cos = np.zeros((n_boot, n_ret))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb = X[idx] - X[idx].mean(axis=0)
        sc = None if scale is None else scale
        try:
            _, _, Vtb, _ = pca_eigs(Xb, standardize, sc)
        except np.linalg.LinAlgError:
            cos[b] = np.nan
            continue
        for k in range(n_ret):
            cos[b, k] = max(cos_abs(Vt[k], Vtb[j]) for j in range(min(topk, Vtb.shape[0])))
    return {'median': np.nanmedian(cos, axis=0), 'p5': np.nanpercentile(cos, 5, axis=0)}


# ----------------------------------------------------------------------------- scores por bloque y acoplamiento
def block_score(Xblock_std, signs):
    """PC1 del bloque (columnas ya estandarizadas) orientado a 'calidad': el signo se elige para que sum(q_j * loading_j) >= 0
    (q=0 -> la variable no orienta). Devuelve (score, alineación en [0,1]: fracción de |loading| con signo coherente)."""
    U, S, Vt = np.linalg.svd(Xblock_std, full_matrices=False)
    v = Vt[0]
    q = np.asarray(signs, float)
    s = float((q * v).sum())
    if s < 0:
        v = -v
    mask = q != 0
    align = float(((q[mask] * v[mask]) > 0).sum() / max(mask.sum(), 1)) if mask.any() else np.nan
    return Xblock_std @ v, align, v


def corr_perm_boot(x, y, n_perm=10000, n_boot=2000, seed=2):
    rng = np.random.default_rng(seed)
    r = float(np.corrcoef(x, y)[0, 1])
    perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
    p = float((np.abs(perm) >= abs(r)).mean())
    n = len(x)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if x[i].std() > 0 and y[i].std() > 0:
            bs.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {'r': r, 'p_perm': (p * n_perm + 1) / (n_perm + 1), 'ci95': [float(lo), float(hi)], 'n': int(n)}


# ----------------------------------------------------------------------------- ridge LOO anidado
def _ridge_loo_press(X, y, alphas):
    """PRESS leave-one-out por alpha (X centrada, forma cerrada vía SVD)."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    uy = U.T @ y
    best = None
    for a in alphas:
        d = S ** 2 / (S ** 2 + a)
        yhat = U @ (d * uy)
        h = (U ** 2 * d).sum(axis=1)
        press = float((((y - yhat) / (1 - h)) ** 2).sum())
        if best is None or press < best[0]:
            best = (press, a)
    return best[1]


def loo_r2(Xd, y, alphas=np.logspace(-2, 4, 25)):
    """R² con validación cruzada dejando un paciente afuera; ridge cuyo alpha se elige por LOO interno (anidado, sin fuga).
    Xd (n,p) descriptores (se estandarizan DENTRO de cada pliegue), y (n,)."""
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = Xd[tr].mean(0), Xd[tr].std(0)
        sd[sd < 1e-9] = 1.0
        Xt, xi = np.clip((Xd[tr] - mu) / sd, -5, 5), np.clip((Xd[i] - mu) / sd, -5, 5)   # recorte: columnas casi constantes en el pliegue daban |z|~1e3
        ym = y[tr].mean()
        a = _ridge_loo_press(Xt, y[tr] - ym, alphas)
        w = Xt.T @ np.linalg.solve(Xt @ Xt.T + a * np.eye(Xt.shape[0]), y[tr] - ym)   # forma dual (n<<p), igual al ridge primal
        pred[i] = ym + xi @ w
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())   # referencia: media muestral (estándar)
    return 1.0 - ss_res / ss_tot, pred


def r2_boot_ci(y, pred, n_boot=2000, seed=3):
    """IC95 bootstrap del R² CV remuestreando pares (y, predicción LOO) de pacientes (sin re-entrenar: no hay fuga por duplicados)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    v = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        sst = ((y[i] - y[i].mean()) ** 2).sum()
        if sst > 0:
            v.append(1.0 - ((y[i] - pred[i]) ** 2).sum() / sst)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def loo_r2_perm_p(Xd, y, r2, n_perm=100, seed=4):
    rng = np.random.default_rng(seed)
    null = [loo_r2(Xd, rng.permutation(y), alphas=np.logspace(-1, 3, 9))[0] for _ in range(n_perm)]
    return float(((np.array(null) >= r2).sum() + 1) / (n_perm + 1))


# ----------------------------------------------------------------------------- vecinos
def knn_from_dist(D, k, pool=None):
    """D (n,n) simétrica; devuelve índices (n,k) de los k vecinos más cercanos (sin sí mismo), dentro de `pool` (índices) si se da."""
    n = D.shape[0]
    pool = np.arange(n) if pool is None else np.asarray(pool)
    out = {}
    for i in pool:
        d = D[i, pool].copy()
        d[pool == i] = np.inf
        out[int(i)] = pool[np.argsort(d)[:k]]
    return out


# ----------------------------------------------------------------------------- Etapa 1b: nula por bloques y pruebas múltiples
def parallel_analysis_blocks(X, groups, n_perm=1000, seed=0, pct=95, scale=None):
    """Análisis paralelo con la nula por BLOQUES: cada bloque (grupo de columnas) se permuta entero entre pacientes, con una
    permutación propia por bloque. Conserva la estructura interna de cada bloque y rompe solo la relación entre bloques."""
    rng = np.random.default_rng(seed)
    ev, *_ = pca_eigs(X, True, scale)
    n = X.shape[0]
    k = ev.size
    nulls = np.empty((n_perm, k))
    for b in range(n_perm):
        Xp = X.copy()
        for g in groups:
            Xp[:, g] = X[rng.permutation(n)][:, g]
        e, *_ = pca_eigs(Xp, True, scale)
        nulls[b] = e[:k]
    thr = np.percentile(nulls, pct, axis=0)
    ret = 0
    for i in range(k):
        if ev[i] > thr[i]:
            ret += 1
        else:
            break
    return {'eig': ev, 'null_p95': thr, 'null_median': np.median(nulls, axis=0), 'n_ret': ret}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def corr_many(y, Xm, n_perm=10000, n_boot=2000, seed=5):
    """Correlación de y (n,) con cada columna de Xm (n,m): r, p de permutación de y (bilateral), IC95 bootstrap de pacientes."""
    rng = np.random.default_rng(seed)
    n, m = Xm.shape

    def zs(a, axis=0):
        s = a.std(axis=axis, ddof=1, keepdims=True)
        return (a - a.mean(axis=axis, keepdims=True)) / np.where(s < 1e-12, 1.0, s)
    Xz, yz = zs(Xm), zs(y[:, None])[:, 0]
    r = Xz.T @ yz / (n - 1)
    Yp = np.column_stack([rng.permutation(yz) for _ in range(n_perm)])
    null = np.abs(Xz.T @ Yp / (n - 1))
    p = ((null >= np.abs(r)[:, None]).sum(axis=1) + 1) / (n_perm + 1)
    bs = np.empty((n_boot, m))
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        bs[b] = (zs(Xm[i]).T @ zs(y[i][:, None])[:, 0]) / (n - 1)
    return r, p, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
