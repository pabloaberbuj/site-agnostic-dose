"""Etapa 1 / entregables: tablas, gráficos, summary.json y resumen.md. Lee los pkl de s3/s4."""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import tools as T  # noqa: E402
import s3_residual as S3  # noqa: E402

STAB_MIN, REPL_MIN, R2_MAX = 0.8, 0.7, 0.2   # umbrales PREDECLARADOS (prompt §9); no se ajustan


def fnum(x, d=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def read_site(site, D, E):
    """Lectura predeclarada para un sitio (tabla del §9). Devuelve (etiquetas, detalle por componente)."""
    comps = []
    coup = D['coupling']
    antag = [k for k, v in coup.items() if v['significativo'] and v['signo'] == 'antagonico']
    sinerg = [k for k, v in coup.items() if v['significativo'] and v['signo'] == 'sinergico']
    for k in range(D['n_ret']):
        stab = D['stab']['median'][k]
        an = D['anatomy'][k] if D['anatomy'] else None
        cos3 = E['k3']['vs_D']['cos_mejor_top3'][k]
        cos5 = E['k5']['vs_D']['cos_mejor_top3'][k]
        comps.append({'comp': k + 1, 'estable': bool(stab > STAB_MIN), 'estabilidad_cos_mediana': fnum(stab),
                      'r2_anatomico': fnum(an['r2_loo']) if an else None, 'r2_bajo': bool(an and an['r2_loo'] < R2_MAX),
                      'replica_cos_k3': fnum(cos3), 'replica_cos_k5': fnum(cos5), 'replica': bool(cos3 > REPL_MIN)})
    tags = []
    if D['n_ret'] == 0:
        tags.append('Nada sobre la nula: sin estructura detectable con este n (evaluar out-of-fold antes de descartar)')
    ok = [c for c in comps if c['estable'] and c['r2_bajo'] and c['replica']]
    if ok and antag:
        tags.append(f'Hay eje de compromiso utilizable como coordenada (comp {[c["comp"] for c in ok]}; antagonico OAR vs {antag})')
    if any(c['estable'] for c in comps) and sinerg:
        tags.append(f'Eje de calidad global (sinergico OAR vs {sinerg}): rankea planes, no es coordenada de preferencia')
    if any(c['estable'] and not c['r2_bajo'] for c in comps):
        tags.append('R² anatomico alto en componente estable: anatomia que A4 no captura (vuelve al eje principal como hallazgo)')
    if D['n_ret'] and not tags:
        tags.append('Componente(s) sobre la nula pero sin cumplir el patron predeclarado de ninguna fila (ver detalle)')
    return tags, comps


def table_eig(res_by_site):
    rows = []
    for s, D in res_by_site.items():
        for i, (e, t, m) in enumerate(zip(D['eig'], D['pa']['null_p95'], D['pa']['null_median'])):
            rows.append({'sitio': s, 'comp': i + 1, 'autovalor': e, 'nula_p95': t, 'nula_mediana': m, 'retenido': i < D['n_ret'],
                         'var_explicada': D['var_expl'][i], 'n': D['n'], 'p': D['p']})
    return pd.DataFrame(rows)


def plot_scree(DD, path):
    fig, axes = plt.subplots(1, len(DD), figsize=(5 * len(DD), 3.8))
    for ax, (s, D) in zip(np.atleast_1d(axes), DD.items()):
        k = min(15, len(D['eig']))
        x = np.arange(1, k + 1)
        ax.fill_between(x, D['pa']['null_median'][:k], D['pa']['null_p95'][:k], color='0.8', label='nula (mediana-p95)')
        ax.plot(x, D['eig'][:k], 'o-', color='C3', label='residual real-A4')
        ax.set_title(f'{s}  (n={D["n"]}, p={D["p"]}, retenidos={D["n_ret"]})')
        ax.set_xlabel('componente')
        ax.set_ylabel('autovalor (matriz de correlación)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_loadings(DD, path):
    fig, axes = plt.subplots(1, len(DD), figsize=(5.5 * len(DD), 11))
    for ax, (s, D) in zip(np.atleast_1d(axes), DD.items()):
        K = max(D['n_ret'], 1) if D['n_ret'] else 3
        order = sorted(range(D['p']), key=lambda j: (C.BLOCKS_ORDER.index(C.block_of(D['cols'][j])), j))
        L = D['load'][order][:, :K]
        im = ax.imshow(L, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([D['cols'][j].replace('OAR|', '').replace('dV_', 'V:') for j in order], fontsize=3.5)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f'PC{i + 1}' + ('' if i < D['n_ret'] else '*') for i in range(K)])
        ax.set_title(f'{s}: corr(score, variable)\n' + ('* = no supera la nula (solo referencia)' if D['n_ret'] < K else ''), fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.3)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_blocks(DD, path):
    fig, axes = plt.subplots(len(DD), 3, figsize=(11, 3.4 * len(DD)))
    axes = np.atleast_2d(axes)
    for r, (s, D) in enumerate(DD.items()):
        sc = D['blocks']['scores']
        for c, o in enumerate(('PTV', 'HOT', 'FALL')):
            ax = axes[r, c]
            if 'OAR' in sc and o in sc:
                ax.scatter(sc['OAR'], sc[o], s=14, alpha=.7)
                v = D['blocks']['corr'].get(f'OAR~{o}')
                ax.set_title(f'{s}: OAR vs {o}  r={v["r"]:.2f} [{v["ci95"][0]:.2f},{v["ci95"][1]:.2f}] p={v["p_perm"]:.3g}', fontsize=8)
            ax.set_xlabel('score OAR (+ = mejor que lo predicho)', fontsize=7)
            ax.set_ylabel(f'score {o} (+ = mejor)', fontsize=7)
            ax.axhline(0, color='0.8', lw=.5)
            ax.axvline(0, color='0.8', lw=.5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def contrast(recs_all, DD):
    rows = []
    comunes = None
    mats = {}
    for s in C.SITES:
        recs = [r for r in recs_all if r['site'] == s and r['split'] in ('val', 'test')]
        R = T.residual_matrix(recs)
        mats[s] = R
        cs = {c for c in R.columns if c.startswith(('HOT|', 'FALL|')) or c.startswith('PTV|PTV_High|')}
        comunes = cs if comunes is None else comunes & cs
    rng = np.random.default_rng(31)
    for v in sorted(comunes):
        for s in C.SITES:
            x = mats[s][v].dropna().values
            if len(x) < 5:
                continue
            bs = [np.std(x[rng.integers(0, len(x), len(x))], ddof=1) for _ in range(500)]
            rows.append({'variable': v, 'sitio': s, 'desvio': float(np.std(x, ddof=1)), 'ic95_lo': float(np.percentile(bs, 2.5)),
                         'ic95_hi': float(np.percentile(bs, 97.5)), 'sesgo': float(x.mean()), 'n': len(x)})
    return pd.DataFrame(rows)


def main():
    C.BLOCKS_ORDER = ['OAR', 'PTV', 'HOT', 'FALL']
    C.GRAFICOS.mkdir(parents=True, exist_ok=True)
    work = json.load(open(C.OUT / 'metrica_trabajo.json', encoding='utf-8'))
    elig = C.load_eligible()
    recs_all = C.load_cache()
    DD = {s: pickle.load(open(C.CACHE / f'resD_{s}.pkl', 'rb')) for s in C.SITES}
    EE = {s: pickle.load(open(C.CACHE / f'resE_{s}.pkl', 'rb')) for s in C.SITES}

    table_eig(DD).to_csv(C.TABLAS / 'autovalores_vs_nula.csv', index=False)
    rows = []
    for s, D in DD.items():
        K = max(D['n_ret'], 3)
        for j, c in enumerate(D['cols']):
            rows.append({'sitio': s, 'variable': c, 'bloque': C.block_of(c), **{f'PC{k + 1}': D['load'][j, k] for k in range(K)}})
    pd.DataFrame(rows).to_csv(C.TABLAS / 'loadings_corr.csv', index=False)
    rows = []
    for s, D in DD.items():
        for k, v in D['blocks']['corr'].items():
            rows.append({'sitio': s, 'analisis': 'residual_real_menos_A4', 'par': k, 'r': v['r'], 'ic95_lo': v['ci95'][0], 'ic95_hi': v['ci95'][1],
                         'p_perm': v['p_perm'], 'n': v['n']})
        for kk in ('k3', 'k5'):
            for k, v in EE[s][kk]['blocks']['corr'].items():
                rows.append({'sitio': s, 'analisis': f'replica_{kk}', 'par': k, 'r': v['r'], 'ic95_lo': v['ci95'][0], 'ic95_hi': v['ci95'][1],
                             'p_perm': v['p_perm'], 'n': v['n']})
    pd.DataFrame(rows).to_csv(C.TABLAS / 'acoplamiento_bloques.csv', index=False)
    rows = [{'sitio': s, 'comp': a['comp'], 'r2_loo': a['r2_loo'], 'r2_ic95_lo': a['r2_ic95'][0], 'r2_ic95_hi': a['r2_ic95'][1], 'p_perm': a['r2_p_perm'],
             'metrica': a['metrica'], 'n_descriptores': a['n_descriptores'], 'r_vecinos': a['corr_vecinos']['r'],
             'r_vecinos_ic_lo': a['corr_vecinos']['ci95'][0], 'r_vecinos_ic_hi': a['corr_vecinos']['ci95'][1],
             'r_d95abs': a['corr_d95_abs']['r'], 'r_d95abs_ic_lo': a['corr_d95_abs']['ci95'][0], 'r_d95abs_ic_hi': a['corr_d95_abs']['ci95'][1],
             'spearman_d95abs': a['spearman_d95_abs'], 'n': DD[s]['n']} for s, D in DD.items() for a in D['anatomy']]
    pd.DataFrame(rows).to_csv(C.TABLAS / 'anatomia_r2.csv', index=False)
    rows = [{'sitio': s, 'k': kk, 'comp_residual': i + 1, 'cos_mejor_top3_replica': E[kk]['vs_D']['cos_mejor_top3'][i],
             'cos_mejor_retenidos_replica': E[kk]['vs_D']['cos_mejor_retenidos'][i], 'n_retenidos_replica': E[kk]['n_ret']}
            for s, E in EE.items() for kk in ('k3', 'k5') for i in range(len(E[kk]['vs_D']['comp_D_evaluados']))]
    pd.DataFrame(rows).to_csv(C.TABLAS / 'replica_coseno.csv', index=False)
    rows = [{'sitio': s, 'razon_mediana_train_sobre_oos': D['optimism']['mediana_ratio'], 'ic95_lo': D['optimism']['ic95'][0],
             'ic95_hi': D['optimism']['ic95'][1], **{f'bloque_{b}': v for b, v in D['optimism']['por_bloque'].items()},
             'n_train': D['optimism']['n_train'], 'n_oos': D['optimism']['n_oos']} for s, D in DD.items()]
    pd.DataFrame(rows).to_csv(C.TABLAS / 'optimismo_train.csv', index=False)
    ct = contrast(recs_all, DD)
    ct.to_csv(C.TABLAS / 'contraste_sitios_varianza.csv', index=False)
    plot_scree(DD, C.GRAFICOS / 'scree_nula.png')
    plot_loadings(DD, C.GRAFICOS / 'heatmap_loadings.png')
    plot_blocks(DD, C.GRAFICOS / 'scores_entre_bloques.png')

    # ---- summary.json ----
    n_scale = {s: {'n_train': sum(r['split'] == 'train' for r in recs_all if r['site'] == s),
                   'n_val': sum(r['split'] == 'val' for r in recs_all if r['site'] == s),
                   'n_test': sum(r['split'] == 'test' for r in recs_all if r['site'] == s)} for s in C.SITES}
    summary = {'etapa': 'spinoff_tradeoff_etapa1', 'modelo': 'exp_a4_003_unet2d_psdm_c176x256 ep113', 'armonizacion_hash': C.HASH_ESPERADO,
               'pacientes': n_scale,
               'uso_del_test': 'Analisis no supervisado, sin seleccion de modelo. No se reportan metricas agregadas de desempeno (MAE ni mean|dV| por banda); solo estructura del residual (val+test, fuera de muestra).',
               'oars_elegibles': {s: {'incluidos': elig[s]['elegibles'], 'descartados_por_presencia<90%': elig[s]['descartados_por_presencia']} for s in C.SITES},
               'metrica_trabajo': work, 'sitios': {}}
    for s in C.SITES:
        D, E = DD[s], EE[s]
        tags, comps = read_site(s, D, E)
        summary['sitios'][s] = {
            'tipo': 'manual' if s in C.MANUALES else 'RapidPlan (contraste)', 'n_val_test': D['n'], 'n_variables': D['p'],
            'variables_descartadas': D['dropped'], 'componentes_retenidos': D['n_ret'],
            'autovalores_top6': [fnum(x) for x in D['eig'][:6]], 'nula_p95_top6': [fnum(x) for x in D['pa']['null_p95'][:6]],
            'var_explicada_top3': [fnum(x, 3) for x in D['var_expl'][:3]],
            'estabilidad_loadings_cos_mediana_p5': {'mediana': [fnum(x) for x in D['stab']['median']], 'p5': [fnum(x) for x in D['stab']['p5']]},
            'sensibilidad_sin_estandarizar_por_unidades': D['sens'],
            'acoplamiento_OAR_vs_otros_calidad': D['coupling'],
            'acoplamiento_todos_los_pares': {k: {'r': fnum(v['r']), 'ic95': [fnum(v['ci95'][0]), fnum(v['ci95'][1])], 'p_perm': fnum(v['p_perm'], 4)} for k, v in D['blocks']['corr'].items()},
            'alineacion_calidad_score_bloque': {b: fnum(a) for b, a in D['blocks']['align'].items()},
            'patron_componentes_por_bloque_calidad': D['comp_block_q'],
            'anatomia': D['anatomy'], 'optimismo_train': D['optimism'],
            'sesgo': {'n_variables_con_sesgo_IC_sin_0': D['bias_summary']['n_vars_sesgo_significativo'], 'escala_renorm_mediana_oos': fnum(D['meta_scale']['escala_mediana_oos'], 3),
                      'escala_p5_p95_oos': [fnum(x, 3) for x in D['meta_scale']['escala_p5_p95_oos']]},
            'replica': {kk: {'n_ret': E[kk]['n_ret'], 'n': E[kk]['n'], 'cos_vs_residual_top3': [fnum(x) for x in E[kk]['vs_D']['cos_mejor_top3']],
                             'cos_vs_residual_retenidos': [fnum(x) for x in E[kk]['vs_D']['cos_mejor_retenidos']],
                             'acoplamiento': {k: {'r': fnum(v['r']), 'signo': v['signo'], 'significativo': v['significativo']} for k, v in E[kk]['coupling'].items()}}
                        for kk in ('k3', 'k5')},
            'lectura_predeclarada': tags, 'detalle_componentes': comps}
    summary['contraste_sitios'] = {'nota': 'Contraste, no control limpio: CyC difiere en generador (RapidPlan) Y en anatomia. Variables comunes: PTV_High, hotspot, camaras, BODY.',
                                   'retenidos': {s: DD[s]['n_ret'] for s in C.SITES},
                                   'desvio_mediano_variables_comunes': {s: fnum(ct[ct.sitio == s].desvio.median(), 3) for s in C.SITES}}
    summary['decisiones'] = DECISIONES
    json.dump(summary, open(C.OUT / 'summary.json', 'w', encoding='utf-8'), indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    write_resumen(summary, ct)


DECISIONES = [
    'Bandas de dosis semiabiertas [lo,hi) salvo la ultima (80-100 incluye 100); dosis en %Rx_High (NPZ ya normalizado a D95(PTV_High)=100%).',
    'Dosis predicha: clip a 0, x BODY, renormalizada a D95(PTV_High)=100% con dvh.renorm_pred (misma funcion del harness); factor guardado por paciente en cache.',
    'PTV_Low de 16 pacientes de Prostata (nivel no declarado, identico a High) ignorado: solo niveles del protocolo.',
    'Bilaterales _L/_R: slot prox/dist por distancia minima al PTV_High; si falta uno de los dos en un paciente, ambos slots quedan NaN (imputados con la media tras centrar).',
    'Variables con >30% NaN o sin varianza se descartan (lista en summary); camaras con <50 voxeles = NaN. NaN restantes imputados con la media (0 tras centrar).',
    'D0.035cm3 = percentil 100*(1-0.035/V_cc) de las dosis de la estructura (maximo si V<0.035cc). ΔDmax solo en OARs con goal tipo Dmax.',
    'PTV Mid/Low: ΔD95 en % de la Rx del propio nivel. Homogeneidad = (D2-D98)/D50. PTV: D50 sin signo de calidad (ambiguo).',
    'Score por bloque = PC1 de las variables estandarizadas del bloque, orientado a calidad (+ = real mejor que predicho): OAR/hotspot/camaras: menos dosis=mejor; PTV: D98/D95 mas alto y D2/homogeneidad mas bajos = mejor. r<0 entre OAR y otro bloque = antagonico; r>0 = sinergico.',
    'Acoplamiento significativo: p de permutacion < 0.05/3 (Bonferroni sobre OAR vs PTV/HOT/FALL) E IC95 bootstrap sin 0.',
    'Sensibilidad "sin estandarizar": cada variable dividida por el desvio RMS de su grupo de unidades (pp, % dosis, cm3, razon), no por el propio.',
    'R² anatomico: ridge con alpha elegido por LOO interno dentro de cada pliegue (LOO externo), descriptores estandarizados en el pliegue; IC95 bootstrap sobre pares (y, pred LOO); p por permutacion de y (100).',
    'Vecinos del §5.5 buscados SOLO dentro de val+test (el score solo existe fuera de muestra), con la distancia de la metrica de trabajo, k=3.',
    'Replica §6: desviacion vs promedio de los k vecinos sobre TODOS los elegibles del sitio (real vs real); loadings comparables = corr(score, variables compactas). Criterio de replica: coseno k=3 > 0.7 (k=5 se reporta).',
    'Criterio de "R² anatomico bajo" para la lectura: R² LOO (estimacion puntual) < 0.2 en el componente; se evalua por componente estable.',
    'Optimismo: razon de RMS por variable (sin centrar) train / val+test, resumida por mediana; IC bootstrap de pacientes en ambos grupos.',
    'Candidata B: roles con peso igual por slot y por referencia (PTV_High / union); curvas y escalares con peso igual (cada grupo normalizado por la mediana de distancias entre pares del sitio). OVH en grilla 0-150 mm cada 2.5 mm; vol OAR/PTV en bruto (no log).',
    'Spread del azar: k=3 vecinos aleatorios del sitio, 100 repeticiones; spread comparado solo en celdas (paciente, slot) validas para todas las candidatas.',
]


def fmt_ci(v):
    return f'{v["r"]:+.2f} [{v["ci95"][0]:+.2f},{v["ci95"][1]:+.2f}]'


def write_resumen(S, ct):
    L = ['# Spin-off trade-off — Etapa 1: estructura del residual (real − A4)', '',
         f'Modelo `{S["modelo"]}`, hash armonización `{S["armonizacion_hash"]}`, 432 pacientes. Residual = real − predicho (renormalizado a D95(PTV_High)=100%).',
         f'**Uso del test:** {S["uso_del_test"]}', '']
    L += ['## Métrica de similitud (k=3, spread del DVH real vs azar)', '']
    for s in C.SITES:
        w = S['metrica_trabajo'][s]
        L.append(f'- {s}: trabajo = **{w["metrica_trabajo"]}**; ratio medio spread/azar {{{", ".join(f"{k}: {v:.2f}" for k, v in w["ratio_medio_vs_azar"].items())}}} (C se reporta, no se usa; detalle en tablas/metrica_eleccion.csv)')
    L += ['', '## Resultado por sitio (val+test, fuera de muestra)', '',
          '| Sitio | n | vars | retenidos | estabilidad (cos) | acoplamiento OAR vs PTV / HOT / FALL (r [IC95], calidad) | R² anat. | réplica cos k3/k5 | optimismo train/oos |',
          '|---|---|---|---|---|---|---|---|---|']
    for s in C.SITES:
        x = S['sitios'][s]
        cp = x['acoplamiento_OAR_vs_otros_calidad']
        cps = ' / '.join(f'{fmt_ci(cp[o])}{"*" if cp[o]["significativo"] else ""}' for o in ('PTV', 'HOT', 'FALL') if o in cp)
        an = ', '.join(f'{a["r2_loo"]:.2f} [{a["r2_ic95"][0]:.2f},{a["r2_ic95"][1]:.2f}]' for a in x['anatomia']) or '—'
        rep = ', '.join(f'{a}/{b}' for a, b in zip(x['replica']['k3']['cos_vs_residual_top3'][:x['componentes_retenidos'] or 0], x['replica']['k5']['cos_vs_residual_top3'][:x['componentes_retenidos'] or 0])) or '—'
        st = ', '.join(str(v) for v in x['estabilidad_loadings_cos_mediana_p5']['mediana']) or '—'
        o = x['optimismo_train']
        L.append(f'| {s} ({x["tipo"]}) | {x["n_val_test"]} | {x["n_variables"]} | {x["componentes_retenidos"]} | {st} | {cps} | {an} | {rep} | {o["mediana_ratio"]:.2f} [{o["ic95"][0]:.2f},{o["ic95"][1]:.2f}] |')
    L += ['', '`*` = significativo (perm p<0.05/3 e IC sin 0). Signo en calidad: r<0 antagónico, r>0 sinérgico.', '', '## Lectura predeclarada', '']
    for s in C.SITES:
        L.append(f'- **{s}**: ' + ' | '.join(S['sitios'][s]['lectura_predeclarada']))
    L += ['', '## Contraste entre sitios (variables comunes)', '',
          f'Retenidos: {S["contraste_sitios"]["retenidos"]}; desvío mediano del residual en variables comunes: {S["contraste_sitios"]["desvio_mediano_variables_comunes"]}. '
          'Salvedad: CyC difiere de los otros dos en generador (RapidPlan) **y** en anatomía; no es un control limpio.', '',
          '## Decisiones no previstas', ''] + [f'- {d}' for d in S['decisiones']]
    (C.OUT / 'resumen.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
