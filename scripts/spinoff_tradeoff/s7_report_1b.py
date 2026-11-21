"""Etapa 1b: corre el análisis por sitio (s6), la réplica, y escribe tablas, gráficos, summary.json y resumen.md."""
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
import s6_analyze_1b as A  # noqa: E402

TAB, OUT, GR = A.TAB_B, A.OUT_B, A.OUT_B / 'graficos'


def f(x, d=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def plot_scree(RES, path):
    fig, axes = plt.subplots(1, len(RES), figsize=(5 * len(RES), 3.8))
    for ax, (s, R) in zip(np.atleast_1d(axes), RES.items()):
        k = min(15, len(R['eig']))
        x = np.arange(1, k + 1)
        ax.fill_between(x, R['pa']['null_median'][:k], R['pa']['null_p95'][:k], color='0.8', label='nula por bloques (mediana-p95)')
        ax.plot(x, R['pa_cols']['null_p95'][:k], '--', color='0.4', lw=1, label='nula por columnas p95 (Etapa 1)')
        ax.plot(x, R['eig'][:k], 'o-', color='C3', label='residual 1b')
        ax.set_title(f'{s} (n={R["n"]}, p={R["p"]}): retenidos bloques={R["n_ret"]} / columnas={R["n_ret_columnas"]}', fontsize=9)
        ax.set_xlabel('componente')
        ax.set_ylabel('autovalor')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_dd95(RES, path):
    fig, axes = plt.subplots(1, len(RES), figsize=(5 * len(RES), 4))
    for ax, (s, R) in zip(np.atleast_1d(axes), RES.items()):
        y = np.array(R['y_dD95'])
        x = R['blocks']['scores']['OAR']
        ax.scatter(x, y, s=16, alpha=.7)
        d = R['directa']
        r = d[d.prueba == 'score_OAR'].iloc[0]
        ax.set_title(f'{s}: dD95(PTV_High) vs score OAR\nr(calidad)={r.r_calidad:+.2f} [{r.ic95_lo:+.2f},{r.ic95_hi:+.2f}] p_holm={r.p_holm:.3g}', fontsize=9)
        ax.set_xlabel('score OAR (+ = menos dosis que lo predicho)')
        ax.set_ylabel('dD95 PTV_High, pp (+ = más cobertura)')
        ax.axhline(0, color='0.8', lw=.5)
        ax.axvline(0, color='0.8', lw=.5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    GR.mkdir(parents=True, exist_ok=True)
    work = json.load(open(C.OUT / 'metrica_trabajo.json', encoding='utf-8'))
    elig = C.load_eligible()
    recs_all = C.load_cache()
    stage1 = {s: pickle.load(open(C.CACHE / f'resD_{s}.pkl', 'rb')) for s in C.SITES}
    RES, REP, RECS = {}, {}, {}
    for site in C.SITES:
        recs1 = sorted([r for r in recs_all if r['site'] == site], key=lambda r: r['pid'])
        res, recs = A.run_site(site, recs1, work, elig, stage1)
        res['y_dD95'] = [r['real'][A.D95] - 100.0 for r in recs if r['split'] in ('val', 'test')]
        RES[site], RECS[site] = res, recs
        REP[site] = A.replica(site, recs, work, res)
        print(site, 'n', res['n'], 'p', res['p'], 'ret bloques', res['n_ret'], 'ret columnas', res['n_ret_columnas'], flush=True)

    # ---- tablas ----
    rows = []
    for s, R in RES.items():
        for i, (e, t, m, tc) in enumerate(zip(R['eig'], R['pa']['null_p95'], R['pa']['null_median'], R['pa_cols']['null_p95'])):
            rows.append({'sitio': s, 'comp': i + 1, 'autovalor': e, 'nula_bloques_p95': t, 'nula_bloques_mediana': m, 'nula_columnas_p95_etapa1': tc,
                         'retenido_bloques': i < R['n_ret'], 'n': R['n'], 'p': R['p']})
    pd.DataFrame(rows).to_csv(TAB / 'autovalores_nula_bloques.csv', index=False)
    rows = []
    for s, R in RES.items():
        K = max(R['n_ret'], 3)
        for j, c in enumerate(R['cols']):
            rows.append({'sitio': s, 'variable': c, 'bloque': C.block_of(c), **{f'PC{k + 1}': R['load'][j, k] for k in range(K)}})
    pd.DataFrame(rows).to_csv(TAB / 'loadings_1b.csv', index=False)
    rows = []
    for s, R in RES.items():
        for d, st in zip(R['comp_block'], R['stab']['median'] + [np.nan] * 99):
            rows.append({'sitio': s, **d, 'estabilidad_cos_mediana': st})
    pd.DataFrame(rows).to_csv(TAB / 'carga_por_bloque_componente.csv', index=False)
    pd.concat([R['directa'].assign(sitio=s) for s, R in RES.items()]).to_csv(TAB / 'prueba_directa_dD95.csv', index=False)
    rows = []
    for s, R in RES.items():
        for k, v in R['blocks']['corr'].items():
            rows.append({'sitio': s, 'analisis': 'residual_1b', 'par': k, 'r': v['r'], 'ic95_lo': v['ci95'][0], 'ic95_hi': v['ci95'][1], 'p_perm': v['p_perm'], 'n': v['n']})
        for kk, E in REP[s].items():
            for k, v in E['blocks']['corr'].items():
                rows.append({'sitio': s, 'analisis': f'replica_{kk}', 'par': k, 'r': v['r'], 'ic95_lo': v['ci95'][0], 'ic95_hi': v['ci95'][1], 'p_perm': v['p_perm'], 'n': v['n']})
    pd.DataFrame(rows).to_csv(TAB / 'acoplamiento_bloques_1b.csv', index=False)
    rows = []
    for s, R in RES.items():
        for pool, a in R['anatomia_dD95'].items():
            rows.append({'sitio': s, 'pool': pool, 'n': a['n'], 'dD95_media': a['dD95_media'], 'dD95_sd': a['dD95_sd'], 'r2_loo': a['r2_loo'], 'r2_ic95_lo': a['r2_ic95'][0],
                         'r2_ic95_hi': a['r2_ic95'][1], 'r2_p_perm': a['r2_p_perm'], 'n_descriptores': a['n_descriptores'], 'r_vecinos': a['corr_vecinos']['r'],
                         'r_vecinos_ic_lo': a['corr_vecinos']['ci95'][0], 'r_vecinos_ic_hi': a['corr_vecinos']['ci95'][1], 'p_perm_vecinos': a['corr_vecinos']['p_perm'],
                         'spearman_vecinos': a['spearman_vecinos']})
    pd.DataFrame(rows).to_csv(TAB / 'anatomia_dD95.csv', index=False)
    rows = []
    for s, R in RES.items():
        v = R['vs_etapa1']
        for i, ci in enumerate(v['cos']):
            for j, c in enumerate(ci):
                rows.append({'sitio': s, 'comp_1b': i + 1, 'comp_etapa1': j + 1, 'coseno': c, 'n_vars_comunes': v['n_vars_comunes'], 'es_eje_derrame_etapa1': v['eje_derrame_etapa1'] == j + 1})
    pd.DataFrame(rows).to_csv(TAB / 'comparacion_etapa1.csv', index=False)
    rows = [{'sitio': s, 'k': kk, 'n': E['n_pacientes'], 'retenidos_bloques': E['n_ret'], 'retenidos_columnas': E['n_ret_columnas'],
             'comp_1b': i + 1, 'cos_mejor_top3': c} for s, d in REP.items() for kk, E in d.items() for i, c in enumerate(E['vs_D']['cos_mejor_top3'])]
    pd.DataFrame(rows).to_csv(TAB / 'replica_1b.csv', index=False)
    plot_scree(RES, GR / 'scree_nula_bloques.png')
    plot_dd95(RES, GR / 'dD95_vs_score_OAR.png')

    # ---- summary.json ----
    san = pd.read_csv(TAB / 'sanity_reescalado.csv')
    summary = {'etapa': 'spinoff_tradeoff_etapa1b', 'espacio': 'real en % Rx_High del protocolo (dosis NPZ x D95_abs/Rx); predicho A4 renormalizado a D95(PTV_High)=100%',
               'sanity_reescalado': {'n': int(len(san)), 'discrepantes_gt_0.5pp_vs_qa_gate': san[san['discrepancia_gt_0.5pp']].patient_id.tolist(),
                                     'max_abs_dif_pp': f(san.dif_vs_qa_pp.abs().max(), 3), 'max_abs_D95_normalizado_menos_100': f(san.dif_norm_vs_100_pp.abs().max(), 3)},
               'uso_del_test': 'Sin metricas agregadas de desempeno sobre test (ni MAE ni mean|dV| por banda); solo estructura del residual en val+test.',
               'sitios': {}}
    for s in C.SITES:
        R = RES[s]
        d = R['directa']
        summary['sitios'][s] = {
            'tipo': 'manual' if s in C.MANUALES else 'RapidPlan (contraste)', 'n_val_test': R['n'], 'n_variables': R['p'], 'variables_descartadas': R['dropped'],
            'retenidos_nula_bloques': R['n_ret'], 'retenidos_nula_columnas_etapa1': R['n_ret_columnas'],
            'autovalores_top6': [f(x) for x in R['eig'][:6]], 'nula_bloques_p95_top6': [f(x) for x in R['pa']['null_p95'][:6]],
            'estabilidad_cos_mediana': [f(x) for x in R['stab']['median']],
            'carga_por_bloque_componente': R['comp_block'],
            'corr_score_componente_vs_dD95': [{'comp': i + 1, **c} for i, c in enumerate(R['comp_vs_dD95'])],
            'acoplamiento_bloques': {k: {'r': f(v['r']), 'ic95': [f(v['ci95'][0]), f(v['ci95'][1])], 'p_perm': f(v['p_perm'], 4)} for k, v in R['blocks']['corr'].items()},
            'HOT~PTV': {'r': f(R['blocks']['corr'].get('PTV~HOT', {}).get('r')), 'ic95': [f(x) for x in R['blocks']['corr'].get('PTV~HOT', {}).get('ci95', [None, None])]},
            'prueba_directa_dD95': {'n_pruebas': int(len(d)), 'n_significativas_holm': int(d.significativo.sum()),
                                    'antagonicas_sig': d[d.significativo & (d.signo == 'antagonico')].prueba.tolist(), 'sinergicas_sig': d[d.significativo & (d.signo == 'sinergico')].prueba.tolist(),
                                    'bloques': d[d.tipo == 'bloque'][['prueba', 'r_calidad', 'ic95_lo', 'ic95_hi', 'p_holm', 'significativo']].round(3).to_dict('records')},
            'anatomia_dD95': R['anatomia_dD95'], 'vs_etapa1': R['vs_etapa1'],
            'replica': {kk: {'n': E['n_pacientes'], 'retenidos_bloques': E['n_ret'], 'cos_vs_residual_top3': [f(x) for x in E['vs_D']['cos_mejor_top3']]} for kk, E in REP[s].items()},
            'lectura_predeclarada': A.read_site(s, R)}
    summary['decisiones'] = DECISIONES
    json.dump(summary, open(OUT / 'summary.json', 'w', encoding='utf-8'), indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    write_resumen(summary, RES)


DECISIONES = [
    'Predicho: la dosis predicha no estaba cacheada en la Etapa 1; se re-infirio en CPU solo val+test (152) para el hotspot fuera de PTV; verificado contra el factor de escala y variables cacheadas de la Etapa 1 (campo pred_check en cache).',
    'Hotspot redefinido fuera de los PTV: D0.035cm3 de (BODY - union de PTV) y volumen >107% fuera de PTV, ambos en real y predicho.',
    'Real = dosis NPZ x (D95_abs_Gy/Rx_High_Gy) de la metadata; sanity contra paso3_qa_gate.csv y recalculo de D95.',
    'Nula por bloques: cada bloque (OAR, PTV, HOT, FALL) se permuta entero entre pacientes con permutacion propia; el acoplamiento usa permutacion del score de un bloque (equivalente). Se reporta tambien el n retenido con la nula por columnas de la Etapa 1.',
    'Prueba directa: familia por sitio = (dV media + dDmean) de cada slot OAR + score de OAR/HOT/FALL; p por permutacion (10000), correccion Holm; significativo = p_holm<0.05 e IC95 bootstrap sin 0. Signo en calidad: r<0 antagonico.',
    'Lectura: "antagonico con OARs" = alguna prueba OAR (slot o score) significativa con r<0 tras Holm; R² anatomico de dD95 evaluado con TODOS los elegibles (se reporta tambien val+test).',
    'Carga por bloque = suma (y media) de loadings corr(score,var) con signo de calidad; "mezcla" si hay bloques con media > +0.15 y otros < -0.15.',
    'Eje de derrame de Pelvis = componente de la Etapa 1 con mayor |loading| medio sobre el bloque de caida de dosis (FALL); sobrevive si coseno>0.7, OAR y FALL con igual signo de calidad y sin acoplamiento con dD95.',
    'Comparacion con Etapa 1 en las variables comunes; el hotspot cambio de definicion (HOT|D0035_out) y no entra en el coseno.',
]


def write_resumen(S, RES):
    L = ['# Spin-off trade-off — Etapa 1b: residual en la normalización original del plan', '',
         f'Espacio: {S["espacio"]}. Residual = real − predicho. {S["uso_del_test"]}', '',
         f'**Sanity del reescalado** (n={S["sanity_reescalado"]["n"]}): discrepantes >0.5 pp vs QA gate = {len(S["sanity_reescalado"]["discrepantes_gt_0.5pp_vs_qa_gate"])}; '
         f'máx |dif| = {S["sanity_reescalado"]["max_abs_dif_pp"]} pp.', '',
         '## Resultado por sitio (val+test)', '',
         '| Sitio | n | Retenidos (bloques / columnas Et.1) | ΔD95 vs score OAR (r calidad [IC95], p Holm) | Pruebas sig. (de n) | R² anat. ΔD95 todos / val+test | r vecinos ΔD95 (todos) | HOT~PTV r |',
         '|---|---|---|---|---|---|---|---|']
    for s in C.SITES:
        x = S['sitios'][s]
        b = [r for r in x['prueba_directa_dD95']['bloques'] if r['prueba'] == 'score_OAR'][0]
        a = x['anatomia_dD95']
        at, av = a['todos_los_elegibles'], a['val+test']
        L.append(f'| {s} | {x["n_val_test"]} | {x["retenidos_nula_bloques"]} / {x["retenidos_nula_columnas_etapa1"]} | {b["r_calidad"]:+.2f} [{b["ic95_lo"]:+.2f},{b["ic95_hi"]:+.2f}], {b["p_holm"]:.3g} | '
                 f'{x["prueba_directa_dD95"]["n_significativas_holm"]} de {x["prueba_directa_dD95"]["n_pruebas"]} | {at["r2_loo"]:.2f} [{at["r2_ic95"][0]:.2f},{at["r2_ic95"][1]:.2f}] (n={at["n"]}) / '
                 f'{av["r2_loo"]:.2f} [{av["r2_ic95"][0]:.2f},{av["r2_ic95"][1]:.2f}] (n={av["n"]}) | {at["corr_vecinos"]["r"]:+.2f} [{at["corr_vecinos"]["ci95"][0]:+.2f},{at["corr_vecinos"]["ci95"][1]:+.2f}] | '
                 f'{x["HOT~PTV"]["r"]} |')
    L += ['', 'Signo en calidad: r<0 antagónico (menos cobertura ↔ OAR mejor).', '', '## Lectura predeclarada', '']
    for s in C.SITES:
        L.append(f'- **{s}**: ' + ' | '.join(S['sitios'][s]['lectura_predeclarada']))
    L += ['', '## Componentes retenidos: carga por bloque (media orientada a calidad)', '']
    for s in C.SITES:
        for c in S['sitios'][s]['carga_por_bloque_componente']:
            bl = ', '.join(f'{b}={c[f"media_{b}"]:+.2f}' for b in ('OAR', 'PTV', 'HOT', 'FALL') if f'media_{b}' in c and c[f'media_{b}'] == c[f'media_{b}'])
            L.append(f'- {s} comp {c["comp"]}: {bl}; ΔD95 carga {c["carga_dD95"]:+.2f}; mezcla de signos opuestos: {c["mezcla_signos_opuestos"]}')
    L += ['', '## Decisiones no previstas', ''] + [f'- {d}' for d in S['decisiones']]
    (OUT / 'resumen.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
