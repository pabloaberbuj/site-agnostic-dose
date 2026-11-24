"""Delta del track cuantilico vs el control MAE (Fase 4). Lee results/f4h_* y results/exp_f4q_*/ckpt_ep*/ y la vara (ep113). Imprime tablas y guarda results/f4q_delta/*.csv
 - PTV-alta: mean|dV| banda alta_80_110, role PTV_High, por sitio (eval_a0), vs vara.
 - DVH-point por differencing: delta pred (cuantilico - MAE) en %Rx, por (sitio, rol, punto) y promedio por sitio de primarios (negativo = menos dosis predicha).
 - OVH: percentil por sitio (primarios), cuantilico vs MAE."""
import glob, os
import numpy as np, pandas as pd
sys_ = pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
OUT = 'results/f4q_delta'; os.makedirs(OUT, exist_ok=True)
VARA = 'results/exp_a4_003_unet2d_psdm_c176x256_vara_f4/ckpt_ep113/metrics.csv'
RUNS = [('0.35', 'exp_f4q_001_pinball_tau0p35', 96), ('0.30', 'exp_f4q_002_pinball_tau0p30', 100), ('0.30', 'exp_f4q_002_pinball_tau0p30', 110),
        ('0.40', 'exp_f4q_003_pinball_tau0p40', 100), ('0.40', 'exp_f4q_003_pinball_tau0p40', 112)]
ctrl = pd.read_csv('results/f4h_mae_control_renorm_none/per_patient_dvhpoint.csv')
def boot(v, n=1000, seed=42):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; rng = np.random.default_rng(seed)
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]; return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def fmt(t): return f'{t[0]:.2f}[{t[1]:.2f},{t[2]:.2f}]'
def ptv_alta(path):
    d = pd.read_csv(path); d = d[d.role == 'PTV_High']
    return {s: boot(g.alta_80_110) for s, g in d.groupby('site')}
vara = ptv_alta(VARA)
rows_ptv, rows_dvh = [], []
for tau, exp, ep in RUNS:
    tag = f'tau{tau}_ep{ep}'; hid = f'f4h_{exp[4:]}_ep{ep}'
    p = f'results/{exp}/ckpt_ep{ep}/metrics.csv'
    if os.path.exists(p):
        q = ptv_alta(p)
        for s in q: rows_ptv.append({'run': tag, 'site': s, 'PTV_alta_quant': fmt(q[s]), 'PTV_alta_vara': fmt(vara[s]), 'delta': round(q[s][0] - vara[s][0], 3)})
    d = pd.read_csv(f'results/{hid}/per_patient_dvhpoint.csv')
    m = d.merge(ctrl, on=['patient_id', 'site', 'role', 'point', 'primary'], suffixes=('_q', '_c'))
    m['d_pred'] = m.pred_q - m.pred_c
    m['bias_real_minus_pred'] = m.real_q - m.pred_q
    for s, g in m[m.primary].groupby('site'):
        pp = g.groupby('patient_id').d_pred.mean()
        rows_dvh.append({'run': tag, 'site': s, 'd_pred_primarios_%Rx': fmt(boot(pp)), 'n': len(pp)})
    for (s, r, pt), g in m.groupby(['site', 'role', 'point']):
        rows_dvh.append({'run': tag, 'site': s, 'role': r, 'point': pt, 'd_pred_%Rx': fmt(boot(g.d_pred)), 'sesgo_real-pred_%Rx': round(g.bias_real_minus_pred.mean(), 2), 'n': len(g)})
ptv_df, dvh_df = pd.DataFrame(rows_ptv), pd.DataFrame(rows_dvh)
ptv_df.to_csv(f'{OUT}/ptv_alta_vs_vara.csv', index=False); dvh_df.to_csv(f'{OUT}/dvhpoint_differencing.csv', index=False)
print('== PTV-alta 80-110 (mean|dV|, menor=mejor) cuantilico vs vara ep113 =='); print(ptv_df.to_string(index=False))
print('\n== DVH-point: delta pred (cuantilico - MAE) en %Rx, primarios por sitio (negativo = menos dosis) =='); print(dvh_df[dvh_df.role.isna()].dropna(axis=1, how='all').to_string(index=False))
