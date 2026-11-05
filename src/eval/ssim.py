"""
SSIM de Fase 3 (vara de coherencia): enmascarado por BODY y por BODY ∩ dosis real > 5 % de Rx_High. Criterio de scripts/ssim_validacion.py.

Mismo SSIM que torchmetrics 1.9 (gauss 11, sigma 1.5, K1=.01, K2=.03, data_range=120, reflect-pad de 5 px SIN recortar el borde,
cortes con body, media por corte y luego entre cortes), pero con los mapas por píxel (luminancia/contraste/estructura) para promediar
sobre una máscara. Las ventanas se calculan sobre la imagen completa; la máscara solo elige qué píxeles se promedian.
SSIM = lum·con·est con C3 = C2/2 (exacto). dose_pct ya está en % de Rx_High (umbral 5 = 5 % Rx_High).

El SSIM de FRAME COMPLETO queda solo como columna de referencia OBSOLETA (`ssim_frame_obsoleto`): 36-71 % de los píxeles están fuera del
BODY, donde la pred no tiene supervisión (la loss enmascara) y domina la métrica (ver results/ssim_validacion/README.md).
"""
import numpy as np
import torch
import torch.nn.functional as F

L, K1, K2, PAD = 120.0, 0.01, 0.03, 5
DOSE_MIN_PCT = 5.0
REGIONES = ('frame', 'body', 'body_dosis5')
COL_FRAME_OBSOLETA = 'ssim_frame_obsoleto'


def postproc_body(pred, body):
    """Post-proceso estándar de inferencia/evaluación/export: dosis predicha * máscara BODY (BODY es input: sin leakage)."""
    return pred * body.astype(pred.dtype)


def _gauss():
    g = torch.exp(-(torch.arange(11, dtype=torch.float64) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_maps(pred, true, dev):
    """(N,H,W) -> mapas (N,H,W): ssim, luminancia, contraste, estructura (float64)."""
    k = _gauss().to(dev)
    p = torch.from_numpy(np.ascontiguousarray(pred)).to(dev).double().unsqueeze(1)
    t = torch.from_numpy(np.ascontiguousarray(true)).to(dev).double().unsqueeze(1)
    p, t = (F.pad(a, (PAD,) * 4, mode='reflect') for a in (p, t))
    cv = lambda a: F.conv2d(a, k)
    mx, my = cv(p), cv(t)
    sxx, syy, sxy = cv(p * p) - mx * mx, cv(t * t) - my * my, cv(p * t) - mx * my
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    sx, sy = sxx.clamp(min=0).sqrt(), syy.clamp(min=0).sqrt()
    lum = (2 * mx * my + C1) / (mx ** 2 + my ** 2 + C1)
    con = (2 * sx * sy + C2) / (sxx + syy + C2)
    est = (sxy + C2 / 2) / (sx * sy + C2 / 2)
    ssim = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return [m[:, 0].cpu().numpy() for m in (ssim, lum, con, est)]


def region_stats(pred, dose, body, dev='cpu'):
    """Por paciente: para cada región {frame, body, body_dosis5}: n píxeles, n cortes, ssim/lum/con/est (media por corte y entre
    cortes) y ssimpix (píxeles agrupados). Más el error abs fuera del BODY. Cortes = los que tienen body."""
    sl = np.where(body.reshape(body.shape[0], -1).any(1))[0]
    s, l, c, e = ssim_maps(pred[sl], dose[sl], dev)
    b = body[sl]
    regs = {'frame': np.ones_like(b), 'body': b, 'body_dosis5': b & (dose[sl] > DOSE_MIN_PCT)}
    out = {}
    for r, m in regs.items():
        cnt = m.sum((1, 2))
        v = cnt > 0
        out[f'{r}_n'] = int(cnt.sum())
        out[f'{r}_nslices'] = int(v.sum())
        for nom, mp in (('ssim', s), ('lum', l), ('con', c), ('est', e)):
            sm = np.where(m, mp, 0.0).sum((1, 2))
            out[f'{r}_{nom}'] = float((sm[v] / cnt[v]).mean()) if v.any() else np.nan
            if nom == 'ssim':
                out[f'{r}_ssimpix'] = float(sm.sum() / cnt.sum()) if v.any() else np.nan
    fuera = ~b
    out['mae_fuera_body'] = float(np.abs(pred[sl] - dose[sl])[fuera].mean()) if fuera.any() else np.nan
    out['frac_fuera_body'] = float(fuera.mean())
    return out
