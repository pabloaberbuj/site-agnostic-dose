"""
Loss de momentos tipo Jhanwar (Fase 4, loss #1), por estructura y restringida a máscara ∩ BODY.

    M_p(S) = ( mean_{i in S} d_i^p )^(1/p)      (d en % Rx_High)
    L_dvh  = (1/|S_pac|) * sum_S sum_p w_{S,p} * ( M_p(pred) - M_p(real) )^2        (w = 1 en los p activos)

Las máscaras de cada estructura llegan empaquetadas en un bitmask int64 (B,Z,H,W) (bit k = estructura k, ya ∩ BODY y
recortadas al marco); `struct_p` (B,S,NP) bool dice qué momento de MOMENTS_P está activo en cada estructura.
Solo batch_size=1 (Z/estructuras varían por paciente).
"""
import torch

MOMENTS_P = (1, 10)           # arranque del prompt: p=1 (media/cobertura), p=10 (~Dmax/hotspot)
MAX_STRUCTS = 62              # bits disponibles en el int64
_EPS = 1e-6


def moment(d: torch.Tensor, p: int) -> torch.Tensor:
    """M_p de un vector de dosis (% Rx). Se calcula sobre d/100 (estabilidad numérica con p=10) y se reescala."""
    x = d / 100.0
    if p == 1:
        return 100.0 * x.mean()
    return 100.0 * x.clamp(min=0.0).pow(p).mean().clamp(min=_EPS).pow(1.0 / p)


def dvh_moment_loss(pred: torch.Tensor, target: torch.Tensor, bitmask: torch.Tensor, struct_p: torch.Tensor, form: str = 'sq'):
    """pred/target (1,Z,H,W) % Rx; bitmask (1,Z,H,W) int64; struct_p (1,S,NP) bool. Devuelve (loss, n_estructuras_usadas)."""
    assert pred.shape[0] == 1, 'dvh_moment_loss: solo batch_size=1'
    assert form in ('sq', 'l1'), form   # sq: (dM)^2 [spec original, %Rx^2]; l1: |dM| [mismas unidades que el MAE]
    pred, target, bm, sp = pred.float(), target.float(), bitmask[0], struct_p[0]
    total, n = pred.new_zeros(()), 0
    for k in range(sp.shape[0]):
        sel = ((bm >> k) & 1).bool()
        if not sel.any():
            continue
        pr, tg = pred[0][sel], target[0][sel].detach()
        term = pred.new_zeros(())
        for j, p in enumerate(MOMENTS_P):
            if sp[k, j]:
                dm = moment(pr, p) - moment(tg, p)
                term = term + (dm ** 2 if form == 'sq' else dm.abs())
        total, n = total + term, n + 1
    return total / max(n, 1), n
