"""
Loss pinball (cuantílica) espacialmente variable, Fase 4 (track de optimización). Spec: docs/PROMPT_code_f4_loss_cuantilica.md.

    r = real - pred ;  l_tau(r) = max(tau*r, (tau-1)*r) ;  se multiplica por 2 -> tau=0.5 == |r| (= masked_mae_body).
    PTV (unión, ∩BODY): tau=0.5 (simétrica, = MAE).  Tejido sano (BODY - PTV): tau_s (< 0.5: sobrepredecir cuesta más).
    Agregación: suma sobre BODY / nº de vóxeles BODY (idéntica a masked_mae).
"""
import torch


def pinball_body(pred: torch.Tensor, target: torch.Tensor, body: torch.Tensor, ptv: torch.Tensor,
                 tau_s: float, tau_ptv: float = 0.5) -> torch.Tensor:
    """pred/target/body/ptv con la misma forma; body y ptv {0,1} (ptv se fuerza ∩ body)."""
    r = target - pred
    ptv = ptv * body
    tau = tau_s + (tau_ptv - tau_s) * ptv                      # tau por vóxel
    l = 2.0 * torch.maximum(tau * r, (tau - 1.0) * r)
    return (l * body).sum() / body.sum().clamp(min=1.0)
