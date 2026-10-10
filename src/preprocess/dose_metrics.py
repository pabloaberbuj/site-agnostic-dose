"""
D95 y normalizacion de dosis (CHARTER §5): D95(PTV_High)=100%, recomputado desde la
grilla de dosis CRUDA (nunca la normalizacion de Eclipse). D95 = dosis tal que el 95%
del volumen de la mascara recibe >= esa dosis -> percentil 5 de los valores de dosis
dentro de la mascara.
"""
import numpy as np


class DoseMetricsError(Exception):
    pass


def d95_gy(dose_gy: np.ndarray, mask: np.ndarray) -> float:
    vals = dose_gy[mask]
    if vals.size == 0:
        raise DoseMetricsError('mascara vacia: no se puede calcular D95')
    return float(np.percentile(vals, 5))


def normalize_dose(dose_gy: np.ndarray, d95_ptv_high_gy: float) -> np.ndarray:
    """-> dosis en % de modo que D95(PTV_High) quede en 100."""
    if d95_ptv_high_gy <= 0:
        raise DoseMetricsError(f'D95(PTV_High)={d95_ptv_high_gy} <= 0, no se puede normalizar')
    return dose_gy * (100.0 / d95_ptv_high_gy)


def denormalize_dose(dose_pct: np.ndarray, d95_ptv_high_gy: float) -> np.ndarray:
    return dose_pct * (d95_ptv_high_gy / 100.0)
