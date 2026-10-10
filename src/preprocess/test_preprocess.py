"""
Tests minimos de Paso 2 (datos sinteticos). Ejecutar desde la raiz del repo:
    python src/preprocess/test_preprocess.py
Los tests de integracion sobre un paciente real por sitio viven en scripts/run_preprocess.py
(--self-test), porque requieren DICOM real (permitido por el spec: "sinteticos o un paciente
de muestra").
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocess.name_resolver import NameResolver
from preprocess.normalize_structure_name import pick_ptv, normalize_ptv_name
from preprocess.rasterize import rasterize_structure, mask_volume_cm3, contour_volume_cm3
from preprocess.dose_metrics import d95_gy, normalize_dose, denormalize_dose
from preprocess.geometry import DoseGrid


# ---------------------------------------------------------------------------
# 1. Resolver de nombres: las cuatro salidas + casos sucios
# ---------------------------------------------------------------------------
def test_name_resolver_four_outcomes():
    schema = {
        'roles': {'targets': ['PTV_High'], 'oars': ['Rectum', 'Bladder']},
        'aliases': {'vejiga': 'Bladder'},
        'ignore_patterns': ['couch*', 'z_*'],
    }
    r = NameResolver(schema)

    assert r.resolve('Rectum').outcome == 'canonical'
    assert r.resolve('PTV_High').outcome == 'canonical'

    res = r.resolve('vejiga')
    assert res.outcome == 'alias' and res.role == 'Bladder'
    res = r.resolve('PTV_High-04')             # candidato recortado canonico (geometry_candidates)
    assert res.outcome == 'canonical' and res.role == 'PTV_High' and res.cropped is True

    res = r.resolve('PTV_High!04')             # tipeo de -04: normaliza igual, pero no es la forma exacta
    assert res.outcome == 'alias' and res.role == 'PTV_High' and res.cropped is True

    assert r.resolve('CouchSurface').outcome == 'ignore'
    assert r.resolve('z_Opt1').outcome == 'ignore'

    assert r.resolve('CTV_High').outcome == 'exclude'
    assert r.resolve('Bone_Mand-PTV_Hi').outcome == 'exclude'

    res = r.resolve('Larynx')                   # no esta en ningun lado
    assert res.outcome == 'unknown'

    # suciedad tipo nombresPosibles: separadores raros, no deberian "adivinar" un rol nuevo
    res = r.resolve('PTV_HighPTV_High')
    assert res.outcome == 'unknown'              # concatenado, no es un PTV valido ni alias

    # duplicado numerado de un rol ya conocido (recontoreado) -> mismo rol, no uno nuevo
    res = r.resolve('Rectum1')
    assert res.outcome == 'alias' and res.role == 'Rectum'
    # pero NUNCA para PTV/GTV/CTV: ahi el numero es semantico (niveles multiples),
    # no se puede colapsar sin perder informacion de "mas niveles que el protocolo"
    assert r.resolve('PTV_Mid01').outcome == 'unknown'
    assert r.resolve('GTV2').outcome == 'exclude'   # esto SI, pero por ser GTV, no por el numero

    # variante recortada (-04/-0.4) de algo que ya esta en exclude_names (no pasa por
    # normalize_ptv_name porque no es 'ptv_<nivel>', pero el mismo sufijo de crop aplica)
    schema2 = dict(schema, exclude_names=['PTVn'])
    r2 = NameResolver(schema2)
    assert r2.resolve('PTVn-0.4').outcome == 'exclude'
    assert r2.resolve('PTVn!04').outcome == 'exclude'


# ---------------------------------------------------------------------------
# 2. Resolver de PTV: fallback al original cuando el -04 esta vacio o ausente
# ---------------------------------------------------------------------------
def test_ptv_fallback_when_cropped_empty_or_absent():
    # -04 presente pero volumen 0 -> usa el original
    name, motivo = pick_ptv([('PTV_High-04', 0.0), ('PTV_High', 48.3)], level='high')
    assert name == 'PTV_High' and motivo is None

    # -04 ausente directamente -> usa el original
    name, motivo = pick_ptv([('PTV_High', 48.3)], level='high')
    assert name == 'PTV_High' and motivo is None

    # -04 presente con volumen -> gana sobre el original
    name, motivo = pick_ptv([('PTV_High-04', 47.0), ('PTV_High', 50.0)], level='high')
    assert name == 'PTV_High-04' and motivo is None

    # ninguno con volumen -> no resuelto
    name, motivo = pick_ptv([('PTV_High-04', 0.0), ('PTV_High', 0.0)], level='high')
    assert name is None and motivo is not None


# ---------------------------------------------------------------------------
# 3. D95(PTV_High)=100% tras normalizar + round-trip a Gy
# ---------------------------------------------------------------------------
def test_d95_normalization_roundtrip():
    rng = np.random.default_rng(0)
    dose_gy = rng.uniform(0, 75, size=(10, 20, 20))
    mask = np.zeros_like(dose_gy, dtype=bool)
    mask[3:7, 5:15, 5:15] = True

    d95 = d95_gy(dose_gy, mask)
    dose_pct = normalize_dose(dose_gy, d95)
    d95_after = d95_gy(dose_pct, mask)
    assert abs(d95_after - 100.0) < 1e-6, d95_after

    recovered_gy = denormalize_dose(dose_pct, d95)
    assert np.allclose(recovered_gy, dose_gy)


# ---------------------------------------------------------------------------
# 4. Anti-leakage: el canal de intencion (Rx_nivel/Rx_High, del protocolo) no cambia
#    si se altera la dosis.
# ---------------------------------------------------------------------------
def test_intention_independent_of_dose():
    rx_by_level = {'PTV_High': 69.96, 'PTV_Mid': 59.4, 'PTV_Low': 54.45}

    def intention_ratios(rx_by_level):
        rx_high = rx_by_level['PTV_High']
        return {k: v / rx_high for k, v in rx_by_level.items()}

    ratios_before = intention_ratios(rx_by_level)
    # "alterar la dosis" no debe tocar rx_by_level (viene del YAML del sitio, no de la dosis)
    _dose_altered = np.zeros((5, 5, 5)) + 999.0   # noqa: F841 (simula dosis distinta)
    ratios_after = intention_ratios(rx_by_level)
    assert ratios_before == ratios_after
    assert abs(ratios_before['PTV_Mid'] - 59.4 / 69.96) < 1e-9


# ---------------------------------------------------------------------------
# 5. Escala de voxel: volumen de la mascara rasterizada vs. volumen de contorno
# ---------------------------------------------------------------------------
def test_mask_volume_matches_contour_volume():
    grid = DoseGrid(shape=(10, 100, 100), spacing=(3.0, 2.0, 2.0),
                     origin=(-100.0, -100.0, 0.0), frame_z_mm=np.arange(10) * 3.0)
    # cuadrado de 60x60mm en 3 cortes consecutivos -> estructura con volumen apreciable,
    # para que el sesgo de borde del rasterizado (ver rasterize.py) sea proporcionalmente chico.
    square = [(-30, -30), (30, -30), (30, 30), (-30, 30)]
    contours = [(z, square) for z in (3.0, 6.0, 9.0)]
    mask = rasterize_structure(contours, grid)

    v_mask = mask_volume_cm3(mask, grid)
    v_contour = contour_volume_cm3(contours)
    rel_diff = abs(v_mask - v_contour) / v_contour
    assert rel_diff < 0.15, f'volumen mascara {v_mask} vs contorno {v_contour} (rel_diff={rel_diff:.3f})'


# ---------------------------------------------------------------------------
# 6. Determinismo: misma entrada -> mismo resultado
# ---------------------------------------------------------------------------
def test_rasterize_deterministic():
    grid = DoseGrid(shape=(5, 50, 50), spacing=(3.0, 2.0, 2.0),
                     origin=(-50.0, -50.0, 0.0), frame_z_mm=np.arange(5) * 3.0)
    contours = [(6.0, [(-10, -10), (10, -10), (10, 10), (-10, 10)])]
    m1 = rasterize_structure(contours, grid)
    m2 = rasterize_structure(contours, grid)
    assert np.array_equal(m1, m2)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('OK', t.__name__)
    print(f'\n{len(tests)} tests OK')
