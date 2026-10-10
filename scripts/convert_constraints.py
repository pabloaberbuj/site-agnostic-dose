#!/usr/bin/env python3
"""
Conversor: plantilla de constraints in-house (JSON de ExploracionPlanes)
          -> YAML de armonización por sitio (OARs -> goals estructurados) + reporte.

Uso:
    python convert_constraints.py zPabloCyC.txt --site CyC \
        --rx PTV_High=69.96 PTV_Mid=59.4 PTV_Low=54.45 \
        --protocol-id cyc_6996_594_5445 --out sites_cyc.yaml

Decisiones embebidas (ver conversación de diseño Fase 0.1):
  - PRVs NO se ignoran: órganos seriales usan el PRV como geometría (es donde vive el constraint).
  - PTV: la geometría se resuelve con nombresPosibles en orden (-04 primero, fallback al original);
    el software in-house ya codifica ese orden.
  - CTV: excluido de la intención (objetivo de optimización), se reporta.
  - Estructuras de resta (X-PTV_*): excluidas, se reportan (flag manual).
  - Constraints condicionales (condicion.EtiquetaRestriccionAnidada): se emite la métrica
    primaria con flag conditional=true.
  - valorEsperado = goal; valorTolerado = tolerancia (se guarda aparte, no es el goal).
  - D<=0.035cm3 se unifica a Dmax (equivalentes). Los casi-máximo mayores (D1cm3, D2%) NO se
    tocan: su canal es la pregunta viva del CHARTER §10.
  - Umbrales en % de Dmax/Dmean se etiquetan pct_rx (% de Rx), igual que las métricas D.
"""
import json, argparse, re, sys, os
from collections import defaultdict, OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess.normalize_structure_name import normalize_ptv_name

# ----------------------------------------------------------------------------
# Clasificación de rol de estructura
# ----------------------------------------------------------------------------

def classify(nombre):
    """Devuelve (clase, rol_canonico, nivel|None)."""
    n = nombre.strip()
    nivel, _cropped = normalize_ptv_name(n)
    if nivel:
        level = 'PTV_' + nivel.capitalize()
        return ('target', level, level)
    if n.upper().startswith('CTV'):
        return ('ctv_excluded', n, None)
    if '-PTV' in n.upper():                      # resta tipo Bone_Mand-PTV_Hi
        return ('subtraction_excluded', n, None)
    if 'PRV' in n.upper():                        # serial con PRV: se queda, geometría = PRV
        return ('oar', n, None)
    return ('oar', n, None)                        # OAR crudo

# ----------------------------------------------------------------------------
# Parseo de una restricción -> métrica estructurada
# ----------------------------------------------------------------------------
def map_unit(u):
    return {'%': 'pct', 'Gy': 'Gy', 'cm3': 'cm3', None: None, 'NaN': None}.get(u, u)

def parse_metric(r):
    """Devuelve dict {type, param?, param_unit?, op, value, value_unit, conditional, tol}."""
    tipo = r['$type'].split('.')[1].split(',')[0]
    op = 'le' if r['esMenorQue'] else 'ge'
    goal = r['valorEsperado']
    tol = r.get('valorTolerado')
    tol = None if (isinstance(tol, str) and tol == 'NaN') else tol
    uval = r['unidadValor']          # unidad del umbral
    ucorr = r['unidadCorrespondiente']  # unidad del eje independiente (nivel)
    cond = r.get('condicion') is not None

    base = {'op': op, 'conditional': cond}
    if tol is not None:
        base['tol'] = tol

    # Umbral en % para Dmax/Dmean = % de Rx (igual que en las métricas D), no un 'pct' ambiguo
    dose_unit = 'pct_rx' if uval == '%' else map_unit(uval)
    if tipo == 'RestriccionDosisMax':
        base.update({'type': 'Dmax', 'value': goal, 'value_unit': dose_unit})
    elif tipo == 'RestriccionDosisMedia':
        base.update({'type': 'Dmean', 'value': goal, 'value_unit': dose_unit})
    elif tipo == 'RestriccionDosis':
        # D a volumen: valorCorrespondiente = volumen, ucorr = % o cm3; umbral en %Rx (o Gy)
        base.update({'type': 'D', 'param': r['valorCorrespondiente'],
                     'param_unit': 'pct_vol' if ucorr == '%' else map_unit(ucorr),
                     'value': goal,
                     'value_unit': 'pct_rx' if uval == '%' else map_unit(uval)})
    elif tipo == 'RestriccionVolumen':
        # V a dosis: valorCorrespondiente = dosis, ucorr = Gy o %; umbral en %vol o cm3
        base.update({'type': 'V', 'param': r['valorCorrespondiente'],
                     'param_unit': 'pct_rx' if ucorr == '%' else map_unit(ucorr),
                     'value': goal,
                     'value_unit': 'pct_vol' if uval == '%' else map_unit(uval)})
    else:
        base.update({'type': 'UNKNOWN_' + tipo, 'value': goal})
    return base

# ----------------------------------------------------------------------------
# Unificación D<=0.035cm3 -> Dmax
# ----------------------------------------------------------------------------
# D0.035cm3 es equivalente a Dmax (decisión del proyecto). En la grilla de dosis 2.5x2.5x3 mm
# un vóxel son 0.01875 cm3, o sea 0.035 cm3 son ~2 vóxeles: indistinguible de Dmax a esta
# resolución. SOLO se unifica este umbral. Los casi-máximo mayores (D1cm3, D2%, D0.5cm3) NO
# se tocan: su canal es la pregunta viva del CHARTER §10.
NEAR_MAX_EQUIV_CM3 = 0.035

def unify_near_max(name, goals, log):
    out = []
    for m in goals:
        if m['type'] == 'D' and m.get('param_unit') == 'cm3' and m['param'] <= NEAR_MAX_EQUIV_CM3:
            orig = m['_label'].split(': ', 1)[-1]          # p.ej. 'D0.035cm3'
            m = {k: v for k, v in m.items() if k not in ('param', 'param_unit')}
            m['type'] = 'Dmax'
            m['unified_from'] = orig
            log.append(f"{name}: {orig} -> Dmax")
        out.append(m)
    # Si la unificación genera un Dmax duplicado (misma dirección), fusionar conservando el más
    # estricto y la mayor prioridad (número menor). Dos Dmax originales no se tocan.
    final = []
    for m in out:
        twin = None
        if m['type'] == 'Dmax':
            twin = next((x for x in final if x['type'] == 'Dmax' and x['op'] == m['op']
                         and ('unified_from' in x or 'unified_from' in m)), None)
        if twin is None:
            final.append(m)
            continue
        if twin.get('value_unit') != m.get('value_unit'):
            final.append(m)
            log.append(f"{name}: AVISO Dmax duplicado con unidades distintas, NO se fusiona (revisar a mano)")
            continue
        pick = min((twin, m), key=lambda x: x['value']) if m['op'] == 'le' \
            else max((twin, m), key=lambda x: x['value'])
        pick['priority'] = min(twin['priority'], m['priority'])
        pick['unified_from'] = twin.get('unified_from') or m.get('unified_from')
        final[final.index(twin)] = pick
        log.append(f"{name}: Dmax duplicado fusionado (se conserva el más estricto)")
    return final

# ----------------------------------------------------------------------------
# Conversión
# ----------------------------------------------------------------------------
def convert(path, site, rx, protocol_id):
    d = json.load(open(path, encoding='utf-8-sig'))
    targets = defaultdict(list)     # nivel -> goals
    target_geom = {}                # nivel -> candidatos de geometría (nombresPosibles)
    oars = defaultdict(list)        # rol -> goals
    aliases = {}                    # nombre_crudo.lower -> rol
    excluded = []                   # (nombre, motivo)
    conditionals = []               # etiquetas de constraints condicionales

    for r in d['listaRestricciones']:
        nombre = r['estructura']['nombre']
        posibles = r['estructura'].get('nombresPosibles', [nombre])
        clase, rol, nivel = classify(nombre)
        metric = parse_metric(r)
        metric['priority'] = int(r['prioridad'])
        metric['_label'] = r['etiquetaInicio']
        if metric['conditional']:
            conditionals.append(r['etiqueta'])

        if clase == 'target':
            targets[nivel].append(metric)
            target_geom.setdefault(nivel, posibles)  # orden de fallback ya codificado
            for p in posibles:
                aliases[p.lower()] = nivel
        elif clase == 'oar':
            oars[rol].append(metric)
            for p in posibles:
                aliases[p.lower()] = rol
        else:  # ctv_excluded | subtraction_excluded
            excluded.append((nombre, clase))

    unify_log = []
    for k in list(targets):
        targets[k] = unify_near_max(k, targets[k], unify_log)
    for k in list(oars):
        oars[k] = unify_near_max(k, oars[k], unify_log)

    return {'site': site, 'protocol_id': protocol_id, 'rx': rx,
            'targets': dict(targets), 'target_geom': target_geom,
            'oars': dict(oars), 'aliases': aliases,
            'excluded': excluded, 'conditionals': conditionals,
            'unify_log': unify_log}

# ----------------------------------------------------------------------------
# Emisión YAML (manual, para control de formato y comentarios)
# ----------------------------------------------------------------------------
def fmt_metric(m):
    parts = [f"type: {m['type']}"]
    if 'param' in m:
        parts.append(f"param: {m['param']}")
        parts.append(f"param_unit: {m['param_unit']}")
    inner = ', '.join(parts)
    lim = f"op: {m['op']}, value: {m['value']}"
    if m.get('value_unit'):
        lim += f", unit: {m['value_unit']}"
    extra = f", priority: {m['priority']}"
    if m.get('conditional'):
        extra += ", conditional: true"
    if 'tol' in m:
        extra += f", tol: {m['tol']}"
    if 'unified_from' in m:
        extra += f", unified_from: {m['unified_from']}"
    return (f"      - {{metric: {{{inner}}}, limit: {{{lim}}}{extra}}}"
            f"   # {m['_label']}")

def emit_yaml(res):
    L = []
    L.append(f"# === sites/{res['site'].lower()}.yaml — generado por convert_constraints.py ===")
    L.append(f"# Revisar antes de usar. Rx inyectada de hojas de protocolo (externa a DICOM).")
    L.append(f"site: {res['site']}")
    L.append("")
    L.append("protocols:")
    rxline = ', '.join(f"{k}: {v}" for k, v in res['rx'].items())
    L.append(f"  - id: {res['protocol_id']}")
    L.append(f"    levels: [{', '.join(res['rx'].keys())}]")
    L.append(f"    rx_gy: {{{rxline}}}")
    L.append("")
    L.append("targets:")
    for nivel in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
        if nivel not in res['targets']:
            continue
        geom = res['target_geom'][nivel]
        L.append(f"  {nivel}:")
        L.append(f"    geometry_candidates: [{', '.join(geom)}]   # orden de fallback (-04 primero)")
        L.append(f"    goals:")
        for m in res['targets'][nivel]:
            L.append(fmt_metric(m))
    L.append("")
    L.append("oars:")
    for rol in sorted(res['oars']):
        L.append(f"  {rol}:")
        for m in res['oars'][rol]:
            L.append(fmt_metric(m))
    return '\n'.join(L)

def emit_report(res):
    R = []
    R.append(f"### Reporte de conversión — {res['site']}")
    R.append(f"- Targets (niveles PTV): {list(res['targets'].keys())}")
    R.append(f"- OARs: {len(res['oars'])}  | goals totales OAR: {sum(len(v) for v in res['oars'].values())}")
    R.append(f"- Alias sembrados: {len(res['aliases'])}")
    if res['excluded']:
        R.append(f"- EXCLUIDOS (no entran a intención):")
        for n, c in res['excluded']:
            R.append(f"    · {n}  ({c})")
    if res['unify_log']:
        R.append(f"- UNIFICADOS a Dmax (D<={NEAR_MAX_EQUIV_CM3}cm3), {len(res['unify_log'])}:")
        for e in res['unify_log']:
            R.append(f"    · {e}")
    if res['conditionals']:
        R.append(f"- CONDICIONALES (flag manual, {len(res['conditionals'])}):")
        for e in res['conditionals']:
            R.append(f"    · {e.strip()}")
    return '\n'.join(R)

# ----------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--site', required=True)
    ap.add_argument('--rx', nargs='+', required=True, help='PTV_High=69.96 ...')
    ap.add_argument('--protocol-id', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rx = OrderedDict()
    for kv in a.rx:
        k, v = kv.split('='); rx[k] = float(v)
    res = convert(a.path, a.site, rx, a.protocol_id)
    open(a.out, 'w', encoding='utf-8').write(emit_yaml(res) + '\n')
    print(emit_report(res))
