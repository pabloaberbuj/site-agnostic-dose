#!/usr/bin/env python3
"""
Consolidador global de armonización. Lee las plantillas in-house de los 3 sitios,
deriva el vocabulario de roles y el mapa de alias (limpiando la suciedad de
nombresPosibles), y emite harmonization_schema.yaml completo.

Las partes estáticas (ignore, qa_gate, metadata) son plantilla acá; las dinámicas
(roles, aliases) se generan de los datos → una sola fuente, regenerable.

Uso:
    python build_harmonization.py zPabloCyC.txt:CyC zPabloPelvis.txt:Pelvis \
        zPabloProstataHipo.txt:Prostata --out harmonization_schema.yaml
"""
import json, argparse, re, sys, os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess.normalize_structure_name import normalize_ptv_name, ALIASES as PTV_NAME_ALIASES

# Alias manuales que no salen de nombresPosibles de las plantillas (se pierden si no se
# fijan acá): confirmados con D95 de Eclipse y, pendiente, con el D95 de Python (ver
# docs/PROMPT_python_pendientes_armonizacion.md, Tarea 4). Misma lista que ALIASES en
# normalize_structure_name.py (fuente de verdad del nivel de PTV) — no diverger.
# NO agregar aquí 'ptv_prostate' ni 'ptvp': son estructuras distintas (más chicas) que
# PTV_High, no alias.
MANUAL_ALIASES = {v: 'PTV_' + lvl.capitalize() for v, lvl in PTV_NAME_ALIASES.items()}

# Alias manuales de OAR (no-PTV) vistos al minar datos reales (Paso 2), no derivables de
# nombresPosibles. Todos son el MISMO órgano/margen que un rol ya canónico, con otro orden
# de palabras, abreviatura o padding de dígito del margen PRV — no una estructura distinta.
MANUAL_OAR_ALIASES = {
    'cavity_oral': 'Oral_Cavity',                      # orden de palabras invertido
    'cabeza_femoral_l': 'FemoralHead_L',                # castellano
    'cabeza_femoral_r': 'FemoralHead_R',
    'brchplex_l_prv05': 'BrachPlex_PRV5_L',             # abrev. "Brch" + orden L/PRV invertido
    'brchplex_r_prv05': 'BrachPlex_PRV5_R',
    'opticnrv_l_prv03': 'OpticNrv_PRV03_L',             # orden L/PRV invertido
    'opticnrv_r_prv03': 'OpticNrv_PRV03_R',
    'opticchiasmprv03': 'OpticChiasm_PRV3',             # mismo margen (3mm), sin guion bajo
    'bowel_bag': 'Bowel_Small',                         # sinonimo frecuente (Prostata), mismo organo
    'bowelbag': 'Bowel_Small',                          # idem, sin guion bajo
    'colon_sigmoid': 'Sigmoid',                         # muy frecuente (Prostata)
    'colon-sigmoid': 'Sigmoid',
    'penilbulb': 'PenileBulb',                          # sin guion bajo + minuscula
    'pituitarygland': 'Pituitary',                      # nombre mas largo, mismo organo
    'tmjoint_l': 'Joint_TM_L',                           # orden invertido
    'tmjoint_r': 'Joint_TM_R',
    'lens left': 'Lens_L',                               # ingles con espacio
    'lens right': 'Lens_R',
    'bladder_p': 'Bladder',                             # sufijo _P visto en un set propio de un plan
    'bowel_p': 'Bowel_Small',
    'rectum_p': 'Rectum',
}

# Estructuras reales (NO ruido de optimización, van a 'exclude' no a 'ignore') que duplican
# un rol ya canónico y no se modelan (CHARTER §5: en seriales "el PRV es la geometría").
# Confirmado en datos reales (Paso 2): el par base/PRV coexiste en el mismo paciente con
# volúmenes distintos (PRV > base, el margen esperado) — no es ambigüedad, es la geometría
# pre-margen que el template no usa. Mismo criterio en ambos sentidos: si el canónico de la
# plantilla es el PRV, el órgano llano sobra; si el canónico es el órgano llano (ej. Cochlea,
# no serial en este protocolo), el PRV extra sobra.
EXCLUDE_NAMES = {
    # canónico = PRV (serial) -> el órgano llano no se modela
    'Brainstem', 'SpinalCord', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R',
    'BrachialPlex_L', 'BrachialPlex_R', 'BrachialPlexus_L', 'BrachialPlexus_R', 'Chiasm',
    # canónico = órgano llano (no serial en este protocolo) -> el PRV extra no se modela
    'Cochlea_PRV03_L', 'Cochlea_PRV03_R', 'Cochlea_L_PRV03', 'Cochlea_R_PRV03', 'Bowel_PRV',
    # margen de PRV NO estandar para este organo (vs. el margen ya canónico de cada uno) — no
    # se funde, sería conflar dos geometrías clínicas distintas.
    'OpticChiasm_PRV2', 'OpticNrv_PRV02_L', 'OpticNrv_PRV02_R', 'SpinalCord_PRV1',
    # subestructuras de PTV que NO son el PTV modelado (ver patient_exceptions.yaml / CHARTER §5):
    # PTVp/PTVn y PTV_Uterus/PTV_LN_* (Pelvis: primario/nodal, PTV_High = suma de ambos),
    # PTV_Prostate/PTV_SeminalVes (Prostata: subestructuras más chicas que PTV_High),
    # PTV_total/PTV_Low_total (combinación de niveles, no un nivel propio).
    # NO aliasar a PTV_High: fundir perdería la geometria de cada subcomponente.
    'PTVp', 'PTVn', 'PTVp_', 'PTVn_', 'PTV_Prostate', 'PTV_SeminalVes',
    'PTV_Uterus', 'PTV_Uterurs', 'PTV_LN_Pelvics', 'PTV_LN_Inguinofe', 'PTV_LN_Iliac',
    'PTV_Pelvics', 'PTV_Pelvis', 'PTV_total', 'PTV_Low_total',
    # resta OAR+PTV concatenada sin separador (mismo patron que '-PTV'/'!PTV', sin signo)
    'RectumPTV', 'BladderPTV',
}

def classify(nombre):
    n = nombre.strip()
    nivel, _cropped = normalize_ptv_name(n)
    if nivel:
        return ('target', 'PTV_' + nivel.capitalize())
    if n.upper().startswith('CTV'):
        return ('exclude', n)
    if '-PTV' in n.upper():
        return ('exclude', n)
    return ('oar', n)

def clean_candidates(cands, canonical):
    """Limpia nombresPosibles: drop vacíos, dedup, descarta concatenaciones repetidas."""
    out = []
    for c in cands:
        c = (c or '').strip()
        if not c:
            continue
        # descartar repetición concatenada tipo 'PTV_HighPTV_High'
        if any(c == base * k for base in {canonical, c[:len(c)//2]} for k in (2, 3, 4) if base):
            continue
        if c not in out:
            out.append(c)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='archivo.txt:Sitio')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    role_sites = defaultdict(set)   # rol -> {sitios que lo usan}
    role_cat = {}                   # rol -> 'target'|'oar'
    aliases = {}                    # variante.lower -> rol  (solo no-identidad)
    excluded = defaultdict(set)     # nombre -> {sitios}
    dirty = []                      # (sitio, nombre, candidatos_crudos) descartados

    for item in a.inputs:
        path, site = item.rsplit(':', 1)
        d = json.load(open(path, encoding='utf-8-sig'))
        for r in d['listaRestricciones']:
            nombre = r['estructura']['nombre']
            raw = r['estructura'].get('nombresPosibles', [nombre])
            cat, rol = classify(nombre)
            if cat == 'exclude':
                excluded[nombre].add(site)
                continue
            role_sites[rol].add(site)
            role_cat[rol] = cat
            clean = clean_candidates(raw, rol)
            if clean != [c for c in raw if c]:           # hubo suciedad
                dirty.append((site, nombre, raw))
            for v in clean:
                if v.lower() != rol.lower():              # solo alias no-identidad
                    aliases[v.lower()] = rol

    for v, rol in MANUAL_ALIASES.items():
        aliases.setdefault(v, rol)
    for v, rol in MANUAL_OAR_ALIASES.items():
        aliases.setdefault(v, rol)

    emit(a.out, role_sites, role_cat, aliases, excluded, dirty)

def emit(out, role_sites, role_cat, aliases, excluded, dirty):
    targets = sorted([r for r, c in role_cat.items() if c == 'target'])
    oars = sorted([r for r, c in role_cat.items() if c == 'oar'])
    L = []
    L += ["# ============================================================================",
          "# Tabla de armonización — SCHEMA GLOBAL",
          "# Roles y aliases GENERADOS por build_harmonization.py de las plantillas reales.",
          "# ignore/qa_gate/metadata son estáticos (editables a mano). Regenerable.",
          "# ============================================================================",
          "schema_version: 2", ""]

    L += ["# ---- 1. VOCABULARIO DE ROLES (sufijo [sitios] = dónde aplica) ----",
          "roles:", "  targets:"]
    for r in targets:
        L.append(f"    - {r}")
    L.append("  oars:")
    for r in oars:
        sites = ','.join(sorted(role_sites[r]))
        note = "   # PRV: geometría del serial" if 'PRV' in r.upper() else ""
        L.append(f"    - {r}    # [{sites}]{note}")
    L.append("")

    L += ["# ---- 2. ALIAS (variante -> rol; derivados de nombresPosibles, ya limpios) ----",
          "#      Extender a mano con variantes vistas al minar (castellano, lateralidad).",
          "aliases:"]
    for v in sorted(aliases):
        if v in MANUAL_ALIASES:
            tag = "   # manual (ver MANUAL_ALIASES)"
        elif v in MANUAL_OAR_ALIASES:
            tag = "   # manual (ver MANUAL_OAR_ALIASES)"
        else:
            tag = ""
        L.append(f'  "{v}": {aliases[v]}{tag}')
    L.append("  # --- extensión manual (ejemplos; completar al minar) ---")
    for ex in ['"recto": Rectum', '"vejiga": Bladder',
               '"parotida_izq": Parotid_L', '"medula": SpinalCord_PRV05']:
        L.append(f"  # {ex}")
    L.append("")

    L += ["# ---- 3. EXCLUDE_NAMES (estructuras REALES, no ruido — van a 'exclude', no a 'ignore'.",
          "#      Se reportan, no se modelan. Distinto de ignore_patterns: esto es lista explícita,",
          "#      no patrón, porque son nombres concretos con semántica clínica, no helpers.) ----",
          "exclude_names:"]
    for n in sorted(EXCLUDE_NAMES):
        L.append(f'  - "{n}"')
    L.append("")

    L += ["# ---- 4. IGNORE (SIN *_prv: los PRV son geometría de seriales, no se ignoran) ----",
          "ignore_patterns:",
          '  - "couch*"',
          '  - "*marker*"',
          '  - "ptv_eval*"',
          '  - "ring*"',
          '  - "z*"               # z_*, zring*, zopt*, pero tambien zArtifact/zShoulders/zCal_*/',
          '                        # zCtrol/zDosis*/zBody/zHigh/zPTV_High57 sin guion bajo (Paso 2,',
          '                        # revision real de datos): mismo convenio interno de estructuras',
          '                        # de optimizacion/trabajo, no se puede enumerar variante por variante.',
          '  - "opt*"              # igual que z*: visto "OPTPTV_Low" concatenado sin guion bajo',
          '  - "avoid*"',
          '  - "*_helper"',
          '  - "*marcador*"       # "marker" en castellano, mismo convenio que *marker*',
          '  - "*control*"        # estructuras de trabajo tipo "Control Region" / "NS_Control"',
          '  - "dose*"',
          '  - "dosis*"           # isodosis convertida a ROI para visualizacion (ej. "Dose 107[%]")',
          '  - "match*"           # puntos/lineas de union de isocentros ("Match points")',
          '  - "ns_*"              # prefijo interno visto en varias estructuras de trabajo',
          '                        # (NS_Control, NS_ControlPrueba, NS_NormalTissue, NS_LN, NS_Ring, NS_UTERUS)',
          '  - "poi*"              # Point Of Interest de Eclipse, no es una estructura',
          '  - "*_eval*"           # evaluacion ("PTV_High_Eval"); ya existe ptv_eval* (prefijo),',
          '                        # esto cubre el sufijo',
          '  - "*marcapaso*"       # dispositivo (castellano "marcapasos"), no anatomia',
          '  - "*caliente*"        # region "hot" de revision (castellano), visto en CyC y Prostata',
          '  - "*cal*"             # Cal/cal_low/cal_high/Low_Cal/Mid_Cal/High_Cal: helper de',
          '                        # calculo, ningun rol canonico contiene "cal"',
          '  - "none"              # artefacto de export (nombre literal "None")',
          '  - "ctrl*"             # forma corta de "control" (ver *control*)',
          ""]

    L += ["# ---- 5. QA GATE (dosis ABSOLUTA cruda Gy; D95 vs Rx_nivel; banda única) ----",
          "qa_gate:",
          "  metric: D95",
          "  dose_space: absolute_gy",
          "  bands:",
          "    default: {floor_pct: 95, ceil_pct: 105}   # provisional; recalibrar si muchos falsos flags",
          "  on_fail: flag_for_manual_review", ""]

    L += ["# ---- METADATA POR PACIENTE (esquema) ----",
          "patient_metadata_schema:",
          "  patient_id: str              # = AnonID del extractor (no se construye mapeo propio)",
          "  site: str",
          "  protocol_id: str",
          "  rx_high_gy: float",
          "  rx_by_level_gy: {}",
          "  ptv_levels_present: []       # niveles realmente resueltos (no todos los pacientes tienen Mid/Low)",
          "  spacing_x_mm: float          # grilla de dosis nativa, sin resamplear",
          "  spacing_y_mm: float",
          "  spacing_z_mm: float          # 3.0 en la mayoría; 2.0 en un subgrupo (ver Paso 1) — SIEMPRE registrar, nunca asumir",
          "  body_bbox_voxels: {}         # {i_min,i_max,j_min,j_max,k_min,k_max} sobre la grilla nativa; recorte/relleno lo hace el datamodule",
          "  out_of_convention: null      # null | motivo -> conjunto reservado (generalización)",
          "  qa_flags: []",
          "  aliases_applied: []",
          "  harmonization_hash: str      # hash de los YAML de armonización usados en este NPZ", ""]

    # apéndice informativo (comentado): exclusiones y suciedad detectada
    L += ["# ---- APÉNDICE (informativo) ----",
          "# Estructuras EXCLUIDAS de intención (CTV / restas):"]
    for n, sites in sorted(excluded.items()):
        L.append(f"#   {n}  [{','.join(sorted(sites))}]")
    if dirty:
        L.append("# nombresPosibles SUCIOS detectados y limpiados (revisar origen in-house):")
        for site, n, raw in dirty:
            L.append(f"#   [{site}] {n}: {raw}")

    open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f"OK -> {out}")
    print(f"  targets: {targets}")
    print(f"  OARs: {len(oars)}")
    print(f"  aliases no-identidad: {len(aliases)}  -> {dict(sorted(aliases.items()))}")
    print(f"  excluidos: {dict((k, sorted(v)) for k,v in excluded.items())}")
    print(f"  nombresPosibles sucios: {len(dirty)}")

if __name__ == '__main__':
    main()
