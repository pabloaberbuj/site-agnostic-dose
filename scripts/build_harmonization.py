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
    'brainstem_prv05': 'Brainstem_PRV05',               # mayuscula distinta en "Stem" (BrainStem)
    'cabeza_femoral_l': 'FemoralHead_L',                # castellano
    'cabeza_femoral_r': 'FemoralHead_R',
    'brchplex_l_prv05': 'BrachPlex_PRV5_L',             # abrev. "Brch" + orden L/PRV invertido
    'brchplex_r_prv05': 'BrachPlex_PRV5_R',
    'opticnrv_l_prv03': 'OpticNrv_PRV03_L',             # orden L/PRV invertido
    'opticnrv_r_prv03': 'OpticNrv_PRV03_R',
    # 'opticchiasmprv03': 'OpticChiasm_PRV3' SACADO 2026-10-15: el unico paciente real
    # con este nombre (PT_e5dde7390b2a4567) lo tiene COEXISTIENDO con 'OpticChiasm_PRV3'
    # (2 contornos reales, ambiguedad) — decision del usuario: retener OpticChiasm_PRV3,
    # desestimar OpticChiasmPRV03. Pasa a EXCLUDE_NAMES (si seguia aca como alias, el
    # chequeo de alias corre ANTES que exclude_names y lo tapaba).
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
    # Prostata (decision del usuario 2026-10-14): "RectumCorregido reemplaza a Rectum" ->
    # mismo organo, version corregida de un recontoreo. Si coexisten ambos con volumen>0,
    # el desempate generico de pipeline.py (ver tambien criterio 'prv') prefiere el que
    # dice "corregido" en el nombre crudo (geometria mas reciente/confiable).
    'rectumcorregido': 'Rectum',
    # Lobe_Temporal_PRV3_L/R: mismo margen (3mm), al menos 4 convenios de nombre distintos
    # vistos en datos reales (abreviatura, prefijo vs sufijo, con/sin guion bajo).
    'lobe_tem_prv3_l': 'Lobe_Temporal_PRV3_L',
    'lobe_tem_prv3_r': 'Lobe_Temporal_PRV3_R',
    'prv_lobetempl_03': 'Lobe_Temporal_PRV3_L',
    'prv_lobetempr_03': 'Lobe_Temporal_PRV3_R',
    'prv_lobetempl03': 'Lobe_Temporal_PRV3_L',
    'prv_lobetempr03': 'Lobe_Temporal_PRV3_R',
    # Pelvis (decision del usuario 2026-10-14): SpinalCord/SpinalCanal, con o sin PRV
    # explicito, es la misma geometria serial que SpinalCord_PRV05 (ya canonico en CyC).
    # El canal oseo es de por si un proxy natural de "cordon + margen" -> se trata igual
    # que el PRV aunque el nombre no diga "PRV".
    'spinalcanal': 'SpinalCord_PRV05',
    'spinalcanal_prv': 'SpinalCord_PRV05',
    'spinalcanalprv05': 'SpinalCord_PRV05',
    'spinalcanal_prv5': 'SpinalCord_PRV05',
    # NOTA: 'spinalcord_prvx' (margen "x" = no especificado, visto en Pelvis) NO se alias
    # aca: decision del usuario 2026-10-14 es "eliminar paciente" (margen no estandar, no
    # se funde con el 50Gy/45Gy de SpinalCord_PRV05) — queda 'unknown' adrede. El paciente
    # real visto con ese nombre (PT_7da7b2507f0ac933) ya estaba out_of_convention por otro
    # motivo (campo extendido); ver patient_exceptions.yaml.
    # 'SpinalCord_PRV1' (CyC, margen 1mm) SI va a EXCLUDE_NAMES mas abajo, a pesar de ser la
    # misma decision ("eliminar paciente"): dejarlo 'unknown' no funciona por un bug de
    # interaccion con la regla de duplicado numerado (name_resolver.py) — le saca el "1"
    # final, "SpinalCord_PRV" (sin digito) YA es alias valido (margen estandar), entonces
    # SpinalCord_PRV1 termina resolviendo de todas formas, pero generando ambiguedad con
    # SpinalCord_PRV05 (PipelineError, el paciente ni siquiera genera NPZ reservado — peor
    # que excluir la estructura). El unico paciente real (PT_caefecceac8746a3) ya esta
    # out_of_convention por otro motivo (PTV_C2), asi que el resultado final (reservado, no
    # en train) es el mismo con exclude_names que con "unknown".
    'spinalcord_prv': 'SpinalCord_PRV05',
    'spinalcord_prv5': 'SpinalCord_PRV05',
    'spinal canal': 'SpinalCord_PRV05',
    # CyC (decision del usuario 2026-10-14, revision de los casos de PRV-con-prefijo):
    # "PRV_<Organo><margen>" o "<Organo>PRV" (sin margen explicito, se asume el margen
    # estandar del organo) en vez del convenio sufijo "<Organo>_PRV<margen>_<Lado>".
    # Mismo margen -> mismo rol, no una estructura distinta. Visto en 2 pacientes con
    # convenio propio (PT_559aefbb26262e5c, PT_8a9d69ea4ef84f91); enumerado explicito
    # (no regex generico) porque las abreviaturas de organo varian demasiado para una
    # regla unica confiable.
    'prv_brainstem05': 'Brainstem_PRV05',
    'brainstemprv': 'Brainstem_PRV05',
    'prv_spinalcord05': 'SpinalCord_PRV05',
    'spinalcordprv': 'SpinalCord_PRV05',
    'prv_brachialpl05': 'BrachPlex_PRV5_L',
    'prv_brachialpr05': 'BrachPlex_PRV5_R',
    'brachialp_l_prv': 'BrachPlex_PRV5_L',
    'brachialp_r_prv': 'BrachPlex_PRV5_R',
    'prv_opticc03': 'OpticChiasm_PRV3',
    'prv_opticnl03': 'OpticNrv_PRV03_L',
    'prv_opticnr03': 'OpticNrv_PRV03_R',
}

# Estructuras reales (NO ruido de optimización, van a 'exclude' no a 'ignore') que duplican
# un rol ya canónico y no se modelan (CHARTER §5: en seriales "el PRV es la geometría").
# Confirmado en datos reales (Paso 2): el par base/PRV coexiste en el mismo paciente con
# volúmenes distintos (PRV > base, el margen esperado) — no es ambigüedad, es la geometría
# pre-margen que el template no usa. Mismo criterio en ambos sentidos: si el canónico de la
# plantilla es el PRV, el órgano llano sobra; si el canónico es el órgano llano (ej. Cochlea,
# no serial en este protocolo), el PRV extra sobra.
# Roles que NO salen de ninguna plantilla (no hay nombresPosibles que minar): se agregan
# a mano tras confirmar en datos reales + criterio clinico del usuario (Paso 2). El goal
# correspondiente vive en MANUAL_OARS de convert_constraints.py (mismo rol, no duplicar
# el valor del constraint en dos lugares).
MANUAL_ROLES = {
    # Lobe_Temporal_L/R contorneado con frecuencia en CyC real, ausente de la plantilla.
    # Dmax<70Gy prioridad 3 sobre el PRV (geometria del serial) — confirmado por el usuario.
    'Lobe_Temporal_PRV3_L': {'CyC'},
    'Lobe_Temporal_PRV3_R': {'CyC'},
    # Pelvis (decision del usuario 2026-10-14, revision de los 49 casos de "campo
    # extendido" de Paso 2): SpinalCord/SpinalCanal con constraint Dmax<50Gy sobre el
    # PRV — mismo rol que CyC, se le agrega el sitio (la union con {'CyC'} ya derivado
    # de la plantilla la hace main() via role_sites[rol] |= sites).
    'SpinalCord_PRV05': {'Pelvis'},
    # Kidney: Dmedia<15Gy. 'Kidneys' (bilateral combinado) si existe esa estructura;
    # si no, Kidney_L/Kidney_R por separado. La mayoria estan lejos del campo
    # (contorneadas por completitud); los casos con campo extendido real (Y del RP
    # >30cm) quedan aparte en patient_exceptions.yaml, no se funden en este rol.
    'Kidneys': {'Pelvis'},
    'Kidney_L': {'Pelvis'},
    'Kidney_R': {'Pelvis'},
}

EXCLUDE_NAMES = {
    # canónico = PRV (serial) -> el órgano llano no se modela
    'Brainstem', 'SpinalCord', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R',
    'BrachialPlex_L', 'BrachialPlex_R', 'BrachialPlexus_L', 'BrachialPlexus_R', 'Chiasm',
    'Lobe_Temporal_L', 'Lobe_Temporal_R', 'TemporalLobe_L', 'TemporalLobe_R',
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
    'PTV_Uterus', 'PTV_Uterurs', 'PTV_LN_Pelvics', 'PTV_LN_Pelvic', 'PTV_LN_Inguinofe',
    'PTV_LN_Inguin', 'PTV_LN_Iliac',
    'PTV_Pelvics', 'PTV_Pelvis', 'PTV_total', 'PTV_Low_total',
    # resta OAR+PTV concatenada sin separador (mismo patron que '-PTV'/'!PTV', sin signo)
    'RectumPTV', 'BladderPTV',
    # Pelvis (decision del usuario 2026-10-14): Liver/Heart/Lung_* "desestimar la
    # estructura" (paciente se mantiene) — a diferencia de Kidney, sin chequeo de tamaño
    # de campo (son casos incidentales, no indicio fuerte de campo extendido real por si
    # solos). OJO: Ovary_* es una decision DISTINTA ("excluirlos" = sacar el PACIENTE, no
    # solo la estructura) — por eso NO va aca; queda sin alias/exclude (unknown) para que
    # el pipeline falle salvo por la excepcion explicita en patient_exceptions.yaml (ver
    # motivo ovario_detectado). Iban mezclados por error en una ronda anterior (corregido
    # 2026-10-14 tras aviso del usuario).
    'Liver', 'Heart', 'Lung_L', 'Lung_R', 'Lungs',
    # CyC (decision del usuario 2026-10-14): estructuras reales, se desestiman pero el
    # PACIENTE se mantiene (no necesitan out_of_convention).
    'Traqueostomia', 'Traqueostoma', 'Pretraqueal', 'Retina_L', 'Retina_R',
    'Cornea_L', 'Cornea_R', 'Eye_R_PRV2mm', 'Parotida_L-0.2', 'Parotida_R-0.2',
    'CuelloDer', 'CuelloIzq', 'Nivel6', 'SCVIZQ',
    # mismo criterio que Cochlea_PRV03_*: el canonico es el organo llano, el PRV sobra
    # (convenio de prefijo de los 2 pacientes con nomenclatura propia).
    'PRV_CochleaL03', 'PRV_CochleaR03', 'CochleaLPRV', 'CochleaRPRV',
    # CyC (decision del usuario 2026-10-14): estructuras reales que SI ameritan sacar al
    # paciente (no solo desestimar la estructura) — ver patient_exceptions.yaml para la
    # lista de patient_id. Se agregan aca tambien para que la estructura en si quede
    # categorizada (documentacion), no solo tolerada por la excepcion del paciente.
    'Trachea', 'EyeBack_L', 'EyeFront_L',
    # Pelvis (decision del usuario 2026-10-14, segunda ronda CyC — la estructura resulto
    # ser de Pelvis, no CyC, al verificar en datos reales): resta entre niveles de PTV, no
    # sobre PTV generico. Paciente ya esta out_of_convention por otro motivo (campo
    # extendido) de todas formas. ('RectumPTV' ya estaba arriba, misma regla.)
    'PTV_Low!High',
    # CyC: PTV_Mid01/PTV_Mid02 con volumen=0 (recontoreo vacio/abandonado) -> desestimar la
    # estructura, el paciente se mantiene (normalmente ya cae en out_of_convention de todas
    # formas por protocolo_distinto_sin_PTV_Mid, ver patient_exceptions.yaml). Con volumen>0
    # el paciente se saca entero (mas niveles de PTV que el protocolo) — eso NO se resuelve
    # aca (es decision por paciente, no por nombre), ver motivo mas_niveles_ptv_que_protocolo
    # en patient_exceptions.yaml. El check de sufijo de crop ya cubre PTV_Mid01-04/!04/etc.
    'PTV_Mid01', 'PTV_Mid02',
    # Prostata (decision del usuario 2026-10-14): subdivisiones de PTV_High por nivel de
    # dosis o variantes puntuales de un solo paciente, no un nivel propio del protocolo.
    'PTV_High100%', 'PTV_High95%', 'PTV_High69', 'PTV_High70', 'PTV_High_mod',
    # Prostata: convencion propia de un paciente puntual (resta Rectum-PTV en castellano,
    # sin separador reconocible) y duplicado de Bladder que ambiguaria con 'Bladder' llano.
    'RectoenPTV', 'PRVRECTOENPTV', 'PTVSINRECTO', 'Bladder1',
    # Resolucion de los 20 pendientes de revision humana (decision del usuario,
    # 2026-10-15, ver paso2_pendientes_20_pacientes_resuelto.csv): "desestimar
    # estructura" en todos estos casos, el paciente se mantiene.
    'Hippocampus_L', 'Hippocampus_R',                   # CyC
    'Lobe_Tem_PRV3_Lb', 'Lobe_Tem_PRV3_Rb',              # CyC: variante "b" sin usar, ya
                                                          # resuelve via Lobe_Tem_PRV3_L/R
    'SpinalCord_Aux',                                    # CyC: ya resuelve via SpinalCord_PRV05
    'Elbow',                                             # CyC
    'amigdala', 'supra',                                 # CyC (castellano, fragmentos)
    'PTV_Boost',                                         # CyC (cubre tambien PTV_boost-04 via crop-suffix)
    # Ambiguedades reales de recontoreo resueltas por el usuario: queda SOLO el nombre
    # que el usuario desestimo, el que retuvo sigue resolviendo por su via normal
    # (canonico o alias ya existente).
    'Cochlea_L1',                                        # retener Cochlea_L
    'Musc_Constrict1',                                   # retener Musc_Constrict
    'OpticChiasmPRV03',                                  # retener OpticChiasm_PRV3 (ver MANUAL_OAR_ALIASES)
    'Sigma',                                              # Pelvis
    'Duodenum', 'Stomach', 'Vulva',                      # Pelvis + Prostata (Stomach)
    'PTV_Uterus1',                                       # Pelvis: mismo criterio que PTV_Uterus
    'PTV_LN', 'PTV_Lesion',                              # Pelvis: PTV_High ya es el PTV real
    'SacralPlexus',                                      # Pelvis
    'Bones',                                             # Pelvis
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
    for rol, sites in MANUAL_ROLES.items():
        role_sites[rol] |= sites
        role_cat.setdefault(rol, 'oar')

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
        if r in MANUAL_ROLES:
            note += " MANUAL (ver MANUAL_ROLES, convert_constraints.py:MANUAL_OARS)"
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
          '  - "*contraste*"       # agente de contraste marcado como ROI (Pelvis), mismo',
          '                        # tratamiento que Cal/Caliente (decision del usuario 2026-10-14)',
          ""]

    L += ["# ---- 5. QA GATE (dosis ABSOLUTA cruda Gy; D95 vs Rx_nivel; banda por sitio) ----",
          "qa_gate:",
          "  metric: D95",
          "  dose_space: absolute_gy",
          "  bands:",
          "    default: {floor_pct: 95, ceil_pct: 105}   # provisional; recalibrar si muchos falsos flags",
          "    Prostata: {floor_pct: 92, ceil_pct: 105}  # Paso 3 (2026-10-16): 8/146 PTV_High",
          "                                              # caian sistematicamente en 92.9-94.6%",
          "                                              # (franja angosta, no dispersion al azar)",
          "                                              # -> piso real del sitio es ~92%, no 95%",
          "                                              # (decision del usuario sobre paso3_qa_gate.csv)",
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
          "  harmonization_hash: str      # hash de los YAML de armonización usados en este NPZ",
          "  warnings: []                 # ej. estructuras desconocidas ignoradas (solo en pacientes con excepción registrada)", ""]

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
