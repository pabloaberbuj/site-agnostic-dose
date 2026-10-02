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
import json, argparse, re
from collections import defaultdict

PTV_RE = re.compile(r'^PTV_(High|Mid|Low)(-\d+)?$', re.IGNORECASE)

def classify(nombre):
    n = nombre.strip()
    m = PTV_RE.match(n)
    if m:
        return ('target', 'PTV_' + m.group(1).capitalize())
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
        L.append(f'  "{v}": {aliases[v]}')
    L.append("  # --- extensión manual (ejemplos; completar al minar) ---")
    for ex in ['"recto": Rectum', '"vejiga": Bladder',
               '"parotida_izq": Parotid_L', '"medula": SpinalCord_PRV05']:
        L.append(f"  # {ex}")
    L.append("")

    L += ["# ---- 3. IGNORE (SIN *_prv: los PRV son geometría de seriales, no se ignoran) ----",
          "ignore_patterns:",
          '  - "couch*"',
          '  - "*marker*"',
          '  - "ptv_eval*"',
          '  - "ring*"',
          '  - "z_*"',
          '  - "opt_*"',
          '  - "avoid*"',
          '  - "*_helper"', ""]

    L += ["# ---- 4. QA GATE (dosis ABSOLUTA cruda Gy; D95 vs Rx_nivel; banda única) ----",
          "qa_gate:",
          "  metric: D95",
          "  dose_space: absolute_gy",
          "  bands:",
          "    default: {floor_pct: 95, ceil_pct: 105}   # provisional; recalibrar si muchos falsos flags",
          "  on_fail: flag_for_manual_review", ""]

    L += ["# ---- METADATA POR PACIENTE (esquema) ----",
          "patient_metadata_schema:",
          "  patient_id: str              # DE-IDENTIFICADO (el mapeo HC->id NO va a git)",
          "  site: str",
          "  protocol_id: str",
          "  rx_high_gy: float",
          "  rx_by_level_gy: {}",
          "  out_of_convention: null      # null | motivo -> conjunto reservado (generalización)",
          "  qa_flags: []",
          "  aliases_applied: []", ""]

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
