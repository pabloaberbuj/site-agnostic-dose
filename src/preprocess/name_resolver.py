"""
Resolucion de nombres de estructura, cuatro salidas (CHARTER §5):
  canonico -> usa; alias conocido -> remapea y loguea; ignore -> saltea;
  desconocido -> FALLA RUIDOSO, va a lista de revision, nunca adivina.

PTV pasa primero por normalize_structure_name.normalize_ptv_name (absorbe el ruido
de separadores / tipeos de "-04" sin enumerar variantes). El resto de los roles
(OARs, PRVs) se resuelven contra el vocabulario + alias de harmonization_schema.yaml.

CTV_* y restas tipo 'Bone_Mand-PTV_Hi' son una quinta categoria de facto (excluidos
de la intencion, se reportan, no fallan): mismo criterio que build_harmonization.py /
convert_constraints.py, para no duplicar esa regla con una definicion distinta.
"""
import fnmatch
import re
from dataclasses import dataclass
from typing import Optional

from .normalize_structure_name import normalize_ptv_name, _CROP

_TRAILING_DIGITS = re.compile(r'\d+$')


@dataclass
class Resolved:
    outcome: str                  # 'canonical' | 'alias' | 'ignore' | 'exclude' | 'unknown'
    role: Optional[str] = None    # rol canonico (None si ignore/exclude/unknown)
    cropped: Optional[bool] = None
    note: str = ''


class NameResolver:
    def __init__(self, schema: dict):
        self.canonical = set(schema['roles']['targets']) | set(schema['roles']['oars'])
        self.aliases = {k.lower(): v for k, v in schema['aliases'].items()}
        self.ignore_patterns = [p.lower() for p in schema['ignore_patterns']]
        self.exclude_names = {n.lower() for n in schema.get('exclude_names', [])}

    def resolve(self, raw_name: str) -> Resolved:
        n = raw_name.strip()

        level, cropped = normalize_ptv_name(n)
        if level:
            role = 'PTV_' + level.capitalize()
            is_exact = n.lower() in (role.lower(), (role + '-04').lower())
            return Resolved('canonical' if is_exact else 'alias', role, cropped,
                             note='' if is_exact else f'normalizado desde "{raw_name}"')

        if n in self.canonical:
            return Resolved('canonical', n)

        if n.lower() in self.aliases:
            return Resolved('alias', self.aliases[n.lower()], note=f'alias de "{raw_name}"')

        for pat in self.ignore_patterns:
            if fnmatch.fnmatch(n.lower(), pat):
                return Resolved('ignore', note=f'matched ignore pattern "{pat}"')

        if n.lower() in self.exclude_names:
            return Resolved('exclude', note='exclude_names (estructura real no modelada)')
        # Las mismas estructuras de exclude_names tambien aparecen con sufijo de crop
        # (ej. 'PTVn-0.4', igual que un PTV real) — no pasan por normalize_ptv_name
        # (no son 'ptv_<nivel>'), asi que se chequea aparte con la misma regla de crop.
        crop_m = _CROP.match(n.lower())
        if crop_m and crop_m.group('base') in self.exclude_names:
            return Resolved('exclude', note=f'exclude_names (variante recortada de "{crop_m.group("base")}")')
        if n.upper().startswith(('CTV', 'GTV')):
            return Resolved('exclude', note='CTV/GTV (objetivo de optimizacion, no intencion)')
        # resta/combinacion sobre PTV: visto con '-' (Bone_Mand-PTV_Hi) y con '!' (Bladder!PTV,
        # PTV!PRV, Rectum!PTV) — mismo patron semantico, distinto separador en los datos reales.
        upper = n.upper()
        if any(sep in upper for sep in ('-PTV', 'PTV-', '!PTV', 'PTV!')):
            return Resolved('exclude', note='resta/combinacion sobre PTV (no se modela)')

        # Duplicado numerado de un rol ya conocido (ej. 'Cochlea_L1', 'SpinalCord2',
        # 'Trachea1'): mismo organo recontoreado, no una estructura nueva. NUNCA se aplica
        # a nombres que empiezan con PTV/GTV/CTV: ahi el numero es semantico (ver
        # normalize_ptv_name y los niveles PTV_Mid01/02 — eso es "mas niveles de PTV que
        # el protocolo", falla ruidoso a proposito, no se colapsa).
        if not upper.startswith(('PTV', 'GTV', 'CTV')):
            stripped = _TRAILING_DIGITS.sub('', n)
            if stripped != n and stripped:
                base = self.resolve(stripped)
                if base.outcome != 'unknown':
                    outcome = 'alias' if base.outcome == 'canonical' else base.outcome
                    return Resolved(outcome, base.role, base.cropped,
                                     note=f'duplicado numerado de "{stripped}"')

        return Resolved('unknown', note='no matchea canonico, alias, ignore ni exclude')
