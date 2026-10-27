"""
Loader de la tabla de armonización versionada (data/harmonization/*.yaml).

La tabla es INPUT del modelo (CHARTER §5): de acá salen los ratios Rx_nivel/Rx_High del canal de
intención y, en A1+, los goals. Cada experimento registra `table.hash`.

El hash replica `harmonization_hash` de scripts/run_preprocess.py (sha256 de nombre+contenido de
cada .yaml del directorio, orden alfabético, 16 hex) — así es comparable con el
`harmonization_hash` que lleva el sidecar JSON de cada NPZ.
"""
import hashlib
import os
from pathlib import Path

import yaml

SITE_FILES = {'Prostata': 'sites_prostata.yaml', 'Pelvis': 'sites_pelvis.yaml', 'CyC': 'sites_cyc.yaml'}
BODY_ROLE = 'BODY'
# Motivo registrado en patient_exceptions.yaml (Paso 3, QA gate): PTV_Low no declarado en el
# protocolo pero idéntico a PTV_High -> máscara inerte, el datamodule la omite (con registro).
INERT_PTV_LOW_MOTIVO = 'qa_ptv_low_identico_a_ptv_high'


class HarmonizationError(Exception):
    pass


def harmonization_hash(harmonization_dir) -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(harmonization_dir)):
        if fn.endswith('.yaml'):
            with open(os.path.join(harmonization_dir, fn), 'rb') as f:
                h.update(fn.encode() + f.read())
    return h.hexdigest()[:16]


class HarmonizationTable:
    def __init__(self, harmonization_dir):
        self.dir = Path(harmonization_dir)
        self.hash = harmonization_hash(self.dir)
        schema = yaml.safe_load(open(self.dir / 'harmonization_schema.yaml', encoding='utf-8'))
        self.target_roles = tuple(schema['roles']['targets'])
        self.oar_roles = tuple(schema['roles']['oars'])
        exc = yaml.safe_load(open(self.dir / 'patient_exceptions.yaml', encoding='utf-8'))
        self.inert_ptv_low = frozenset(e['id'] for e in (exc.get('reviewed_ok') or [])
                                       if e.get('motivo') == INERT_PTV_LOW_MOTIVO)
        self._sites = {site: yaml.safe_load(open(self.dir / fn, encoding='utf-8'))
                       for site, fn in SITE_FILES.items()}

    @property
    def known_roles(self) -> set:
        return set(self.target_roles) | set(self.oar_roles) | {BODY_ROLE}

    def protocol(self, site: str, protocol_id: str) -> dict:
        """Protocolo `{id, levels, rx_gy}` de la tabla. Falla ruidoso si sitio/protocolo no existen."""
        if site not in self._sites:
            raise HarmonizationError(f'sitio desconocido: {site!r} (tabla tiene {sorted(self._sites)})')
        for p in self._sites[site]['protocols']:
            if p['id'] == protocol_id:
                return p
        raise HarmonizationError(f'protocol_id {protocol_id!r} no está en la tabla de {site}')

    def intent_ratios(self, site: str, protocol_id: str) -> dict:
        """{rol de PTV: Rx_nivel/Rx_High} PRESCRIPTOS de la tabla (nunca de la dosis: leakage)."""
        rx = self.protocol(site, protocol_id)['rx_gy']
        high = rx['PTV_High']
        return {lvl: float(v) / float(high) for lvl, v in rx.items()}

    # ---- goals de OAR (A1): se construyen al cargar desde la tabla (CHARTER §5) ----
    def oar_goals(self, site: str, protocol_id: str) -> dict:
        """{rol OAR: [goal]} con valores normalizados a fracción de Rx_High (1.0 = Rx_High).

        goal = {'channels': [...], 'ceiling': float, 'budget_frac': float|None, 'budget_cm3': float|None,
                'conditional': bool, 'label': str}. `budget_*` solo en los volumétricos (fracción del volumen
        que PUEDE superar el techo; la preservada es 1-budget). Usa `limit` (no `tol`).
        """
        rx_high = float(self.protocol(site, protocol_id)['rx_gy']['PTV_High'])

        def frac(v, unit):
            if unit == 'Gy':
                return float(v) / rx_high
            if unit == 'pct_rx':
                return float(v) / 100.0
            raise HarmonizationError(f'unidad de dosis no soportada: {unit!r}')

        def budget(v, unit):
            if unit == 'pct_vol':
                return float(v) / 100.0, None
            if unit == 'cm3':
                return None, float(v)
            raise HarmonizationError(f'unidad de volumen no soportada: {unit!r}')

        out = {}
        for role, goals in self._sites[site].get('oars', {}).items():
            for g in goals:
                m, lim = g['metric'], g['limit']
                label = f"{site}/{role}: {m['type']}" + (f"{m['param']}{m['param_unit']}" if 'param' in m else '')
                if lim['op'] != 'le':
                    raise HarmonizationError(f'{label}: op {lim["op"]!r} no soportado en OAR (solo le)')
                t = m['type']
                b_frac = b_cm3 = None
                if t == 'Dmax':
                    ch, ceil = ['maximos'], frac(lim['value'], lim['unit'])
                elif t == 'Dmean':
                    ch, ceil = ['medias'], frac(lim['value'], lim['unit'])
                elif t == 'V':       # techo = dosis del V; presupuesto = límite de volumen
                    ch, ceil = ['volumetrico'], frac(m['param'], m['param_unit'])
                    b_frac, b_cm3 = budget(lim['value'], lim['unit'])
                elif t == 'D':       # techo = límite de dosis; presupuesto = volumen del Dx
                    if not g.get('channels'):
                        raise HarmonizationError(f'{label}: casi-máximo volumétrico sin `channels` explícito (CHARTER §10)')
                    ch, ceil = list(g['channels']), frac(lim['value'], lim['unit'])
                    b_frac, b_cm3 = budget(m['param'], m['param_unit'])
                else:
                    raise HarmonizationError(f'{label}: tipo de métrica no soportado')
                bad = set(ch) - {'maximos', 'medias', 'volumetrico'}
                if bad:
                    raise HarmonizationError(f'{label}: canales inválidos {sorted(bad)}')
                out.setdefault(role, []).append({'channels': ch, 'ceiling': ceil, 'budget_frac': b_frac,
                                                 'budget_cm3': b_cm3, 'conditional': bool(g.get('conditional')),
                                                 'label': label})
        return out

    def conditional_goals(self) -> list:
        """Goals `conditional: true` de todos los sitios: tratados como normales en A1, diferidos a A3."""
        return sorted(f"{site}/{role}: {g['metric']['type']}" + (f"{g['metric']['param']}{g['metric']['param_unit']}" if 'param' in g['metric'] else '')
                      for site, t in self._sites.items() for role, gl in t.get('oars', {}).items()
                      for g in gl if g.get('conditional'))
