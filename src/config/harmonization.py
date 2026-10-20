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
