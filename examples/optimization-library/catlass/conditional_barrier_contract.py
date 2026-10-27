"""L05 actual per-case helper dimensions; bounded represented-input oracle."""
from pathlib import Path
from unit_flags_contract import check_spec as base_check, make_case as base_case, validate_actual
CASE_SOURCES = [Path(__file__), Path(__file__).with_name('unit_flags_contract.py')]
TILES = ((64,64),(32,64),(64,80),(32,48),(32,80))


def check_spec(spec, cores=None):
    base_check(spec, cores)
    tile = (spec.get('tile_m'), spec.get('tile_n'))
    if tile not in TILES or tile != (spec['m'],spec['n']):
        raise ValueError('declared resource-safe tile must match actual helper dimensions')
    return 1


def make_case(spec):
    check_spec(spec)
    return base_case(spec)
