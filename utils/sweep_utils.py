import ast
import itertools
import json
import re

from utils.get_optimizers import get_supported_optimizer_params

def parse_sweep_params(sweep_params):
    if not sweep_params:
        return {}
    if isinstance(sweep_params, dict):
        return sweep_params
    try:
        parsed = json.loads(sweep_params)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(sweep_params)
    if not isinstance(parsed, dict):
        raise ValueError('--sweep_params must parse to a dictionary.')
    return parsed


def validate_sweep_params(optimizer_name, param_grid):
    allowed = set(get_supported_optimizer_params(optimizer_name))
    unknown = sorted(set(param_grid.keys()) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported sweep parameters for optimizer '{optimizer_name}': {unknown}. Allowed parameters: {sorted(allowed)}"
        )
    for key, values in param_grid.items():
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"Sweep parameter '{key}' must map to a non-empty list/tuple of candidate values.")


def expand_sweep_grid(param_grid):
    if not param_grid:
        return []
    keys = list(param_grid.keys())
    values_product = itertools.product(*(param_grid[key] for key in keys))
    return [dict(zip(keys, values)) for values in values_product]


def _normalize_value(value):
    if isinstance(value, list):
        return tuple(value)
    return value


def apply_sweep_combo(args, optimizer_name, combo):
    args.optimizer = optimizer_name.lower()
    for key, value in combo.items():
        setattr(args, key, _normalize_value(value))
    return args


def _sanitize_fragment(text):
    return re.sub(r'[^a-zA-Z0-9_.=-]+', '-', text)


def format_sweep_tag(combo_index, combo):
    fragments = [f'run{combo_index:03d}']
    for key, value in combo.items():
        fragments.append(f'{key}={value}')
    return _sanitize_fragment('_'.join(str(fragment) for fragment in fragments))