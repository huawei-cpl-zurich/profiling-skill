"""Trusted benchmark adapter to the pinned, unmodified cann bot verifier."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

VENDOR = Path(__file__).resolve().parents[1] / 'third_party/cannbot_verifier'


def verifier_identity():
    """Bind policy and exact dependency bytes into reports and private caches."""
    provenance = json.loads((VENDOR / 'provenance.json').read_text())
    for name, digest in provenance['files'].items():
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != digest:
            raise ValueError('cann bot verifier dependency changed: ' + name)
    provenance['adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return provenance


@lru_cache(maxsize=1)
def _verifier():
    verifier_identity()
    # Upstream uses sibling absolute imports. Isolate these names from other
    # tools, while keeping the original files and their numerical policy intact.
    names = ('_log_utils', '_common_utils', 'eco_limits', 'npu_preflight',
             '_baseline_integrity', 'hidden_state_probe', '_cannbot_verify')
    previous = {name: sys.modules.get(name) for name in names}
    import_path = sys.path[:]
    try:
        for name in names:
            file = 'verify.py' if name == '_cannbot_verify' else name + '.py'
            spec = importlib.util.spec_from_file_location(name, VENDOR / file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        # verify.py inserts its own folder; do not change callers' import order.
        sys.path[:] = import_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def snapshot(value):
    """Freeze reference values before a candidate can reuse or mutate storage."""
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item) for item in value)
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    return value


def verify_outputs(expected, actual, inputs=(), *, non_compute=False):
    """Check output structure, then delegate every tensor to upstream compare().

    Shape is checked before the upstream comparator flattens tensors. No extra
    Model call is made. Existing private validators can add semantic checks.
    """
    verifier = _verifier()
    def leaves(value):
        if isinstance(value, dict):
            return [leaf for item in value.values() for leaf in leaves(item)]
        if isinstance(value, (tuple, list)):
            return [leaf for item in value for leaf in leaves(item)]
        return [value]

    input_type, input_dtype = verifier._infer_input_type(leaves(inputs))
    input_spec = verifier.InputSpec(input_type, input_dtype)
    torch = sys.modules.get('torch')

    def visit(reference, result, path):
        if torch is not None and isinstance(reference, torch.Tensor):
            if not isinstance(result, torch.Tensor):
                raise AssertionError(f'验证失败，{path} 输出类型不一致')
            if reference.shape != result.shape:
                raise AssertionError(f'验证失败，{path} 输出形状不一致: '
                                     f'framework={reference.shape}, impl={result.shape}')
            try:
                verifier.compare(reference, result, reference.dtype, input_spec, non_compute=non_compute)
            except verifier.AccuracyError as error:
                raise verifier.AccuracyError(f'{path}: {error}', error.metrics) from error
            except AssertionError as error:
                raise AssertionError(f'{path}: {error}') from error
        elif isinstance(reference, (tuple, list)):
            if not isinstance(result, (tuple, list)) or len(reference) != len(result):
                raise AssertionError(f'验证失败，{path} 输出数量不一致')
            for index, (left, right) in enumerate(zip(reference, result)):
                visit(left, right, f'{path}[{index}]')
        elif isinstance(reference, dict):
            if not isinstance(result, dict) or reference.keys() != result.keys():
                raise AssertionError(f'验证失败，{path} 输出字段不一致')
            for key in reference:
                visit(reference[key], result[key], f'{path}.{key}')
        elif type(reference) is not type(result) or reference != result:
            raise AssertionError(f'验证失败，{path} 输出不一致: framework={reference!r}, impl={result!r}')

    visit(expected, actual, 'output')


def failure_details(error):
    """Retain upstream structured metrics alongside its original Chinese error."""
    return dict(error_type=type(error).__name__, error_msg=str(error),
                metrics=getattr(error, 'metrics', None))
