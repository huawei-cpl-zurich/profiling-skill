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



# Translation is confined to pinned upstream diagnostic text; numerical policy,
# metric keys/values and the original chained exception remain untouched.
_DIAGNOSTIC_ENGLISH = {
    '非计算类不支持的浮点 dtype': 'unsupported floating dtype for non-compute comparison',
    '非计算类实现输出必须是 Tensor，实际为': 'non-compute implementation output must be a Tensor; got',
    '生态开源精度标准': 'ecosystem open-source precision standard',
    '非计算类验证失败': 'non-compute verification failed',
    '验证失败': 'verification failed',
    '输出形状不一致': 'output shape mismatch',
    '输出 dtype 不一致': 'output dtype mismatch',
    'NaN 位置不匹配': 'NaN position mismatch',
    'Inf 位置不匹配': 'Inf position mismatch',
    'Inf 符号不匹配': 'Inf sign mismatch',
    '非计算类，要求二进制完全一致': 'non-compute; bitwise equality required',
    'bool 输出，要求严格相等': 'bool output; exact equality required',
    '量化类 fp->int，要求|diff|<=1': 'fp->int quantization; |diff|<=1 required',
    '整数计算类，要求严格相等': 'integer compute; exact equality required',
    '个量化误差超限位置': 'quantization-error violations',
    '个 bit 不一致位置': 'bit-mismatch positions',
    '个不一致位置': 'mismatch positions',
    '个 max_error_cap 违例位置': 'max_error_cap violations',
    '个 matched 未通过位置': 'unmatched positions',
    '个越限位置': 'limit violations',
    '逐元素上限': 'elementwise limit',
    '元素不一致': 'elements differ',
    '元素超限': 'elements exceed the limit',
    '相对误差': 'relative error',
    '小值域': 'small-value range',
    '正常域': 'normal-value range',
    '越限': 'violations',
    '常数': 'constant ',
    '允许': 'allowed',
    '位置': 'position',
    '前 ': 'first ',
    '，': ', ',
}


def english_diagnostic(message):
    """Translate known pinned comparison messages without losing numeric detail."""
    for original, translated in _DIAGNOSTIC_ENGLISH.items():
        message = message.replace(original, translated)
    return message


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
                raise AssertionError(f'verification failed, {path}: output type mismatch')
            if reference.dtype != result.dtype:
                raise AssertionError(f'verification failed, {path}: output dtype mismatch: '
                                     f'framework={reference.dtype}, impl={result.dtype}')
            if reference.shape != result.shape:
                raise AssertionError(f'verification failed, {path}: output shape mismatch: '
                                     f'framework={reference.shape}, impl={result.shape}')
            try:
                verifier.compare(reference, result, reference.dtype, input_spec, non_compute=non_compute)
            except verifier.AccuracyError as error:
                raise verifier.AccuracyError(f'{path}: {english_diagnostic(str(error))}', error.metrics) from error
            except AssertionError as error:
                raise AssertionError(f'{path}: {english_diagnostic(str(error))}') from error
        elif isinstance(reference, (tuple, list)):
            if not isinstance(result, (tuple, list)) or len(reference) != len(result):
                raise AssertionError(f'verification failed, {path}: output count mismatch')
            for index, (left, right) in enumerate(zip(reference, result)):
                visit(left, right, f'{path}[{index}]')
        elif isinstance(reference, dict):
            if not isinstance(result, dict) or reference.keys() != result.keys():
                raise AssertionError(f'verification failed, {path}: output fields mismatch')
            for key in reference:
                visit(reference[key], result[key], f'{path}.{key}')
        elif type(reference) is not type(result) or reference != result:
            raise AssertionError(f'verification failed, {path}: output mismatch: framework={reference!r}, impl={result!r}')

    visit(expected, actual, 'output')


def failure_details(error):
    """Retain upstream structured metrics alongside the English diagnostic."""
    return dict(error_type=type(error).__name__, error_msg=str(error),
                metrics=getattr(error, 'metrics', None))
