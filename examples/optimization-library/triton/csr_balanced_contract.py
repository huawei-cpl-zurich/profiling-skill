"""F05 CPU contract: beta=0 CSR SpMV, repeated columns and empty rows."""
from pathlib import Path
import torch

ATOL = RTOL = 2e-3
FAST_MAX_ROW = 1024


def dimensions(spec):
    names = ('rows', 'cols', 'nnz', 'max_row', 'seed')
    values = tuple(spec.get(k) for k in names)
    if any(type(v) is not int for v in values):
        raise ValueError('integer rows, cols, nnz, max_row and seed required')
    rows, cols, nnz, maximum, seed = values
    if not (1 <= rows <= 65536 and 1 <= cols <= 65536 and 0 <= nnz <= rows * maximum
            and 0 <= maximum <= 2048 and seed >= 0):
        raise ValueError('CSR dimensions outside declared envelope')
    if spec.get('distribution') not in ('uniform', 'long-tail', 'long-tail-with-empty-rows', 'empty', 'long-row-guard'):
        raise ValueError('unsupported distribution')
    return values


def row_lengths(spec):
    rows, _, nnz, maximum, seed = dimensions(spec)
    if spec['distribution'] == 'uniform':
        if nnz % rows or nnz // rows != maximum:
            raise ValueError('uniform rows must have the declared exact length')
        return torch.full((rows,), maximum, dtype=torch.int64)
    if spec['distribution'] == 'empty':
        if nnz:
            raise ValueError('empty distribution requires nnz=0')
        return torch.zeros(rows, dtype=torch.int64)
    # Bounded long tail with exact nnz. Held-out input explicitly reserves empty
    # rows; remaining rows start short and receive whole bounded long rows.
    empty = rows // 8 if spec['distribution'] == 'long-tail-with-empty-rows' else 0
    base = 0 if spec['distribution'] == 'long-row-guard' else min(8, nnz // rows)
    lengths = torch.full((rows,), base, dtype=torch.int64)
    lengths[:empty] = 0
    remaining = nnz - int(lengths.sum())
    for row in range(empty, rows):
        add = min(maximum - int(lengths[row]), remaining)
        lengths[row] += add
        remaining -= add
    if remaining:
        raise ValueError('requested nnz cannot fit reserved empty rows')
    permutation = torch.randperm(rows, generator=torch.Generator().manual_seed(seed))
    return lengths[permutation]


def reference(indptr, columns, values, x):
    products = values.double() * x[columns.long()].double()
    row = torch.repeat_interleave(torch.arange(indptr.numel() - 1), indptr.diff().long())
    return torch.zeros(indptr.numel() - 1, dtype=torch.float64).index_add_(0, row, products)


def validate_inputs(fixture, spec):
    rows, cols, nnz, maximum, _ = dimensions(spec)
    ptr, col, val, x = (fixture[k] for k in ('indptr', 'columns', 'values', 'x'))
    for tensor, dtype, shape in ((ptr, torch.int32, (rows + 1,)), (col, torch.int32, (nnz,)),
                                 (val, torch.float32, (nnz,)), (x, torch.float32, (cols,))):
        if tensor.device.type != 'cpu' or tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError('contiguous CPU CSR int32 indices/FP32 values with declared shape required')
    if ptr[0] != 0 or ptr[-1] != nnz or bool((ptr.diff() < 0).any()) or int(ptr.diff().max()) > maximum:
        raise ValueError('invalid CSR offsets or max-row certificate')
    if nnz and (bool((col < 0).any()) or bool((col >= cols).any())):
        raise ValueError('column out of range')
    if not bool(torch.isfinite(val).all() and torch.isfinite(x).all()) or bool((val.abs() > .25).any() or (x.abs() > .25).any()):
        raise ValueError('finite values/x in [-0.25,0.25] required')
    return int(ptr.diff().max())


def make_case(spec):
    rows, cols, nnz, _, seed = dimensions(spec)
    generator = torch.Generator().manual_seed(seed)
    lengths = row_lengths(spec)
    ptr = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0))).int()
    columns = torch.randint(cols, (nnz,), generator=generator, dtype=torch.int32)
    # Adjacent repeated column indices retain repeated independent contributions.
    columns[1::2] = columns[::2][:nnz // 2]
    values = torch.rand(nnz, generator=generator) * .5 - .25
    x = torch.rand(cols, generator=generator) * .5 - .25
    result = dict(indptr=ptr, columns=columns, values=values, x=x,
                  initial_y=torch.full((rows,), float('nan')))
    validate_inputs(result, spec)
    result['expected'] = reference(ptr, columns, values, x)
    return result


def diagnostics(actual, fixture):
    expected = fixture['expected']
    if actual.device.type != 'cpu' or actual.dtype != torch.float32 or actual.shape != expected.shape:
        return dict(passed=False, reason='expected CPU FP32 output of rows length')
    if not bool(torch.isfinite(actual).all()):
        return dict(passed=False, reason='nonfinite output; beta=0 must overwrite NaN initial y')
    error = (actual.double() - expected).abs()
    allowed = ATOL + RTOL * expected.abs()
    return dict(passed=bool((error <= allowed).all()), atol=ATOL, rtol=RTOL,
                max_absolute_error=float(error.max()), max_tolerance_fraction=float((error / allowed).max()),
                empty_rows_zero=bool((actual[fixture['indptr'].diff() == 0] == 0).all()))


CASE_SOURCES = [Path(__file__)]
