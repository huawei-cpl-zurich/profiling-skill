"""Export fused Triton workloads without optimization history or answers."""
import argparse
import json
from pathlib import Path
import shutil


TASKS = {
    'task01': dict(family='gather', rows=4096, width=256, table=4194304, rounds=4),
    'task02': dict(family='gather', rows=1024, width=256, table=16384, rounds=32),
    'task03': dict(family='masked', rows=2048, width=1024, density=.125),
    'task04': dict(family='masked', rows=256, width=4096, density=.75),
    'task05': dict(family='matmul', m=1024, n=1024, k=256, rounds=4),
    'task06': dict(family='matmul', m=256, n=256, k=2048, rounds=16),
}
HEADER = 'import triton\nimport triton.language as tl\n\n'
GATHER = '''@triton.jit
def task_kernel(x, index, weight, coeff, output, WIDTH: tl.constexpr,
                ROUNDS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    address = tl.load(index + row * WIDTH + col, col < WIDTH, 0)
    value = tl.load(x + address, col < WIDTH, 0)
    w = tl.load(weight + row * WIDTH + col, col < WIDTH, 0)
    for step in range(ROUNDS):
        a = tl.load(coeff + step * 2)
        b = tl.load(coeff + step * 2 + 1)
        value = value + a * (2.0 / (1.0 + tl.exp(-(value * b))) - 1.0)
    mean = tl.sum(tl.where(col < WIDTH, value * w, 0), 0) / WIDTH
    energy = tl.sum(tl.where(col < WIDTH, value * value * w, 0), 0) / WIDTH
    tl.store(output + row * 2, mean)
    tl.store(output + row * 2 + 1, energy)

def launch(inputs, output):
    return task_kernel[(output.shape[0],)](
        inputs['x'], inputs['index'], inputs['weight'], inputs['coeff'], output,
        inputs['index'].shape[1], inputs['coeff'].shape[0], 256, num_warps=4)
'''
MASKED = '''@triton.jit
def task_kernel(x, selected, gain, output, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    keep = tl.load(selected + row * WIDTH + col, valid, 0).to(tl.int1)
    value = tl.load(x + row * WIDTH + col, valid, 0)
    scale = tl.load(gain + col, valid, 0)
    score = value * (2.0 / (1.0 + tl.exp(-value * scale)) - 1.0)
    score = tl.where(valid & keep, score, -float('inf'))
    maximum = tl.max(score, 0)
    maximum = tl.where(maximum == -float('inf'), 0.0, maximum)
    probability = tl.exp(score - maximum)
    denominator = tl.sum(probability, 0)
    result = probability / tl.maximum(denominator, 1.0e-20)
    tl.store(output + row * WIDTH + col, result, valid)

def launch(inputs, output):
    return task_kernel[(output.shape[0],)](
        inputs['x'], inputs['selected'], inputs['gain'], output,
        output.shape[1], triton.next_power_of_2(output.shape[1]), num_warps=4)
'''
MATMUL = '''@triton.jit
def task_kernel(lhs, rhs, bias, residual, coeff, output, M: tl.constexpr,
                N: tl.constexpr, K: tl.constexpr, ROUNDS: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        kk = base * BK + inner
        a = tl.load(lhs + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(rhs + kk[:, None] * N + cols[None, :],
                    (kk[:, None] < K) & (cols[None, :] < N), 0)
        acc = tl.dot(a, b, acc)
    value = acc + tl.load(bias + cols, cols < N, 0)[None, :]
    for step in range(ROUNDS):
        a = tl.load(coeff + step * 2)
        b = tl.load(coeff + step * 2 + 1)
        value = value + a * (2.0 / (1.0 + tl.exp(-value * b)) - 1.0)
    skip = tl.load(residual + rows[:, None] * N + cols[None, :],
                   (rows[:, None] < M) & (cols[None, :] < N), 0)
    value = skip + value / (1.0 + tl.exp(-value))
    tl.store(output + rows[:, None] * N + cols[None, :], value,
             (rows[:, None] < M) & (cols[None, :] < N))

def launch(inputs, output):
    return task_kernel[(triton.cdiv(output.shape[0], 32), triton.cdiv(output.shape[1], 64))](
        inputs['lhs'], inputs['rhs'], inputs['bias'], inputs['residual'], inputs['coeff'],
        output, output.shape[0], output.shape[1], inputs['lhs'].shape[1],
        inputs['coeff'].shape[0], 32, 64, 32, num_warps=4)
'''


def source(task):
    return HEADER + {'gather': GATHER, 'masked': MASKED, 'matmul': MATMUL}[TASKS[task]['family']]


def qualification_variants(task):
    """Private scoped alternatives; never copied into exported public files."""
    baseline = source(task)
    family = TASKS[task]['family']
    if family == 'gather':
        return (baseline.replace('num_warps=4', 'num_warps=8'),
                baseline.replace('range(ROUNDS)', 'tl.static_range(ROUNDS)'))
    if family == 'masked':
        return (baseline.replace('value = tl.load(x + row * WIDTH + col, valid, 0)',
                                 'value = tl.load(x + row * WIDTH + col, valid & keep, 0)'),
                baseline.replace('num_warps=4', 'num_warps=8'))
    return (baseline.replace('output.shape[0], 32)', 'output.shape[0], 16)')
                    .replace('32, 64, 32, num_warps', '16, 64, 32, num_warps'),
            baseline.replace('32, 64, 32, num_warps', '32, 64, 64, num_warps'))


def export(task, output):
    spec = dict(TASKS[task], id=task, language='triton', selector='task_kernel',
                single_kernel=True, entrypoint='launch(inputs, output)',
                tolerances=dict(rtol=.003, atol=.003),
                public_files=['baseline.py', 'candidate.py', 'task.json', 'TASK.md'])
    spec.setdefault('n', spec.get('rows', 0))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('baseline.py', 'candidate.py'):
        (output / name).write_text(source(task))
    shutil.copyfile(Path(__file__).with_name('natural_task_support') / 'runner.py', output / 'runner.py')
    shutil.copyfile(Path(__file__).with_name('blind_task_support') / 'native_host.asc', output / 'native_host.asc')
    (output / 'task.json').write_text(json.dumps(spec, indent=2) + '\n')
    (output / 'TASK.md').write_text(
        'Optimize candidate.py using launch(inputs, output), returning its compiled task_kernel.\n'
        'Preserve one fused kernel, every input-dependent stage, output shape and tolerances.\n'
        'No host output computation or auxiliary computation kernels. All tensors contiguous.\n'
        'Gather: float32 table/weights/coefficients, int32 arbitrary valid indices; weighted\n'
        'mean and energy after the coefficient-driven nonlinear recurrence, shape (rows,2).\n'
        'Masked: float32 x/gain and bool selected; softmax of the nonlinear score,\n'
        'zero at excluded entries and on fully excluded rows. Masks may be arbitrary.\n'
        'Matmul: float16 lhs/rhs, float32 bias/residual/coefficients and output;\n'
        'float32 accumulation, coefficient-driven recurrence, SiLU and residual addition.\n'
        'Support the shapes/round count in task.json with arbitrary finite normal inputs.\n'
        'Keep baseline.py, task.json and trusted support files unchanged.\n'
        'Use experiment-tool help for correctness, timing, optional pipes and compiler evidence.\n')
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export'])
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.task, args.output)))
