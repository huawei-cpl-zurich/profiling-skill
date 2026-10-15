"""Export standalone starting kernels without optimization history or answers."""
import argparse
import json
from pathlib import Path
import shutil


TASKS = {
    'task01': dict(n=256, work=64, language='triton'),
    'task02': dict(n=1048576, language='triton'),
    'task03': dict(n=32, language='triton'),
    'task04': dict(n=1024, repetitions=128, language='triton'),
    'task05': dict(n=4096, language='triton'),
    'task06': dict(n=4096, rounds=32, language='ascendc'),
}

HEADER = 'import triton\nimport triton.language as tl\n\n'
KERNELS = {
    'task01': '''@triton.jit
def task_kernel(x, work_ptr, output, n: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + offset, offset < n, 0)
    work = tl.load(work_ptr)
    state = value * 0.0
    for index in range(256):
        state += tl.where(index < work, value * 0.000001, value * 0.0)
    tl.store(output + offset, (value + state) * 1.5 + 2.0, offset < n)

def launch(inputs, output):
    return task_kernel[(triton.cdiv(output.numel(), 128),)](
        inputs['x'], inputs['work'], output, output.numel(), 128)
''',
    'task02': '''@triton.jit
def task_kernel(x, index, output, n: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    address = tl.load(index + offset, offset < n, 0)
    value = tl.load(x + address, offset < n, 0)
    tl.store(output + offset, value, offset < n)

def launch(inputs, output):
    return task_kernel[(triton.cdiv(output.numel(), 256),)](
        inputs['x'], inputs['index'], output, output.numel(), 256)
''',
    'task03': '''@triton.jit
def task_kernel(lhs, rhs, output, size: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, size)
    left = tl.load(lhs + row * size + cols)
    for column in range(size):
        right = tl.load(rhs + cols * size + column)
        tl.store(output + row * size + column, tl.sum(left * right))

def launch(inputs, output):
    return task_kernel[(output.shape[0],)](
        inputs['lhs'], inputs['rhs'], output, output.shape[0])
''',
    'task04': '''@triton.jit
def task_kernel(x, selected, repeat_ptr, output, n: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    repetitions = tl.load(repeat_ptr)
    for _ in range(repetitions):
        chosen = tl.load(selected + offset, offset < n, 0, volatile=True).to(tl.int1)
        value = tl.load(x + offset, offset < n, 0.0, volatile=True)
        result = tl.where(chosen, value * 1.5 + 2.0, 0.0)
        tl.store(output + offset, result, offset < n)

def launch(inputs, output):
    return task_kernel[(triton.cdiv(output.numel(), 128),)](
        inputs['x'], inputs['selected'], inputs['repetitions'], output, output.numel(), 128)
''',
    'task05': '''@triton.jit
def task_kernel(x, output, n: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + offset, offset < n, 0)
    tl.store(output + offset, value * 1.5 + 2.0, offset < n)

def launch(inputs, output):
    return task_kernel[(triton.cdiv(output.numel(), 128),)](
        inputs['x'], output, output.numel(), 128, num_warps=4, num_stages=1)
''',
}

NATIVE = '''#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t N = 4096;
extern "C" __global__ __aicore__ void task_kernel(GM_ADDR input, GM_ADDR output) {
    TPipe p;
    TBuf<TPosition::VECCALC> storage;
    p.InitBuffer(storage, 3 * N * sizeof(float));
    auto a = storage.Get<float>();
    auto b = a[N];
    auto w = a[2 * N];
    GlobalTensor<float> x, y;
    x.SetGlobalBuffer((__gm__ float*)input, 2 * N);
    y.SetGlobalBuffer((__gm__ float*)output, 2 * N);
    Duplicate(w, 0.0f, N);
    PipeBarrier<PIPE_V>();
    DataCopy(a, x, N);
    SetFlag<HardEvent::MTE2_V>(0);
    DataCopy(b, x[N], N);
    SetFlag<HardEvent::MTE2_V>(1);
    WaitFlag<HardEvent::MTE2_V>(0);
    for (int round = 0; round < 32; ++round) {
        Adds(w, w, 1.0f, N);
        PipeBarrier<PIPE_V>();
    }
    WaitFlag<HardEvent::MTE2_V>(1);
    Add(a, a, b, N);
    PipeBarrier<PIPE_V>();
    Add(a, a, w, N);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(y, a, N);
    SetFlag<HardEvent::V_MTE3>(1);
    WaitFlag<HardEvent::V_MTE3>(1);
    DataCopy(y[N], w, N);
    PipeBarrier<PIPE_ALL>();
}
'''


def export(task, output):
    spec = dict(TASKS[task], id=task, selector='task_kernel', single_kernel=True)
    spec['tolerances'] = (dict(rtol=.02, atol=.02) if task == 'task03' else
                          dict(rtol=0., atol=0.) if task in ('task02', 'task06') else
                          dict(rtol=1e-6, atol=1e-6))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    extension = 'asc' if spec['language'] == 'ascendc' else 'py'
    spec['public_files'] = ['baseline.' + extension, 'candidate.' + extension, 'task.json', 'TASK.md']
    spec['entrypoint'] = 'task_kernel(input, output)' if extension == 'asc' else 'launch(inputs, output)'
    source = NATIVE if extension == 'asc' else HEADER + KERNELS[task]
    for name in ('baseline', 'candidate'):
        (output / f'{name}.{extension}').write_text(source)
    templates = Path(__file__).with_name('blind_task_support')
    for name in ('runner.py', 'native_host.asc'):
        shutil.copyfile(templates / name, output / name)
    (output / 'task.json').write_text(json.dumps(spec, indent=2) + '\n')
    (output / 'TASK.md').write_text(
        'Optimize candidate without changing runner.py, native_host.asc, task.json or baseline.\n'
        'Preserve all input-dependent outputs, float32 arithmetic accuracy and one fused kernel.\n'
        'For task01 support runtime work in [0,256]. For task02 support arbitrary valid runtime indices.\n'
        'For task04 preserve the runtime repetition loop and volatile input accesses;\n'
        'repeat count can be 1,32,128. Other tasks use shapes/constants in task.json.\n'
        'Public API: Python launch(inputs, output) returns compiled kernel; native task_kernel(input,output).\n'
        'Use experiment-tool help for checking, timing, pipe metrics and compiler inspection.\n'
        'The controller supplies independent seeded correctness checks through that tool.\n'
        'Every output must match task.json tolerances (indexed copy and native outputs exact).\n')
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export'])
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.task, args.output)))


if __name__ == '__main__':
    main()
