"""Standalone native before tasks; no paired after implementations or profiles."""
import argparse
import json
from pathlib import Path
import shutil

TASKS = {'intercore': 1, 'intra': 2, 'phase': 3}
HEADER = '''#include "kernel_operator.h"
using namespace AscendC;
'''
SETUP = '''
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 3*n*sizeof(float));
    auto a=storage.Get<float>(); auto b=a[n]; auto w=a[2*n];
    auto load=pipe.AllocEventID<HardEvent::MTE2_V>();
    auto store=pipe.AllocEventID<HardEvent::V_MTE3>();
    auto reuse=pipe.AllocEventID<HardEvent::V_MTE2>();
    GlobalTensor<float> x,y;
    x.SetGlobalBuffer((__gm__ float*)input,2*n);
    y.SetGlobalBuffer((__gm__ float*)output,4*n);
    Duplicate(w,0.0f,n); PipeBarrier<PIPE_V>();
'''
WORK = '''
    for(int r=0;r<rounds;++r) { Adds(w,w,1.0f,n); PipeBarrier<PIPE_V>(); }
'''
FINISH = '''
    pipe.ReleaseEventID<HardEvent::MTE2_V>(load);
    pipe.ReleaseEventID<HardEvent::V_MTE3>(store);
    pipe.ReleaseEventID<HardEvent::V_MTE2>(reuse);
    PipeBarrier<PIPE_ALL>();
}
'''


def source(task):
    inter = task == 'intercore'
    signature = ('extern "C" __global__ __aicore__ ' + ('__schedmode__(1) ' if inter else '') +
                 'void task_kernel(GM_ADDR input, GM_ADDR output, int n, int rounds, '
                 'int repetitions, int imbalanced) {\n')
    if inter:
        body = '''
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
''' + SETUP + '''
    int block=GetBlockIdx();
    DataCopy(a,x[block*n],n);
    SetFlag<HardEvent::MTE2_V>(load); WaitFlag<HardEvent::MTE2_V>(load);
    if(imbalanced && block==0) for(int r=0;r<32;++r) {
        Adds(a,a,1.0f,n); PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(store); WaitFlag<HardEvent::V_MTE3>(store);
    DataCopy(y[block*2*n],a,n);
    CrossCoreSetFlag<0,PIPE_MTE3>(8);
    CrossCoreWaitFlag(8);
''' + WORK + '''
    DataCopy(b,y[(1-block)*2*n],n);
    SetFlag<HardEvent::MTE2_V>(load); WaitFlag<HardEvent::MTE2_V>(load);
    Add(w,w,b,n);
    SetFlag<HardEvent::V_MTE3>(store); WaitFlag<HardEvent::V_MTE3>(store);
    DataCopy(y[block*2*n+n],w,n);
'''
    elif task == 'intra':
        body = SETUP + '''
    DataCopy(a,x,n); DataCopy(b,x[n],n);
    SetFlag<HardEvent::MTE2_V>(load); WaitFlag<HardEvent::MTE2_V>(load);
''' + WORK + '''
    Add(a,a,b,n); PipeBarrier<PIPE_V>(); Add(a,a,w,n);
    SetFlag<HardEvent::V_MTE3>(store); WaitFlag<HardEvent::V_MTE3>(store);
    DataCopy(y,a,n); DataCopy(y[n],w,n);
'''
    else:
        body = SETUP + '''
    SetFlag<HardEvent::V_MTE2>(reuse);
    for(int rep=0;rep<repetitions;++rep) {
        WaitFlag<HardEvent::V_MTE2>(reuse);
        DataCopy(a,x,n);
        SetFlag<HardEvent::MTE2_V>(load); WaitFlag<HardEvent::MTE2_V>(load);
''' + WORK + '''
        SetFlag<HardEvent::V_MTE2>(reuse);
    }
    WaitFlag<HardEvent::V_MTE2>(reuse);
    SetFlag<HardEvent::V_MTE3>(store); WaitFlag<HardEvent::V_MTE3>(store);
    DataCopy(y,a,n); DataCopy(y[n],w,n);
'''
    if not inter:
        body = body.replace('output,4*n)', 'output,2*n)')
    return HEADER + signature + body + FINISH


def export(task, output):
    kind = TASKS[task]
    variants = dict(default=dict(n=4096, rounds=32, repetitions=1, imbalanced=0))
    if task == 'intercore':
        variants['imbalanced'] = dict(variants['default'], imbalanced=1)
        variants['little-work'] = dict(variants['default'], rounds=1)
    if task == 'phase':
        variants = dict(tiny=dict(n=1024, rounds=1, repetitions=1, imbalanced=0),
                        long=dict(n=8192, rounds=32, repetitions=16, imbalanced=0))
    spec = dict(id='native-' + task, language='ascendc', selector='task_kernel',
                native_wait_kind=kind, variants=variants, default_variant=next(iter(variants)),
                blocks=2 if kind == 1 else 1, single_kernel=True,
                public_files=['baseline.asc', 'candidate.asc', 'task.json', 'TASK.md'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('baseline', 'candidate'):
        (output / (name + '.asc')).write_text(source(task))
    support = Path(__file__).with_name('blind_task_support')
    shutil.copyfile(support / 'native_wait_runner.py', output / 'runner.py')
    shutil.copyfile(support / 'native_wait_host.asc', output / 'native_host.asc')
    shutil.copyfile(Path(__file__).with_name('a3_native_build.py'), output / 'a3_native_build.py')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'scripts/profile_device.py', output / 'profile_device.py')
    (output / 'task.json').write_text(json.dumps(spec, indent=2) + '\n')
    (output / 'TASK.md').write_text(
        'Optimize candidate.asc while preserving task_kernel and all task.json variants.\n'
        'The trusted host checks both complete outputs against seeded CPU references.\n'
        'Preserve one fused launch, input dependence, repetitions and useful work; do not\n'
        'replace computation with host work, constants or auxiliary kernels.\n'
        'Do not edit baseline, runner, host or contract. Use experiment-tool help.\n')
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.task, args.output)))
