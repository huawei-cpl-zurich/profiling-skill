"""Compile a source-owned, A2/A3-only certified TopP without launching it."""
import argparse
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from optimization_native_bridge import build_shared


def compile_kernel(output, arm):
    if arm not in ('before','after'): raise ValueError('known arm required')
    compiler=Path(shutil.which('bisheng')).resolve()
    roots=[Path(os.environ[key]).resolve() for key in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME') if os.environ.get(key)] + list(compiler.parents)
    cann=next((path for path in roots if (path/'include/acl/acl.h').is_file()),None)
    if cann is None: raise RuntimeError('CANN root unavailable in selected runtime')
    operators=list((cann/'include').rglob('kernel_operator.h'))
    if len(operators)!=1: raise RuntimeError('one unambiguous installed kernel_operator.h required')
    api=operators[0].resolve().parent
    include=[cann/'include',api.parent,api,api/'interface',api/'impl']
    source=Path(__file__).with_suffix('.cpp').with_name('top_p_prefix.cpp')
    output=Path(output).resolve()
    command=[str(compiler),'-xcce','-std=c++17','-O2','-fPIC','-shared',f'-DAFTER={int(arm=="after")}','--cce-aicore-arch=dav-c220',
             *[f'-I{path}' for path in include],str(source),'-o',str(output/f'{arm}.so')]
    try:
        return build_shared(command,source_paths=[source,Path(__file__)],binary=output/f'{arm}.so',
                            output=output/f'build-{arm}',source_revision='F04-v2-independent-heap-cache-publication')
    except Exception:
        log=output/f'build-{arm}/build.log'
        if log.exists(): print(log.read_text()[-15000:],flush=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--arm',choices=('before','after'),action='append')
    options=parser.parse_args()
    for arm in options.arm or ('before','after'): compile_kernel(options.output,arm)
