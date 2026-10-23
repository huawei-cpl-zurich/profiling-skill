"""Compile-only native pilot; environment/tool paths come from the chosen runtime."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from optimization_native_bridge import build_shared, digest

PIN = '40aced927fdaaa0f3eaa9c6f5c5d859e06392f55'


def compile_pair(catlass, output, *, source=None, variants=None):
    catlass, output = Path(catlass).resolve(), Path(output).resolve()
    manifest = json.loads((catlass/'manifest.json').read_text())
    for entry in manifest['files']:
        if digest(catlass/entry['file']) != entry['sha256']:
            raise ValueError(f'filtered source checksum mismatch: {entry["file"]}')
    source = Path(source or Path(__file__).parent/'tile_matmul.cpp').resolve()
    variants = variants or {'before': ['-DAFTER=0'], 'after': ['-DAFTER=1']}
    compiler = Path(shutil.which('bisheng')).resolve()
    candidates = [Path(os.environ[k]).resolve() for k in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME')
                  if os.environ.get(k)] + list(compiler.parents)
    cann = next((p for p in candidates if (p/'include/acl/acl.h').is_file()), None)
    if cann is None:
        raise RuntimeError('runtime has no discoverable CANN root')
    operators = list((cann/'include').rglob('kernel_operator.h'))
    if len(operators) != 1:
        raise RuntimeError('runtime must expose exactly one AscendC kernel_operator.h')
    api = operators[0].resolve().parent
    includes = [catlass/'include',source.parent,cann/'include',api.parent,api]
    includes.extend(p for p in (api/'interface',api/'impl') if p.is_dir())
    for arm, defines in variants.items():
        if not arm or Path(arm).name != arm or not isinstance(defines, list):
            raise ValueError('variants map simple artifact names to compiler argv lists')
        binary = output/f'{arm}.so'
        command = [str(compiler),'-xcce','-std=c++17','-O2','-fPIC','-shared',
            '--cce-aicore-arch=dav-c220','-DCATLASS_ARCH=2201','-D_GLIBCXX_USE_CXX11_ABI=1',
            *defines,*[f'-I{p}' for p in includes],str(source),'-o',str(binary)]
        try:
            result = build_shared(command, source_paths=[source,
                source.parent/'catlass_bridge.h',catlass/'manifest.json',
                *[catlass/entry['file'] for entry in manifest['files']]],
                binary=binary, output=output/f'build-{arm}', source_revision=PIN)
        except Exception:
            log = output/f'build-{arm}/build.log'
            if log.exists():
                print(log.read_text()[-12000:], flush=True)
            raise
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catlass', required=True, help='filtered pinned dependency closure')
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    compile_pair(options.catlass, options.output)
