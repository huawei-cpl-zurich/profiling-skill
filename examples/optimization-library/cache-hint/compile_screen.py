"""Compile F12's native hint consumer; never launch or qualify performance.

A binary difference is evidence of a compiler difference only. A supported
runtime enable contract and device correctness remain separate requirements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess


DEVICE_SECTIONS = ('.aicore_binary', '__aicore_rel_binary')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def device_sections(data):
    """Hash nonempty device payloads from a little-endian ELF64 host library."""
    if len(data) < 64 or data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected a little-endian ELF64 library')
    offset = struct.unpack_from('<Q', data, 40)[0]
    width, count, strings = struct.unpack_from('<HHH', data, 58)
    if width != 64 or not count or strings >= count or offset + width * count > len(data):
        raise ValueError('invalid ELF section table')
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * width) for i in range(count)]
    names_header = headers[strings]
    start, size = names_header[4:6]
    if start + size > len(data):
        raise ValueError('invalid ELF section names')
    names = data[start:start + size]
    found = {}
    for header in headers:
        name_offset, _, _, _, start, size, *_ = header
        if name_offset >= len(names):
            raise ValueError('invalid ELF section name offset')
        end = names.find(b'\0', name_offset)
        if end < 0:
            raise ValueError('unterminated ELF section name')
        name = names[name_offset:end].decode('ascii')
        if name not in DEVICE_SECTIONS:
            continue
        if not size or start + size > len(data) or name in found:
            raise ValueError('invalid or duplicate device payload')
        found[name] = dict(bytes=size, sha256=sha256(data[start:start + size]))
    if not found:
        raise ValueError('library contains no nonempty device payload')
    return found


def compile_screen(sdk, output, *, compiler='bisheng', timeout=120, runner=subprocess.run):
    """Retain four bounded builds and their failures without using a device.

    Output must be new so failed versions cannot be silently overwritten.
    The macro-enabled builds are a diagnostic, not a supported launch recipe.
    """
    sdk, output = Path(sdk).resolve(), Path(output).resolve()
    executable = shutil.which(str(compiler))
    if executable is None:
        raise FileNotFoundError(compiler)
    compiler = Path(executable).resolve()
    if not (sdk / 'include/acl/acl.h').is_file():
        raise ValueError('SDK must provide include/acl/acl.h')
    operator = sdk / 'include/ascendc/basic_api/kernel_operator.h'
    if not operator.is_file():
        raise ValueError('SDK must provide the native AscendC kernel_operator.h')
    # CANN 9.1 public headers are symlinks into the matching asc implementation.
    api = operator.resolve().parent
    include = [sdk / 'include', operator.parent, operator.parent / 'interface',
               api, api.parent, api.parent.parent,
               sdk / 'include/ascendc/highlevel_api']
    source = Path(__file__).with_name('probe.cpp')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, family='F12', outcome='runtime_contract_unqualified',
                  source_sha256=sha256(source.read_bytes()), compiler_sha256=sha256(compiler.read_bytes()),
                  architecture='dav-c220', builds=[], comparisons={}, scientific_forwards=0,
                  cpu_oracles=0, performance_cases={f'F12-0{i}': 'unattempted' for i in range(1, 5)})
    (output / 'probe.cpp').write_bytes(source.read_bytes())
    common = [str(compiler), '-xcce', '-std=c++17', '-O2', '-fPIC', '-shared',
              '--cce-aicore-arch=dav-c220', *[f'-I{p}' for p in include]]
    # Capture the actual device predefines, not the host preprocessing path.
    macros = common[:1] + ['-xcce', '--cce-aicore-arch=dav-c220', '--cce-aicore-only',
                           '-E', '-dM', '-']
    try:
        run = runner(macros, input='', stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, timeout=timeout)
        (output / 'device-macros.log').write_text(run.stdout or '')
        result['device_preprocessing'] = dict(command=macros, exit_code=run.returncode,
            macros=sorted(set(line for line in (run.stdout or '').splitlines()
                if line.startswith(('#define __NPU_', '#define __CCE_AICORE__')))))
    except subprocess.TimeoutExpired:
        result['device_preprocessing'] = dict(command=macros, timed_out=True)
    for enabled in (False, True):
        mode = 'macro' if enabled else 'default'
        pair = []
        for hint in (0, 1):
            name = f'{mode}-{hint}'
            binary = output / f'{name}.so'
            command = [*common, f'-DF12_HINT={hint}', *(['-DL2_CACHE_HINT'] if enabled else []),
                       str(output / 'probe.cpp'), '-o', str(binary)]
            build = dict(name=name, command=command)
            try:
                run = runner(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=timeout)
                (output / f'{name}.log').write_text(run.stdout or '')
                build['exit_code'] = run.returncode
                if run.returncode == 0:
                    data = binary.read_bytes()
                    build.update(binary_sha256=sha256(data), device_sections=device_sections(data))
            except subprocess.TimeoutExpired as exc:
                build['timed_out'] = True
                log = exc.stdout or b''
                (output / f'{name}.log').write_text(log.decode(errors='replace') if isinstance(log, bytes) else log)
            except (OSError, ValueError) as exc:
                build['artifact_error'] = str(exc)
            result['builds'].append(build)
            pair.append(build)
        if all('device_sections' in build for build in pair):
            result['comparisons'][mode] = ('identical_device_artifacts'
                if pair[0]['device_sections'] == pair[1]['device_sections'] else 'different_device_artifacts')
        else:
            result['comparisons'][mode] = 'build_or_artifact_failed'
    (output / 'screen.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', default=os.environ.get('ASCEND_HOME_PATH'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--compiler', default='bisheng')
    args = parser.parse_args()
    if not args.sdk:
        parser.error('--sdk or ASCEND_HOME_PATH is required')
    report = compile_screen(args.sdk, args.output, compiler=args.compiler)
    print(json.dumps(report, indent=2))
    # Unqualified runtime is the expected screen outcome. Build failures remain
    # machine-visible failures; this command cannot certify a runtime gate.
    raise SystemExit(int(any(v == 'build_or_artifact_failed' for v in report['comparisons'].values())))
