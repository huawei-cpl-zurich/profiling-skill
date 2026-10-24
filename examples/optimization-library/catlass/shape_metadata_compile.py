"""Device-less metadata screen. No fixture generation or NPU API is imported."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from compile_tile import compile_pair


def select_metadata(request_shape_info=False):
    """Call before any generic fixture cache lookup for this restricted pair."""
    if type(request_shape_info) is not bool:
        raise ValueError('request_shape_info must be a boolean')
    if request_shape_info:
        raise ValueError('ShapeInfo consumers require a separate SDK-default implementation')
    return {'before': [], 'after': ['-DW006_ZERO_SHAPE_INFO=1']}


def device_sections(path):
    """Read executable/data and allocation notes from a native little-endian ELF64.

    These sections diagnose changes but cannot prove equivalence: raw relocation
    symbol indices must be resolved through tables outside this selection. A no-op
    additionally requires full extracted device ELF identity.
    Stack records are raw compiler records, never certified physical peak usage.
    """
    blob = Path(path).read_bytes()
    if blob[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected little-endian ELF64 device object')
    offset = struct.unpack_from('<Q', blob, 40)[0]
    width, count, names_index = struct.unpack_from('<HHH', blob, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', blob, offset+i*width) for i in range(count)]
    names = headers[names_index]
    table = blob[names[4]:names[4]+names[5]]
    result = {}
    for h in headers:
        name = table[h[0]:].split(b'\0', 1)[0].decode()
        if h[2] & 2 or name.startswith(('.ascend.', '__CCE_', '.rela.')):
            data = blob[h[4]:h[4]+h[5]] if h[1] != 8 else b''
            result[name] = dict(type=h[1], flags=h[2], size=h[5], align=h[8],
                                sha256=hashlib.sha256(data).hexdigest())
            if h[5] <= 32:
                result[name]['raw_hex'] = data.hex()
    if '.text' not in result or not result['.text']['size']:
        raise ValueError('no executable device code')
    if '.ascend.stack.size.record' not in result:
        raise ValueError('missing compiler stack allocation record')
    return result


def classify(default, zero, before, after, *, device_object_sha256):
    if type(default) is not int or default < 0 or zero != 0:
        raise ValueError('actual SDK dimension evidence required')
    if not before or not after:
        raise ValueError('device code/resource evidence required')
    if (set(device_object_sha256) != {'before', 'after'} or any(
            not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
            for value in device_object_sha256.values())):
        raise ValueError('full extracted device ELF SHA256 identities required')
    same_elf = device_object_sha256['before'] == device_object_sha256['after']
    same_sections = before == after
    if same_elf and not same_sections:
        raise ValueError('section evidence contradicts full device ELF identity')
    # Relocation/name/metadata differences alone establish neither equivalence
    # nor a physical change. Stop admission until those targets are resolved.
    physical_change = any(before.get(name) != after.get(name)
                          for name in ('.text', '.data', '.ascend.stack.size.record'))
    outcome = ('compile-noop' if same_elf else
               'physical-change-requires-runtime' if physical_change and default != 0 else
               'inconclusive-device-equivalence')
    return dict(outcome=outcome, sdk_default=default, override=zero,
                device_sections_equal=same_sections, device_elf_equal=same_elf,
                device_object_sha256=device_object_sha256,
                scientific_oracles=0, npu_api_forwards=0, profiles=0,
                cases={f'W006-L03-{i:02}': 'unattempted' for i in range(1, 5)},
                latency_claim=None, physical_stack_peak_claim=None)


def compile_screen(catlass, output, *, request_shape_info=False):
    variants = select_metadata(request_shape_info)  # precedes all cache/build access
    catlass = Path(catlass).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('shape_metadata.cpp')
    evidence, dimensions, objects = {}, {}, {}
    entry = Path.cwd()
    try:
        for arm, flags in variants.items():
            destination = output/arm
            destination.mkdir()
            os.chdir(destination)
            compile_pair(catlass, destination, source=source,
                         variants={arm: flags+['-save-temps=obj']})
            cube = list(destination.glob('*-cube.o'))
            preprocessed = list(destination.glob('*-host-*.ccei'))
            if len(cube) != 1 or len(preprocessed) != 1:
                raise ValueError('ambiguous saved compiler artifacts')
            text = preprocessed[0].read_text()
            values = re.findall(r'catlass_shape_dimension\(\)\s*\{\s*return\s+(\d+)\s*;', text)
            if len(values) != 1:
                raise ValueError('cannot establish actual expanded SDK shape dimension')
            dimensions[arm] = int(values[0])
            objects[arm] = hashlib.sha256(cube[0].read_bytes()).hexdigest()
            evidence[arm] = device_sections(cube[0])
            (destination/'device-sections.json').write_text(json.dumps(evidence[arm], indent=2)+'\n')
        result = classify(dimensions['before'], dimensions['after'], evidence['before'], evidence['after'],
                          device_object_sha256=objects)
        result['source_bindings'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (source, source.with_name('shape_metadata_abi.h'))}
        (output/'screen.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        return result
    finally:
        os.chdir(entry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catlass', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--request-shape-info', action='store_true')
    args = parser.parse_args()
    compile_screen(args.catlass, args.output, request_shape_info=args.request_shape_info)
