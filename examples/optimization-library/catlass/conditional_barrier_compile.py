"""Compile exact stock AFTER and one-predicate conservative BEFORE in private trees."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from compile_tile import compile_pair, PIN
from conditional_barrier_contract import TILES

HEADER = 'include/catlass/gemm/tile/tile_mmad.hpp'
PREDICATE = 'if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare_dependency(catlass, destination, arm):
    """Validate immutable input, clone, and prove the only conservative delta."""
    if arm not in ('before','after'):
        raise ValueError('unknown arm')
    catlass, destination = Path(catlass), Path(destination)
    manifest = json.loads((catlass/'manifest.json').read_text())
    if manifest['revision'] != PIN:
        raise ValueError('wrong pinned revision')
    for entry in manifest['files']:
        if sha((catlass/entry['file']).read_bytes()) != entry['sha256']:
            raise ValueError('filtered source checksum mismatch')
    stock = (catlass/HEADER).read_bytes()
    text = stock.decode()
    if text.count(PREDICATE) < 1:
        raise ValueError('stock no-bias predicate absent')
    # Only the first (no-bias) overload changes. Bias overload remains immutable.
    changed = text.replace(PREDICATE, 'if (true) {', 1).encode()
    shutil.copytree(catlass,destination)  # Existing destinations are rejected.
    (destination/HEADER).write_bytes(changed if arm=='before' else stock)
    for entry in manifest['files']:
        if entry['file'] == HEADER:
            entry['sha256'] = sha((destination/HEADER).read_bytes())
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    proof = dict(revision=PIN, arm=arm, stock_header_sha256=sha(stock),
        compiled_header_sha256=sha((destination/HEADER).read_bytes()),
        stock_no_bias_body_sha256=sha(text[text.index('template <'):text.index('    CATLASS_DEVICE\n    void operator()(',text.index('AscendC::Mmad(l0CTensor'))].encode()),
        modification='first no-bias predicate -> true' if arm=='before' else 'none',
        stock_predicate=PREDICATE, original_manifest_sha256=sha((catlass/'manifest.json').read_bytes()))
    (destination/'barrier-patch.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


def compile_variants(catlass, output):
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    for arm in ('before','after'):
        dependency = output/f'dependency-{arm}'
        prepare_dependency(catlass, dependency, arm)
        for m,n in TILES:
            scratch = output/f'{m}x{n}'/f'build-{arm}'
            scratch.mkdir(parents=True,exist_ok=False)
            previous = Path.cwd()
            try:
                os.chdir(scratch)
                compile_pair(dependency, output/f'{m}x{n}',
                    source=Path(__file__).with_name('conditional_barrier.cpp'),
                    variants={arm:['-save-temps=obj',f'-DW006_STOCK_BARRIER={int(arm=="after")}',
                        f'-DW006_TILE_M={m}', f'-DW006_TILE_N={n}']})
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catlass',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    compile_variants(args.catlass,args.output)
