#!/usr/bin/env python3
"""Export sealed diagnostic packets from retained natural-task qualification."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile


TASKS = [f'task{x:02}' for x in range(1, 7)]
PUBLIC = ['task.json', 'TASK.md', 'baseline.py', 'candidate-a.py', 'candidate-b.py', 'timing.json']
CONTRACT = '''Analyze the baseline and two proposed source changes for this task.
The baseline implements launch(inputs, output) in a single fused Triton kernel.
Preserve every input-dependent stage, output shape and declared tolerances.
Inputs may be arbitrary finite normal values; gather indices and masks may be arbitrary valid values.
task.json gives the shape and input contract; baseline.py is the executable specification.
candidate-a.py and candidate-b.py are proposed alternatives, not measured winners.
timing.json contains baseline measurements only. This is a diagnostic task, not a request to run kernels.
'''
LESSONS = {
    'gather': dict(supported=[
        'Retained lowering keeps a scalarized 256-element gather loop.',
        'Warp hint has unchanged retained early IR, not proven binary equivalence.',
        'Static unroll reduces small vector/stall activity; overall timing direction is not robust.'],
        forbidden=['Scalar ratio proves capacity saturation.', 'Memory cost is excluded.',
                   'A noisy null proves equivalence.'], candidate_directions=['indeterminate', 'indeterminate']),
    'masked': dict(supported=[
        'Predicate remains in early IR, but retained lower IR has dense copy and select.',
        'Predicated alternative adds vector activity.',
        'Warp-hint latency direction reverses between batches; task04 predicate medians are slower but noisy.'],
        forbidden=['Logical masking proves reduced physical bytes.', 'Early IR proves binary identity.',
                   'All masks are always unhelpful.'], candidate_directions=['task04 slower medians; otherwise indeterminate', 'indeterminate']),
    'matmul': dict(supported=[
        'BK64 halves reduction loops: task05 eight to four, task06 sixty-four to thirty-two.',
        'BK64 reduces reported Cube MTE2/stall and vector wait with near-constant vector arithmetic.',
        'BK64 is faster in both batches; BM16 is slower and increases program/activity costs.'],
        forbidden=['A larger vector ratio proves more vector work.', 'Pipe times are additive latency costs.',
                   'A named wait counter alone identifies a particular source flag or dependency.'],
        candidate_directions=['slower', 'faster']),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def one_csv(members, prefix, suffix):
    selected = [data for name, data in members.items() if name.startswith(prefix + '/') and name.endswith(suffix)]
    require(len(selected) == 1, 'expected one capture CSV: ' + prefix)
    return selected[0]


def verify_case(case, archives, bundles):
    archive = archives / case['archive']
    require(sha(archive.read_bytes()) == case['archive_sha256'], 'archive hash mismatch: ' + archive.name)
    with tarfile.open(archive, 'r:gz') as stream:
        files = [item for item in stream.getmembers() if item.isfile()]
        require(len({item.name for item in files}) == len(files), 'duplicate archive members')
        members = {item.name: stream.extractfile(item).read() for item in files}
    held_out, public_pipes = {}, []
    for index in range(3):
        variant = f'variant-{index}'
        declared = case['variants'][variant]
        source = (bundles / case['task'] / variant / 'candidate.py').read_bytes()
        spec = json.loads((bundles / case['task'] / variant / 'task.json').read_text())
        require(spec['id'] == case['task'] and
                all(spec.get(k) == v for k, v in case['task_spec'].items()), 'task specification mismatch')
        require(sha(source) == declared['source_sha256'], 'source hash mismatch: ' + variant)
        prefix = f"batch-{case['batch']}/{variant}"
        actual = json.loads(members[prefix + '/result.json'])
        require(actual['state'] == declared['state'] == 'completed', 'incomplete candidate')
        require(actual['source_sha256'] == declared['source_sha256'], 'recorded source mismatch')
        selector = declared['selector']
        discovery = rows(one_csv(members, prefix + '/discovery', 'OpBasicInfo.csv'))
        require(actual['selector'] == selector and len(discovery) == 1 and discovery[0]['Op Name'] == selector,
                'selector/discovery mismatch')
        require(declared['compute_discovery_count'] == 1, 'multiple compute kernels')
        checks = actual['correctness']
        require(len(checks) == 4 and all(x['correctness'] is True for x in checks) and
                [x['seed'] for x in checks] == declared['correctness_seeds'] == [1, 2, 3, 4] and
                declared['correctness_checks_passed'] == 4, 'incorrect seeded candidate')
        samples = []
        for capture in range(5):
            basic = rows(one_csv(members, prefix + f'/basic-{capture}', 'OpBasicInfo.csv'))
            require(len(basic) == 1 and basic[0]['Op Name'] == selector, 'basic selector mismatch')
            samples.append(float(basic[0]['Task Duration(us)']))
        require(all(math.isfinite(x) and x > 0 for x in samples) and
                samples == declared['timing']['samples_us'] == actual['timing']['samples_us'] and
                statistics.median(samples) == declared['timing']['median_us'], 'timing sample mismatch')
        captures = []
        for capture in range(3):
            pipe_basic = rows(one_csv(members, prefix + f'/pipe-{capture}', 'OpBasicInfo.csv'))
            require(len(pipe_basic) == 1 and pipe_basic[0]['Op Name'] == selector,
                    'pipe selector binding mismatch')
            raw = one_csv(members, prefix + f'/pipe-{capture}', 'PipeUtilization.csv')
            parsed = rows(raw)
            require(len(parsed) == declared['pipe_capture_row_counts'][capture] and len(parsed) > 0 and
                    all(x.get('sub_block_id') and x.get('block_id') is not None for x in parsed),
                    'pipe capture integrity mismatch')
            if 'pipe_captures' in actual:
                require(actual['pipe_captures'][capture]['selector'] == selector and
                        actual['pipe_captures'][capture]['pipe_rows'] == parsed, 'pipe result mismatch')
            captures.append(dict(sha256=sha(raw), row_count=len(parsed)))
            if index == 0:
                public_pipes.append(raw)
        compiler = {}
        for kind, provenance in declared['compiler_artifacts'].items():
            data = members[provenance['member']]
            require(sha(data) == provenance['sha256'], 'compiler hash mismatch')
            compiler[kind] = dict(provenance, text=data.decode())
        require(set(compiler) == {'ttir', 'ttadapter'}, 'compiler evidence missing')
        held_out[variant] = dict(result=actual, compiler=compiler, pipe_captures=captures,
                                 source_sha256=sha(source))
    return held_out, public_pipes


def export(qualification, archives, bundles, output):
    qualification, archives, bundles, output = map(Path, (qualification, archives, bundles, output))
    require(not output.exists(), 'output must be fresh')
    qualification_bytes = qualification.read_bytes()
    document = json.loads(qualification_bytes)
    cases = document['cases']
    require(len(cases) == 12 and {(x['task'], x['batch']) for x in cases} ==
            {(task, batch) for task in TASKS for batch in (1, 2)}, 'expected six tasks and twelve unique batches')
    staged, held_out = {}, {}
    for case in sorted(cases, key=lambda x: (x['task'], x['batch'])):
        require(case['state'] == 'completed', 'incomplete batch')
        task = case['task']
        validated, pipe_csvs = verify_case(case, archives, bundles)
        held_out[f"{task}-batch{case['batch']}"] = validated
        if case['batch'] != 1:
            continue
        spec = json.loads((bundles / task / 'variant-0/task.json').read_text())
        spec['public_files'] = PUBLIC
        public = Path('public') / task
        staged[public / 'task.json'] = encoded(spec)
        staged[public / 'TASK.md'] = CONTRACT.encode()
        for index, filename in enumerate(('baseline.py', 'candidate-a.py', 'candidate-b.py')):
            staged[public / filename] = (bundles / task / f'variant-{index}/candidate.py').read_bytes()
        baseline = case['variants']['variant-0']
        staged[public / 'timing.json'] = encoded(dict(baseline['timing'], units='us',
            measurement='msprof op BasicInfo task duration; diagnostic, not canonical latency',
            batch=1, selector=baseline['selector'], source_sha256=baseline['source_sha256']))
        captures = []
        for index, data in enumerate(pipe_csvs):
            filename = f'pipe-{index}.csv'
            staged[Path('pipes') / task / filename] = data
            captures.append(dict(file=filename, sha256=sha(data), row_count=len(rows(data))))
        staged[Path('pipes') / task / 'pipe-manifest.json'] = encoded(dict(
            task=task, batch=1, selector=baseline['selector'], source_sha256=baseline['source_sha256'],
            metric='PipeUtilization', captures=captures))
    gold = dict(cases=cases, held_out=held_out, lessons=LESSONS,
                qualification_sha256=sha(qualification_bytes),
                skill_revision='294f76fc602f9d121cb9826bfcdc6c9e524c1bd2')
    staged[Path('private/gold.json')] = encoded(gold)
    manifest = dict(schema_version=1, tasks=TASKS, qualification_sha256=sha(qualification_bytes),
                    files={str(path): sha(data) for path, data in sorted(staged.items())})
    # Every input has been checked before creating any output.
    output.mkdir(parents=True)
    for path, data in staged.items():
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (output / 'manifest.json').write_bytes(encoded(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qualification', 'archives', 'bundles', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.qualification, args.archives, args.bundles, args.output)))


if __name__ == '__main__':
    main()
