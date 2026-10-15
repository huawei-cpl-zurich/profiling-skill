"""Retained sanitizer controls: reports are evidence, exit status alone is not."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_device import bounded_run, wait_for_device


def classify(text, exit_code, *, timeout=False):
    if timeout:
        return 'timeout'
    if 'Usage: mssanitizer' in text or re.search(r'param .*invalid|invalid (option|argument)', text, re.I):
        return 'instrumentation-failure'
    if re.search(r'unsupported|not support', text, re.I):
        return 'unsupported'
    if re.search(r'(RAW|WAR|WAW)\s+hazard|(?:ERROR|WARNING|Error|Warning)[^\n]*(?:data race|race condition|out.of.bounds|invalid (?:read|write))', text):
        return 'detected'
    if exit_code or re.search(r'instrument.*(fail|missing)|no instrument|failed to', text, re.I):
        return 'instrumentation-failure'
    if re.search(r'Sanitizer finished[^\n]*See all detected errors above', text):
        return 'other-finding'
    if re.search(r'(error|hazard)s?\s*[:=]\s*0|0 (errors|hazards)|no (error|hazard|race)|checking.*(finish|complete)', text, re.I):
        return 'no-finding'
    return 'unqualified-report'


def memory_events(text):
    """Instrumented accesses provide positive lowering evidence, not timings."""
    events = []
    for line in text.splitlines():
        if not line.startswith('serialNo:'):
            continue
        row = dict(field.split(':', 1) for field in line.split(', ') if ':' in field)
        if row.get('type') in ('LOAD', 'STORE') and row.get('coreId', '-1') != '-1':
            events.append(row)
    return events


def qualification(checks, *, families, canary=False):
    races = [row for row in checks if row['tool'] == 'racecheck']
    expected = {(family, mutant, repeat) for family in families
                for mutant in ((False,) if canary else (False, True))
                for repeat in range(1 if canary else 3)}
    observed = [(row['family'], row['mutant'], row['repeat']) for row in races]
    if not expected or set(observed) != expected or len(observed) != len(expected):
        return 'inconclusive'
    if any(not row['analysis_finished'] or row['classification'] !=
           ('detected' if row['mutant'] else 'no-finding') for row in races):
        return 'inconclusive'
    return 'passed'


def run_check(command, folder, env, tool):
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(env, TRITON_ENABLE_SANITIZER='1', TRITON_ALWAYS_COMPILE='1',
               PYTORCH_NO_NPU_MEMORY_CACHING='1', TRITON_CACHE_DIR=str(folder / 'cache'))
    argv = ['mssanitizer', '--tool=' + tool, '--log-level=info',
            '--log-file=' + str(folder / 'sanitizer.log'), '--', *command]
    (folder / 'command.json').write_text(json.dumps(argv))
    timeout, code = False, None
    try:
        result = bounded_run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=300)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        timeout = True
        stdout, stderr = exc.stdout or '', exc.stderr or ''
    except OSError as exc:
        stdout, stderr, code = '', str(exc), 127
    for name, value in [('stdout', stdout), ('stderr', stderr)]:
        if isinstance(value, bytes):
            value = value.decode(errors='replace')
        (folder / (name + '.log')).write_text(value)
    reports = []
    for path in sorted(folder.glob('sanitizer.log*')):
        if path.is_file():
            reports.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                text=path.read_text(errors='replace')))
    text = '\n'.join([str(stdout), str(stderr), *(item['text'] for item in reports)])
    kernels = re.findall(r'Start \w+ sanitizer on kernel (\S+)', text)
    completed = [name.rstrip('.') for name in re.findall(r'Sanitizer finished on kernel (\S+)', text)]
    target_kernels = [name for name in kernels if name in ('a3_sanitizer_control', 'a3_sanitizer_write')]
    started = bool(target_kernels)
    finished = any(name in completed for name in target_kernels)
    sections = re.findall(r'(Start \w+ sanitizer on kernel (a3_sanitizer_\S+).*?'
                          r'Sanitizer finished on kernel \2[^\n]*)', text, re.S)
    target_text = '\n'.join(section for section, _ in sections)
    classification = classify(target_text or text, code, timeout=timeout)
    if classification in ('detected', 'other-finding', 'no-finding') and not (started and finished):
        classification = 'unqualified-report'
    relevant = [line for line in text.splitlines() if re.search(
        r'hazard|race|\b(?:RAW|WAR|WAW|UB|GM|MTE[123]|PIPE_V)\b|0x[\da-f]+|warning|error', line, re.I)]
    correctness = None
    for line in str(stdout).splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get('correctness'), bool):
            correctness = record['correctness']
    result = dict(classification=classification, exit_code=code, workload_correctness=correctness,
                  timed_out=timeout, reports=reports, diagnostic_lines=relevant,
                  analysis_started=started, analysis_finished=finished,
                  analyzed_kernels=kernels, target_kernels=target_kernels,
                  target_report=target_text,
                  memory_events=memory_events(target_text),
                  limitation='no-finding is not proof of race freedom; instrumentation must be audited')
    (folder / 'result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--language', choices=('triton', 'native'), required=True)
    parser.add_argument('--canary', action='store_true', help='one repaired control, before a full qualification')
    args = parser.parse_args()
    if args.repeats != 3:
        parser.error('qualification requires exactly three repeats')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = None
    if args.language == 'native':
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
        from a3_native_build import build
        executable = build(output / 'build', source=Path(__file__).resolve().parents[1] /
                           'benchmarks' / 'a3_sanitizer_control.asc', extra_flags=('--sanitizer',))
    def probe(env):
        p = bounded_run([sys.executable, '-c', 'import torch,torch_npu; torch.empty(1,device="npu"); torch.npu.synchronize()'],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        return p.returncode == 0, p.stderr[-1000:]
    physical, env, lease = wait_for_device(probe)
    summary = dict(physical_device=physical, logical_device=0, instrumented_performance=False,
                   qualification='running', checks={})
    try:
        for family in ((0,) if args.canary else range(5)) if executable else (None,):
            for mutant in ((False,) if args.canary else (False, True)):
                for repeat in range(1 if args.canary else 3):
                    for tool in ('memcheck', 'racecheck', 'synccheck'):
                        label = f'{args.language}-{family}-{int(mutant)}-{repeat}-{tool}'
                        folder = output / label
                        if executable:
                            command = [str(executable), str(family), str(int(mutant))]
                        else:
                            command = ['python', str(Path(__file__).resolve().parents[1] / 'benchmarks' /
                                       'a3_sanitizer_triton.py'), '--artifacts', str(folder / 'compiler')]
                            if mutant:
                                command.append('--mutant')
                        summary['checks'][label] = run_check(command, folder, env, tool)
                        summary['checks'][label].update(tool=tool, mutant=mutant, family=family, repeat=repeat)
                        failure_code = 0
                        if summary['checks'][label]['classification'] in ('instrumentation-failure', 'unqualified-report', 'timeout'):
                            failure_code = 2
                        elif not mutant and summary['checks'][label]['workload_correctness'] is not True:
                            failure_code = 3
                        elif not mutant and tool == 'racecheck' and summary['checks'][label]['classification'] == 'detected':
                            failure_code = 4
                        if failure_code:
                            summary['qualification'] = 'inconclusive'
                        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
                        print(json.dumps({'check': label, 'classification': summary['checks'][label]['classification']}), flush=True)
                        if failure_code:
                            return failure_code
        families = ((0,) if args.canary else range(5)) if executable else (None,)
        summary['qualification'] = qualification(summary['checks'].values(), families=families, canary=args.canary)
        summary['canary_only'] = args.canary
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    finally:
        lease.close()
    return 0 if summary['qualification'] == 'passed' else 5


if __name__ == '__main__':
    raise SystemExit(main())
