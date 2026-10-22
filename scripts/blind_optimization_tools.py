#!/usr/bin/env python3
"""Neutral task tools; remote operations use the managed cpl-remote executable."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import tarfile
import uuid


REMOTE_PROGRAM = r'''
import csv, hashlib, json, os, re, shlex, signal, subprocess, sys, tarfile, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_device import bounded_run, wait_for_device
from phase_deadline import wait_for_activation
from a3_memory_batch import execute, capture, duration, summarize, validate_capture
root = Path(__file__).resolve().parent
request = json.loads((root / 'request.json').read_text())
output = root / 'results'
output.mkdir()
summary = {'operation': request['verb'], 'implementation': request['implementation'],
           'source_sha256': request.get('source_sha256'),
           'launch_contract': dict(runner_checked=False, profile_stream_verified=False)}
started = time.monotonic()
relative_budget = float(request.get('host_remaining_seconds', 1800))
relative_budget = min(relative_budget,
                      float(os.environ.get('CPL_ROLE_REMAINING_SECONDS', relative_budget)))
summary['budget_evidence'] = dict(original_host_utc_deadline=request.get('utc_deadline'),
    host_remaining_seconds=request.get('host_remaining_seconds'),
    dispatch_remaining_seconds=relative_budget,
    policy='host-owned deadline; remote monotonic elapsed time only')
def remaining():
    # The host has already subtracted staging time. Remote UTC may differ by
    # minutes, so comparing it with a host timestamp rejects healthy work.
    value = relative_budget - (time.monotonic() - started)
    if value <= 0:
        raise TimeoutError('original role deadline expired')
    return value
def expire(*unused):
    raise TimeoutError('original role deadline expired')
original_execute = execute
def execute(argv, folder, env, *, timeout=600):
    return original_execute(argv, folder, env, timeout=min(timeout, remaining()))
lease = None
stage = 'environment'
try:
    if request.get('activation_required'):
        relative_budget = wait_for_activation(root / 'activation.json', remaining()) - started
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining())
    def query(*arguments):
        result = bounded_run(['npu-smi', 'info', *arguments], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=min(120, remaining()))
        if result.returncode:
            raise RuntimeError('npu-smi query failed: ' + result.stderr[-2000:])
        return result.stdout
    def probe(env):
        p = bounded_run([sys.executable, '-c', 'import torch,torch_npu; torch.empty(1,device="npu"); torch.npu.synchronize()'],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=min(180, remaining()))
        return p.returncode == 0, p.stderr[-1000:]
    physical, env, lease = wait_for_device(probe, query=query)
    summary.update(physical_device=physical, logical_device=0)
    command = [sys.executable, str(root / 'runner.py'), '--implementation', request['implementation']]
    stage = 'candidate'
    checked = execute(command + ['--checks=4'], output / 'check', env, timeout=300)
    if checked.returncode:
        raise RuntimeError('correctness or compilation failed: ' + checked.stderr[-12000:])
    records = []
    for line in checked.stdout.splitlines():
        try: records.append(json.loads(line))
        except ValueError: pass
    if not any(isinstance(r, dict) and r.get('correctness') is True for r in records):
        raise RuntimeError('trusted runner omitted correctness evidence')
    summary['correctness'] = records
    summary['launch_contract']['runner_checked'] = True
    if request['verb'] in ('timing', 'pipes', 'sanitize'):
        stage = 'profiling'
        wrapper = root / 'application.sh'
        wrapper.write_text('#!/bin/sh\nexec ' + shlex.join(command) + '\n')
        wrapper.chmod(0o700)
        folder = output / 'discovery'
        argv = ['msprof', 'op', '--application=' + str(wrapper), '--output=' + str(folder / 'report'),
                '--aic-metrics=BasicInfo', '--warm-up=0', '--launch-count=5000', '--kill=off']
        discovery = execute(argv, folder, env)
        validate_capture(discovery.returncode, discovery.stdout + discovery.stderr)
        rows = []
        for path in (folder / 'report').rglob('*OpBasicInfo*.csv'):
            with path.open(encoding='utf-8-sig', newline='') as stream:
                rows.extend(csv.DictReader(stream))
        names = {value for row in rows for key,value in row.items() if value and
                 'name' in key.lower() and ('kernel' in key.lower() or 'op' in key.lower())}
        summary['discovered_compute_rows'] = rows
        if len(rows) != 1 or len(names) != 1:
            stage = 'candidate-contract'
            raise RuntimeError('single-kernel contract requires exactly one compute launch in the complete bounded stream; observed ' + str(len(rows)) + ' rows: ' + repr(sorted(names)))
        selector = names.pop()
        summary['selector'] = selector
        summary['launch_contract']['profile_stream_verified'] = True
        stage = 'profiling'
        if request['verb'] in ('timing', 'pipes'):
            metric, count = (('BasicInfo', request.get('timing_captures', 5)) if request['verb'] == 'timing'
                             else ('PipeUtilization', request.get('pipe_captures', 3)))
            summary['capture_count'] = count
            captures = [capture(command, output / ('capture-' + str(i)), env, metric, selector) for i in range(count)]
            summary['captures'] = captures
            if request['verb'] == 'timing':
                summary['timing'] = summarize([duration(item['basic_rows'][0]) for item in captures])
        elif request['verb'] == 'sanitize':
            stage = 'sanitizer'
            spec = json.loads((root / 'task.json').read_text())
            if spec['language'] != 'triton':
                raise RuntimeError('sanitizer instrumentation is currently supported for Triton tasks only')
            tool = request['sanitizer']
            env.update(TRITON_ENABLE_SANITIZER='1', TRITON_ALWAYS_COMPILE='1',
                       PYTORCH_NO_NPU_MEMORY_CACHING='1', TRITON_CACHE_DIR=str(root / 'sanitizer-cache'))
            log = output / 'sanitizer-report.log'
            result = execute(['mssanitizer', '--tool=' + tool, '--log-level=info', '--log-file=' + str(log),
                              '--', *command], output / 'sanitizer', env, timeout=300)
            text = result.stdout + result.stderr + ''.join(p.read_text(errors='replace') for p in output.glob('sanitizer-report.log*') if p.is_file())
            starts = re.findall(r'Start \w+ sanitizer on kernel (\S+)', text)
            finishes = [x.rstrip('.') for x in re.findall(r'Sanitizer finished on kernel (\S+)', text)]
            target = [x for x in starts if x == selector]
            qualified = bool(target) and all(x in finishes for x in target)
            summary['sanitizer'] = dict(tool=tool, exit_code=result.returncode, analysis_completed=qualified,
                analyzed_kernels=starts, target_kernels=target, instrumented_performance=False,
                diagnostic_lines=[x for x in text.splitlines() if re.search(r'hazard|race|warning|error|sanitizer', x, re.I)])
            if result.returncode or not qualified:
                raise RuntimeError('sanitizer did not complete analysis of the selected task kernel; see retained logs')
    if request['verb'] == 'ir':
        stage = 'compiler-evidence'
        artifacts = root / 'compiler-evidence' / request['implementation']
        summary['compiler_artifacts'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.glob('*') if p.is_file()}
        if not summary['compiler_artifacts']:
            raise RuntimeError('no compiler artifacts returned by this implementation')
    summary['state'] = 'completed'
except BaseException as error:
    summary.update(state='failed', failure_stage='budget' if isinstance(error, TimeoutError) else stage,
                   error=str(error), traceback=traceback.format_exc())
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    if lease: lease.close()
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    # Compact reports and inventoried compiler outputs, never raw profiler payloads.
    with tarfile.open(root / 'evidence.tar.gz', 'w:gz') as archive:
        for base in (output, root / 'compiler-evidence', root / 'build'):
            if base.exists():
                for path in base.rglob('*'):
                    if path.is_file() and (base.name == 'compiler-evidence' or path.suffix in
                            ('.json', '.log', '.csv', '.txt', '.mlir', '.ll', '.s', '.asci', '.stdout', '.stderr')):
                        archive.add(path, arcname=str(path.relative_to(root)))
    print(json.dumps(summary), flush=True)
sys.exit(0 if summary['state'] == 'completed' else 1)
'''


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def positive_count(value):
    count = int(value)
    if count < 1:
        raise ValueError('capture counts must be positive integers')
    return count


class Tools:
    def __init__(self, bundle, workspace, target='bz-a3-1', runtime='py311-torch'):
        self.bundle, self.workspace = Path(bundle).resolve(), Path(workspace).resolve()
        self.client = (Path.home() / '.agents/skills/remote-access/scripts/cpl-remote').resolve(strict=True)
        module = runpy.run_path(str(self.client.with_name('cpl_remote.py')))
        registry = module['load_registry'](Path(os.path.expanduser(os.environ.get('CPL_REMOTE_CONFIG', '~/.config/cpl/remotes.yaml'))))
        config = module['get_target'](registry, target)
        self.cwd = config['workspace']
        self.target, self.runtime = target, runtime
        self.state = self.workspace.parent / 'tool-state'
        self.state.mkdir(exist_ok=True)

    def remote(self, *arguments):
        command = [str(self.client), '--json', *map(str, arguments)]
        timeout = 240
        deadline = getattr(self, '_staging_deadline', None)
        if deadline and arguments[0] in ('targets', 'capabilities', 'preflight', 'upload', 'run'):
            timeout = min(timeout, (datetime.fromisoformat(deadline) - datetime.now(timezone.utc)).total_seconds())
            if timeout <= 0:
                raise TimeoutError('original role deadline expired during staging')
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            deadline = getattr(self, '_staging_deadline', None)
            if deadline and datetime.fromisoformat(deadline) <= datetime.now(timezone.utc):
                raise TimeoutError('original role deadline expired during transport') from error
            raise
        with (self.state / 'transport.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(command=command, code=result.returncode,
                                         stdout=result.stdout, stderr=result.stderr)) + '\n')
        try:
            payload = json.loads(result.stdout)
        except ValueError:
            raise RuntimeError('transport response unavailable; inspect host transport journal: ' + result.stderr[-2000:])
        if result.returncode and not payload.get('handle'):
            raise RuntimeError('transport operation failed: ' + json.dumps(payload))
        return payload

    def start(self, verb, implementation, sanitizer, *, timing_captures=5, pipe_captures=3):
        timing_captures, pipe_captures = positive_count(timing_captures), positive_count(pipe_captures)
        # A candidate edit never licenses duplicate work while a prior call lives.
        for path in sorted(self.state.glob('*/receipt.json')):
            previous = json.loads(path.read_text())
            if previous.get('state') not in ('completed', 'failed', 'cancelled'):
                return self.observe(previous.get('call_id', path.parent.name))
        spec = json.loads((self.bundle / 'task.json').read_text())
        if spec['language'] != 'triton' and verb in ('ir', 'sanitize'):
            return dict(state='unsupported', failure_stage='capability', operation=verb,
                        implementation=implementation, error=verb + ' is supported for Triton tasks only')
        ident = uuid.uuid4().hex[:16]
        folder = self.state / ident
        folder.mkdir()
        extension = 'py' if spec['language'] == 'triton' else 'asc'
        source = (self.workspace if implementation == 'candidate' else self.bundle) / (implementation + '.' + extension)
        # Copy bytes once: subsequent edits cannot alter the submitted experiment.
        if source.is_symlink() or not source.is_file():
            raise ValueError('implementation must be a regular non-symlink file')
        for name in ('runner.py', 'native_host.asc', 'task.json'):
            shutil.copyfile(self.bundle / name, folder / name)
        shutil.copyfile(source, folder / source.name)
        helpers = Path(__file__).resolve().parent
        for name in ('profile_device.py', 'a3_memory_batch.py', 'phase_deadline.py'):
            shutil.copyfile(helpers / name, folder / name)
        request = dict(verb=verb, implementation=implementation, sanitizer=sanitizer,
                       timing_captures=timing_captures, pipe_captures=pipe_captures,
                       source_sha256=hashlib.sha256((folder / source.name).read_bytes()).hexdigest())
        context = self.workspace / 'execution.json'
        if context.exists():
            deadline = json.loads(context.read_text())['utc_deadline']
            budget = (datetime.fromisoformat(deadline) - datetime.now(timezone.utc)).total_seconds()
            if budget <= 0:
                return dict(state='failed', failure_stage='budget', error='original role deadline expired',
                            source_sha256=request['source_sha256'])
            request.update(utc_deadline=deadline, host_remaining_seconds=budget, activation_required=True)
            self._staging_deadline = deadline
        save(folder / 'request.json', request)
        (folder / 'execute.py').write_text(REMOTE_PROGRAM)
        archive = folder / (ident + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as out:
            for path in folder.iterdir():
                if path != archive: out.add(path, arcname=path.name)
        remote_archive = self.cwd.rstrip('/') + '/' + archive.name
        remote_folder = self.cwd.rstrip('/') + '/blind-tools-' + ident
        receipt = dict(call_id=ident, target=self.target, runtime=self.runtime, cwd=self.cwd,
                       remote_archive=remote_archive, remote_folder=remote_folder, request=request,
                       source_sha256=request['source_sha256'], state='prepared')
        save(folder / 'receipt.json', receipt)
        script = folder / 'run.sh'
        script.write_text('#!/bin/bash\nset -eu\nmkdir -p ' + shlex.quote(remote_folder) + '\n' +
                          'tar -xzf ' + shlex.quote(remote_archive) + ' -C ' + shlex.quote(remote_folder) + '\n' +
                          'cd ' + shlex.quote(remote_folder) + '\nexec python execute.py\n')
        try:
            for command in (('targets',), ('capabilities', self.target), ('preflight', self.target)):
                self.remote(*command)
            receipt['upload'] = self.remote('upload', self.target, archive, remote_archive)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            receipt.update(state='failed', failure_stage='budget' if isinstance(error, TimeoutError) else 'infrastructure',
                           error=str(error))
            save(folder / 'receipt.json', receipt)
            return receipt
        save(folder / 'receipt.json', receipt)
        if receipt['upload'].get('state') != 'completed':
            return self.observe(ident)
        return self.dispatch(folder, receipt)

    def dispatch(self, folder, receipt):
        if receipt.get('dispatch_intent'):
            return self.unconfirmed(receipt)
        request = receipt.get('request', {})
        if request.get('utc_deadline'):
            self._staging_deadline = request['utc_deadline']
            budget = (datetime.fromisoformat(request['utc_deadline']) - datetime.now(timezone.utc)).total_seconds()
            if budget <= 0:
                receipt.update(state='failed', failure_stage='budget', error='original role deadline expired',
                               source_sha256=request.get('source_sha256'))
                save(folder / 'receipt.json', receipt)
                return receipt
            receipt['dispatch_remaining_seconds'] = min(budget, request['host_remaining_seconds'])
            script = folder / 'run.sh'
            text = script.read_text()
            script.write_text(text.replace('set -eu\n', 'set -eu\nexport CPL_ROLE_REMAINING_SECONDS=' +
                shlex.quote(str(receipt['dispatch_remaining_seconds'])) + '\n', 1))
        # Commit the per-call intent before crossing the external dispatch boundary.
        # A lost response must never turn an observation into a second run.
        receipt['dispatch_intent'] = True
        save(folder / 'receipt.json', receipt)
        try:
            dispatch = self.remote('run', receipt['target'], '--runtime', receipt['runtime'],
                                   '--cwd', receipt['cwd'], '--file', folder / 'run.sh')
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            receipt.update(state='dispatch-unconfirmed', error=str(error))
            save(folder / 'receipt.json', receipt)
            return self.unconfirmed(receipt)
        if not dispatch.get('handle'):
            receipt.update(state='dispatch-unconfirmed', dispatch_response=dispatch)
            save(folder / 'receipt.json', receipt)
            return self.unconfirmed(receipt)
        receipt.update(state=dispatch['state'], handle=dispatch['handle'])
        save(folder / 'receipt.json', receipt)
        return self.observe(receipt['call_id'])

    @staticmethod
    def unconfirmed(receipt):
        return dict(receipt, state='dispatch-unconfirmed',
                    recovery='DO NOT REDISPATCH: dispatch was attempted but its handle is unknown. '
                    'Recover the original handle from managed remote-access diagnostics before continuing; '
                    'this call will never automatically submit another workload.')

    def transfer(self, receipt, name):
        transfer = receipt[name]
        if transfer.get('state') not in ('completed', 'failed', 'cancelled'):
            if not transfer.get('handle'):
                return dict(state='failed', error='nonterminal transfer omitted retained handle')
            transfer = self.remote('observe', transfer['handle'], '--wait', '--timeout', '120')
        receipt[name] = transfer
        return transfer

    def observe(self, ident):
        if not ident.isalnum(): raise ValueError('invalid call id')
        folder = self.state / ident
        receipt = json.loads((folder / 'receipt.json').read_text())
        if not receipt.get('handle'):
            if receipt.get('dispatch_intent'):
                return self.unconfirmed(receipt)
            if 'upload' not in receipt:
                return dict(receipt, state='infrastructure-unavailable', error='submission has no retained job or transfer')
            try:
                transfer = self.transfer(receipt, 'upload')
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                return dict(receipt, error=str(error), next='experiment-tool observe ' + ident)
            save(folder / 'receipt.json', receipt)
            if transfer['state'] in ('failed', 'cancelled'):
                receipt.update(state='failed', failure_stage='infrastructure',
                               error='upload ' + transfer['state'], transfer=transfer)
                save(folder / 'receipt.json', receipt)
                return receipt
            if transfer['state'] != 'completed':
                return dict(receipt, state='transfer-pending', next='experiment-tool observe ' + ident)
            return self.dispatch(folder, receipt)
        try:
            activate = runpy.run_path(str(Path(__file__).with_name('phase_deadline.py')))['activate']
            if not activate(self, folder, receipt):
                return dict(receipt, next='experiment-tool observe ' + ident)
        except TimeoutError as error:
            receipt.update(state='failed', failure_stage='budget', error=str(error))
            save(folder / 'receipt.json', receipt)
            return receipt
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            return dict(receipt, activation_error=str(error), next='experiment-tool observe ' + ident)
        try:
            result = self.remote('observe', receipt['handle'], '--wait', '--timeout', '120')
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            receipt.update(state='observation-unavailable', error=str(error))
            save(folder / 'receipt.json', receipt)
            return dict(receipt, next='experiment-tool observe ' + ident)
        receipt['state'] = result['state']
        save(folder / 'receipt.json', receipt)
        if result['state'] not in ('completed', 'failed', 'cancelled'):
            return dict(receipt, next='experiment-tool observe ' + ident)
        evidence = self.workspace / 'evidence' / ident
        evidence.mkdir(parents=True, exist_ok=True)
        try:
            if 'download' not in receipt:
                target = receipt.get('target', receipt['handle'].split(':', 3)[1])
                receipt['download'] = self.remote('download', target, receipt['remote_folder'] + '/evidence.tar.gz', evidence / 'evidence.tar.gz')
                save(folder / 'receipt.json', receipt)
            download = self.transfer(receipt, 'download')
            save(folder / 'receipt.json', receipt)
            if download.get('state') == 'completed' and download.get('handle'):
                fetched = self.remote('fetch', download['handle'], evidence / 'evidence.tar.gz')
                if fetched.get('state') != 'completed':
                    return dict(receipt, state='failed', failure_stage='infrastructure', error='evidence fetch did not complete', fetch=fetched)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            # A completed workload and a failed evidence transfer are different events.
            return dict(receipt, evidence_error=str(error), next='experiment-tool observe ' + ident)
        if download.get('state') != 'completed':
            if download.get('state') in ('failed', 'cancelled'):
                return dict(receipt, state='failed', failure_stage='infrastructure', error='evidence transfer ' + download['state'])
            return dict(receipt, evidence_transfer=download, next='experiment-tool observe ' + ident)
        with tarfile.open(evidence / 'evidence.tar.gz') as archive:
            for entry in archive.getmembers():
                if entry.issym() or entry.islnk() or not entry.isfile() or not (evidence / entry.name).resolve().is_relative_to(evidence.resolve()):
                    raise ValueError('invalid evidence archive member')
            archive.extractall(evidence, filter='data')
        summary = json.loads((evidence / 'results/summary.json').read_text())
        summary.update(call_id=ident, handle=receipt['handle'], evidence='evidence/' + ident)
        save(evidence / 'receipt.json', receipt)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--target', default='bz-a3-1')
    parser.add_argument('--runtime', default='py311-torch')
    parser.add_argument('verb', choices=('help', 'check', 'timing', 'pipes', 'ir', 'sanitize', 'observe'))
    parser.add_argument('call_id', nargs='?')
    parser.add_argument('--implementation', choices=('baseline', 'candidate'), default='candidate')
    parser.add_argument('--sanitizer', choices=('memcheck', 'racecheck', 'synccheck'), default='racecheck')
    parser.add_argument('--timing-captures', type=positive_count, default=5)
    parser.add_argument('--pipe-captures', type=positive_count, default=3)
    args = parser.parse_args(argv)
    if args.verb == 'help':
        print('experiment-tool {check,timing,pipes,ir,sanitize} [--implementation baseline|candidate]\n'
              'sanitize [--sanitizer memcheck|racecheck|synccheck] (Triton only)\n'
              'experiment-tool observe CALL_ID resumes an existing retained job.\n'
              'All execution operations include four correctness checks and the trusted runner launch contract.\n'
              'timing/pipes/sanitize additionally verify exactly one exported compute launch in a bounded profiler stream;\n'
              'check/ir do not verify that stream. launch_contract reports each guarantee separately.\n'
              'A separate check call is unnecessary before timing.\n'
              'Results and diagnostics appear under evidence/CALL_ID. --timing-captures N defaults to five BasicInfo samples;\n'
              '--pipe-captures N defaults to three PipeUtilization captures; both must be positive integers.\n'
              'ir returns Triton compiler artifacts; native ir is unsupported.\n'
              'check and ir do not require a profiler. Optional execution.json utc_deadline bounds original role work,\n'
              'including staging and device waits; observation resumes the same call without resetting its budget.\n'
              'Candidate failures are retained; nonterminal infrastructure observations must be resumed, not duplicated.')
        return 0
    try:
        tools = Tools(args.bundle, args.workspace, args.target, args.runtime)
        if args.verb == 'observe' and not args.call_id: parser.error('observe requires CALL_ID')
        result = tools.observe(args.call_id) if args.verb == 'observe' else tools.start(
            args.verb, args.implementation, args.sanitizer,
            timing_captures=args.timing_captures, pipe_captures=args.pipe_captures)
        print(json.dumps(result, indent=2))
        return 1 if result.get('state') == 'failed' else 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(json.dumps(dict(state='infrastructure-unavailable', error=str(error))), file=sys.stderr)
        return 74


if __name__ == '__main__':
    raise SystemExit(main())
