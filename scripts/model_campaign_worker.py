"""Trusted retained worker; task adapters and all evidence stay outside sandboxes."""
from datetime import datetime, timezone
import json
import hashlib
import importlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from profile_device import bounded_run, wait_for_device


def qualification_key(root, adapter, runtime, source_sha256):
    """Identity of a proof reusable only inside this retained private runtime."""
    if runtime is None or not (root / 'candidate.py').is_file():
        return None
    if hashlib.sha256((root / 'candidate.py').read_bytes()).hexdigest() != source_sha256:
        return None
    from model_campaign_task import trusted_bundle_root
    bundle = trusted_bundle_root(root)
    digest = hashlib.sha256()
    for name in ('scripts', 'reference', 'benchmarks', 'skills', 'private-task.json', 'task.json', 'API.md',
                 'candidate.py', 'execute.py', 'execute-phase.py'):
        path = bundle / name
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        for item in files:
            if item.is_file() and '__pycache__' not in item.parts:
                digest.update(str(item.relative_to(bundle)).encode())
                digest.update(item.read_bytes())
    context = (str(bundle.resolve()), digest.hexdigest(), id(adapter),
               tuple(id(getattr(adapter, name, None)) for name in ('check', 'benchmark', 'pipes', 'screen')),
               runtime['physical'], runtime['torch_module_id'], id(sys.modules.get('torch')),
               runtime['runtime_initializations'], tuple(sorted(runtime['env'].items())))
    return hashlib.sha256(repr((source_sha256, context)).encode()).hexdigest()


def execute(root, adapter, *, acquire=None, monotonic=time.monotonic, runtime=None):
    """Adapter check(root,env,remaining) must validate the task's required variants.

    It returns {passed: bool, variants: [...], ...} with candidate-only safe
    diagnostics from candidate_probe. benchmark/pipes/screen have the same
    signature and may return only private artifacts; export happens on host.
    Every adapter subprocess must use remaining() as its timeout.
    """
    root = Path(root)
    request = json.loads((root / 'request.json').read_text())
    budget = min(float(request['host_remaining_seconds']),
                 float(os.environ.get('CAMPAIGN_REMAINING_SECONDS', request['host_remaining_seconds'])))
    started = monotonic()
    def remaining():
        # Host has already subtracted staging; remote UTC can be skewed.
        return max(0., budget - (monotonic() - started))
    def expire(*_):
        raise TimeoutError('original round deadline expired')
    summary = dict(state='failed', failure_stage='infrastructure', operation=request['operation'],
                   source_sha256=request['source_sha256'], worker_pid=os.getpid())
    from model_campaign_task import check_variants
    summary['required_variants'] = check_variants(root)
    lease, proof_key, proof_cache = None, None, {}
    try:
        if runtime is None and request.get('activation_required'):
            from phase_deadline import wait_for_activation
            cutoff = wait_for_activation(root / 'activation.json', remaining())
            budget = min(budget, cutoff - started)
            if remaining() <= 0:
                raise TimeoutError('original round deadline expired before activation')
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, remaining() or .001)
        def probe(env):
            summary.update(physical_device=int(env['ASCEND_RT_VISIBLE_DEVICES']), logical_device=0,
                           progress='runtime-probe')
            print('MODEL_STAGE=runtime-probe', flush=True)
            # Import once in the retained worker, after selecting its visibility.
            # A subprocess probe pays the64s native import twice before any work.
            # The selector aborts on a failed probe rather than switching devices
            # inside a process with an initialized device runtime.
            os.environ.update(env)
            signal.setitimer(signal.ITIMER_REAL, min(180., remaining()) or .001)
            try:
                torch = importlib.import_module('torch')
                importlib.import_module('torch_npu')
                torch.npu.set_device(0)
                torch.ones(1, device='npu')
                torch.npu.synchronize()
                return True, ''
            finally:
                signal.setitimer(signal.ITIMER_REAL, remaining() or .001)
        if runtime:
            physical, env = runtime['physical'], runtime['env']
        elif acquire is None:
            from model_device import acquire as select
            physical, env, lease = select(probe, remaining)
        else:
            physical, env, lease = acquire(probe)
        if runtime is not None and not runtime:
            runtime.update(physical=physical, env=env, lease=lease,
                           runtime_initializations=1, torch_module_id=id(sys.modules.get('torch')))
        os.environ.update(env)
        if runtime:
            summary.update({key: runtime[key] for key in ('runtime_initializations', 'torch_module_id')})
        summary.update(physical_device=physical, logical_device=0, failure_stage='candidate')
        summary['progress'] = 'correctness-check'
        print('MODEL_STAGE=correctness-check', flush=True)
        proof_key = qualification_key(root, adapter, runtime, request['source_sha256'])
        proof_cache = runtime.setdefault('correctness_cache', {}) if runtime is not None else {}
        reused = proof_cache.get(proof_key) if request['operation'] in ('benchmark', 'pipes') else None
        if reused:
            checked = dict(passed=True, variants=summary['required_variants'], diagnostics=[])
            summary['correctness_provenance'] = dict(reused, kind='phase-local-reuse')
        else:
            if proof_key:
                proof_cache.pop(proof_key, None)
            checked = adapter.check(root, env, remaining)
            summary['correctness_provenance'] = dict(kind='fresh', original_call=root.name,
                source_sha256=request['source_sha256'])
        summary['check'] = checked
        if checked.get('passed') is not True or set(checked.get('variants', [])) != set(summary['required_variants']):
            if runtime:
                try:
                    torch = sys.modules['torch']
                    torch.ones(1, device='npu')
                    torch.npu.synchronize()
                except TimeoutError:
                    raise
                except Exception:
                    summary.update(failure_stage='infrastructure',
                        diagnostic=dict(kind='phase-runtime-unavailable'))
            return summary
        if request['operation'] != 'check':
            summary['failure_stage'] = 'measurement'
            summary['progress'] = request['operation']
            print('MODEL_STAGE=' + request['operation'], flush=True)
            summary['result'] = getattr(adapter, request['operation'])(root, env, remaining)
            if summary['result'].get('state') not in ('complete', 'completed'):
                proof_cache.pop(proof_key, None)
                return summary
        if proof_key and not reused:
            proof_cache[proof_key] = dict(original_call=root.name, source_sha256=request['source_sha256'])
        summary.update(state='completed')
        summary.pop('failure_stage')
        return summary
    except Exception as error:
        proof_cache.pop(proof_key, None)
        summary.update(error=str(error), traceback=traceback.format_exc())
        summary['diagnostic'] = dict(kind='worker-failure', error_type=type(error).__name__,
                                    stage=summary.get('progress', 'device-acquisition'))
        if isinstance(error, TimeoutError):
            summary['failure_stage'] = 'budget'
        return summary
    finally:
        print('MODEL_REPORT_WRITER=' + json.dumps(dict(owner_pid=summary['worker_pid'],
            writer_pid=os.getpid(), operation=request['operation'], state=summary['state'])), flush=True)
        if os.getpid() == summary['worker_pid']:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if lease and runtime is None:
                lease.close()
            output = root / 'results'
            output.mkdir(exist_ok=True)
            (output / 'summary.json').write_text(json.dumps(summary, indent=2))
            # Bundle compact evidence only. Raw profile payloads and all providers
            # remain remote; even this private archive is never directly agent-mounted.
            with tarfile.open(root / 'evidence.pending', 'w:gz') as archive:
                for path in root.rglob('*'):
                    if path.is_file() and (path.suffix in ('.json', '.csv', '.stdout', '.stderr', '.log')
                            or path.suffix == '.txt' and path.is_relative_to(root / 'diagnostics')):
                        if path.is_relative_to(output) or path.is_relative_to(root / 'measurement') or path.is_relative_to(root / 'diagnostics'):
                            archive.add(path, arcname=str(path.relative_to(root)))
            (root / 'evidence.pending').replace(root / 'evidence.tar.gz')


def phase(root, adapter):
    """One private process/device lease for the original role deadline."""
    root = Path(root)
    deadline = float(os.environ.get('CAMPAIGN_DEADLINE_MONOTONIC',
        time.monotonic() + float(os.environ['CAMPAIGN_REMAINING_SECONDS'])))
    runtime = {}
    phase_pid = os.getpid()
    start = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    (root / 'worker.json').write_text(json.dumps(dict(pid=phase_pid, start=start)))
    def terminate(signum, *_):
        if os.getpid() != phase_pid:
            os._exit(128 + signum)
        print('MODEL_PHASE_SIGNAL=' + json.dumps(dict(owner_pid=phase_pid,
            receiver_pid=os.getpid(), signal=signum)), flush=True)
        raise SystemExit(0)
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        while time.monotonic() < deadline and not (root / 'stop').exists():
            for archive in sorted((root / 'inbox').glob('*.tar.gz')):
                call = root / 'calls' / archive.name[:-7]
                if (call / 'evidence.tar.gz').exists():
                    continue
                try:
                    # rsync may still be writing. Never admit an incomplete gzip.
                    with tarfile.open(archive) as stream:
                        members = stream.getmembers()
                        if {m.name for m in members} != {'candidate.py', 'request.json'}:
                            raise ValueError('invalid request members')
                        candidate = stream.extractfile('candidate.py').read()
                        request = json.loads(stream.extractfile('request.json').read())
                    import hashlib
                    if hashlib.sha256(candidate).hexdigest() != request['source_sha256']:
                        continue
                except (EOFError, OSError, tarfile.TarError, ValueError):
                    continue
                call.mkdir(parents=True, exist_ok=True)
                (call / 'candidate.py').write_bytes(candidate)
                request['host_remaining_seconds'] = min(request['host_remaining_seconds'],
                    max(.001, deadline - time.monotonic()))
                (call / 'request.json').write_text(json.dumps(request))
                # A distinct source directory prevents timestamp/pyc cache reuse.
                sys.modules.pop('candidate', None)
                sys.path.insert(0, str(call))
                try:
                    result = execute(call, adapter, runtime=runtime)
                    if result.get('failure_stage') in ('infrastructure', 'budget'):
                        return
                    if any(token in result.get('error', '').lower() for token in
                           ('device-side assert', 'device lost', 'acl error', 'rt error')):
                        return
                finally:
                    sys.path.remove(str(call))
                    sys.modules.pop('candidate', None)
            time.sleep(.1)
    finally:
        if os.getpid() == phase_pid:
            if runtime.get('lease'):
                runtime['lease'].close()
            signal.signal(signal.SIGTERM, previous)
