"""Trusted retained worker; task adapters and all evidence stay outside sandboxes."""
from datetime import datetime, timezone
import json
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


def execute(root, adapter, *, acquire=None, monotonic=time.monotonic):
    """Adapter check(root,env,remaining) must validate variants 0 and 1.

    It returns {passed: bool, variants: [0,1], ...} with candidate-only safe
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
                   source_sha256=request['source_sha256'])
    lease = None
    try:
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
        if acquire is None:
            from model_device import acquire as select
            physical, env, lease = select(probe, remaining)
        else:
            physical, env, lease = acquire(probe)
        os.environ.update(env)
        summary.update(physical_device=physical, logical_device=0, failure_stage='candidate')
        summary['progress'] = 'correctness-check'
        print('MODEL_STAGE=correctness-check', flush=True)
        checked = adapter.check(root, env, remaining)
        summary['check'] = checked
        if checked.get('passed') is not True or set(checked.get('variants', [])) != {0, 1}:
            return summary
        if request['operation'] != 'check':
            summary['failure_stage'] = 'measurement'
            summary['progress'] = request['operation']
            print('MODEL_STAGE=' + request['operation'], flush=True)
            summary['result'] = getattr(adapter, request['operation'])(root, env, remaining)
            if summary['result'].get('state') not in ('complete', 'completed'):
                return summary
        summary.update(state='completed')
        summary.pop('failure_stage')
        return summary
    except Exception as error:
        summary.update(error=str(error), traceback=traceback.format_exc())
        summary['diagnostic'] = dict(kind='worker-failure', error_type=type(error).__name__,
                                    stage=summary.get('progress', 'device-acquisition'))
        if isinstance(error, TimeoutError):
            summary['failure_stage'] = 'budget'
        return summary
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if lease:
            lease.close()
        output = root / 'results'
        output.mkdir(exist_ok=True)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        # Bundle compact evidence only. Raw profile payloads and all providers
        # remain remote; even this private archive is never directly agent-mounted.
        with tarfile.open(root / 'evidence.tar.gz', 'w:gz') as archive:
            for path in root.rglob('*'):
                if path.is_file() and path.suffix in ('.json', '.csv', '.stdout', '.stderr', '.log'):
                    if path.is_relative_to(output) or path.is_relative_to(root / 'measurement') or path.is_relative_to(root / 'diagnostics'):
                        archive.add(path, arcname=str(path.relative_to(root)))
