"""Trusted retained worker; task adapters and all evidence stay outside sandboxes."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from profile_device import bounded_run, wait_for_device


def execute(root, adapter, *, acquire=wait_for_device, monotonic=time.monotonic):
    """Adapter check(root,env,remaining) must validate variants 0 and 1.

    It returns {passed: bool, variants: [0,1], ...} with candidate-only safe
    diagnostics from candidate_probe. benchmark/pipes/screen have the same
    signature and may return only private artifacts; export happens on host.
    Every adapter subprocess must use remaining() as its timeout.
    """
    root = Path(root)
    request = json.loads((root / 'request.json').read_text())
    deadline = datetime.fromisoformat(request['utc_deadline'])
    budget = min(float(request['host_remaining_seconds']),
                 float(os.environ.get('CAMPAIGN_REMAINING_SECONDS', request['host_remaining_seconds'])))
    started = monotonic()
    def remaining():
        return max(0., min((deadline - datetime.now(timezone.utc)).total_seconds(),
                          budget - (monotonic() - started)))
    def expire(*_):
        raise TimeoutError('original round deadline expired')
    summary = dict(state='failed', failure_stage='infrastructure', operation=request['operation'],
                   source_sha256=request['source_sha256'])
    lease = None
    try:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, remaining() or .001)
        def probe(env):
            result = bounded_run([sys.executable, '-c',
                'import torch,torch_npu;torch.npu.set_device(0);torch.ones(1,device="npu");torch.npu.synchronize()'],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=min(30., remaining()))
            return result.returncode == 0, result.stderr[-2000:]
        physical, env, lease = acquire(probe)
        os.environ.update(env)
        summary.update(physical_device=physical, logical_device=0, failure_stage='candidate')
        checked = adapter.check(root, env, remaining)
        summary['check'] = checked
        if checked.get('passed') is not True or set(checked.get('variants', [])) != {0, 1}:
            return summary
        if request['operation'] != 'check':
            summary['failure_stage'] = 'measurement'
            summary['result'] = getattr(adapter, request['operation'])(root, env, remaining)
            if summary['result'].get('state') not in ('complete', 'completed'):
                return summary
        summary.update(state='completed')
        summary.pop('failure_stage')
        return summary
    except Exception as error:
        summary.update(error=str(error), traceback=traceback.format_exc())
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
