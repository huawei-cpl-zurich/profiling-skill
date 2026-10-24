#!/usr/bin/env python3
"""Private host backend for interactive and terminal coding evaluations."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path
import subprocess
import sys
import tarfile

from simple_experiment_runner import CplRemote, ObservationLost, _write_json

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'completed', 'failed', 'cancelled', 'missing'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invalid(request, cases, reason):
    return dict(schema='optimization-coding-evaluation-v1',
                measurement_kind='profiled_device_operation_span_us',
                candidate_sha256=request['candidate_sha256'], outcome='infrastructure_failure',
                error=reason, cases=[dict(id=c['id'], correct=False, identity_verified=False,
                                        status='not-run') for c in cases])


def evaluate(config, request, *, remote=None):
    """A cache intent is never redispatched; an existing handle is only observed."""
    remote = remote or CplRemote()
    cases = config['cases'][request['phase']]
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('distinct configured cases required')
    source = Path(request['candidate_path'])
    if source.is_symlink() or sha(source) != request['candidate_sha256']:
        raise ValueError('submitted candidate digest mismatch')
    source = source.resolve()
    for case in cases:
        values = case['baseline_samples_us']
        if len(values) != 10 or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('ten trusted baseline samples required per case')
    worker = ROOT / 'scripts/optimization_demo_worker.py'
    helpers = [worker, ROOT/'scripts/optimization_pair.py', ROOT/'scripts/optimization_profile_evidence.py',
               *(ROOT/'skills/ascend-model-benchmarking/scripts').glob('*.py')]
    identity = dict(config=config, phase=request['phase'], source=request['candidate_sha256'],
                    helpers={str(p.relative_to(ROOT)):sha(p) for p in helpers})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(config['cache_directory']).resolve() / key
    cache.mkdir(parents=True, exist_ok=True)
    with (cache/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt_path, report_path = cache/'receipt.json', cache/'report.json'
        if report_path.exists():
            return json.loads(report_path.read_text())
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        try:
            if receipt and not receipt.get('handle'):
                transfer = receipt.get('transfer', {})
                if receipt.get('state') == 'transfer-intent' and transfer.get('handle'):
                    observed = remote._call(['observe',transfer['handle'],'--wait'])
                    remote._identity(observed, transfer['handle'], 'transfer observation')
                    receipt['transfer_observation'] = observed
                    if observed.get('state') == 'completed': receipt['state'] = 'uploaded'
                    _write_json(receipt_path, receipt)
                if receipt.get('state') != 'uploaded':
                    return invalid(request, cases, 'uncertain dispatch or transfer; inspect retained intent, never resubmit')
            if not receipt:
                remote.preflight(dict(target=config['target']))
                payload = dict(request, cases=cases, operation_id=config['operation_id'],
                    provider_path=config['provider_path'], provider_sha256=config['provider_sha256'],
                    physical_die=config['physical_die'], device_query=config['device_query'],
                    lease_directory=config['lease_directory'], export_directory=config.get('export_directory',str(Path(config['lease_directory'])/'exports')),
                    runtime_identity=config.get('runtime_identity',{}))
                bundle = cache/'bundle'; (bundle/'scripts').mkdir(parents=True)
                (bundle/'candidate.py').write_bytes(source.read_bytes())
                for name in ('optimization_demo_worker.py','optimization_pair.py','optimization_profile_evidence.py'):
                    (bundle/'scripts'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
                helpers = bundle/'skills/ascend-model-benchmarking/scripts'; helpers.mkdir(parents=True)
                for path in (ROOT/'skills/ascend-model-benchmarking/scripts').glob('*.py'):
                    (helpers/path.name).write_bytes(path.read_bytes())
                _write_json(bundle/'request.json', payload)
                archive = cache/('demo-'+key+'.tar.gz')
                with tarfile.open(archive, 'w:gz') as tar:
                    for path in bundle.rglob('*'):
                        if path.is_file(): tar.add(path, arcname=str(path.relative_to(bundle)))
                receipt.update(state='transfer-intent', archive_sha256=sha(archive))
                _write_json(receipt_path, receipt)
                transfer = remote._call(['upload',config['target'],str(archive),
                    str(Path(config['remote_workspace'])/archive.name)])
                receipt['transfer'] = transfer; _write_json(receipt_path, receipt)
                if transfer.get('state') != 'completed':
                    if transfer.get('handle'):
                        receipt['transfer_observation'] = remote._call(['observe',transfer['handle'],'--wait'])
                        _write_json(receipt_path, receipt)
                    if receipt.get('transfer_observation',{}).get('state') != 'completed':
                        raise ObservationLost('transfer incomplete; retain its handle')
                receipt['state']='uploaded'; _write_json(receipt_path, receipt)
            if not receipt.get('handle'):
                archive = cache/('demo-'+key+'.tar.gz')
                if sha(archive) != receipt['archive_sha256']:
                    raise ValueError('retained upload archive changed')
                destination = str(Path(config['remote_root'])/key)
                script = cache/'run.py'
                script.write_text("#!/usr/bin/env python3\nimport hashlib,json,sys,tarfile,signal\nfrom pathlib import Path\n"
                    "signal.alarm(600)\n" + f"archive=Path({archive.name!r});root=Path({destination!r})\n"
                    + f"assert hashlib.sha256(archive.read_bytes()).hexdigest()=={receipt['archive_sha256']!r}\n"
                    "root.mkdir(parents=True,exist_ok=False)\n"
                    "with tarfile.open(archive) as t:t.extractall(root,filter='data')\n"
                    "sys.path.insert(0,str((root/'scripts').resolve()))\n"
                    "from optimization_demo_worker import remote_main\n"
                    "request=json.loads((root/'request.json').read_text())\n"
                    "request['candidate_path']=str((root/'candidate.py').resolve())\n"
                    "request['output_directory']=str((root/'evidence').resolve())\nremote_main(request)\n")
                receipt['state'] = 'dispatch-intent'; _write_json(receipt_path, receipt)
                receipt['handle'] = remote.submit(dict(target=config['target'], runtime=config['runtime'],
                    remote_workspace=config['remote_workspace']), script)
                _write_json(receipt_path, receipt)
            deadline = time.monotonic() + config.get('observation_seconds', 500)
            while True:
                observed = remote._call(['observe',receipt['handle'],'--wait','--timeout','50'])
                remote._identity(observed, receipt['handle'], 'observe')
                receipt['observation'] = observed; _write_json(receipt_path, receipt)
                if observed['state'] in TERMINAL: break
                if time.monotonic() >= deadline:
                    return invalid(request, cases, 'retained job is nonterminal; observe this same candidate again')
            stdout = remote._logs(receipt['handle'], 'stdout')
            (cache/'remote.stdout').write_text(stdout)
            lines = [line[len('DEMO_RESULT '):] for line in stdout.splitlines() if line.startswith('DEMO_RESULT ')]
            if not lines:
                report = invalid(request, cases, 'terminal job has no trusted result; see retained handle/logs')
            else:
                report = json.loads(lines[-1])
                if (report.get('candidate_sha256') != request['candidate_sha256']
                        or report.get('schema') != 'optimization-coding-evaluation-v1'
                        or report.get('measurement_kind') != 'profiled_device_operation_span_us'
                        or len(report.get('cases',[])) != len(cases)
                        or {c.get('id') for c in report['cases']} != {c['id'] for c in cases}):
                    raise ValueError('remote result source/case/measurement identity differs')
                if report.get('outcome') == 'success' and (observed['state'] != 'completed' or observed.get('exit') != 0):
                    report = invalid(request, cases, 'contradictory remote success and terminal state')
            report['remote_handle'] = receipt['handle']; report['cache_key'] = key
            _write_json(report_path, report)
            return report
        except Exception as error:
            receipt.update(error_type=type(error).__name__, error=str(error))
            _write_json(receipt_path, receipt)
            return invalid(request, cases, str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('action', nargs='?', choices=['evaluate'])
    args = parser.parse_args()
    if args.workspace:
        candidate = args.workspace.resolve()/'candidate.py'
        request = dict(action='evaluate', phase='development', candidate_path=str(candidate),
                       candidate_sha256=sha(candidate))
    else:
        request = json.load(sys.stdin)
    if request.get('action') != 'evaluate' or request.get('phase') not in ('development','terminal'):
        parser.error('evaluate with a development or terminal phase required')
    config = json.loads(args.config.read_text())
    if request.get('task_id',config['task_id']) != config['task_id']:
        parser.error('private task identity differs')
    print(json.dumps(evaluate(config, request), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
