"""Phase-local file requests over the installed remote-access transport."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import uuid

from blind_optimization_tools import save

TERMINAL = {'completed', 'failed', 'cancelled'}


def start(service, operation):
    tools = service.tools
    for path in sorted(tools.state.glob('*/receipt.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('state') not in TERMINAL:
            return service.observe(receipt['call_id'])
    context = json.loads((service.public / 'execution.json').read_text())
    if service.remaining(context) <= 0:
        return dict(state='failed', failure_stage='budget', retryable=False)
    tools._staging_deadline = context['utc_deadline']
    source = service.public / 'candidate.py'
    if source.is_symlink() or not source.is_file():
        raise ValueError('candidate must be a regular file')
    ident = uuid.uuid4().hex[:16]
    folder = tools.state / ident
    folder.mkdir()
    shutil.copyfile(source, folder / 'candidate.py')
    request = dict(operation=operation, utc_deadline=context['utc_deadline'],
        host_remaining_seconds=service.remaining(context),
        source_sha256=hashlib.sha256((folder / 'candidate.py').read_bytes()).hexdigest())
    save(folder / 'request.json', request)
    archive = folder / 'request.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name in ('candidate.py', 'request.json'):
            stream.add(folder / name, arcname=name)
    statefile = tools.state / 'phase.json'
    phase = json.loads(statefile.read_text()) if statefile.exists() else None
    if phase and phase['deadline'] != context['utc_deadline']:
        raise RuntimeError('phase workspace reused with another deadline')
    if phase and not phase.get('handle'):
        return dict(state='dispatch-unconfirmed', call_id=phase['first_call'],
                    recovery='Recover the original phase handle; never redispatch.')
    if phase:
        observed = tools.remote('observe', phase['handle'])
        if observed['state'] in TERMINAL:
            # Only confirmed terminal workers may be replaced, never live or
            # unavailable ones. The original phase deadline still applies.
            phase = None
    if phase is None:
        for args in (('targets',), ('capabilities', tools.target), ('preflight', tools.target)):
            result = tools.remote(*args)
            if args[0] == 'preflight' and result.get('state') != 'completed':
                raise RuntimeError('preflight unavailable')
        remote = tools.cwd.rstrip('/') + '/model-phase-' + ident
        phase = dict(remote_folder=remote, first_call=ident, deadline=context['utc_deadline'])
        staging = folder / 'phase'
        shutil.copytree(service.bundle, staging)
        (staging / 'inbox').mkdir()
        shutil.copyfile(archive, staging / 'inbox' / (ident + '.tar.gz'))
        (staging / 'execute-phase.py').write_text(
            'from pathlib import Path\nimport sys\nroot=Path(__file__).resolve().parent\n'
            'sys.path[:0]=[str(root/"scripts"),str(root/"skills/ascend-model-benchmarking/scripts")]\n'
            'import model_campaign_native as adapter\nfrom model_campaign_worker import phase\nphase(root,adapter)\n')
        bundle = folder / 'phase.tar.gz'
        with tarfile.open(bundle, 'w:gz') as stream:
            for path in staging.iterdir():
                stream.add(path, arcname=path.name)
    receipt = dict(call_id=ident, target=tools.target, runtime=tools.runtime, cwd=tools.cwd,
        state='prepared', request=request, warm=True,
        remote_folder=phase['remote_folder'] + '/calls/' + ident)
    save(folder / 'receipt.json', receipt)
    if phase.get('handle'):
        receipt['handle'] = phase['handle']
        save(folder / 'receipt.json', receipt)
        receipt['upload'] = tools.remote('upload', tools.target, archive,
            phase['remote_folder'] + '/inbox/' + ident + '.tar.gz')
        save(folder / 'receipt.json', receipt)
    else:
        save(statefile, phase)
        receipt['bootstrap'] = True
        receipt['upload'] = tools.remote('upload', tools.target, bundle,
                                        phase['remote_folder'] + '.tar.gz')
        save(folder / 'receipt.json', receipt)
    return service.observe(ident)


def launch(tools, receipt, folder):
    phasefile = tools.state / 'phase.json'
    phase = json.loads(phasefile.read_text())
    if phase.get('dispatch_intent'):
        return dict(receipt, state='dispatch-unconfirmed')
    upload = tools.transfer(receipt, 'upload')
    save(folder / 'receipt.json', receipt)
    if upload['state'] != 'completed':
        return dict(receipt, state='transfer-pending' if upload['state'] not in TERMINAL else 'failed')
    deadline = receipt['request']['utc_deadline']
    remaining = (datetime.fromisoformat(deadline) - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        receipt.update(state='failed', failure_stage='budget')
        save(folder / 'receipt.json', receipt)
        return receipt
    tools._staging_deadline = deadline
    script = '#!/bin/bash\nset -eu\nexport CAMPAIGN_REMAINING_SECONDS=' + shlex.quote(str(remaining))
    script += '\nmkdir -p ' + shlex.quote(phase['remote_folder'])
    script += '\ntar -xzf ' + shlex.quote(phase['remote_folder'] + '.tar.gz')
    script += ' -C ' + shlex.quote(phase['remote_folder'])
    script += '\ncd ' + shlex.quote(phase['remote_folder']) + '\nexec python execute-phase.py\n'
    (folder / 'run.sh').write_text(script)
    phase['dispatch_intent'] = receipt['dispatch_intent'] = True
    save(phasefile, phase)
    save(folder / 'receipt.json', receipt)
    try:
        dispatched = tools.remote('run', tools.target, '--runtime', tools.runtime,
            '--cwd', tools.cwd, '--file', folder / 'run.sh')
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
        return dict(receipt, state='dispatch-unconfirmed')
    if dispatched.get('handle'):
        phase['handle'] = receipt['handle'] = dispatched['handle']
        save(phasefile, phase)
        save(folder / 'receipt.json', receipt)
    return receipt


def observe(tools, ident):
    folder = tools.state / ident
    receipt = json.loads((folder / 'receipt.json').read_text())
    if not receipt.get('handle'):
        receipt = launch(tools, receipt, folder)
        if not receipt.get('handle'):
            return dict(receipt, next='experiment-tool observe ' + ident)
    if 'upload' in receipt and receipt['upload'].get('state') not in TERMINAL:
        transfer = tools.transfer(receipt, 'upload')
        save(folder / 'receipt.json', receipt)
        if transfer['state'] != 'completed':
            return dict(receipt, state='transfer-pending' if transfer['state'] not in TERMINAL else 'failed',
                        next='experiment-tool observe ' + ident)
    try:
        observed = tools.remote('observe', receipt['handle'])
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
        return dict(receipt, state='observation-unavailable', next='experiment-tool observe ' + ident)
    evidence = tools.workspace / 'evidence' / ident
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        download = receipt.get('download')
        if download and download.get('state') not in TERMINAL:
            download = tools.transfer(receipt, 'download')
            save(folder / 'receipt.json', receipt)
        if download is None:
            download = tools.remote('download', tools.target,
                receipt['remote_folder'] + '/evidence.tar.gz', evidence / 'evidence.tar.gz')
        if download.get('state') not in TERMINAL:
            receipt['download'] = download
            save(folder / 'receipt.json', receipt)
            return dict(receipt, state='running', next='experiment-tool observe ' + ident)
        if download['state'] != 'completed':
            receipt.pop('download', None)
            raise RuntimeError('evidence not yet available')
        if download.get('handle'):
            fetched = tools.remote('fetch', download['handle'], evidence / 'evidence.tar.gz')
            if fetched.get('state') != 'completed':
                return dict(receipt, state='running', next='experiment-tool observe ' + ident)
        with tarfile.open(evidence / 'evidence.tar.gz') as stream:
            for entry in stream.getmembers():
                if not entry.isfile() or not (evidence / entry.name).resolve().is_relative_to(evidence.resolve()):
                    raise ValueError('invalid evidence member')
            stream.extractall(evidence, filter='data')
        summary = json.loads((evidence / 'results/summary.json').read_text())
        receipt['state'] = summary['state']
        save(folder / 'receipt.json', receipt)
        return dict(summary, call_id=ident, handle=receipt['handle'])
    except (OSError, RuntimeError, ValueError, tarfile.TarError, subprocess.TimeoutExpired):
        receipt['state'] = observed['state'] if observed['state'] in TERMINAL else 'running'
        save(folder / 'receipt.json', receipt)
        return dict(receipt, next='experiment-tool observe ' + ident) if receipt['state'] not in TERMINAL else receipt
