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
    devices = getattr(tools, 'devices', None)
    if phase and not phase.get('handle'):
        return dict(state='dispatch-unconfirmed', call_id=phase['first_call'],
                    recovery='Recover the original phase handle; never redispatch.')
    if phase:
        observed = tools.remote('observe', phase['handle'])
        if phase.get('devices') != devices and observed['state'] not in TERMINAL:
            raise ValueError('device allowance differs from retained phase; observe/cancel its handle first')
        if phase['deadline'] != context['utc_deadline'] and observed['state'] not in TERMINAL:
            return dict(state='phase-cleanup-required', handle=phase['handle'],
                        diagnostic=dict(kind='previous-phase-not-terminal'))
        if observed['state'] in TERMINAL:
            stopped = shutdown_phase(tools)
            if stopped.get('state') != 'terminal' or stopped.get('worker_closed') is not True:
                return dict(state='phase-cleanup-required', handle=phase['handle'],
                            worker_shutdown=stopped)
            # Only confirmed terminal workers may be replaced, never live or
            # unavailable ones. The original phase deadline still applies.
            statefile.replace(tools.state / phase['first_call'] / 'phase.json')
            phase = None
    if phase is None:
        try:
            for args in (('targets',), ('capabilities', tools.target), ('preflight', tools.target)):
                result = tools.remote(*args)
                accepted = {'completed', 'available'} if args[0] == 'capabilities' else {'completed'}
                if result.get('state') not in accepted:
                    raise RuntimeError('discovery unavailable')
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            receipt = dict(call_id=ident, target=tools.target, runtime=tools.runtime, cwd=tools.cwd,
                state='failed', request=request, warm=True, dispatch_attempted=False,
                failure_stage='budget' if isinstance(error, TimeoutError) else 'infrastructure')
            save(folder / 'receipt.json', receipt)
            return service.publish(receipt)
        remote = tools.cwd.rstrip('/') + '/model-phase-' + ident
        phase = dict(remote_folder=remote, first_call=ident, deadline=context['utc_deadline'],
                     activation_required=True, devices=devices, runtime=tools.runtime)
        staging = folder / 'phase'
        shutil.copytree(service.bundle, staging)
        (staging / 'inbox').mkdir()
        shutil.copyfile(archive, staging / 'inbox' / (ident + '.tar.gz'))
        (staging / 'execute-phase.py').write_text(
            'from pathlib import Path\nimport os,sys\nroot=Path(__file__).resolve().parent\n'
            'sys.path[:0]=[str(root/"scripts"),str(root/"skills/ascend-model-benchmarking/scripts")]\n'
            'from phase_deadline import wait_for_activation\n'
            'os.environ["CAMPAIGN_DEADLINE_MONOTONIC"]=str(wait_for_activation(root/"activation.json",'
            'float(os.environ["CAMPAIGN_REMAINING_SECONDS"])))\n'
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
        try:
            receipt['upload'] = tools.remote('upload', tools.target, archive,
                phase['remote_folder'] + '/inbox/' + ident + '.tar.gz')
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            receipt['failure_stage'] = 'budget' if isinstance(error, TimeoutError) else 'infrastructure'
            return service.publish(cancel_request(tools, receipt, folder))
        save(folder / 'receipt.json', receipt)
    else:
        save(statefile, phase)
        receipt['bootstrap'] = True
        try:
            receipt['upload'] = tools.remote('upload', tools.target, bundle,
                                            phase['remote_folder'] + '.tar.gz')
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            receipt.update(state='failed', failure_stage=
                           'budget' if isinstance(error, TimeoutError) else 'infrastructure')
            statefile.replace(folder / 'phase.json')
            save(folder / 'receipt.json', receipt)
            return service.publish(receipt)
        save(folder / 'receipt.json', receipt)
    return service.observe(ident)


def cancel_request(tools, receipt, folder):
    """An uncertain request transfer cannot leave an executable late request."""
    try:
        result = tools.remote('cancel', receipt['handle'])
        receipt['state'] = 'failed' if result.get('state') in TERMINAL else 'observation-unavailable'
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
        receipt['state'] = 'observation-unavailable'
    receipt.setdefault('failure_stage', 'infrastructure')
    save(folder / 'receipt.json', receipt)
    return receipt


def launch(tools, receipt, folder):
    phasefile = tools.state / 'phase.json'
    phase = json.loads(phasefile.read_text())
    if phase.get('dispatch_intent'):
        return dict(receipt, state='dispatch-unconfirmed')
    upload = tools.transfer(receipt, 'upload')
    save(folder / 'receipt.json', receipt)
    if upload['state'] != 'completed':
        if upload['state'] in TERMINAL:
            receipt.update(state='failed', failure_stage='infrastructure')
            phasefile.replace(folder / 'phase.json')
            save(folder / 'receipt.json', receipt)
        return dict(receipt, state='transfer-pending' if upload['state'] not in TERMINAL else 'failed')
    deadline = receipt['request']['utc_deadline']
    remaining = (datetime.fromisoformat(deadline) - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        receipt.update(state='failed', failure_stage='budget')
        phasefile.replace(folder / 'phase.json')
        save(folder / 'receipt.json', receipt)
        return receipt
    tools._staging_deadline = deadline
    script = '#!/bin/bash\nset -eu\nexport CAMPAIGN_REMAINING_SECONDS=' + shlex.quote(str(remaining))
    if phase.get('devices') is not None:
        script += '\nexport CAMPAIGN_ALLOWED_DEVICES=' + shlex.quote(json.dumps(phase['devices']))
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
    evidence = tools.workspace / 'evidence' / ident
    if (evidence / 'results/summary.json').is_file():
        summary = json.loads((evidence / 'results/summary.json').read_text())
        return dict(summary, call_id=ident, handle=receipt.get('handle'))
    if not receipt.get('handle'):
        receipt = launch(tools, receipt, folder)
        if not receipt.get('handle'):
            return receipt if receipt.get('state') in TERMINAL else dict(
                receipt, next='experiment-tool observe ' + ident)
    phase = json.loads((tools.state / 'phase.json').read_text())
    if phase.get('activation_required'):
        from phase_deadline import activate
        activation_folder = tools.state / phase['first_call'] / 'activation'
        activation_folder.mkdir(exist_ok=True)
        activation_receipt = activation_folder / 'receipt.json'
        activation = json.loads(activation_receipt.read_text()) if activation_receipt.exists() else dict(
            target=tools.target, handle=phase['handle'], remote_folder=phase['remote_folder'],
            request=dict(utc_deadline=phase['deadline'], activation_required=True))
        save(activation_receipt, activation)
        try:
            ready = activate(tools, activation_folder, activation)
        except TimeoutError:
            tools.remote('cancel', phase['handle'])
            receipt.update(state='failed', failure_stage='budget')
            save(folder / 'receipt.json', receipt)
            return receipt
        if not ready:
            return dict(receipt, state='running', next='experiment-tool observe ' + ident)
    if 'upload' in receipt and receipt['upload'].get('state') not in TERMINAL:
        transfer = tools.transfer(receipt, 'upload')
        save(folder / 'receipt.json', receipt)
        if transfer['state'] != 'completed':
            if transfer['state'] in TERMINAL:
                return cancel_request(tools, receipt, folder)
            return dict(receipt, state='transfer-pending' if transfer['state'] not in TERMINAL else 'failed',
                        next='experiment-tool observe ' + ident)
    if receipt.get('upload', {}).get('state') in {'failed', 'cancelled'}:
        return cancel_request(tools, receipt, folder)
    try:
        observed = tools.remote('observe', receipt['handle'])
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
        return dict(receipt, state='observation-unavailable', next='experiment-tool observe ' + ident)
    evidence.mkdir(parents=True, exist_ok=True)
    terminal_evidence_failure = False
    try:
        download = receipt.get('download')
        if download and download.get('state') not in TERMINAL:
            download = tools.transfer(receipt, 'download')
            save(folder / 'receipt.json', receipt)
        if download is None:
            download = tools.remote('download', tools.target,
                receipt['remote_folder'] + '/evidence.tar.gz', evidence / 'evidence.tar.gz')
            receipt['download'] = download
            save(folder / 'receipt.json', receipt)
        if download.get('state') not in TERMINAL:
            receipt['download'] = download
            save(folder / 'receipt.json', receipt)
            return dict(receipt, state='running', next='experiment-tool observe ' + ident)
        if download['state'] != 'completed':
            terminal_evidence_failure = True
            receipt.pop('download', None)
            raise RuntimeError('evidence not yet available')
        if download.get('handle'):
            fetched = tools.remote('fetch', download['handle'], evidence / 'evidence.tar.gz')
            if fetched.get('state') != 'completed':
                return dict(receipt, state='running', next='experiment-tool observe ' + ident)
        terminal_evidence_failure = True  # Completed transfer: any remaining error is local evidence.
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
        if observed['state'] == 'completed' and terminal_evidence_failure:
            # A completed phase cannot produce a missing call report later.
            # Preserve its handle, but do not advertise endless evidence_pending.
            receipt.update(state='failed', failure_stage='infrastructure',
                           diagnostic=dict(kind='terminal-evidence-unavailable',
                               phase_state='completed'))
        elif observed['state'] == 'completed':
            receipt['state'] = 'observation-unavailable'
        save(folder / 'receipt.json', receipt)
        return dict(receipt, next='experiment-tool observe ' + ident) if receipt['state'] not in TERMINAL else receipt


def _shutdown_phase(tools):
    """Stop the actual owned worker even if its outer job reports terminal."""
    phasefile = tools.state / 'phase.json'
    if not phasefile.exists():
        return dict(state='terminal', worker_closed=True)
    phase = json.loads(phasefile.read_text())
    control = phase.get('shutdown', {})
    if not control.get('handle'):
        if control.get('dispatch_intent'):
            return dict(state='observation-required', worker_closed=False, reason='shutdown dispatch handle unknown')
        source = Path(__file__).with_name('model_phase_shutdown.py').read_text()
        script = tools.state / 'shutdown-phase.sh'
        script.write_text("#!/bin/bash\nset -eu\npython - " + shlex.quote(phase['remote_folder'])
                          + " <<'PY'\n" + source + "\nPY\n")
        phase['shutdown'] = dict(dispatch_intent=True)
        save(phasefile, phase)
        control = tools.remote('run', tools.target, '--runtime', tools.runtime,
                               '--cwd', tools.cwd, '--file', script)
        phase['shutdown'].update(control)
        save(phasefile, phase)
    else:
        control = tools.remote('observe', control['handle'], '--wait', '--timeout', '15')
        phase['shutdown'].update(control)
        save(phasefile, phase)
    if control.get('handle') and control.get('state') not in TERMINAL:
        control = tools.remote('observe', control['handle'], '--wait', '--timeout', '15')
        phase['shutdown'].update(control)
        save(phasefile, phase)
    acknowledgment = phase['shutdown'].get('acknowledgment')
    if control.get('state') == 'completed' and control.get('exit') == 0 and not acknowledgment:
        logs = tools.remote('logs', phase['shutdown']['handle'], '--tail', '20')
        for line in reversed(logs.get('content', '').splitlines()):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get('state') == 'terminal' and value.get('worker_closed') is True:
                acknowledgment = value
                phase['shutdown']['acknowledgment'] = value
                save(phasefile, phase)
                break
    closed = bool(acknowledgment and acknowledgment.get('worker_closed') is True)
    return dict(state='terminal' if closed else 'observation-required', worker_closed=closed,
                control_handle=phase['shutdown'].get('handle'), acknowledgment=acknowledgment)


def shutdown_phase(tools):
    try:
        return _shutdown_phase(tools)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        return dict(state='observation-required', worker_closed=False, diagnostic=type(error).__name__)


def shutdown_owned_phase(phasefile):
    """Controller boundary: recover registered transport from an owned receipt."""
    from blind_optimization_tools import Tools
    phasefile = Path(phasefile)
    phase = json.loads(phasefile.read_text())
    target = phase['handle'].split(':')[1]
    tools = Tools(phasefile.parent, phasefile.parent / 'shutdown-private', target,
                  phase.get('runtime', 'py311-torch'))
    tools.state = phasefile.parent
    return shutdown_phase(tools)
