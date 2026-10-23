"""Acquire the required current-source profile before launching the analyzer."""
from datetime import datetime, timedelta
import json
import time


TERMINAL = {'completed', 'failed', 'cancelled', 'unsupported', 'infrastructure-unavailable'}


def acquire(launcher, run, save, *, clock=time.monotonic, pause=time.sleep):
    """Use only existing managed tools; preserve intent before any dispatch.

    The remote tool receives a subdeadline ending at reporting reserve. The
    original execution context is restored for the analyzer, whose launch uses
    only the remainder of that same original deadline.
    """
    context_path = launcher.campaign.history / 'context.json'
    context = json.loads(context_path.read_text())
    if context['status'] == 'unchanged':
        return
    path = launcher.root / 'required-profile.json'
    execution_path = launcher.workspace / 'execution.json'
    original = json.loads(execution_path.read_text())
    state = json.loads(path.read_text()) if path.exists() else dict(
        source_sha256=context['source_sha256'], state='prepared', dispatch_intent=False)
    if state['source_sha256'] != context['source_sha256']:
        raise ValueError('required profile source identity changed')
    cutoff = launcher.deadline_monotonic - launcher.finish_reserve_seconds
    # execute() allows five seconds for process-group termination after timeout.
    stop_grace = 6.
    bounded = dict(original, utc_deadline=(datetime.fromisoformat(original['utc_deadline']) -
        timedelta(seconds=launcher.finish_reserve_seconds)).isoformat(),
        original_utc_deadline=original['utc_deadline'])

    def recover_call():
        matches = []
        for receipt_path in (launcher.root / 'tool-state').glob('*/receipt.json'):
            receipt = json.loads(receipt_path.read_text())
            request = receipt.get('request', {})
            if request.get('source_sha256') == state['source_sha256'] and request.get(
                    'verb', request.get('operation')) == 'pipes':
                matches.append((receipt_path.parent.name, receipt))
        if len(matches) == 1:
            state.update(call_id=matches[0][0], handle=matches[0][1].get('handle'))

    try:
        save(execution_path, bounded)
        if state['dispatch_intent'] and not state.get('call_id'):
            recover_call()
            if not state.get('call_id'):
                state.update(state='observation-required', reason='Existing dispatch intent has no recoverable call; do not redispatch')
                return
        while state['state'] not in TERMINAL and clock() < cutoff - stop_grace:
            arguments = ['observe', state['call_id']] if state.get('call_id') else ['pipes']
            state['dispatch_intent'] = True
            save(path, state)
            command = [*launcher.tool_command, '--workspace', str(launcher.workspace), *arguments]
            result = run(command, launcher.workspace, max(.001, cutoff - clock() - stop_grace))
            state['command_result'] = {key: result[key] for key in ('returncode', 'timed_out', 'stderr')}
            state['command_dispatched'] = True
            if arguments == ['pipes']:
                state['profile_attempt_arguments'] = arguments
            try:
                payload = json.loads(result['stdout'])
            except ValueError:
                payload = dict(state='observation-required', reason='Tool response unavailable; recover original receipt')
            state.update({key: payload[key] for key in ('state', 'call_id', 'handle', 'reason', 'error') if key in payload})
            recover_call()
            with launcher.journal.open('a') as journal:
                journal.write(json.dumps(dict(controller_required_profile=True, command=command, **result)) + '\n')
            save(path, state)
            if state['state'] in TERMINAL or not state.get('call_id') or result['timed_out']:
                break
            pause(min(1., max(0., cutoff - clock())))
        if state['state'] not in TERMINAL:
            state.update(state='observation-required', reason=state.get('reason', 'Acquisition allowance ended; retain same call for observation'))
        if state.get('call_id'):
            summary = launcher.workspace / 'evidence' / state['call_id'] / 'results/summary.json'
            if summary.is_file():
                state['evidence'] = str(summary.relative_to(launcher.workspace))
    finally:
        save(execution_path, original)
        save(path, state)
        context['required_profile'] = state
        save(context_path, context)
