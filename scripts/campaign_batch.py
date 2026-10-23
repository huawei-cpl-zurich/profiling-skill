"""Trusted nine-cell admission and fail-stop decisions, independent of transport.

Persist the returned state before dispatch/cancellation. Queued cells have no
role deadline: invoke the controller only after admission. Cancellation must
use the supplied skill-owned cpl-remote boundary and retained owned handles.
"""
from copy import deepcopy

CONFIGURATIONS = ('cannbot', 'cannbot-profile', 'guarded-profile')
CELLS = tuple(f'{task}/{config}' for task in ('matmul', 'gdn', 'bsa') for config in CONFIGURATIONS)
NONTERMINAL = ('running', 'reconnecting', 'observation-unavailable', 'observation-required')


def gate(state, round_number=1):
    actor = state.get('phases', {}).get(f'round-{round_number}-actor', {})
    analyzer = state.get('phases', {}).get(f'round-{round_number}-analyzer', {})
    return (actor.get('state') == analyzer.get('state') == 'terminal'
            and actor.get('candidate_commit') == analyzer.get('candidate_commit')
            and bool(actor.get('candidate_commit')) and analyzer.get('analysis_valid') is True
            and all(entry.get('cleanup', {}).get('state') == 'terminal'
                    for entry in (actor, analyzer)))


def failure(state):
    if state.get('state') in ('failed', 'cancelled'):
        return 'terminal cell failure'
    for name, entry in state.get('phases', {}).items():
        if name.endswith('-analyzer') and entry.get('state') == 'skipped-no-verified-candidate':
            return 'no verified candidate'
        if entry.get('state') != 'terminal':
            continue
        if name.endswith('-actor') and entry.get('candidate_verified') is False:
            return 'unverified candidate'
        result = entry.get('result', {})
        if name.endswith('-analyzer') and (entry.get('analysis_valid') is not True
                or result.get('exit_code', 0) != 0):
            return 'invalid or failed analyzer'
    if state.get('state') == 'completed' and not gate(state, 4):
        return 'completed without verified final analysis'
    return None


def reconcile(previous, observations, capacity):
    """Return durable decisions; never dispatch, cancel, or reset a role clock."""
    if capacity < 0:
        raise ValueError('capacity must be nonnegative')
    unknown = set(observations) - set(CELLS)
    if unknown:
        raise ValueError('unknown campaign cells')
    state = deepcopy(previous or dict(released=False, stopped=False, admitted=[], handles={}, cancelled=[]))
    state['admit'] = []
    for cell in CELLS[:3]:
        reason = failure(observations.get(cell, {}))
        if reason:
            state.update(stopped=True, reason=f'{cell}: {reason}')
    if state['stopped']:
        state['cancel'] = sorted({handle for handles in state['handles'].values() for handle in handles}
                                 - set(state['cancelled']))
        return state
    if all(gate(observations.get(cell, {})) for cell in CELLS[:3]):
        state['released'] = True
    state.setdefault('resumed', [])
    active = [cell for cell in state['admitted'] if observations.get(cell, {}).get('state')
              not in ('completed', 'failed', 'cancelled', 'checkpoint')
              or (cell in state['resumed'] and observations.get(cell, {}).get('state') == 'checkpoint')]
    eligible = CELLS if state['released'] else CELLS[:3]
    for cell in eligible:
        observed = observations.get(cell, {})
        resume = observed.get('state') == 'checkpoint' and state['released'] and cell not in active and cell not in state['resumed']
        if cell in state['admitted'] and not resume:
            continue
        if len(active) >= capacity:
            break
        state['admit'].append(dict(cell=cell, stop_after_round=None if state['released'] else 1))
        if resume:
            state['resumed'].append(cell)
        if cell not in state['admitted']:
            state['admitted'].append(cell)
        active.append(cell)
    state['cancel'] = []
    return state


def cancel_owned(state, cancel):
    """Call cancel(handle) for batch-owned handles; retain failures for observation."""
    for handle in state.get('cancel', []):
        result = cancel(handle)
        if result.get('state') in ('completed', 'failed', 'cancelled'):
            state['cancelled'].append(handle)
    return state
