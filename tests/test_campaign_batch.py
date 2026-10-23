import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_batch import CELLS, reconcile, cancel_owned


def ready(round_number=1):
    return dict(state='checkpoint', phases={
        f'round-{round_number}-actor': dict(state='terminal', candidate_commit='abc', cleanup=dict(state='terminal')),
        f'round-{round_number}-analyzer': dict(state='terminal', candidate_commit='abc', analysis_valid=True, cleanup=dict(state='terminal'))})


def test_capacity_and_gate_release():
    state = reconcile(None, {}, 2)
    assert [v['cell'] for v in state['admit']] == list(CELLS[:2])
    assert all(v['stop_after_round'] == 1 for v in state['admit'])
    observed = {cell: ready() for cell in CELLS[:2]}
    state = reconcile(state, observed, 2)
    assert state['admit'] == [dict(cell=CELLS[2], stop_after_round=1)]
    observed[CELLS[2]] = ready()
    state = reconcile(state, observed, 4)
    assert state['released']
    assert len(state['admit']) == 4
    assert all(v['stop_after_round'] is None for v in state['admit'])


@pytest.mark.parametrize('status', ['running', 'reconnecting', 'observation-unavailable'])
def test_observation_loss_never_duplicates(status):
    state = reconcile(None, {}, 1)
    again = reconcile(state, {CELLS[0]: dict(state=status)}, 1)
    assert again['admit'] == [] and not again['stopped']


def test_matmul_failure_freezes_and_cancels_only_owned_handles():
    state = reconcile(None, {}, 3)
    state['handles'] = {CELLS[0]: ['remote:gz-a3:job:ours']}
    bad = ready()
    bad['phases']['round-1-analyzer']['analysis_valid'] = False
    state = reconcile(state, {CELLS[0]: bad}, 9)
    assert state['stopped'] and state['admit'] == []
    calls = []
    def cancel(handle):
        calls.append(handle)
        return dict(state='reconnecting')
    cancel_owned(state, cancel)
    assert state['cancelled'] == []
    cancel_owned(state, lambda handle: dict(state='cancelled'))
    assert reconcile(state, {}, 9)['cancel'] == []
    assert calls == ['remote:gz-a3:job:ours']


def test_checkpoint_admission_is_durable_before_observation_changes():
    observations = {cell: ready() for cell in CELLS[:3]}
    state = reconcile(None, observations, 1)
    first = state['admit'][0]['cell']
    assert reconcile(state, observations, 1)['admit'] == []
    assert first in state['resumed']


def test_failed_candidate_is_terminal_even_when_older_candidate_exists():
    bad = ready()
    bad['phases']['round-1-actor']['candidate_verified'] = False
    assert reconcile(None, {CELLS[0]: bad}, 9)['stopped']


@pytest.mark.parametrize('role', ['actor', 'analyzer'])
@pytest.mark.parametrize('cleanup', [None, 'observation-required'])
def test_first_round_gate_waits_for_each_worker_cleanup(role, cleanup):
    observations = {cell: ready() for cell in CELLS[:3]}
    pending = observations[CELLS[0]]
    pending['state'] = 'observation-required'
    entry = pending['phases']['round-1-' + role]
    entry['cleanup'] = {} if cleanup is None else dict(state=cleanup)
    state = reconcile(None, observations, 9)
    assert not state['released'] and not state['stopped']
    assert all(row['cell'].startswith('matmul/') for row in state['admit'])
    entry['cleanup'] = dict(state='terminal')
    pending['state'] = 'checkpoint'
    state = reconcile(state, observations, 9)
    assert state['released']
    assert any(row['cell'].startswith('gdn/') for row in state['admit'])
