"""Exercise transport path conversion through the managed CLI boundary."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('campaign_tools_paths',
    Path(__file__).parents[1] / 'scripts/blind_optimization_tools.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def tools(tmp_path, backend='remote-agent'):
    result = object.__new__(module.Tools)
    result.cwd = '/configured/workspace'
    result.backend, result.target = backend, 'registered-target'
    result.state = tmp_path
    result.client = tmp_path / 'managed-client'
    result.client.write_text('#!/usr/bin/env python3\nimport json,sys\n'
        'print(json.dumps({"state":"completed","arguments":sys.argv[1:]}))\n')
    result.client.chmod(0o700)
    return result


@pytest.mark.parametrize(('arguments', 'expected'), [
    (['run', 'registered-target', '--runtime', 'named', '--cwd', '/configured/workspace', '--file', '/local/script'],
     ['run', 'registered-target', '--runtime', 'named', '--cwd', '.cpl-remote', '--file', '/local/script']),
    (['run', 'registered-target', '--cwd', '/configured/workspace/phase'],
     ['run', 'registered-target', '--cwd', 'phase']),
    (['upload', 'registered-target', '/local/bundle', '/configured/workspace/phase.tar.gz'],
     ['upload', 'registered-target', '/local/bundle', 'phase.tar.gz']),
    (['download', 'registered-target', '/configured/workspace/phase/evidence.tar.gz', '/local/evidence'],
     ['download', 'registered-target', 'phase/evidence.tar.gz', '/local/evidence']),
])
def test_listener_paths_are_relative_without_changing_local_paths(tmp_path, arguments, expected):
    result = tools(tmp_path).remote(*arguments)
    assert result['arguments'] == ['--json', *expected]
    journal = json.loads((tmp_path / 'transport.jsonl').read_text())
    assert journal['command'][2:] == expected


def test_ssh_keeps_absolute_workspace_paths(tmp_path):
    arguments = ['run', 'registered-target', '--cwd', '/configured/workspace', '--file', '/local/script']
    assert tools(tmp_path, 'ssh').remote(*arguments)['arguments'] == ['--json', *arguments]


@pytest.mark.parametrize('arguments', [
    ['run', 'registered-target', '--cwd', 'retained/phase', '--file', '/local/script'],
    ['upload', 'registered-target', '/local/archive', 'retained/phase/archive'],
    ['download', 'registered-target', 'retained/phase/evidence', '/local/evidence'],
])
def test_listener_accepts_already_relative_transport_paths(tmp_path, arguments):
    assert tools(tmp_path).remote(*arguments)['arguments'] == ['--json', *arguments]


@pytest.mark.parametrize('arguments', [
    ['upload', 'registered-target', '/local/file', '/configured/workspace'],
    ['download', 'registered-target', '/configured/workspace', '/local/evidence'],
])
def test_listener_rejects_transfer_of_workspace_root(tmp_path, arguments):
    with pytest.raises(ValueError, match='must name an item'):
        tools(tmp_path).remote(*arguments)
    assert not (tmp_path / 'transport.jsonl').exists()


@pytest.mark.parametrize('path', ['/other/workspace/item', '/configured/workspace/../outside'])
def test_listener_rejects_remote_paths_outside_workspace_before_dispatch(tmp_path, path):
    with pytest.raises(ValueError, match='workspace'):
        tools(tmp_path).remote('upload', 'registered-target', '/local/file', path)
    assert not (tmp_path / 'transport.jsonl').exists()
