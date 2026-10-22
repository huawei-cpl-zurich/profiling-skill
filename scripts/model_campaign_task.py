"""Trusted operator routing. Never expose this module or its fixtures to workers."""
import importlib
import json
from pathlib import Path


def configuration(root):
    path = Path(root) / 'private-task.json'
    return json.loads(path.read_text()) if path.exists() else dict(task='matmul', profile_index=7, check_indices=list(range(8)))


def case(root, index=None, variant=0):
    config = configuration(root)
    index = config['profile_index'] if index is None else index
    if config['task'] == 'matmul':
        from private_campaign_cases import make_case, validate
        return make_case('matmul', index, variant=variant), validate
    adapter = importlib.import_module('comparison_' + config['task'])
    return adapter.make_case(Path(root) / 'reference', index, variant=variant), adapter.validate


def check_indices(root):
    return configuration(root)['check_indices']
