"""Exercise the installed folder without repository modules or working-directory assumptions."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1] / 'skills/ascend-profiling'


def test_isolated_skill_query_and_capture(tmp_path):
    installed = tmp_path / 'installed'
    shutil.copytree(SOURCE, installed, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    work = tmp_path / 'work'
    work.mkdir()
    env = dict(os.environ, PYTHONPATH='')
    query = subprocess.run([sys.executable, '-I', str(installed / 'scripts/query_triton_pipe_attribution.py'),
                            '--construct', 'gm_to_l1_copy', '--product', 'a3'],
                           cwd=work, env=env, text=True, capture_output=True, check=True)
    assert json.loads(query.stdout)['status'] == 'direct'
    profiler = work / 'msprof'
    profiler.write_text('''#!/usr/bin/env python3
import csv
from pathlib import Path
import sys
out = Path(next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--output=')))
out.mkdir(parents=True, exist_ok=True)
with (out / 'OpBasicInfo.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['Op Name', 'Task Duration(us)'])
    writer.writerow(['example_kernel', '2.5'])
print('Profiling running finished. All task success.')
''')
    profiler.chmod(0o755)
    capture = subprocess.run([sys.executable, '-I', str(installed / 'scripts/profile_a3.py'),
        '--msprof', str(profiler), '--output', str(work / 'capture'),
        '--kernel-name', 'example_kernel', '--warm-up', '0', '--launch-count', '1',
        '--', sys.executable, '-c', 'pass'], cwd=work, env=env,
        text=True, capture_output=True, check=True)
    evidence = json.loads(capture.stdout)
    assert evidence['status'] == 'success'
    assert evidence['kernels'][0]['duration_us']['median'] == 2.5
    capacity = work / 'capacity.json'
    capacity.write_text(json.dumps(dict(schema_version=1, product='a5',
        provenance=dict(capture_id='isolated-capture', source_sha256='a' * 64),
        phases=[dict(phase_id='compute', start_ns=0, end_ns=10,
                     metrics=dict(accumulated_cube_block_cycles=100,
                                  whole_device_cube_capacity_cycles=100))])))
    analyzed = subprocess.run([sys.executable, '-I', str(installed / 'scripts/analyze_pipe_saturation.py'),
        '--product', 'a5', '--model', str(installed / 'references/a5-pipe-capacity-model.json'),
        '--evidence', str(capacity)], cwd=work, env=env, text=True,
        capture_output=True, check=True)
    assert json.loads(analyzed.stdout)['phases'][0]['resources'][0]['state'] == 'saturated'
    for script in (installed / 'scripts').glob('*.py'):
        result = subprocess.run([sys.executable, '-I', str(script), '--help'],
                                cwd=work, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
