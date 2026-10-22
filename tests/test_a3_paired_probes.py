"""Device-less functional contract for generic examples; hardware runs validate kernels."""
import json
from pathlib import Path
import subprocess
import sys

EXAMPLE = Path(__file__).resolve().parents[1] / 'skills/ascend-profiling/examples/a3_paired_probes.py'


def test_manifest_has_matched_work_and_explicit_added_traffic():
    run = subprocess.run([sys.executable, '-I', str(EXAMPLE), '--manifest',
                          '--size', '513', '--matrix-size', '512'],
                         text=True, capture_output=True, check=True)
    records = {(item['case'], item['variant']): item for item in json.loads(run.stdout)}
    assert len(records) == 6
    before, after = records['copy', 'baseline'], records['copy', 'candidate']
    assert before['logical_input_bytes'] == after['logical_input_bytes'] == 4 * 513
    assert before['logical_output_bytes'] == after['logical_output_bytes']
    assert before['input_allocation_bytes'] == after['input_allocation_bytes'] == 8 * 513
    assert after['accessed_input_span_bytes'] > before['accessed_input_span_bytes']
    assert records['gather', 'baseline']['logical_index_bytes'] == 0
    assert records['gather', 'candidate']['logical_index_bytes'] == 4 * 513
    assert records['matmul', 'baseline']['requested_num_stages'] == 1
    assert records['matmul', 'candidate']['requested_num_stages'] == 2
    assert records['matmul', 'baseline']['useful_flops'] == 2 * 512**3


def test_invalid_parameters_fail_without_loading_device_packages():
    run = subprocess.run([sys.executable, '-I', str(EXAMPLE), '--manifest', '--size', '0'],
                         text=True, capture_output=True)
    assert run.returncode == 2
    assert 'sizes and iterations must be positive' in run.stderr
