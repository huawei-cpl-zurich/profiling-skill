import importlib.util
from pathlib import Path


def test_compiler_evidence_preserves_variants_repetitions_and_repeated_runs(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'benchmarks/compile_evidence.py'
    spec = importlib.util.spec_from_file_location('compile_evidence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captures = []
    for variant, repeat, source in [('before', 32, 'scf.for before'),
                                    ('after', 32, 'scf.for after'),
                                    ('before', 128, 'scf.for amplified'),
                                    ('before', 32, 'scf.for repeat invocation')]:
        result = module.retain_compiler_evidence({'ttir': source, 'binary': b'raw'},
                                                  tmp_path, 'mask', variant, repeat)
        assert len(result) == 1
        assert result[0]['loop_evidence'] == [source]
        captures.append((Path(result[0]['path']), source))
    assert len({path for path, _ in captures}) == 4
    assert all(path.read_text() == source for path, source in captures)
