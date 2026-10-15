"""Retain separate compiler artifacts for every invocation of a causal probe."""
import hashlib
from pathlib import Path
import tempfile


def retain_compiler_evidence(assembly, root, experiment, variant, repetitions):
    parent = Path(root) / experiment / variant / f'repetitions-{repetitions}'
    parent.mkdir(parents=True, exist_ok=True)
    invocation = Path(tempfile.mkdtemp(prefix='invocation-', dir=parent))
    artifacts = []
    for stage, value in assembly.items():
        if not isinstance(value, str): continue
        path = invocation / (stage + '.txt')
        path.write_text(value)
        artifacts.append({'stage': stage, 'path': str(path),
                          'sha256': hashlib.sha256(value.encode()).hexdigest(),
                          'bytes': len(value.encode()),
                          'loop_evidence': [line.strip() for line in value.splitlines() if
                                           any(token in line for token in
                                               ('scf.for', 'scf.while', 'cf.cond_br',
                                                'llvm.cond_br', 'volatile', 'br i1'))][:20]})
    return artifacts
