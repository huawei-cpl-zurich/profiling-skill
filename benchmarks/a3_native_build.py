"""Compile the self-contained Ascend C synchronization control, retaining errors."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from profile_device import bounded_run


def build(output, *, source=None, compiler=None, extra_flags=()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(source).resolve() if source is not None else Path(__file__).with_name('a3_sync_control.asc')
    compiler = compiler or shutil.which('bisheng')
    if not compiler:
        raise RuntimeError('bisheng unavailable in selected named runtime')
    executable = output / 'a3_sync_control'
    command = [compiler, str(source), '-O2', '-g', '-save-temps=obj', '--npu-arch=dav-2201',
               '-lascendcl', '-o', str(executable), *extra_flags]
    identity = {'command': command, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (output / 'build.json').write_text(json.dumps(identity, indent=2))
    try:
        result = bounded_run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=300, cwd=output)
    except (OSError, subprocess.TimeoutExpired) as exc:
        for name in ('stdout', 'stderr'):
            value = getattr(exc, name, '') or ''
            if isinstance(value, bytes):
                value = value.decode(errors='replace')
            (output / ('build.' + name)).write_text(value)
        identity['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        (output / 'build.json').write_text(json.dumps(identity, indent=2))
        raise RuntimeError(f'native compilation failed; see {output}/build.json') from exc
    (output / 'build.stdout').write_text(result.stdout)
    (output / 'build.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'native compilation failed ({result.returncode}); see {output}/build.stderr')
    identity['binary_sha256'] = hashlib.sha256(executable.read_bytes()).hexdigest()
    identity['compiler_artifacts'] = [
        {'path': str(path), 'bytes': path.stat().st_size,
         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(output.rglob('*')) if path.suffix in ('.bc', '.ll', '.s')]
    (output / 'build.json').write_text(json.dumps(identity, indent=2))
    return executable
