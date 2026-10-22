"""Compact blind-task evidence without publishing a trusted native host."""
import json
from pathlib import Path
import tarfile


def compact_evidence(root, output, summary):
    root, output = Path(root), Path(output)
    spec = json.loads((root / 'task.json').read_text())
    native = spec.get('language') != 'triton'
    if native:
        # Native -save-temps compiles the trusted host translation unit, not
        # candidate.asc alone. Names/extensions cannot prove device-only scope;
        # even build.json may embed source-annotated disassembly.
        summary['compiler_evidence'] = dict(
            available=False, role='unproven_native_translation_unit',
            reason='candidate/device-only IR provenance unavailable; trusted-host artifacts retained remotely',
            diagnostics='compiler stdout/stderr retained')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    archive_path = root / 'evidence.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for base in (output, root / 'compiler-evidence', root / 'build'):
            if not base.exists():
                continue
            for path in base.rglob('*'):
                if not path.is_file() or path.is_symlink():
                    continue
                if native and base.name == 'compiler-evidence':
                    continue
                if native and base.name == 'build':
                    allowed = path.suffix in ('.stdout', '.stderr')
                else:
                    allowed = base.name == 'compiler-evidence' or path.suffix in (
                        '.json', '.log', '.csv', '.txt', '.mlir', '.ll', '.s', '.asci', '.stdout', '.stderr')
                if allowed:
                    archive.add(path, arcname=str(path.relative_to(root)))
    return archive_path
