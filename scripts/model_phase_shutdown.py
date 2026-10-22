"""Trusted shutdown control for one retained private phase, never public-mounted."""
import json
import os
from pathlib import Path
import signal
import sys
import time


def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start=fields[19], ppid=int(fields[1]), state=fields[0],
                    cwd=str((proc / 'cwd').resolve(strict=True)),
                    command=(proc / 'cmdline').read_bytes().split(b'\0'))
    except (OSError, ValueError):
        return None


def shutdown(root, grace=3):
    root = Path(root).resolve(strict=True)
    # The owned, unique phase path is supplied only by its trusted receipt.
    (root / 'stop').touch()
    record = root / 'worker.json'
    if record.exists():
        pinned = json.loads(record.read_text())
        worker = identity(pinned['pid'])
        if worker and (worker['start'] != pinned['start'] or worker['cwd'] != str(root)):
            raise RuntimeError('phase worker identity changed; refusing termination')
    else:
        # Compatibility with retained pre-identity workers: exact owned cwd and
        # entrypoint, never a process-name or device-wide search/termination.
        matches = [value for path in Path('/proc').iterdir() if path.name.isdigit()
                   if (value := identity(int(path.name))) and value['cwd'] == str(root)
                   and value['command'][:2] == [b'python', b'execute-phase.py']]
        if len(matches) > 1:
            raise RuntimeError('ambiguous phase worker identity')
        worker = matches[0] if matches else None
    owned = {worker['pid']: worker} if worker else {}
    # Capture attributable descendants before reparenting. Never select peers.
    changed = True
    while changed:
        changed = False
        for path in Path('/proc').iterdir():
            if path.name.isdigit() and int(path.name) not in owned:
                value = identity(int(path.name))
                if value and value['ppid'] in owned and value['cwd'] == str(root):
                    owned[value['pid']] = value
                    changed = True
    def alive():
        result = []
        for pid, pinned in owned.items():
            now = identity(pid)
            if now and now['state'] != 'Z' and now['start'] == pinned['start'] and now['cwd'] == str(root):
                result.append(pid)
        return result
    for sig in (None, signal.SIGTERM, signal.SIGKILL):
        for pid in alive():
            if sig:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        cutoff = time.monotonic() + grace
        while alive() and time.monotonic() < cutoff:
            time.sleep(.1)
        if not alive():
            return dict(state='terminal', worker_closed=True, owned_pids=sorted(owned))
    return dict(state='observation-required', remaining=alive())


if __name__ == '__main__':
    result = shutdown(sys.argv[1])
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['state'] == 'terminal' else 1)
