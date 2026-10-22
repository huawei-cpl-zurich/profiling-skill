"""A device-less activation handshake using only the retained worker's clock."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


def wait_for_activation(path, wait_seconds):
    """Advertise an anchor, then refuse device work without a live cutoff.

    The host adds its remaining allowance *after receiving* this anchor. Its
    clock need not match this machine's clock; all subsequent delays consume
    the fixed remote cutoff. A lost launch response never activates device work.
    """
    anchor = time.monotonic()
    print('PHASE_READY=' + json.dumps(dict(monotonic=anchor)), flush=True)
    while time.monotonic() < anchor + wait_seconds:
        try:
            cutoff = float(json.loads(Path(path).read_text())['cutoff_monotonic'])
        except (FileNotFoundError, ValueError, KeyError):
            time.sleep(.02)
            continue
        if not math.isfinite(cutoff) or not time.monotonic() < cutoff <= anchor + wait_seconds:
            raise TimeoutError('original role deadline expired before activation')
        return cutoff
    raise TimeoutError('original role deadline expired without activation')


def activate(tools, folder, receipt):
    """Activate one existing handle, preserving its cutoff across retries.

    Return False while readiness/transfer is pending, True when activated.
    The caller persists all other receipt state as usual. No new job is issued.
    """
    deadline = receipt.get('request', {}).get('utc_deadline')
    if (not deadline or not receipt.get('request', {}).get('activation_required')
            or receipt.get('activation_state') == 'activated'):
        return True
    if datetime.fromisoformat(deadline) <= datetime.now(timezone.utc):
        raise TimeoutError('original role deadline expired before activation')
    path = Path(folder) / 'activation.json'
    if 'activation_upload' not in receipt:
        if not path.exists():
            try:
                logs = tools.remote('logs', receipt['handle'], '--tail', '100')
            except RuntimeError:
                # Listener log files may not exist immediately after admission.
                # Confirm the SAME retained job is pending before waiting; a
                # terminal failure or unrelated malformed response stays visible.
                observed = tools.remote('observe', receipt['handle'])
                if observed.get('state') in ('queued', 'dispatching', 'running',
                                             'reconnecting', 'observation-unavailable'):
                    return False
                raise
            anchors = [json.loads(line.split('=', 1)[1])['monotonic']
                       for line in logs.get('content', '').splitlines() if line.startswith('PHASE_READY=')]
            if not anchors:
                return False
            remaining = (datetime.fromisoformat(deadline) - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                raise TimeoutError('original role deadline expired before activation')
            path.write_text(json.dumps(dict(cutoff_monotonic=anchors[-1] + remaining)))
        # Save the immutable payload before upload. Resume must never recalculate
        # an allowance from a later observation or issue another retained job.
        receipt['activation_upload'] = tools.remote('upload', receipt['target'], path,
            receipt['remote_folder'].rstrip('/') + '/activation.json')
        (Path(folder) / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    transfer = tools.transfer(receipt, 'activation_upload')
    receipt['activation_upload'] = transfer
    (Path(folder) / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if transfer.get('state') in ('failed', 'cancelled'):
        raise RuntimeError('activation upload ' + transfer['state'])
    if transfer.get('state') != 'completed':
        return False
    receipt['activation_state'] = 'activated'
    (Path(folder) / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return True
