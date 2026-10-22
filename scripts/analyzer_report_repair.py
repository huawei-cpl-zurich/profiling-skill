"""One report-only semantic repair, bounded by the original analyzer deadline."""
import hashlib
import json
from pathlib import Path
import shutil
import time


def permitted_citations(launcher):
    from actor_analyzer_controller import citation_identities, correct, digest
    expected = digest(launcher.workspace / 'candidate.py')
    entries = []
    for path in sorted((launcher.workspace / 'evidence').glob('*/results/summary.json')):
        item = dict(path=str(path.relative_to(launcher.workspace)), kinds=[])
        try:
            item['sha256'] = digest(path)
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('summary must be an object')
        except (ValueError, OSError):
            item['unavailable_reason'] = 'Unreadable original summary'
            entries.append(item)
            continue
        if value.get('source_sha256') == expected and correct(value) and value.get('implementation', 'candidate') == 'candidate':
            kernels, cases = citation_identities(value)
            item.update(kernels=kernels, cases=cases)
            if kernels and cases:
                if any(c.get('pipe_rows') for c in value.get('captures', [])): item['kinds'].append('profile')
                if any(value.get(k) for k in ('timing', 'latency', 'busy_time')): item['kinds'].append('timing')
                if value.get('compiler_artifacts'): item['kinds'].append('compiler')
            item['compiler_artifacts'] = value.get('compiler_artifacts', {})
        if not item['kinds']:
            item['unavailable_reason'] = 'No supported same-source attributed measurement/compiler citation'
        entries.append(item)
    return entries


def repair(launcher, result, run, save, *, clock=time.monotonic):
    if result['returncode'] != 0 or result['timed_out']:
        return result
    directory = launcher.root / 'report-repair'
    directory.mkdir(exist_ok=True)
    def validate():
        try:
            launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
            return None
        except (ValueError, OSError) as error:
            return str(error)
    error = validate()
    if error is None:
        if not (directory / 'dispatch.json').exists():
            save(directory / 'validation.json', dict(status='valid', attempts=0))
        return result
    repair_cutoff = min(clock() + 90., launcher.deadline_monotonic - 10.)
    report = launcher.reports / 'analysis.json'
    initial = directory / 'initial-report.json'
    if not initial.exists() and report.is_file(): shutil.copyfile(report, initial)
    budget = repair_cutoff - clock() - 6.  # Include execute() termination grace in the 90s cap.
    if (directory / 'dispatch.json').exists():
        if not (directory / 'validation.json').exists():
            save(directory / 'validation.json', dict(status='observation-required', attempts=1, error=error))
        return result
    if budget <= 0:
        save(directory / 'validation.json', dict(status='invalid', attempts=0, error=error,
            reason='No repair allowance inside the original deadline'))
        return result
    source = hashlib.sha256((launcher.workspace / 'candidate.py').read_bytes()).hexdigest()
    context = json.loads((launcher.campaign.history / 'context.json').read_text())
    feedback = dict(error=error, source_sha256=source, trusted_context=context,
                    source_diff_available=bool(context.get('source_diff')),
                    optimization_attribution=('No source diff exists: observed optimization entries are unsupported; use none with an explanation'
                        if not context.get('source_diff') else 'Observed optimizations must cite exact provided source_diff lines'),
                    allowed_citations=permitted_citations(launcher),
                    policy='Exactly one report-only repair; no remote/device commands or source changes')
    save(directory / 'feedback.json', feedback)
    shutil.copyfile(launcher.frozen / 'prompt.md', directory / 'original-instructions.md')
    config_path = launcher.root / 'tool-config.json'
    config = json.loads(config_path.read_text());config['report_repair_only'] = True;save(config_path, config)
    launcher.report_repair_only = True
    if not report.exists(): report.touch()
    prompt = ('Repair only /analysis/analysis.json. Correct ALL reported issues, not just the first citation. Read /report-repair/feedback.json for the exact validator '
        'error and authoritative permitted citation metadata, /report-repair/initial-report.json if present, '
        'and /report-repair/original-instructions.md for the report contract. Preserve supported analysis; '
        'remove unsupported citations and dependent claims, and state missing evidence honestly. '
        'Do not recapture, invoke remote/device tools, edit the kernel, invent metadata or reset the deadline. '
        f'You have at most {budget:.1f} seconds inside the ORIGINAL role allowance.')
    command = [*launcher.base_command(), '/runtime/node/bin/codex', 'exec', '--json', '--ignore-user-config',
        '--dangerously-bypass-approvals-and-sandbox', '-m', launcher.model['name'], '-c',
        f'model_reasoning_effort="{launcher.model["reasoning_effort"]}"', prompt]
    budget = repair_cutoff - clock() - 6.
    if budget <= 0:
        save(directory / 'validation.json', dict(status='invalid', attempts=0, error=error, reason='Repair allowance expired'))
        return result
    save(directory / 'dispatch.json', dict(dispatch_intent=True, attempt=1,
        attempt_id='report-repair-1', journal='report-repair/model.jsonl',
        original_execution=json.loads((launcher.root / 'role-execution.json').read_text()),
        allowance_seconds=budget, source_sha256=source))
    terminal = run(command, launcher.root, budget,
                   (directory / 'model.jsonl', directory / 'model.stderr'))
    save(directory / 'terminal.json', {key: terminal[key] for key in ('returncode', 'timed_out')})
    if report.is_file(): shutil.copyfile(report, directory / 'repaired-report.json')
    unchanged = hashlib.sha256((launcher.workspace / 'candidate.py').read_bytes()).hexdigest() == source
    final_error = validate() if unchanged else 'Frozen candidate changed during report repair'
    save(directory / 'validation.json', dict(status='valid' if final_error is None else 'invalid',
        attempts=1, initial_error=error, error=final_error, candidate_unchanged=unchanged))
    result['report_repair'] = dict(attempts=1, valid=final_error is None)
    if terminal['returncode'] != 0 or terminal['timed_out']:
        result.update(returncode=terminal['returncode'], timed_out=terminal['timed_out'])
    return result
