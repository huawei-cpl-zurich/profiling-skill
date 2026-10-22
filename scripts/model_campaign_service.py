"""Private campaign dispatch using the existing managed Tools lifecycle.

The trusted bundle contains execute.py and private providers. Only the exporter
may cross the agent boundary. This module is never mounted into agent folders.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import traceback
import uuid

from blind_optimization_tools import Tools, save

TERMINAL = {'completed', 'failed', 'cancelled'}
OPERATIONS = {'check', 'benchmark', 'pipes', 'screen'}


class CampaignTools(Tools):
    """Keep the existing handle/transfer recovery, cap every dispatch by deadline."""
    def observe(self, ident):
        receipt = json.loads((self.state / ident / 'receipt.json').read_text())
        if receipt.get('warm'):
            from model_campaign_warm import observe
            return observe(self, ident)
        return super().observe(ident)

    def dispatch(self, folder, receipt):
        if receipt.get('dispatch_intent') or receipt.get('handle'):
            return super().dispatch(folder, receipt)
        context = receipt['request']
        remaining = max(0., (datetime.fromisoformat(context['utc_deadline']) - datetime.now(timezone.utc)).total_seconds())
        if not remaining:
            receipt.update(state='failed', failure_stage='budget', dispatch_attempted=False)
            save(folder / 'receipt.json', receipt)
            output = self.workspace / 'evidence' / receipt['call_id'] / 'results'
            output.mkdir(parents=True, exist_ok=True)
            save(output / 'summary.json', dict(state='failed', failure_stage='budget'))
            return receipt
        script = '#!/bin/bash\nset -eu\nexport CAMPAIGN_REMAINING_SECONDS=' + shlex.quote(str(remaining))
        script += '\nmkdir -p ' + shlex.quote(receipt['remote_folder'])
        script += '\ntar -xzf ' + shlex.quote(receipt['remote_archive']) + ' -C ' + shlex.quote(receipt['remote_folder'])
        script += '\ncd ' + shlex.quote(receipt['remote_folder']) + '\nexec python execute.py\n'
        (folder / 'run.sh').write_text(script)
        return super().dispatch(folder, receipt)


class CampaignService:
    """An instance belongs to exactly one isolated branch/agent workspace.

    tools is a CampaignTools instance using the established Tools lifecycle. exporter
    accepts (private_evidence, fresh_public_directory), returns public JSON,
    and must allowlist values rather than redact arbitrary trusted strings.
    """
    def __init__(self, tools, bundle, public_workspace, exporter, *, clock=None):
        self.tools, self.bundle = tools, Path(bundle)
        self.public = Path(public_workspace)
        self.exporter = exporter
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def remaining(self, context):
        deadline = datetime.fromisoformat(context['utc_deadline'])
        if deadline.tzinfo is None:
            raise ValueError('deadline must be timezone-aware')
        return max(0., (deadline - self.clock()).total_seconds())

    def _terminal(self, folder, receipt, stage):
        receipt.update(state='failed', failure_stage=stage, dispatch_attempted=False)
        save(folder / 'receipt.json', receipt)
        private = self.tools.workspace / 'evidence' / receipt['call_id'] / 'results'
        private.mkdir(parents=True, exist_ok=True)
        save(private / 'summary.json', dict(state='failed', failure_stage=stage))
        return self.publish(receipt)

    def publish(self, result):
        ident = result['call_id']
        private = self.tools.workspace / 'evidence' / ident
        public = self.public / 'evidence' / ident
        pending_transfer = any(isinstance(result.get(key), dict) and
            result[key].get('state') not in TERMINAL for key in ('upload', 'download', 'evidence_transfer'))
        if not (private / 'results/summary.json').exists() and result.get('state') in {'failed', 'cancelled'} \
                and not result.get('next') and not pending_transfer:
            # A confirmed terminal job/transfer with no recoverable worker report
            # must release the cell for debugging. Unknown/live handles never do.
            receipt_path = self.tools.state / ident / 'receipt.json'
            receipt = json.loads(receipt_path.read_text())
            if not any(isinstance(receipt.get(key), dict) and receipt[key].get('state') not in TERMINAL
                       for key in ('upload', 'download')):
                stage = result.get('failure_stage', 'infrastructure')
                receipt.update(state=result['state'], failure_stage=stage)
                save(receipt_path, receipt)
                (private / 'results').mkdir(parents=True, exist_ok=True)
                save(private / 'results/summary.json', dict(state=result['state'],
                    failure_stage=stage, source_sha256=receipt['request']['source_sha256'],
                    diagnostic=dict(kind='terminal-evidence-unavailable',
                                    reason='retained job or transfer ended without a worker report')))
        if (private / 'results/summary.json').is_file():
            if not public.exists():
                public.parent.mkdir(parents=True, exist_ok=True)
                # Failed export must not leave a partial public directory that
                # prevents the same retained call from being observed again.
                with tempfile.TemporaryDirectory(dir=public.parent) as temporary:
                    staging = Path(temporary) / 'export'
                    self.exporter(private, staging)
                    json.loads((staging / 'results/summary.json').read_text())
                    staging.rename(public)
            payload = json.loads((public / 'results/summary.json').read_text())
            # Navigation metadata is trusted controller metadata, not worker text.
            payload.update(call_id=ident, evidence='evidence/' + ident)
            if result.get('handle'):
                payload['handle'] = result['handle']
            return payload
        payload = {key: result[key] for key in ('state', 'call_id', 'handle') if key in result}
        payload['next'] = 'experiment-tool observe ' + ident
        if result.get('state') in TERMINAL:
            payload['evidence_pending'] = True
        return payload

    def observe(self, ident):
        # Observation must not depend on the current round deadline. Finishing
        # evidence collection is allowed even after optimizer time expires.
        if not isinstance(ident, str) or not ident.isalnum():
            raise ValueError('invalid call id')
        return self.publish(self.tools.observe(ident))

    def start(self, operation):
        if operation not in OPERATIONS:
            raise ValueError('unknown operation')
        if isinstance(self.tools, CampaignTools):
            from model_campaign_warm import start
            return start(self, operation)
        # Resuming a prior call is lifecycle safety, not a trial/round quota.
        for path in sorted(self.tools.state.glob('*/receipt.json')):
            prior = json.loads(path.read_text())
            private = self.tools.workspace / 'evidence' / prior['call_id'] / 'results/summary.json'
            if prior['state'] not in TERMINAL or not private.exists():
                return self.observe(prior['call_id'])
        context = json.loads((self.public / 'execution.json').read_text())
        if self.remaining(context) <= 0:
            return dict(state='failed', failure_stage='budget', retryable=False)
        source = self.public / 'candidate.py'
        if source.is_symlink() or not source.is_file():
            raise ValueError('candidate must be a regular file')
        ident = uuid.uuid4().hex[:16]
        folder = self.tools.state / ident
        # The trusted bundle is a directory, never a caller-selected archive.
        shutil.copytree(self.bundle, folder)
        shutil.copyfile(source, folder / 'candidate.py')
        request = dict(operation=operation, utc_deadline=context['utc_deadline'],
                       host_remaining_seconds=self.remaining(context),
                       source_sha256=hashlib.sha256((folder / 'candidate.py').read_bytes()).hexdigest())
        save(folder / 'request.json', request)
        remote_folder = self.tools.cwd.rstrip('/') + '/model-campaign-' + ident
        receipt = dict(call_id=ident, target=self.tools.target, runtime=self.tools.runtime,
                       cwd=self.tools.cwd, state='prepared', request=request,
                       remote_folder=remote_folder, remote_archive=remote_folder + '.tar.gz')
        save(folder / 'receipt.json', receipt)
        archive = folder / 'submission.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            for path in folder.iterdir():
                if path != archive:
                    stream.add(path, arcname=path.name)
        try:
            for args in (('targets',), ('capabilities', self.tools.target), ('preflight', self.tools.target)):
                result = self.tools.remote(*args)
                if args[0] == 'preflight' and result.get('state') != 'completed':
                    raise RuntimeError('preflight unavailable')
            if not self.remaining(context):
                return self._terminal(folder, receipt, 'budget')
            receipt['upload'] = self.tools.remote('upload', self.tools.target, archive, receipt['remote_archive'])
            save(folder / 'receipt.json', receipt)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            # No run dispatch was attempted. Preserve full transport diagnostics
            # privately, not strings possibly containing trusted source paths.
            (folder / 'service-error.txt').write_text(traceback.format_exc())
            return self._terminal(folder, receipt, 'infrastructure')
        if receipt['upload'].get('state') != 'completed':
            return self.observe(receipt['call_id'])
        return self.publish(self.tools.dispatch(folder, receipt))
