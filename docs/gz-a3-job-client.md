# GZ-A3 retained JSON jobs

`scripts/gz_a3_job_client.py` accepts the existing version 1 benchmark job JSON
on standard input and emits one JSON result. It uses the same native worker,
receipt validation and compact profiling evidence as the BZ client, but routes
only to `gz-a3` through the installed global `$remote-access` client.

Read the current remote-access skill and discover the target's capabilities,
preflight and registry workspace before choosing a healthy idle physical device.
The job's `device` selects that physical device; its `logical_device` is zero.
The named runtime is `py311-torch`, supplied through the job's `runtime` field.
Runtime activation scripts, workspace adapters and direct SSH are not supported.

```bash
python scripts/gz_a3_job_client.py \
  --cpl-remote-sha256 <installed-client-sha256> \
  --remote-root <existing-workspace-relative-staging-child> \
  --state-dir <private-local-state-directory> < job.json
```

The GZ route requires a non-root, normalized path relative to its registry
workspace (for example `benchmark-checks`), not an absolute host path or `.`.
Upload paths stay in that namespace; the worker resolves its payload and run
paths relative to the selected staging cwd before changing directories. Native
BZ targets retain their distinct absolute-root policy.

The client pins the installed global transport digest and validates the remote
root before dispatch. Use the same private state directory and exact request
after an observer interruption: retained `remote:gz-a3:job:<id>` handles are
observed rather than redispatched. Completed results are cached against source,
request and execution provenance; a changed calibration phase is a new request.
The controller's observe-only mode never submits new work.

Compilation, runtime, correctness and submission failures remain counted
candidate outcomes with retained diagnostics. Transport or observer failures are
structured infrastructure results, not kernel failures. Successful profiling
keeps compact captures and their exact selector/case bindings; it does not
download raw profiling directories. Invalid configuration produces JSON and a
nonzero exit status, rather than an unhandled configuration traceback.
