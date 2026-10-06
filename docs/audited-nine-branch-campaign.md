# Audited nine-branch campaign

This campaign crosses three benchmark tasks (`matmul`, `gdn`, and `bsa`) with
three isolated skill treatments (`cannbot`, `project-cannbot`, and
`project-guarded`). Each of the nine unmerged branches performs four audited
experiments with a 24-request budget. The seed plus four experiment commits
therefore produces five commits per branch and 45 commits in the complete run.

## Frozen manifest

Create a provenance JSON document before generating the manifest. It must pin
the 40-character source revision; SHA-256 digests for the controller, all
three benchmark baselines, and all treatment skill bundles; the immutable
runtime image digest; and the model name and reasoning effort. A representative
shape is:

```json
{
  "source_revision": "<40 hex characters>",
  "controller_sha256": "<64 hex characters>",
  "runtime_image_digest": "sha256:<64 hex characters>",
  "model": {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
  "baselines": {"matmul": "<sha256>", "gdn": "<sha256>", "bsa": "<sha256>"},
  "skills": {
    "cannbot": "<sha256>",
    "ascend-profiling": "<sha256>",
    "triton-guarded-kernel": "<sha256>"
  }
}
```

Generate the manifest from the invariant audited prompt and the three task-only
files. The ordering seed is recorded and produces a deterministic balanced
Latin order: each consecutive block of three cells contains each task and each
treatment once.

```bash
python scripts/audited_campaign.py generate \
  --run-id RUN_ID \
  --prompt prompts/audited-three-experiment.md \
  --matmul-task experiments/audited-tasks/matmul.md \
  --gdn-task experiments/audited-tasks/gdn.md \
  --bsa-task experiments/audited-tasks/bsa.md \
  --provenance /absolute/frozen-provenance.json \
  --ordering-seed RUN_ID \
  --output /absolute/campaign/manifest.json
```

The manifest intentionally contains no target or device number. Every task
uses one byte-identical task document across its three treatments, and every
cell carries the exact treatment skill allowlist. The runtime must prepare the
isolated branches/workspaces from those declarations; agents must never merge
or push their experiment branches.

## Dynamic execution and recovery

The production resource provider calls `cpl-remote capabilities` and
`cpl-remote preflight` for `bz-a3-1` and `bz-a3-2`, then returns every healthy,
idle physical device. `run_campaign` launches one cell per unique admitted
target/device and fills all available slots. It has no batch barrier: whenever
any cell finishes, it refreshes admission and immediately fills that free slot
in the recorded fair order. An infrastructure failure checkpoints only its
cell; independent queued cells continue. The campaign pauses with failed cells
pending only after otherwise runnable work is exhausted.

Each launcher result must include a terminal status, durable handle, completed
round count, and four round evidence records. Candidate compilation, runtime,
correctness, or budget failure is a terminal `candidate_failed` result and is
not retried. Infrastructure failure before submission leaves the cell pending
for a later resume. If a durable handle exists, resume calls `observe` for that
exact handle and cannot dispatch a replacement.

The ledger is atomically updated before dispatch and after every result. Repeat
the same operation with `resume=True` after infrastructure recovery; completed
and candidate-failed cells are never launched again. Large `msprof` trees stay
remote, while receipts retain only compact in-place analysis and exact handles.

The controller-adapter integration supplies resource discovery and the real
launcher. The following local simulation validates the scheduler, checkpoint,
and report path without hardware:

```bash
python scripts/audited_campaign.py simulate \
  --manifest /absolute/campaign/manifest.json \
  --ledger /absolute/campaign/ledger.json \
  --slots 4

python scripts/audited_campaign.py report \
  --manifest /absolute/campaign/manifest.json \
  --ledger /absolute/campaign/ledger.json \
  --output /absolute/campaign/report.json
```

## Acceptance

Before the measured run, validate manifest generation and a fake-controller
end-to-end run locally. On BZ-A3, validate a known-good matmul candidate, an
intentional compilation failure, observer reconnection to the same durable
handle, and measurement-only resume. Then run one matmul canary per treatment.
Do not start the nine branches while an infrastructure fault remains
unclassified.

The version-two final report contains one row per branch with status, attempt
count, unchanged raw round receipts, per-case evidence, controls and policy,
calibration-normalized timing, best raw and normalized medians, frozen baseline
evidence, speedup, and retained failure detail. A separate discarded-attempts
table records every infrastructure exclusion with its target, device, handle,
and diagnostic. Infrastructure-pending cells remain visibly separate from
candidate failures. Each round's normalized samples, median, calibration,
baseline, and speedup are propagated from the validated controller receipt.
Best-round and cross-device/cross-run comparison fields use normalized timing
when present; raw timing remains available for device-local diagnosis.
