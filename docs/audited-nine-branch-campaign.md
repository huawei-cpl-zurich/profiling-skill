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
  "resource_admission_sha256": "<64 hex characters>",
  "admission_provider_id": "campaign-operator",
  "admission_allowlist_sha256": "<64 hex characters>",
  "cpl_remote_closure_sha256": "<64 hex characters>",
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

Resource discovery is supplied by the prerequisite documented in
`docs/audited-resource-admission.md`. Production imports that module rather
than duplicating its placement parser or global-client probes. Both the exact
module file and the global `cpl-remote` client are hash-pinned in runtime
configuration and manifest provenance before any cell can launch.

`run_campaign` launches one cell per unique admitted target/device and fills
all available slots. It has no batch barrier: whenever any cell finishes, it
refreshes admission and immediately fills that free slot in the recorded fair
order. An infrastructure failure checkpoints only its cell; independent queued
cells continue. The campaign pauses with failed cells pending only after
otherwise runnable work is exhausted.

Each launcher result must include a terminal status, durable handle, completed
round count, four distinct experiment commits, and four ordered controller
receipts. This contract applies equally to `complete` and `candidate_failed`.
A candidate-failed result must contain at least one fully classified
`candidate_error` round; partial or contradictory evidence remains
infrastructure-pending rather than terminalizing. Candidate compilation,
runtime, correctness, or budget failure is not retried. Infrastructure failure
before submission leaves the cell pending for a later resume. If a durable
handle exists, resume calls `observe` for that exact handle and cannot dispatch
a replacement.

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

Newly generated manifests use `schema_version: 2`, which requires starter
provenance. Verification and reporting continue to accept historical
`schema_version: 1` manifests without starter bindings. Production launch is
v2-only: create a hash-pinned runtime configuration with schema
`profiling-skill/audited-campaign-runtime/v2`. It declares the run root, a
pinned source repository and commit for each task, exact per-skill source trees
and hashes, the pinned runtime scripts closure containing
`audited_bz_controller.py`, its sibling benchmark-assets tree and hash, the
exact frozen timing-baseline bindings, invariant prompt and task bindings,
exact task-specific starter candidate and manifest bindings,
remote staging root, Codex authentication home, model, reasoning effort,
immutable runtime-image digest, and the five timeout-budget fields described
below. It also binds the fixed sibling `audited_resource_admission.py` path and
digest to `provenance.resource_admission_sha256`; pins the static provider and
target/device allowlist identities; and binds both the authenticated global
client launcher and its sibling implementation to
`provenance.cpl_remote_closure_sha256`. Each timing baseline uses schema
`profiling-skill/baseline-timing/v1`, names the benchmark, lists positive
per-case medians in exact development-case order, records a positive control
median, and includes the SHA-256 of the canonical JSON for those four fields.
`runtime_mode` must be
`docker`; direct Codex execution is rejected. The configuration repeats the
manifest provenance object exactly, and every pin is checked against the
source revision, complete controller closure, baseline files, composite
CANNBot skill bundle, project skill trees, model, and resolved Docker image.

`starter_sources` and `provenance.starters` must be identical maps covering
`matmul`, `gdn`, and `bsa`. Each task maps `candidate` and `manifest` to an
absolute frozen `path` and its 64-character `sha256`, for example:

```json
{
  "matmul": {
    "candidate": {"path": "/frozen/matmul/candidate.py", "sha256": "..."},
    "manifest": {
      "path": "/frozen/matmul/candidate.manifest.json",
      "sha256": "..."
    }
  }
}
```

The launcher verifies all six files and each manifest contract before creating
an agent. For each cell it first checks out and verifies the common clean source
revision, then copies that task's exact pair. The lifecycle commits the pair in
the seed commit before the first agent turn. Resume never recopies the files:
it re-hashes the frozen sources and verifies the seed commit's candidate and
manifest blobs, while preserving any checkpointed in-progress candidate.
Initial manifests may use `REPLACE_WITH_EXACT_EXPORTED_KERNEL`; the invariant
prompt requires the agent to replace that sentinel with the actual exported
kernel selector before profiling. Lifecycle validation rejects an invalid
schema, blank selector, or unresolved sentinel and repairs it in the same agent
session before any controller request.

The selector stored in the manifest remains the immutable experiment identity.
Profiling first requests that exact name. If a successful `msprof op` capture
reports a machine-readable selector miss and the declared name ends in
`_mix_aic` or `_mix_aiv`, the batch profiler strips that one terminal suffix
and tries once more. A successful resolution is pinned for every later case
and repeat. No other name rewriting is allowed. Compact evidence records both
the declared and resolved selectors; legacy evidence without those fields is
accepted only when its exact selector matches the declaration.

A clean final selector miss is a counted `submission_error`. Profiler startup,
timeout, exit, terminal-marker, report, or malformed-evidence failures remain
infrastructure errors. Failure receipts retain the requested cases and every
completed capture, so a selector mistake cannot be hidden by a missing
success-only evidence field.

If the host stops after the seed commit but before writing a blocked
checkpoint, resume is allowed only while `HEAD` is exactly that seed commit.
The worktree may be clean or may modify only `candidate.py` and
`candidate.manifest.json`. Recovery preserves those bytes, starts a new agent
session to inspect and locally check them, and continues experiment one.
Additional history or unrelated worktree changes fail closed.

The runtime has five separate positive-integer timeout settings:
`agent_turn_timeout`, `controller_transaction_timeout`, `verifier_timeout`,
`backend_job_timeout`, and `timeout_grace`. The BZ job client receives
`backend_job_timeout`; the controller backend receives that value plus one
grace interval; and `controller_transaction_timeout` must exceed the backend
job timeout plus two grace intervals. Agent and verifier budgets are
independent. The ambiguous legacy `timeout` setting is rejected, so changing
one boundary cannot silently shorten a different subprocess or remote job.

The BZ job client is part of that pinned controller closure. It accepts only
the fixed global `remote-access/scripts/cpl-remote` boundary and its required
SHA-256. Runtime configuration cannot provide an adapter command, remote
command, or alternate `cpl-remote` path. Its placement file uses
`{"logical-id":{"target":"bz-a3-1","device":N}}`; physical placement never
enters the agent prompt. Launch with:

```bash
python scripts/audited_campaign_production.py \
  --manifest /absolute/campaign/manifest.json \
  --runtime-config /absolute/campaign/runtime.json \
  --runtime-config-sha256 RUNTIME_CONFIG_SHA256 \
  --admission /absolute/campaign/admission.json \
  --admission-sha256 ADMISSION_SHA256 \
  --ledger /absolute/campaign/ledger.json
```

Repeat the identical command with `--resume` after infrastructure recovery.
Every successful admission refresh is atomically recorded before its slots are
returned in `RUN_ROOT/state/admission-evidence.json`, including its sequence,
complete receipt SHA-256, provider and allowlist identities, and exact
generation and expiration timestamps. An expired v2 receipt admits no slot, so
the scheduler pauses without dispatching duplicate work; resume requires a new
hash-pinned receipt with the same provider and allowlist identities.
Each cell retains its isolated clone, experiment branch, Codex state,
controller checkpoint, physical placement, and BZ job receipts. A retained
handle is reobserved through `CommandController`; it is never replaced by a
new dispatch. Runtime controller configuration uses logical device zero and
stores the physical target/device only in the private placement evidence.
Resume is permitted only from a lifecycle-validated blocked checkpoint. A
four-round cell requires `profiling-skill/audited-blocked/v2` with immutable
`round_count: 4`; the v1 checkpoint remains a legacy three-round format and is
rejected here. If a ledger is lost after a branch completed, the launcher
independently runs `validate_audited_experiment.py` and reconstructs the
terminal receipt instead of restarting the agent. Every new completion also
passes that independent verifier before it can be recorded as successful.

Initial cell materialization uses a sibling initializing directory with an
atomically written, exact cell-identity marker. Clone, checkout, skill copy,
and identity creation finish there before one atomic rename publishes the
cell. A restart may delete and recreate only a partial directory whose marker
matches that exact cell identity; malformed, absent, or different markers fail
closed and are preserved for inspection. Every fresh start and resume also
requires the repository-local skill directory to contain exactly the declared
treatment allowlist and re-hashes each copied tree against its pinned source
binding before Codex is invoked.

## Stacked integration order

Production execution is intentionally a dependent stack. PR #48 supplies the
configurable four-round lifecycle and v2 checkpoint contract. PR #50 supplies
the pinned global `cpl-remote` BZ job-client boundary, and PR #51 supplies the
controller and compact timing evidence contract. This production integration
PR (#52) lands only after #48, #50, and #51. Its real-composition test may be
dependency-gated on the isolated branch, but must run unskipped in a checkout
containing all four heads before #52 is merged.

Cell receipts retain each complete compact controller receipt, including
per-case measurements, admission and post-run controls, infrastructure
attempts, baseline fields, and calibration fields. Codex, controller,
verification, and local lifecycle failures are infrastructure failures.
Compilation, runtime, and correctness failures are candidate evidence only
when they arrive in a contract-valid `candidate_error` controller receipt.
The agent may repair a failed candidate twice inside the same numbered round.
Each attempted candidate remains hash-bound in that round's evidence, but only
the final attempt determines the round status. A round is terminal
`candidate_error` only after all three attempts fail.

## Repair-aware acceptance and rollout

New manifests default to 48 controller operations per cell; frozen legacy
manifests with 24 operations remain valid. Production runtime configuration
sets `max_candidate_repairs_per_round` to `2` (omission means the same value
for legacy configuration compatibility). Run one matmul canary for each of the
three treatments before the measured campaign. Together they must demonstrate
an in-round compile/smoke repair, exact checkpoint/session resume, independent
offline verification, and a positive `msprof op` timing. Dynamic admission
chooses the physical BZ-A3 devices; canary definitions never encode devices.

Freeze runtime scripts and benchmark assets from the reviewed merge commit,
pin the unchanged prompt, tasks, starters, baselines, treatment skills, model,
image, and source revision, then generate a new run ID and 48-operation
manifest. Validate all hashes with `ProductionCellLauncher` before producing
an admission receipt. Do not copy a prior ledger or cell worktree into the new
run root, and do not launch the nine cells until all three canaries pass.

The version-two final report contains one row per branch with status, attempt
count, unchanged raw round receipts, per-case evidence, controls and policy,
calibration-normalized timing, best raw and normalized medians, frozen baseline
evidence, speedup, and retained failure detail. A separate discarded-attempts
table records every infrastructure exclusion with its target, device, handle,
and diagnostic. Infrastructure-pending cells remain visibly separate from
candidate failures. Each round's normalized samples, median, calibration,
baseline, and speedup are propagated from the validated controller receipt.
Best-round and cross-device/cross-run comparison fields use normalized timing
when present; raw timing remains available for device-local diagnosis. The
report also aggregates every candidate-error round, including earlier rounds,
with its round number, failure type, and reason.

Repair-aware reports additionally expose raw successful attempts over all
candidate attempts, successful and repaired round counts, total repairs,
ordered failure transitions, and the final round timing. Legacy branches are
reported as one candidate attempt per round.
