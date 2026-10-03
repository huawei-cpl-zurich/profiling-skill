# A3 profiling experiment campaign

This runbook defines the intended measured campaign and identifies the gates
that prevent an invalid production run. The design has nine cells: three skill
treatments for each of GDN, BSA, and the streaming matmul-add control. No
measured campaign starts until the campaign PR is merged and devices 0-3 pass
managed preflight.

## Frozen inputs

The candidate baselines come from
`huawei-cpl-zurich/ascendc-kernelgen-data`, branch `br_perf_baseline`, commit
`a42c54b916189500e2f7cb47640980f230f2eb65`. The checked-in copies and case
files live under `benchmarks/gdn` and `benchmarks/bsa`. Every cell for a given
benchmark starts from the same byte-identical baseline and receives the same
byte-identical prompt. `prompts/kernel-optimization.md` is the canonical prompt;
copy it byte-for-byte and never customize it by benchmark or treatment.
`scripts/campaign.py` records SHA-256 digests for both the prompt and baseline
in the campaign manifest and checks them before every wave.

The development cases are the five longest cases from three warm-ups and
seven synchronized timing repetitions of every case:

| Benchmark | Source | Physical NPU | Development cases |
| --- | --- | ---: | --- |
| GDN | `30_ChunkGatedDeltaRule.py` | 0 | 40, 49, 47, 46, 45 |
| BSA | `54_BlockSparseAttnFwd.py` | 1 | 47, 46, 49, 44, 43 |
| Matmul | `streaming_matmul_add.py` | 0-3 | 7, 8, 9 |

GDN and BSA retain all 50 cases as their final correctness gates; matmul uses
its 10 frozen cases. A submitted GZ-A3 job sees its selected physical device
as logical device 0.

Immediately before freezing a measured campaign, `freeze-cannbot` resolves
the current `master` of the configured CANNBot repository, checks out that
exact commit, rejects symlinks and special files, and copies regular files
into an immutable bundle. `freeze.json` pins the commit and hashes every
copied skill and the Triton plugin support tree. Do not refresh the bundle
during a campaign.

## Treatments and isolation

Every agent runs in a fresh Bubblewrap namespace. Its writable workspace
contains only the selected baseline and its treatment-local
`.agents/skills`; host and global skill directories are not mounted. The
prompt, model (`gpt-5.6-sol`), reasoning effort (`low`), baseline, devices,
controller protocol, request budget, and time budget are identical across
treatments.

The treatments are:

- `cannbot`: the complete CANNBot Triton skill set
  (`triton-task-extractor`, `triton-op-designer`, `triton-op-coding`,
  `triton-op-verifier`, `triton-latency-optimizer`, and
  `triton-simulator-optimizer`), its `npu-arch` dependency and Triton plugin
  support, plus CANNBot `ops-profiling`.
- `project-cannbot`: the same complete CANNBot Triton set, dependency and
  plugin support, but project `ascend-profiling` replaces `ops-profiling`.
- `project-guarded`: project `ascend-profiling` plus the frozen
  `triton-guarded-kernel`; no CANNBot skills or plugin support are visible.

`campaign.py preflight` verifies the prompt, baseline, skill manifests, skill
hashes, plugin-support presence, and the absence of symlinks before launch.
The production launcher exposes an opaque controller socket as the only route
from an isolated agent to its host controller command; the agent cannot see
that implementation or the other treatments.

## Adaptive diagnostic waves

The short BZ diagnostic campaign is advanced one wave at a time with
`scripts/one_shot_bz_campaign.py --action run-wave --wave N`. A successful
wave atomically pauses its ledger in `awaiting_curation`; it does not launch
the next wave. Submit the curator-produced JSON with `--action
acknowledge-curation --curation-receipt /absolute/receipt.json`. The receipt
must identify the completed wave, set `accepted` to true, and contain nonempty
`stable_ref_citations` (`ref://...`) and `librarian_query_ids` arrays. It also
binds the curator operation to the ledger's `campaign_id` and canonical
`wave_sha256`; receipts from another campaign or evidence revision are rejected.

Acknowledgement moves waves 1-3 to `ready_for_next` and Wave 4 to `complete`.
Only the prompt path and digest may change between waves. The CLI rejects
model, treatment, skill, timeout, asset, or placement drift before launching
an agent, and completed countable cells are never relaunched. The legacy
`run-all` action remains available for fixed non-adaptive tests.

If a local terminal invocation times out without returning a durable handle,
the cell pauses for manual reconciliation and must not be replayed. Recover it
with `--action reconcile-terminal`, the original `--cell-id`,
`--agent-attempt`, and `--terminal-attempt`, plus exactly one of
`--terminal-handle HANDLE` or `--terminal-result /absolute/result.json`.
Handle recovery locates the exact durable BZ dispatch receipt and observes it;
it fails closed when no unique receipt owns the handle. A supplied result must
carry the campaign, request, cell, attempt, job-handle, and frozen-candidate
identities recorded by the uncertain receipt.

## Two-shot smoke gate

Before a longer campaign, run the configured smoke schedule with
`one_shot_bz_campaign.py --action run-smoke` and
`experiments/two-shot-matmul.json`. Waves 1 and 2 run all three treatments;
wave 3 runs only `cannbot`. Each cell is one persistent Codex session:
Round 1 writes and checks a candidate, Round 2 receives the same context and
diagnostics, must change `candidate.py`, and checks it once more. The host then
runs the configured correctness cases against a read-only frozen submission.
Profiling is unavailable in this mode.

Infrastructure results are excluded and retried once; compiler, runtime,
correctness, protocol, budget, and agent-time failures count. All configured
trials run even after a treatment reaches its minimum. The gate requires at
least one success in three counted `cannbot` trials and at least one success in
two counted trials for each project treatment. Once all exact trial targets and
their independent thresholds are satisfied, BSA can start with
`experiments/two-shot-bsa.json` and `--matmul-gate` pointing to that ledger.
Both configurations use the same prompt and treatment isolation. The usual
manifest, placements, BZ adapter/state, agent-command, and run-root arguments
remain required; use a fresh run root for each benchmark.

The ledger retains raw cell evidence and adds a bounded structured summary for
every counted failure, including phase, diagnostics, candidate hashes, and
durable handles. In the existing failed `cannbot` matmul cell, round 1 omitted
the required `Model` entry point. Round 2 repaired that entry point but used
`triton.cdiv` inside the JIT kernel, which is invalid kernel-language code; the
result was not caused by a missing `@triton.jit` decorator.

The smoke ledger is written before dispatch and after every completed cell.
If a run stops or ends in `infrastructure_pending`, repeat the same
`run-smoke` command with `--resume-smoke`; resume verifies the frozen inputs,
preserves successful and counted cells, and creates fresh attempts only for
ordinary infrastructure exclusions. `reconciliation_required` means a durable
terminal observation is uncertain and must be reconciled through its retained
handle; resume will not redispatch that cell.

## Freeze and configure

Use a new output directory for every freeze and campaign. The examples use
placeholders deliberately; machine-specific locations and credentials must
not enter the manifest or repository.

Set `ABS_REPOSITORY` to an operator-approved, frozen operational checkout and
validate that it is absolute before running any command. It must not be the
managed canonical checkout or a task implementation worktree: those locations
remain subject to the workspace's no-build/no-run and task-isolation rules.

```bash
ABS_REPOSITORY=/absolute/approved/profiling-skill
case "$ABS_REPOSITORY" in /*) ;; *) exit 2 ;; esac
test -f "$ABS_REPOSITORY/scripts/campaign.py"
test -f "$ABS_REPOSITORY/scripts/experimentctl.py"
```

```bash
python "$ABS_REPOSITORY/scripts/campaign.py" \
  freeze-cannbot \
  --repository https://gitcode.com/cann/cannbot-skills.git \
  --output /absolute/campaign-inputs/cannbot-freeze

python "$ABS_REPOSITORY/scripts/generate_benchmark_config.py" \
  --job-client-json '["python","/absolute/approved/profiling-skill/scripts/gz_a3_job_client.py","--adapter-json","[\"/absolute/orchestration/execution-profiles/catlass-validation.sh\"]","--state-dir","/absolute/campaign-state/gz-a3-jobs"]' \
  --candidate candidate.py \
  --candidate-manifest candidate.manifest.json \
  --output /absolute/campaign-inputs/controller.json
```

Generate the campaign manifest with the checked-in executable CLI. The
following commands are deliberately absolute so the manifest records resolved
inputs and the host controller never depends on an agent's working directory.
Replace only the external artifact roots with absolute paths on the campaign
host; do not use relative paths.

```bash
python "$ABS_REPOSITORY/scripts/campaign.py" \
  generate-manifest \
  --prompt "$ABS_REPOSITORY/prompts/kernel-optimization.md" \
  --gdn-baseline /absolute/campaign-inputs/baselines/gdn \
  --bsa-baseline /absolute/campaign-inputs/baselines/bsa \
  --matmul-baseline /absolute/campaign-inputs/baselines/matmul \
  --project-skill "$ABS_REPOSITORY" \
  --guarded-skill /absolute/cpl-skills/skills/triton-guarded-kernel \
  --guarded-skill-revision 7fe1230a68487a954a479c5c1ed4473b760c7ac6 \
  --cannbot-freeze /absolute/campaign-inputs/cannbot-freeze \
  --controller-config /absolute/campaign-inputs/controller.json \
  --controller-json '["python","/absolute/approved/profiling-skill/scripts/experimentctl.py","--config","/absolute/campaign-inputs/controller.json","--cell","{cell_id}"]' \
  --output /absolute/campaign-inputs/campaign.json \
  --rounds 3 \
  --request-budget 18
```

Manifest generation creates a version 2 manifest and a sibling, self-contained
controller bundle. The bundle preserves `scripts/` and `benchmarks/` layout and
contains the controller, benchmark adapter, managed GZ-A3 job client, remote
runner, profiler, and all three pinned benchmark assets. Every regular file is declared and
hashed recursively alongside the path-independent controller argv and Python
runtime identity. Backend and nested job-client commands are rewritten to
bundle-relative templates; only the approved neutral-adapter argv and private,
writable job-state directory remain external. The manifest and ledger therefore
contain no mutable operational-checkout script paths. The
generated controller config contains all nine cell IDs, hard-binding their benchmark,
treatment, device, benchmark-specific development and correctness cases, and
backend command. Preserve generated JSON files as campaign evidence; never
hand-edit them.

The scheduler uses fixed 4/4/1 waves with no device collision:

| Wave | NPU 0 | NPU 1 | NPU 2 | NPU 3 |
| ---: | --- | --- | --- | --- |
| 1 | `gdn-cannbot` | `bsa-cannbot` | `matmul-cannbot` | `bsa-project-guarded` |
| 2 | `matmul-project-guarded` | `gdn-project-cannbot` | `bsa-project-cannbot` | `matmul-project-cannbot` |
| 3 | — | — | `gdn-project-guarded` | — |

Immediately before and after every wave, the host profiles frozen matmul case 7
on all four devices through the same managed `msprof op` route. Drift above the
frozen threshold (10% by default), missing handles, or invalid latency discards
and reschedules the whole wave. Results retain raw latency and report
`raw_candidate_latency * C0 / Cd`, where each `C` is the geometric mean of its
device's bracketing calibration and device 0 is canonical.

## Readiness and production launch

Before measured work, use the streaming K-tiled matmul-add kernel as the
development and readiness control. Its deterministic contract has seven
correctness cases (tiny, dimension tails, wide and tall shapes) and three
separate performance cases (256, 1024 and 2048 square matrices). Three fresh
agents must each retrieve an injected compilation diagnostic, repair the
kernel, pass all correctness cases, and complete profiling without an
infrastructure failure.

Mutable candidates are staged with `gz_a3_job_client.py` through the neutral
profile adapter's content-addressed managed-bundle actions. Upload, command,
and result-transfer receipts live below the configured private state
directory. Repeating an identical request resumes those receipts and never
submits a duplicate command. The client has no raw SSH, SCP, Docker, or
remote-agent route.

Production accepts only a version 2 controller-bound manifest. Before any
cell starts, it verifies the runtime and bundle, copies the bundle into the
campaign's private evidence directory, makes the complete tree read-only, and constructs the
controller command from that private copy. Resume reuses and verifies the
same copy. Changes to the original config or repository scripts after staging
cannot affect later waves; missing, added, or changed bundled assets stop the
campaign before launch. Mutable transfer receipts and job results remain in the
configured external state directory rather than the controller evidence tree.

The checked-in production client returns candidate compilation, runtime, and
correctness failures as counted results, including compiler tracebacks. A
profile/device/service/transfer failure is an infrastructure exclusion. Every
successful profile response binds the exact case and kernel name and includes
compact `msprof op` evidence plus the durable `gz-a3:<job-id>` handle.

Once those gates are implemented, first exercise the exact frozen manifest
with `--dry-run`. Dry-run preparation uses host temporary roots named
`campaign-dry-run-*`, outside the campaign output root; interrupted runs can
leave those roots behind for inspection, and operators are responsible for
retention or cleanup under their site's artifact policy. The ledger records
all nine planned cells with status `dry_run`. The production invocation may
safely reuse the same campaign output root because production creates fresh
attempt directories and replaces the dry-run ledger state.

```bash
python "$ABS_REPOSITORY/scripts/campaign.py" run \
  --manifest /absolute/campaign-inputs/campaign.json \
  --output /absolute/campaign-results/run-001 \
  --python /absolute/frozen/python \
  --forbid /absolute/host-skill-root \
  --dry-run

python "$ABS_REPOSITORY/scripts/campaign.py" run \
  --manifest /absolute/campaign-inputs/campaign.json \
  --output /absolute/campaign-results/run-001 \
  --python /absolute/frozen/python \
  --forbid /absolute/host-skill-root
```

Each cell is one persistent Codex session with exactly three optimization
rounds, a budget of 18 billed agent controller requests, and a 60-minute
wall-clock limit. Local `help` and `budget` requests are free. Invalid
controller arguments are rejected before billing. Thus a successful cell can
issue all 18 billed agent requests plus the two mandatory host-owned terminal
gates (20 billed or host-owned controller executions total), as well as any
number of free local introspection requests. Candidate compilation, runtime,
correctness, and time-budget failures are retained as candidate failures; they
do not abort later cells or waves. Exhausting the request budget is recorded as
metadata and prevents further agent requests, but does not suppress the
host-owned terminal gates or invalidate a candidate that passes them. After
three agent rounds, the host issues two terminal requests outside the
18-request agent budget. It first runs exactly
`check --scope full`, which must identify the operation, cell, benchmark, and
device, report the configured benchmark-specific full case set in order (50
for GDN/BSA and 10 for matmul), set `passed=true`, and retain
at least one durable handle. It then runs exactly
`profile --repeats 3 --round 3`, which must carry the same exact identity,
report the configured development cases in order, retain three samples per
case, retain exactly one durable handle for the managed batch job, and report
three repeats. A cell is
complete only after both host gates pass; their full JSON, stdout, stderr,
diagnostics, handles, and artifact paths are retained under the attempt and in
`ledger.json`.

The first Codex turn receives the canonical prompt and is explicitly limited
to Round 1. The two resume turns each name exactly one current round and require
the agent to stop before beginning a later round. This preserves one persistent
session without allowing a single turn to consume multiple experimental rounds.

The implemented profiling interface uses `msprof op`. The terminal profile
gate requests three captures for each development case, reports their median
kernel latency, and computes the score as the geometric mean of the five case
medians. The kernel name comes from the candidate manifest and is passed
explicitly to `msprof`; host-observed timing is diagnostic only and is not the
score. Completion therefore does not depend on trusting that the agent chose
to profile during its optimization turns.

## Failures, evidence, and replay

Compilation, runtime, correctness, and time-budget failures count as candidate
failures. Request-budget exhaustion is retained as experiment metadata; the
host still judges the frozen submission through its terminal gates. Transport
or service failures, unhealthy or lost devices, model-service failures, and
terminal-gate protocol or transport failures are infrastructure exclusions.
Never manually resubmit merely
because observing a durable `gz-a3:<job-id>` was interrupted; resume that
handle through the checked-in profile.

`ledger.json` is atomically checkpointed after every completed attempt and on
failure or interruption. Candidate failures remain terminal observations and
the scheduler continues through later cells. Infrastructure results are also
retained, and their cell IDs alone are placed in `reschedule`. An explicit
`--resume` runs only those unresolved infrastructure cells in fresh attempt
directories; it neither repeats successful cells nor candidate failures.

```bash
python "$ABS_REPOSITORY/scripts/campaign.py" run \
  --manifest /absolute/campaign-inputs/campaign.json \
  --output /absolute/campaign-results/run-001 \
  --python /absolute/frozen/python \
  --forbid /absolute/host-skill-root \
  --resume
```

The complete evidence set to preserve across the ledger and its referenced
artifacts is:

- campaign manifest and its recorded hashes, including the canonical prompt
  SHA-256, configured request budget, and exact controller-help text or digest,
  plus the private frozen
  controller bundle, normalized command template, and runtime identity;
- prompt, baseline, project-skill, and frozen-CANNBot hashes and CANNBot commit;
- cell, treatment, model configuration, session and attempt IDs;
- physical device, controller request count, remote job handles, diagnostics,
  correctness outcomes, and `msprof op` evidence;
- exclusion classification and the replacement attempt, when applicable.

Replay uses the same frozen directories and campaign manifest with a fresh
campaign output directory and the same production command. `campaign.py
preflight` can audit a retained cell sandbox. Compare the hashes that the
campaign manifest records before aggregating results. Also compare the frozen
18-request budget and controller-help metadata so a replay cannot silently use
a different agent-facing protocol. The controller config,
bundle closure, command template, Python runtime identity, benchmarks, prompt,
and skill trees are all manifest-bound. Native matmul calibration has passed
on A3 with its exact selector. The remaining operational gate is successful
preflight of devices 0-3 immediately before measured work. No measured
nine-cell campaign has run yet.
