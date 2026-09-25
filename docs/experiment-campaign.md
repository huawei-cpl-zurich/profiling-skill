# A3 profiling experiment campaign

This runbook defines the intended measured campaign and identifies the gates
that still prevent a reproducible production run. The design has six cells:
three profiling treatments for each of the pinned GDN and BSA kernels. It does
not include the streaming matmul-add development control, and no measured
campaign has started yet.

## Frozen inputs

The candidate baselines come from
`huawei-cpl-zurich/ascendc-kernelgen-data`, branch `br_perf_baseline`, commit
`a42c54b916189500e2f7cb47640980f230f2eb65`. The checked-in copies and case
files live under `benchmarks/gdn` and `benchmarks/bsa`. Every cell for a given
benchmark starts from the same byte-identical baseline and receives the same
byte-identical prompt. `scripts/campaign.py` records SHA-256 digests for both
in the campaign manifest and checks them before every wave.

The development cases are the five longest cases from three warm-ups and
seven synchronized timing repetitions of every case:

| Benchmark | Source | Physical NPU | Development cases |
| --- | --- | ---: | --- |
| GDN | `30_ChunkGatedDeltaRule.py` | 0 | 40, 49, 47, 46, 45 |
| BSA | `54_BlockSparseAttnFwd.py` | 1 | 47, 46, 49, 44, 43 |

All 50 cases remain the final correctness gate. A submitted GZ-A3 job sees
its selected physical device as logical device 0.

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
- `project-only`: project `ascend-profiling` only; no CANNBot skills or plugin
  support are visible.

`campaign.py preflight` verifies the prompt, baseline, skill manifests, skill
hashes, plugin-support presence, and the absence of symlinks before launch.
The production launcher exposes an opaque controller socket as the only route
from an isolated agent to its host controller command; the agent cannot see
that implementation or the other treatments.

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
  --job-client-json '["<absolute-production-gz-a3-job-client>"]' \
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
  --prompt /absolute/campaign-inputs/prompt.md \
  --gdn-baseline /absolute/campaign-inputs/baselines/gdn \
  --bsa-baseline /absolute/campaign-inputs/baselines/bsa \
  --project-skill "$ABS_REPOSITORY" \
  --cannbot-freeze /absolute/campaign-inputs/cannbot-freeze \
  --controller-config /absolute/campaign-inputs/controller.json \
  --controller-json '["python","/absolute/approved/profiling-skill/scripts/experimentctl.py","--config","/absolute/campaign-inputs/controller.json","--cell","{cell_id}"]' \
  --output /absolute/campaign-inputs/campaign.json \
  --rounds 3 \
  --request-budget 12
```

Manifest generation creates a version 2 manifest and a sibling controller
bundle containing the normalized config, `experimentctl.py`, and
`benchmark_backend.py`. It hashes all three files, the path-independent argv
template, and the Python implementation, version, and executable bytes.
Backend commands in the config are rewritten to bundle-relative templates.
The manifest and ledger therefore contain no controller host paths. The
generated controller config contains all
six cell IDs, hard-binding their benchmark,
treatment, device, five development cases, all 50 correctness cases, and
backend command. Preserve generated JSON files as campaign evidence; never
hand-edit them.

The scheduler uses three fixed two-cell waves so no more than two agents run
at once and each benchmark stays on its assigned NPU:

| Wave | NPU 0 | NPU 1 |
| ---: | --- | --- |
| 1 | `gdn-cannbot` | `bsa-project-cannbot` |
| 2 | `gdn-project-cannbot` | `bsa-project-only` |
| 3 | `gdn-project-only` | `bsa-cannbot` |

## Readiness and production launch

Before measured work, use the streaming K-tiled matmul-add kernel as the
development and readiness control. Its deterministic contract has seven
correctness cases (tiny, dimension tails, wide and tall shapes) and three
separate performance cases (256, 1024 and 2048 square matrices). Three fresh
agents must each retrieve an injected compilation diagnostic, repair the
kernel, pass all correctness cases, and complete profiling without an
infrastructure failure.

The control's native A3 gate is currently pending because the managed GitHub
proxy prevented staging its branch into the one-shot runtime. This is an
environment gate, not a kernel result. Do not start or claim measured
experiments until the branch is available through the managed profile and the
three-agent readiness gate passes.

Production accepts only a version 2 controller-bound manifest. Before any
cell starts, it verifies the runtime and bundle, copies the bundle into the
campaign's private evidence directory, makes it read-only, and constructs the
controller command from that private copy. Resume reuses and verifies the
same copy. Changes to the original config or repository scripts after staging
cannot affect later waves.

There is currently no checked-in production GZ-A3 job client for
`benchmark_backend.py`. The placeholder used by config generation cannot run
a benchmark, so production execution is blocked and is not yet reproducible.
Do not substitute ad hoc SSH, Docker, transfer, or remote-agent calls. A future
job client must use the checked-in `$gz-a3` profile and preserve durable job
handles.

Once those gates are implemented, first exercise the exact frozen manifest
with `--dry-run`. Dry-run preparation uses host temporary roots named
`campaign-dry-run-*`, outside the campaign output root; interrupted runs can
leave those roots behind for inspection, and operators are responsible for
retention or cleanup under their site's artifact policy. The ledger records
all six planned cells with status `dry_run`. The production invocation may
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
rounds, a budget of 12 agent controller requests, and a 60-minute wall-clock
limit. Thus a successful cell can issue all 12 budgeted agent requests plus
the two mandatory host-owned terminal gates (14 controller executions total).
Candidate compilation, runtime, correctness, request-budget, and time-budget
failures are retained as candidate failures; they do not abort later cells or
waves. After three successful agent rounds, the host issues two additional
terminal requests outside the 12-request agent budget. It first runs exactly
`check --scope full`, which must identify the operation, cell, benchmark, and
device, report the configured 50 cases in order, set `passed=true`, and retain
at least one durable handle. It then runs exactly
`profile --repeats 3 --round 3`, which must carry the same exact identity,
report the five configured development cases in order, retain three samples
per case and 15 distinct durable handles, and report three repeats. A cell is
complete only after both host gates pass; their full JSON, stdout, stderr,
diagnostics, handles, and artifact paths are retained under the attempt and in
`ledger.json`.

The implemented profiling interface uses `msprof op`. The terminal profile
gate requests three captures for each development case, reports their median
kernel latency, and computes the score as the geometric mean of the five case
medians. The kernel name comes from the candidate manifest and is passed
explicitly to `msprof`; host-observed timing is diagnostic only and is not the
score. Completion therefore does not depend on trusting that the agent chose
to profile during its optimization turns.

## Failures, evidence, and replay

Compilation, runtime, correctness, request-budget, and time-budget failures
count as candidate failures. Transport or service failures, unhealthy or lost
devices, model-service failures, and terminal-gate protocol or transport
failures are infrastructure exclusions. Never manually resubmit merely
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

- campaign manifest and its recorded hashes, plus the private frozen
  controller bundle, normalized command template, and runtime identity;
- prompt, baseline, project-skill, and frozen-CANNBot hashes and CANNBot commit;
- cell, treatment, model configuration, session and attempt IDs;
- physical device, controller request count, remote job handles, diagnostics,
  correctness outcomes, and `msprof op` evidence;
- exclusion classification and the replacement attempt, when applicable.

Replay uses the same frozen directories and campaign manifest with a fresh
campaign output directory and the same production command. `campaign.py
preflight` can audit a retained cell sandbox. Compare the hashes that the
campaign manifest actually records before aggregating results. Until the
controller config and command are manifest-bound, operators must retain and
compare them separately; a changed value defines a new campaign.

Three operational gates remain:

- no checked-in production GZ-A3 JSON job client and source-staging route yet
  connects `benchmark_backend.py` to the managed profile;
- the streaming matmul-add native A3 gate is still pending;
- three fresh agents have not yet completed the required readiness runs.

In addition, the controller config and exact controller command are not yet
bound and hashed by the campaign manifest, which is a scheduler-level
reproducibility blocker. No measured six-cell campaign has run; that is the
current outcome, not a fourth gate.

Until those gates are cleared, the battery remains a functionally tested
scaffold rather than a completed reproducible measurement.
