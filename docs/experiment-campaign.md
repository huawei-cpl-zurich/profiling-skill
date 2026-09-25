# A3 profiling experiment campaign

This runbook defines the reproducible measured campaign. It has six cells:
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
The opaque controller socket is the only route from an isolated agent to
kernel checking or profiling; the agent cannot see its implementation or the
other treatments.

## Freeze and configure

Use a new output directory for every freeze and campaign. The examples use
placeholders deliberately; machine-specific locations and credentials must
not enter the manifest or repository.

```bash
python scripts/campaign.py freeze-cannbot \
  --repository "$CANNBOT_REPOSITORY" \
  --output "$CANNBOT_FREEZE"

python scripts/generate_benchmark_config.py \
  --job-client-json '["<checked-in-gz-a3-job-client>"]' \
  --candidate candidate.py \
  --candidate-manifest candidate.manifest.json \
  --output "$CONTROLLER_CONFIG"
```

Create the manifest with `campaign.write_manifest(...)`, passing the one
frozen prompt, the `gdn` and `bsa` baseline directories, the project skill
root, and the frozen CANNBot directory. The generated controller config has
all six cell IDs and hard-binds their benchmark, treatment, device, five
development cases, all 50 correctness cases, and backend command. Preserve
both generated JSON files as campaign evidence; never hand-edit them.

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

The production launch has no implicit backend. Supply the generated cell
controller as a JSON string array through `--controller-json`; placeholders
are expanded per cell. First exercise the exact frozen manifest with
`--dry-run`, then remove that flag for production:

```bash
python scripts/campaign.py run \
  --manifest "$CAMPAIGN_MANIFEST" \
  --output "$CAMPAIGN_OUTPUT" \
  --controller-json '["python","scripts/experimentctl.py","--config","<controller-config>","--cell","{cell_id}"]' \
  --forbid "$HOST_SKILL_ROOT" \
  --dry-run
```

Each cell is one persistent Codex session with exactly three optimization
rounds, at most 12 controller requests, and a 60-minute wall-clock limit.
Compilation and the first launch are outside latency samples. Development
checks use the five pinned cases; a successful candidate must pass the final
50-case correctness check.

Performance is collected with `msprof op`. For each development case the
controller requests three captures, reports their median kernel latency, and
computes the cell score as the geometric mean of the five case medians. The
kernel name comes from the candidate manifest and is passed explicitly to
`msprof`; host-observed timing is diagnostic only and is not the score.

## Failures, evidence, and replay

Compilation, runtime, correctness, request-budget, and time-budget failures
belong to the candidate and count in success-rate results. Transport or
service failures, unhealthy or lost devices, model-service failures, and a
profiler failure also reproduced by a known-good canary are infrastructure
failures. Exclude those attempts, retain their evidence, and reschedule the
same cell in a new sandbox. Never submit a duplicate merely because observing
a durable `gz-a3:<job-id>` was interrupted; resume that handle through the
checked-in profile.

`ledger.json` is atomically checkpointed after every completed cell and on
failure or interruption. Preserve it together with:

- campaign and controller manifests and their hashes;
- prompt, baseline, project-skill, and frozen-CANNBot hashes and CANNBot commit;
- cell, treatment, model configuration, session and attempt IDs;
- physical device, controller request count, remote job handles, diagnostics,
  correctness outcomes, and `msprof op` evidence;
- exclusion classification and the replacement attempt, when applicable.

To replay, use the same frozen directories and manifests, choose a fresh
campaign output directory, and invoke the same production command. First run
`campaign.py preflight` on any retained cell sandbox that is being audited.
Compare manifest hashes before aggregating results; a changed prompt,
baseline, skill bundle, controller config, or CANNBot freeze defines a new
campaign rather than a replay.
