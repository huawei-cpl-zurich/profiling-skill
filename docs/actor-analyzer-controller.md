# Alternating actor and analyzer (opt-in)

This runner leaves the existing optimization launcher/default unchanged. Each
round gives an actor 1,200 seconds, then a distinct analyzer 600 seconds; four
rounds are the default. All tool waits consume that role's allowance. Actor-only
controls receive the same actor allowance and zero analyzer allowance. Report
the additional analyzer time/cost rather than claiming equal-total-budget wins.

Run from a development worktree, placing experiment output outside the
orchestration checkout:

```sh
python scripts/actor_analyzer_controller.py \
  --output /tmp/actor-analyzer-example --bundle /tmp/exported-public-task \
  --model-json /tmp/model.json --tool-command-json /tmp/tools.json \
  --instruction /tmp/task.md --skill /tmp/pinned-profiling-skill \
  --coding-skill /tmp/pinned-coding/triton-guarded-kernel
```

`model.json` declares `name` and `reasoning_effort`. `tools.json` is a JSON array
containing the trusted tool CLI command, including its private `--bundle` and
remote profile arguments. The controller appends `--workspace` and the agent's
operation. The CLI must implement the existing neutral `help`, `check`, `timing`,
`pipes`, and `observe CALL_ID` interface and return correctness-qualified results
with authoritative `source_sha256`. Native Tools implements this interface;
transport remains exclusively the installed remote-access client. A public
`task.json` manifest declares only the initial `candidate.py` and API files.
Private harnesses, input fixtures, comparators, and tools remain outside mounts.

Use `--actor-only` for the paired control; `--rounds`, `--actor-budget`, and
`--analyzer-budget` are independently configurable. Model and task configuration
cannot change on resume. Each phase gets a separate session, deadline, transcript,
command journal, and receipt. Candidates and compact evidence are retained on one
local `experiment/actor-analyzer` Git branch; nothing is merged or pushed.
Repeatable `--coding-skill DIRECTORY` freezes explicitly selected coding or
verification skill directories (each needs `SKILL.md` and no symlinks). Actors
alone see read-only copies under `/workspace/.agents/skills/<directory-name>`;
the analyzer never receives them. No global skill roots are mounted. Use the
same explicit coding snapshots in actor-only and paired cells.
Profiling guidance is likewise copied once into the campaign, with a persisted
content manifest; every role receives that frozen copy, never a fresh copy of
the original directory. Original-source edits cannot alter an ongoing campaign.
Resume with changed source contents fails explicitly rather than changing the
treatment, and a modified retained profiling snapshot is rejected.
The full trusted bundle (including private runner/fixture files) is content-pinned
and checked before every phase and on resume. Retained coding snapshots are
likewise verified after copying and before use; input drift stops the campaign.

The actor writes `/workspace/candidate.py` and experiment logs. The analyzer sees
a read-only clone at the last correctness-qualified commit, including read-only
Git metadata. It writes `/analysis/analysis.json` using the contract supplied in
its prompt. Tool evidence appears read-only beneath `/workspace/evidence`; cite
paths such as `evidence/CALL_ID/results/summary.json`. The controller copies and
hash-checks those trusted files into the retained report. A claimed acquired
profile must include actual pipe rows for the frozen candidate source, not merely
an agent-created CSV or a report claiming that collection worked. Unsupported
profiling must be reported as unavailable, with the failure and uncertainties.
Trusted artifacts are archived separately under each experiment's
`trusted-evidence`, so a malformed `/analysis/evidence` file or symlink cannot
overwrite them or prevent raw analyzer output from being retained.

For single-kernel tasks, correctness-only `check`/`ir` results do not qualify the
final launch contract. Finish with `timing` or `pipes` returning
`launch_contract.profile_stream_verified: true`; ordinary timing verification
does not require the actor to perform diagnostic profiling analysis.

An invalid latest candidate does not replace the verified revision. If none has
passed, analysis is explicitly skipped. Failed or timed-out actors still have
their code, logs, and diagnostics committed. Analyzer failure does not erase an
earlier valid analysis; reports identify exactly which candidate they describe.

Rerun the same command to resume. A terminal receipt is adopted without another
agent dispatch. An interrupted phase without a terminal receipt returns
`observation-required`: inspect the existing process/transcript rather than
starting a second session or renewing its deadline. Pending native tool jobs
block new remote work; `observe CALL_ID` routes to the original tool workspace
and retains its original deadline and handle, even across role transitions.
The controller's host-side watchdog uses the original role deadline to cancel
phase-owned outstanding jobs through the installed managed client, including
jobs still queued remotely. It also performs cleanup when a role exits early.
Unknown handles or unconfirmed cancellation stop progression before another
role starts; resume observes/cleans up that same work, without renewing budgets.
Malformed/incomplete analyzer reports and compact diagnostics are still archived
and committed, but cannot replace the last valid actor handoff.

Promotion to a default requires the three paired natural-task gate (gather,
masked compute, matmul): real profile acquisition, defensible interpretation,
and demonstrable evaluation by the next actor. Timing improvement is not itself
required, and truthful negative findings remain useful evidence.

## Reproducing the three paired natural-task gate

Use `benchmarks/natural_before_tasks.py`, **not** the older
`blind_before_tasks.py`: their task identifiers describe different workloads.
Prepare an ignored export outside this checkout. The controller commit, model
JSON, complete profiling skill snapshot, task instructions, and tool arguments
must be identical in the two arms of each task. Do not copy an optimized answer
into either arm. Record the exact source commit before exporting its files:

```sh
test -z "$(git status --porcelain --untracked-files=no)"
gate_dir=$(mktemp -d /tmp/actor-analyzer-gate.XXXXXXXX)
mkdir "$gate_dir/source"
git rev-parse HEAD > "$gate_dir/source-commit.txt"
git archive HEAD | tar -x -C "$gate_dir/source"
```

Before the following preparation, set `PROFILE_SKILL` to the chosen complete
profiling-skill snapshot, `CODING_SKILL` to the managed source resolved from the
installed `triton-guarded-kernel` skill, and `MODEL_JSON` to the approved model JSON.
Set `A3_TARGET` and `A3_RUNTIME` from the ignored remote-access registry; these
machine-specific choices are not source defaults. Copying freezes the skill and
model for all six cells. Retain the printed gate directory in the central task
record so resume uses the same output directories, not a fresh campaign.

```sh
cp "$MODEL_JSON" "$gate_dir/model.json"
cp -aL "$PROFILE_SKILL" "$gate_dir/skill"
mkdir "$gate_dir/coding"
cp -aL "$CODING_SKILL" "$gate_dir/coding/triton-guarded-kernel"
for task in task01 task03 task05; do
  python "$gate_dir/source/benchmarks/natural_before_tasks.py" export \
    --task "$task" --output "$gate_dir/$task"
done
python - "$gate_dir" "$A3_TARGET" "$A3_RUNTIME" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
for task in ('task01', 'task03', 'task05'):
    command = [sys.executable, str(root/'source/scripts/blind_optimization_tools.py'),
               '--bundle', str(root/task), '--target', sys.argv[2], '--runtime', sys.argv[3]]
    (root/(task+'-tools.json')).write_text(json.dumps(command))
files = [p for p in root.rglob('*') if p.is_file()]
(root/'inputs.json').write_text(json.dumps({
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
}, indent=2))
PY
```

First check remote-access capabilities/preflight and `npu-smi` responsiveness,
then run the starting candidates through the same trusted tool CLI on eligible
idle devices. Retain correctness and actual pipe-acquisition evidence before
spending model budgets. A responsive SSH connection alone is not readiness.
Do not run six model cells against unavailable/busy devices. Once enough idle
capacity is confirmed, this loop specifies the exact six independent runs:

```sh
for task in task01 task03 task05; do
  for arm in actor-only actor-analyzer; do
    control=()
    if [ "$arm" = actor-only ]; then control=(--actor-only); fi
    python "$gate_dir/source/scripts/actor_analyzer_controller.py" \
      --output "$gate_dir/cells/$task-$arm" --bundle "$gate_dir/$task" \
      --model-json "$gate_dir/model.json" \
      --tool-command-json "$gate_dir/$task-tools.json" \
      --instruction "$gate_dir/$task/TASK.md" --skill "$gate_dir/skill" \
      --coding-skill "$gate_dir/coding/triton-guarded-kernel" \
      --rounds 4 --actor-budget 1200 --analyzer-budget 600 "${control[@]}" \
      > "$gate_dir/$task-$arm.log" 2>&1 &
  done
done
wait
```

This is a Bash example. If fewer than six devices are available, launch only as
many paired cells as fit, then run the remaining task pairs with the same
arguments and previously unused output paths. Do not rerun completed cells to
fill capacity. A controller interrupted after dispatch must observe its existing
phase/receipt rather than starting a duplicate. Each cell has its own unmerged
Git repository/branch, retained with all phase outputs; the operator never merges
an experimental candidate as part of this gate.

Keep a manually reviewed `assessment.json` beside `inputs.json`, containing one
entry for each task with both cell paths, exact branch heads, native handles,
devices, kernel selectors/cases, retained report paths, and these judgments:

- Four actor phases per arm completed or have explicit failure/timeout evidence;
  all four paired analyzer phases have a retained report or explicit failure.
- At least one paired analysis per task acquired real candidate-specific pipe
  metrics. Verify its source digest/commit, exact metric rows and units against
  the trusted summary—not just an agent's assertion that profiling happened.
- Inspect the actual analyzer reasoning for defensible bottleneck conclusions,
  confidence, roofline assumptions/bounds, and honest unknowns. Resource activity
  is not achieved bandwidth; unknown ceilings remain unknown. An analyzer that
  only prescribes tuning configurations has not satisfied its role.
- Inspect the *next actor's* transcript, commands, and code diff for an actual
  evaluation of those findings. Cite the relevant evidence and experiment
  commit; a supported rejection counts, reading the report alone does not.
- Retain negative results and infrastructure interruptions separately from
  candidate compilation/numerical failures. Include correctness and timing for
  both arms without selecting only successful runs.

Set `accepted` only when all three task entries pass acquisition,
interpretation, and next-actor evaluation. Otherwise retain `accepted: false`
with reasons; do not promote the controller to a default. Each arm has 4,800
configured actor seconds. The paired arm additionally has 2,400 configured
analyzer seconds: report configured and observed elapsed time separately,
including timeouts and infrastructure waiting. Assessment is a retained human
audit artifact, not an automated claim of reasoning quality or speedup.

### Structured round history (contract 2)

Fresh campaigns use analysis contract 2. Campaigns whose saved configuration
predates that version resume with their original report contract. The role
budgets and tool capture counts are unchanged.

The controller exposes read-only `/history/context.json`, including the frozen
source hash, source diff, previous distinct verified round, repeated-round
identity, and a compact index. Each `round-N` retains exact `candidate.py`, its
commit/hash, original compact evidence, validation status, and the validated
`analysis.json` when available. Invalid reports remain in phase diagnostics;
the latest handoff explicitly identifies failure instead of silently retaining
an older report.

The analyzer adds `contract_version: 2`, `source_sha256`, nonempty `optimizations`
and `utilization` lists, nonempty roofline `ceilings`, and a `comparison` object.
See the supplied history prompt for field shapes. Observed optimization entries
cite exact changed diff lines. Utilization entries cover every cited profile
case/kernel with measured evidence or explicit unavailability. Bottlenecks carry
confidence and original evidence references. A new kernel requires a recorded
`pipes` tool attempt; failed acquisitions produce honest incomplete reports.
Unchanged kernels can reuse their original retained evidence without recapture.

Comparisons must include the previous distinct verified kernel and may select
one additional older distinct kernel with a reason. Controller code computes
numeric deltas from original summaries, including non-runtime pipeline metrics;
it does not accept analyzer-authored numeric pairs. Pairing requires common
workload, node, runtime, cases, and capture settings. Different physical devices
on the same node are equivalent. Unattributed multi-case kernel rows are not
averaged. Changed kernel decomposition retains workload timing comparisons and
explicitly reports missing kernel pairs. Roofline fractions are estimates whose
assumptions and sources remain visible.

The next actor reads the latest status, validated report, selected historical
reports and kernel sources through `/history`, and records which findings it
considered. The analyzer interprets evidence; the actor chooses optimizations.

Compatible successful acquisitions within a round are reduced to one median per
metric before pairing, with acquisition counts and original call/handle provenance
retained in each pair. Standalone CSV metrics accept finite numeric strings;
seeded correctness IDs identify the validation workload, while a verified
single-launch profile stays explicitly labeled as a workload aggregate rather
than being attributed to each seed. Historical comparison selections must
represent different source hashes, including when a kernel recurs in later rounds.

### Multi-operator batches and checkpoints

`model_campaign_tools.py prepare --task matmul|gdn|bsa --output DIR`
accepts `--reference-root` for private comparison references. Optional
`--profile-index` and `--check-indices` select the explicit supplied case and
bounded per-call correctness subset. Full baseline/case qualification remains
a prerequisite outside worker calls. Only API, task descriptor and original
naive starter are released; reference sources and fixtures stay private.

`actor_analyzer_controller.py --stop-after-round 1` checkpoints a configured
four-round campaign without changing its immutable budgets or round count.
Resume using the same output and configuration without the checkpoint option.
The existing phase records prevent replaying completed roles.

The trusted `campaign_batch.reconcile` boundary takes retained controller states
and device capacity, then returns durable admissions or cancellation requests.
Persist decisions before acting. Only admitted cells may start role clocks.
Three successful matmul round-one checkpoints release the remaining cells and
matmul continuations. Terminal matmul candidate/analyzer failures permanently
freeze admission. Record batch-owned remote handles in `handles`; invoke
`cancel_owned` with the installed cpl-remote cancellation boundary and observe
nonterminal cancellation receipts. Lost observation never permits redispatch.

Use the trusted tool CLI `--devices 2,5` to constrain physical device admission.
The comma-separated list must contain distinct nonnegative IDs. Inventory is
filtered before per-device queries, leases, or probes; missing configured IDs
fail closed. Allowed busy devices wait for isolation. The application still
uses logical device zero. A retained live phase keeps its original allowance;
observe/cancel that handle before starting a phase with a different allowance.

Trusted cleanup must invoke the tool CLI `stop` and require `worker_closed:true`
before releasing a phase. The controller does this automatically. A terminal
outer transport handle alone does not prove its Python worker has exited.
Shutdown retains its own control handle, verifies private PID/start-time/cwd
identity, requests cooperative exit, and escalates only verified owned processes
when necessary. Repeat `stop` to observe a pending control; do not redispatch it.
This operation is unavailable through the public actor/analyzer tool bridge.

Within one retained phase, timing and pipe measurements may reuse a passed
source qualification. Explicit checks always run the full configured case and
variant subset. Reuse requires identical source, trusted bundle fingerprint,
adapter, physical device and runtime context. Measurement still executes fresh
profile-case preparation and validation. Exported `correctness_provenance`
identifies the original call and source hash; reused reports contain no fresh
backend-check diagnostics. Failed checks or measurements invalidate that proof.
Final benchmark case coverage, samples, and role deadlines remain unchanged.

For a new distinct contract-2 kernel the controller now acquires the required
profile before starting the analyzer model. The managed tool performs staging,
checks, capture and same-handle observation inside an acquisition subdeadline
that ends at the original reporting reserve. Process termination allowance is
also reserved. The model receives only the remaining original role allowance;
its execution context keeps the original role deadline. The warm tool worker may
expire at the earlier acquisition cutoff, so further device tools are not
promised during report writing.

`required_profile` in the read-only current context identifies the exact capture
or retained failure/observation state. A successful capture must be cited in the
report. `required-profile.json` is retained with experiment diagnostics and the
immutable round record, including the actual current-source tool attempt. A lost
observer resumes the original call; dispatch intent without a recoverable call
requires observation and never permits another dispatch. Unchanged kernels use
retained evidence, while actors, legacy campaigns and preflight checks skip this
acquisition. Failed acquisitions remain honest unavailable reports.

If the controller stops during required acquisition, `Campaign.run` can reattach
that analyzer before model dispatch. It verifies the retained frozen inputs,
restores the private original execution context, and observes the original call
without granting a new budget. An expired deadline produces a terminal timeout
and owned-work cleanup without starting a model. The controller atomically saves
model-dispatch intent before launching it; an existing intent or missing trusted
original clock remains observation-required and never authorizes a second model.
Both clock and dispatch audit files are retained with experiment diagnostics.

A terminal successful analyzer model now passes the same strict report validator
before phase cleanup and history publication. If its report is invalid and time
remains, the controller permits one report-only repair within the original
analyzer deadline, capped at 90 seconds including process termination allowance.
There is no new role clock, capture, or kernel change. The repair receives the
exact error and original citation metadata; all device/remote tool commands are
rejected by the trusted bridge, and only `analysis.json` is writable among report
artifacts. Missing kernel/case attribution or compiler inventory remains invalid.

The first raw report, error, permitted metadata, repair dispatch intent, output,
terminal result and validated disposition are retained with the experiment and
round record. An uncertain repair intent never permits a second model launch.
Initial failed/timed-out models and exhausted allowances skip repair. A failed
repair still stops the phase; final publication reuses the identical validator.
