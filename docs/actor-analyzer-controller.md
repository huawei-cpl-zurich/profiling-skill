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
  --instruction /tmp/task.md --skill /tmp/pinned-profiling-skill
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

The actor writes `/workspace/candidate.py` and experiment logs. The analyzer sees
a read-only clone at the last correctness-qualified commit, including read-only
Git metadata. It writes `/analysis/analysis.json` using the contract supplied in
its prompt. Tool evidence appears read-only beneath `/workspace/evidence`; cite
paths such as `evidence/CALL_ID/results/summary.json`. The controller copies and
hash-checks those trusted files into the retained report. A claimed acquired
profile must include actual pipe rows for the frozen candidate source, not merely
an agent-created CSV or a report claiming that collection worked. Unsupported
profiling must be reported as unavailable, with the failure and uncertainties.

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
profiling-skill snapshot and `MODEL_JSON` to the approved explicit model JSON.
Set `A3_TARGET` and `A3_RUNTIME` from the ignored remote-access registry; these
machine-specific choices are not source defaults. Copying freezes the skill and
model for all six cells. Retain the printed gate directory in the central task
record so resume uses the same output directories, not a fresh campaign.

```sh
cp "$MODEL_JSON" "$gate_dir/model.json"
cp -aL "$PROFILE_SKILL" "$gate_dir/skill"
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
