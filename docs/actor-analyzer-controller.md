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
