# Simple experiment runner

`scripts/simple_experiment_runner.py` is the only supported experiment campaign
entrypoint. It runs one cell with one readable folder, isolated Git clone,
branch, private agent home, and append-only attempt history. Git commits are
the round history. A remote handle is only a pointer used to resume one retained
job.

## Configuration

```json
{
  "campaign": "profiler-v2",
  "product": "a3",
  "task": "matmul",
  "treatment": "guarded-new-profiler",
  "prompt": "The exact common experiment prompt.\n",
  "task_path": "/absolute/path/to/benchmark-task.md",
  "skill_paths": ["/absolute/path/to/guarded", "/absolute/path/to/profiler"],
  "source_repo": "/absolute/path/to/source",
  "base_ref": "main",
  "target": "bz-a3-1",
  "runtime": "py311-torch",
  "remote_workspace": "/remote/workspace",
  "terminal_gate_path": "/absolute/path/to/host-owned-gate.sh",
  "terminal_gate_bundle_path": "/absolute/path/to/a3-terminal-gate.tar",
  "candidate_manifest_path": "/absolute/path/to/matmul.manifest.json",
  "physical_device": 3,
  "billed_checks": 8,
  "rounds": 4,
  "infrastructure_retries": 2,
  "candidate_repairs": 2,
  "agent_command": ["experiment-agent", "--json"]
}
```

Use `py311-torch` for BZ-A3 cells and `cann91` for BZ-A5 cells.

`prompt` is required and is passed byte-for-byte to every agent phase.
`task_path` is optional. Use it when the common prompt directs the agent to
read `TASK.md`; initialization snapshots the exact UTF-8 file into the cell
repository at that name. Prompts that contain their complete task need not set
it, so generic cells remain unchanged.
`skill_paths` is an explicit allowlist. At initialization those directories
are copied, not linked, into the cell-private
`agent-home/.agents/skills/<name>`. Every agent invocation uses the cell repo
as its working directory and sets `HOME` to `agent-home` and `CODEX_HOME` to
`agent-home/.codex`. This prevents accidental treatment crossover for normal
agent workflows; it is not an adversarial filesystem sandbox.

Initialization builds the clone, branch, prompt, private home, and skills in a
temporary sibling directory and publishes the cell atomically. Any failure
removes the temporary cell, so the same `init` command is retryable.
`terminal_gate_path` must name a host-owned file and `billed_checks` must be a
positive integer. Initialization snapshots both the exact gate content and the
check limit inside the cell. When configured, it also records the task digest.
An A3 campaign gate additionally requires all three of
`terminal_gate_bundle_path`, `candidate_manifest_path`, and `physical_device`.
The runner snapshots the binary bundle and exact v2 fully-fused declaration,
records their digests, and rejects later replacement. The physical placement
is host configuration rather than gate source: the wrapper exposes only that
physical device and maps it to logical device 0.
Every `run` and `resume`, including the boundaries after agent prepare and
analyze calls, rejects a missing, replaced, or changed cell `TASK.md`. Later
edits to source inputs or configuration cannot change these snapshots.

## Two agent phases

The runner sends one JSON object on stdin and reads one JSON object on stdout.
The prepare request contains `phase=prepare`, exact prompt, treatment, allowed
skill names, round, repo, target/runtime, and optional `session_id`. A
same-round repair also contains the prior candidate, full terminal result,
exact attempt path, and diagnostics.

Prepare must return:

```json
{
  "candidate": "complete candidate source",
  "commands": ["python candidate.py"],
  "hypothesis": "why this candidate should work",
  "retrieved_knowledge": ["ref://stable-citation"],
  "session_id": "optional durable agent session"
}
```

Candidates are limited to 1 MiB. Candidate fields and the generated wrapper
are validated before an attempt number or directory is created, so rejected
prepare output can be repaired without poisoning the round. The runner, not
the agent, creates the exact submitted `run.sh`. It embeds the candidate,
materializes it as `candidate.py` in a private remote temporary directory,
exports `CANDIDATE_PATH`, changes to that directory, and executes the retained
host-owned terminal gate. The agent cannot select or replace the executed
fragment. The exact candidate, gate, and generated wrapper are retained and
committed.

After a successful terminal result, the runner invokes the same agent command
with `phase=analyze`, the session ID, candidate, exact attempt path, and full
result. Analyze must return final `reasoning` and `postmortem`. Its output is
persisted before finalization. If analysis or finalization is interrupted,
`resume` reuses the terminal result, analysis, evidence directory, and existing
commit without rerunning the remote workload or duplicating the round commit.

## Remote result contract

Before each new `run` invocation that may dispatch work, the runner records
global `cpl-remote` `capabilities` and `preflight` receipts and requires `run`,
`observe`, and `logs`. A failed probe is retained with its phase and diagnostics,
classified as infrastructure, and consumes neither a candidate attempt nor a
round. Production execution uses only the user-wide global client, a named
runtime, and a file-backed script.

The last nonblank stdout line must be a JSON envelope:

```json
{"schema":"simple-experiment-result/v1","outcome":"success","profile_summary":{"duration_us":12.5}}
```

Success requires all three facts to agree: terminal state `completed`, exit
code `0`, and that structured success envelope. Failed or cancelled jobs,
missing envelopes, and contradictory success envelopes fail closed. A valid
envelope may instead report `candidate_failure` or `infrastructure_failure`
with diagnostics. Every stdout/stderr receipt must repeat the requested handle
and matching target before its content is accepted.

The retained gate runs after candidate materialization and owns the terminal
result protocol. Candidate output cannot grant success: the gate must leave its
result envelope as the last nonblank stdout line.

Known infrastructure failures retry the same prepared candidate twice without
advancing the round. Candidate failures return all repair evidence to the
agent and consume a same-round repair. Exhausted budgets are terminal and a
later `run` does not dispatch again.

Every possible remote submission also consumes one billed check. The runner
checks the snapshotted limit before preparing or dispatching replacement work
and charges at the dispatch boundary, before asking `cpl-remote` to run. Thus a
lost dispatch receipt is still charged and can never cause the cell to exceed
its limit. Preflight-only failures and observation of an existing durable
handle are not charged. Exhaustion is reported as `billed_checks_exhausted`;
`report` includes both the immutable limit and used count.

## Recovery and evidence

```bash
python scripts/simple_experiment_runner.py init --root /path/to/runs --config cell.json
python scripts/simple_experiment_runner.py run --root /path/to/runs --config cell.json
python scripts/simple_experiment_runner.py resume --root /path/to/runs --config cell.json
python scripts/simple_experiment_runner.py report --root /path/to/runs --config cell.json
```

After dispatch, the durable handle is the first persisted artifact. Any
`observing`, `infrastructure_unknown`, `analyzing`, or `analysis_pending` cell
blocks `run`; use `resume`. It observes the exact stored handle when the result
is unknown, or resumes analysis/finalization when the terminal result is
already present. An uncertain dispatch with no returned handle requires manual
reconciliation because an automatic replacement could duplicate live work.

Each successful round adds exactly one commit under `repo/rounds/NN/` with the
candidate, generated wrapper, command log, compact profile and terminal result,
host terminal gate, prompt/treatment/skill inputs, billed-check limit,
hypothesis, retrieved knowledge, final reasoning, and post-mortem. When a task
is configured, every round commit also contains its exact `TASK.md` bytes and
digest. Large profiler report trees remain remote. The
runner has no campaign admission, receipt graph, runtime-closure hash, sealed
manifest, migration attestation, or generalized plugin layer.

## A3 host-owned gates

Build the reusable deterministic harness once:

```bash
python scripts/build_a3_gate_bundle.py \
  --source "$PWD" --output /ignored/runtime/a3-terminal-gate.tar
```

Select `scripts/a3-gates/{gdn,bsa,matmul}.sh` as `terminal_gate_path` and its
matching `.manifest.json` as `candidate_manifest_path`. The task prompt must
require the fixed entrypoint declared by that host manifest. Each gate runs the
complete benchmark domain first (50 GDN, 50 BSA, or 10 matmul cases), validates
the trusted single-logical-launch evidence, and only then runs three `msprof op`
captures for every ranked development case. Its final result contains compact
case samples and capture digests; vendor profiler trees remain remote.
