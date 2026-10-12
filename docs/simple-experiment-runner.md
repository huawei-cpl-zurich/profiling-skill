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
  "skill_bindings": [
    {"name": "triton-guarded-kernel", "path": "/frozen/guarded", "sha256": "..."},
    {"name": "ascend-profiling", "path": "/frozen/profiler-new", "sha256": "..."}
  ],
  "agent_auth_path": "/ignored/private/auth.json",
  "source_repo": "/absolute/path/to/source",
  "base_ref": "main",
  "target": "bz-a3-1",
  "runtime": "py311-torch",
  "remote_workspace": "/remote/workspace",
  "terminal_gate_path": "/absolute/path/to/host-owned-gate.sh",
  "terminal_gate_bundle_path": "/absolute/path/to/a3-terminal-gate.tar",
  "candidate_manifest_path": "/absolute/path/to/matmul.manifest.json",
  "billed_checks": 8,
  "autotune": true,
  "rounds": 4,
  "infrastructure_retries": 2,
  "candidate_repairs": 2,
  "agent_command": ["python", "/absolute/path/to/codex_json_adapter.py", "--model", "gpt-5.6-sol"]
}
```

Use `py311-torch` for BZ-A3 cells and `cann91` for BZ-A5 cells.

`prompt` is required and is passed byte-for-byte to every agent phase.
`task_path` is optional. Use it when the common prompt directs the agent to
read `TASK.md`; initialization snapshots the exact UTF-8 file into the cell
repository at that name. Prompts that contain their complete task need not set
it, so generic cells remain unchanged.
`skill_bindings` is the production allowlist. Each binding gives the canonical
agent-visible name, frozen source directory, and production tree SHA-256.
Initialization verifies the tree and copies it, without links, into
`agent-home/.agents/skills/<name>`. Legacy `skill_paths` remains supported for
existing local fixtures but provides neither aliases nor a declared digest.

`agent_auth_path` names one ignored, mode-0600 authentication file. The runner
copies only those bytes to the private `CODEX_HOME/auth.json`; it never copies
global configuration, rules, or skills. Agent subprocesses receive a minimal
environment allowlist plus cell-private `HOME` and `CODEX_HOME`, so ambient
token and skill variables are not inherited. The production adapter also
rejects a private home containing `config.toml` or an unsafe auth file. This
prevents accidental treatment crossover for normal agent workflows; it is not
an adversarial filesystem sandbox.

Initialization builds the clone, branch, prompt, private home, and skills in a
temporary sibling directory and publishes the cell atomically. Any failure
removes the temporary cell, so the same `init` command is retryable.
`terminal_gate_path` must name a host-owned file and `billed_checks` must be a
positive integer. Initialization snapshots both the exact gate content and the
check limit inside the cell. When configured, it also records the task digest.
An A3 campaign gate requires `terminal_gate_bundle_path` and
`candidate_manifest_path` together. The runner snapshots the binary bundle and
exact v2 fully-fused declaration, records their digests, and rejects later
replacement. Initialization is device-neutral. The scheduler adds a
non-negative `physical_device` to the configuration for each `run`; the runner
rejects missing or invalid placement before remote preflight or dispatch. The
launch wrapper exposes only that physical device and maps it to logical device
0, while the resulting round evidence records the device actually used.
Every `run` and `resume`, including the boundaries after agent prepare and
analyze calls, rejects a missing, replaced, or changed cell `TASK.md`. Later
edits to source inputs or configuration cannot change these snapshots.

## Three agent phases

The runner sends one JSON object on stdin and reads one JSON object on stdout.
The prepare request contains `phase=prepare`, exact prompt, treatment, allowed
skill names, round, repo, target/runtime, and optional `session_id`. A
same-round repair also contains the prior candidate, full terminal result,
exact attempt path, and diagnostics. When a host-owned campaign gate has
snapshotted a validated v2 candidate manifest, every initial and repair request
also contains its exact normalized `candidate_contract`: `schema`,
`entrypoint`, `kernel_name`, and `fusion`. The adapter's common instruction
requires the agent to define and launch exactly `candidate_contract.entrypoint`,
preserve `candidate_contract.kernel_name` as the host selector, and implement
the declared complete operator as one fused logical launch. Task-specific names
remain request data, so the instruction stays byte-identical across treatments.

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

The supplied `scripts/codex_json_adapter.py` implements this protocol over
`codex exec --json` and exact-thread `codex exec resume`. It ignores user
configuration and rules, requests a schema-constrained final response, emits
only the response JSON on stdout, and returns the durable Codex thread as
`session_id` after prepare. Configure its model and reasoning effort explicitly
in `agent_command`.

`autotune=true` is an opt-in runner/gate contract. The configured terminal
gate must consume `AUTOTUNE_REQUEST_PATH` and emit the validated autotune
result schema described below; a legacy gate fails closed instead of silently
running only the incumbent. A capable gate declares the exact line
`# simple-experiment-capability: managed-autotune/v1`; the runner does not
infer support from a shipped filename, target, or product. Gates acquire
managed-search support only when their own snapshot carries that declaration.
With that option enabled, prepare must define one
top-level literal `AUTOTUNE_CONFIG` mapping. Before dispatch, the same durable agent thread
receives `phase=tune`, the declared mapping, prior-round profile summary when
available, and the limits of 128 proposed configurations and three profiled
finalists. It returns zero through 128 `{id, parameters, hypothesis}` objects;
parameters may override only declared JSON-scalar keys. The runner
deterministically materializes the incumbent and every proposal and embeds one
compressed, hash-bound request in the retained job.

The prepare response, including its candidate and durable session, is
atomically checkpointed before tune. If the agent turn is interrupted, `resume`
reuses that exact checkpoint and thread; it does not prepare a replacement.
Source/configuration checks that the controller can perform run before that
checkpoint. Invalid tune output becomes an explicit same-round candidate
repair and discards the checkpoint; input-integrity and gate-contract failures
are terminal rather than misleadingly resumable.
An accepted materialized plan is then stored in a second atomic `tuned` state
before attempt creation. Resume validates and dispatches those exact options,
source hashes, and policy without another agent turn. If interruption leaves a
pre-dispatch attempt directory before its state transition, resume rebuilds it
from the retained plan only when no dispatch evidence exists.

A compatible host gate screens all options in that job across its ranked
correctness and fusion cases using ordinary synchronized device timing.
Candidate failures eliminate only their option. The three fastest successful
options receive at least three repeated `msprof op` timings; every finalist
profile reports an integer `repeats >= 3`, and equal cheap timings retain
request order. The fastest official finalist then receives the complete
correctness/fusion gate and diagnostic pipe capture; equal official geomeans
retain finalist order. Duplicate materialized sources are rejected before
dispatch. Observation loss reuses the same durable handle, and completed
option screens are checkpointed inside the retained job. Tuning options do not
increment the optimization round.

## Reproducible seed repositories

`scripts/materialize_experiment_seed.py SPEC.json` verifies hash-bound common
prompt, task, and seed files before atomically creating a clean Git repository,
`TASK.md`, `cell.json`, and `receipt.json`. Its seed commit has fixed author and
time metadata, so identical file bytes produce the same commit and tree across
output locations. `cell.json` embeds the exact common prompt and points the
runner at the generated task snapshot and seed repository. Add task-specific
baseline, case inventory, starter candidate, and host candidate manifest as
explicit `files` bindings; remote gates remain independently host-owned.

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
`upload`, `observe`, and `logs`. A failed probe is retained with its phase and diagnostics,
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

Every possible remote job submission also consumes one billed check. The
runner first stages the content-addressed gate payload with `cpl-remote upload`,
then submits a small launcher that verifies its hash. A proven upload failure
cannot have created a job, so the runner preserves the candidate and evidence
without charging a check. Once job dispatch begins, a lost receipt is charged
because automatically replacing it could duplicate live work. Preflight-only
failures and observation of an existing durable handle are also not charged.
Exhaustion is reported as `billed_checks_exhausted`; `report` includes both the
immutable limit and used count.

## Recovery and evidence

```bash
python scripts/simple_experiment_runner.py init --root /path/to/runs --config cell.json
python scripts/simple_experiment_runner.py run --root /path/to/runs --config cell.json
python scripts/simple_experiment_runner.py resume --root /path/to/runs --config cell.json
python scripts/simple_experiment_runner.py recover --root /path/to/runs --config cell.json
python scripts/simple_experiment_runner.py report --root /path/to/runs --config cell.json
```

After dispatch, the durable handle is the first persisted artifact. Any
`observing`, `infrastructure_unknown`, `analyzing`, or `analysis_pending` cell
blocks `run`; use `resume`. It observes the exact stored handle when the result
is unknown, or resumes analysis/finalization when the terminal result is
already present. An uncertain dispatch with no returned handle requires manual
reconciliation because an automatic replacement could duplicate live work.
`recover` is deliberately narrower: it only rolls back a checkpoint whose
retained evidence proves dispatch never began (including the legacy local
`E2BIG` process-start failure). It archives that evidence under
`pre-dispatch/`, restores the attempt/check budget, and retains the exact
prepared candidate for the next `run`.

Each successful round adds exactly one commit under `repo/rounds/NN/` with the
selected and structural candidates, compact tuning request/result, generated
wrapper, agent and host command logs, compact profile and terminal result,
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
