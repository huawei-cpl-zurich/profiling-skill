# Simple experiment runner

`scripts/simple_experiment_runner.py` runs one experiment cell without the
audited campaign control plane. A cell has one readable folder, isolated Git
clone, branch, private agent home, and append-only attempt history. Git commits
are the round history. A remote handle is only a pointer used to resume one
retained job.

## Configuration

```json
{
  "campaign": "profiler-v2",
  "product": "a3",
  "task": "matmul",
  "treatment": "guarded-new-profiler",
  "prompt": "The exact common experiment prompt.\n",
  "skill_paths": ["/absolute/path/to/guarded", "/absolute/path/to/profiler"],
  "source_repo": "/absolute/path/to/source",
  "base_ref": "main",
  "target": "bz-a3-1",
  "runtime": "cann",
  "remote_workspace": "/remote/workspace",
  "rounds": 4,
  "infrastructure_retries": 2,
  "candidate_repairs": 2,
  "agent_command": ["experiment-agent", "--json"]
}
```

`prompt` is required and is passed byte-for-byte to every agent phase.
`skill_paths` is an explicit allowlist. At initialization those directories
are copied, not linked, into the cell-private
`agent-home/.agents/skills/<name>`. Every agent invocation uses the cell repo
as its working directory and sets `HOME` to `agent-home` and `CODEX_HOME` to
`agent-home/.codex`. This prevents accidental treatment crossover for normal
agent workflows; it is not an adversarial filesystem sandbox.

Initialization builds the clone, branch, prompt, private home, and skills in a
temporary sibling directory and publishes the cell atomically. Any failure
removes the temporary cell, so the same `init` command is retryable.

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
  "script_fragment": "python \"$CANDIDATE_PATH\"",
  "commands": ["python candidate.py"],
  "hypothesis": "why this candidate should work",
  "retrieved_knowledge": ["ref://stable-citation"],
  "session_id": "optional durable agent session"
}
```

Candidates are limited to 1 MiB. The runner, not the agent, creates the exact
submitted `run.sh`. It embeds the candidate, materializes it as `candidate.py`
in a private remote temporary directory, exports `CANDIDATE_PATH`, changes to
that directory, and executes `script_fragment`. The exact candidate and exact
generated wrapper are retained and committed.

After a successful terminal result, the runner invokes the same agent command
with `phase=analyze`, the session ID, candidate, exact attempt path, and full
result. Analyze must return final `reasoning` and `postmortem`. Its output is
persisted before finalization. If analysis or finalization is interrupted,
`resume` reuses the terminal result, analysis, evidence directory, and existing
commit without rerunning the remote workload or duplicating the round commit.

## Remote result contract

Before each new `run` invocation that may dispatch work, the runner records global `cpl-remote`
`capabilities` and `preflight` receipts and requires `run`, `observe`, and
`logs`. Production execution uses only the user-wide global client, a named
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

Known infrastructure failures retry the same prepared candidate twice without
advancing the round. Candidate failures return all repair evidence to the
agent and consume a same-round repair. Exhausted budgets are terminal and a
later `run` does not dispatch again.

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
prompt/treatment/skill inputs, hypothesis, retrieved knowledge, final
reasoning, and post-mortem. Large profiler report trees remain remote. The
runner has no campaign admission, receipt graph, runtime-closure hash, sealed
manifest, migration attestation, or generalized plugin layer.
