# Simple experiment runner

`scripts/simple_experiment_runner.py` runs one experiment cell without the
audited campaign control plane. A cell is identified by four readable names:
campaign, product, task, and treatment. It gets its own folder, Git clone, and
branch. Git commits are the experiment history; remote handles only reconnect
an interrupted observation.

## Cell layout

For this configuration:

```json
{
  "campaign": "profiler-v2",
  "product": "a3",
  "task": "matmul",
  "treatment": "guarded-new-profiler"
}
```

the runner creates:

```text
runs/profiler-v2/a3-matmul-guarded-new-profiler/
  repo/                         # isolated experiment branch
    rounds/01/                  # evidence committed for round 1
  state.json
  attempts/1/
    request.json
    agent-result.json
    run.sh
    remote-job-id.txt
    stdout.log
    stderr.log
    result.json
```

The branch is
`experiment/profiler-v2/a3-matmul-guarded-new-profiler`. Each successful round
adds exactly one commit containing:

- the candidate;
- every agent-reported command;
- the compact profile summary and terminal result;
- reasoning, post-mortem, and retrieved-knowledge citations.

Large profiler report trees remain remote. Only compact evidence belongs in
the round commit.

## Configuration and agent contract

A complete configuration adds the following fields:

```json
{
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

The runner sends one JSON request to `agent_command` on stdin. The request
includes the round, isolated repository, selected target/runtime, and the last
candidate diagnostic when a repair is needed. The command returns one JSON
object on stdout:

```json
{
  "candidate": "complete candidate source",
  "remote_script": "#!/bin/bash\n...self-contained check...",
  "commands": ["python candidate.py"],
  "reasoning": "why this candidate should work",
  "postmortem": "expected result and interpretation",
  "retrieved_knowledge": ["ref://stable-citation"]
}
```

`remote_script` must be self-contained for the selected remote workspace. It
must print a final JSON line with one of these outcomes:

```json
{"outcome":"success","profile_summary":{"duration_us":12.5}}
{"outcome":"candidate_failure","diagnostics":"compiler output"}
{"outcome":"infrastructure_failure","diagnostics":"device unavailable"}
```

Compilation, runtime, and correctness errors are candidate failures. Target
reachability, runtime provisioning, device allocation, and transport failures
are infrastructure failures. Do not label an ordinary kernel error as
infrastructure.

## Commands

```bash
python scripts/simple_experiment_runner.py init \
  --root /absolute/path/to/runs --config cell.json
python scripts/simple_experiment_runner.py run \
  --root /absolute/path/to/runs --config cell.json
python scripts/simple_experiment_runner.py resume \
  --root /absolute/path/to/runs --config cell.json
python scripts/simple_experiment_runner.py report \
  --root /absolute/path/to/runs --config cell.json
```

`run` automatically retries a known infrastructure failure twice without
advancing the round or asking the agent to regenerate the candidate. A
candidate failure sends its diagnostics back to the same agent and consumes a
same-round repair, not a new round.

If observation is interrupted or returns a nonterminal state, the cell becomes
`infrastructure_unknown`. The runner has already written the durable job ID to
`remote-job-id.txt`; use `resume`. It observes that exact ID and never submits
a replacement. If dispatch itself becomes uncertain before an ID is returned,
the runner stops for manual reconciliation because a safe automatic retry is
impossible.

The production adapter invokes only the user-wide global `cpl-remote` client,
uses a named runtime and file-backed script, and collects compact stdout and
stderr after terminal observation. It does not use admission hashes, receipt
graphs, sealed manifests, migration attestations, or campaign identity hashes.
