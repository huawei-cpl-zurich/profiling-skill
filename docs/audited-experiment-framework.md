# Audited experiment framework

This framework produces branches that can answer not only which candidate won,
but what the agent tried, what it observed, and why it retained or reverted the
change. Keep `prompts/audited-three-experiment.md` byte-for-byte unchanged.
Describe each new benchmark only in `TASK.md`; the acceptance helper rejects a
battery whose three prompt or task files differ.

## Isolation and treatments

Start every agent from an independent clone of the same baseline commit. The
runner creates `experiment/<run-id>/<agent-id>` and does not merge or push it.
Do not reuse a Git directory, Codex state directory, or session between agents.
The Docker runtime is the default: it exposes only the agent repository, that
agent's state directory, and the read-only Node/Codex runtime. It does not copy
global configuration, plugins, or skills. Only `auth.json` is copied into the
private state directory and it is removed in `finally`, on success or failure.

Prepare treatment repositories before the run. Put only the skills assigned to
that treatment in its baseline and record their names, versions, content
hashes, and expected entry points in the baseline documentation. The control
baseline must contain none of those skill files. Keep all non-treatment files,
the invariant prompt, `TASK.md`, model, reasoning effort, controller, and
runtime image identical. The agent never receives another treatment's checkout
or state directory. Because `--ignore-user-config` is used and no global skills
are mounted, repository contents are the complete treatment boundary.

## Lifecycle

The host, not the agent, owns the lifecycle:

1. Seed: require a clean repository, create the experiment branch, copy the
   immutable prompt and task, record runtime/controller provenance, and commit.
2. Prepare: ask the persistent agent to make one material candidate change and
   run local checks. Repair a missing, unchanged, stale, schema-invalid, blank,
   or unresolved-sentinel candidate manifest
   in that same session before any controller submission.
3. Controller: freeze candidate and manifest hashes and submit those exact
   hashes. The agent does not see or select hosts or physical devices.
4. Finalize: give the agent the compact receipt. The candidate and manifest are
   read-only in this phase. Repair malformed report JSON in the same session.
5. Commit: materialize evidence and make one host-controlled, agent-attributed
   commit. A revert restores the prior candidate but archives the tested one.

After the host-declared round count completes, the branch is exactly one seed
plus that many linear commits. Every evidence record cites one persistent
session. There are no agent-created commits, merges, or pushes. Run the
independent verifier before comparing results.

If the run is interrupted, preserve the repository and private state directory.
The runner commits `.experiment/blocked.json` with the session, stage, frozen
hashes, prior candidate, commands, and receipt. Resume with the same `run-id`,
`agent-id`, prompt, task, repository, and state directory plus `--resume`. The
checkpoint commit is removed from the final linear history; it does not consume
an experiment.

## Controller contract and infrastructure policy

The configured command is called with:

```text
CONTROLLER --experiment N --candidate-sha256 HASH --manifest-sha256 HASH
```

It returns one JSON receipt no larger than 64 KiB. `status` is `ok`,
`candidate_error`, `measurement_pending`, or a nonterminal infrastructure
classification. Completed receipts include the hashes, durable `handle`,
selected `device`, compact result, and a `policy` object with schema
`profiling-skill/controller-policy/v1`.

The controller—not the prompt or agent—selects an eligible idle device. Before
admission it runs a warmed known-good control and records healthy/idle/warmed
proof. Compilation errors, runtime errors, and correctness mismatches from the
candidate count as candidate results. A mismatch does not implicate the device
unless the known-good control also fails. Quarantine requires two recorded
known-good control failures on that device.

Transport, scheduler, and controller failures are infrastructure failures.
Use a fixed retry budget and the identical candidate. Once a nonterminal
durable handle exists, re-observe it after a lost observer connection; never
submit a duplicate merely because the client timed out. Receipts record every
submitted and observed handle. Exhausting the infrastructure budget creates a
blocked checkpoint, not a failed experiment, so it can resume without losing
work or adding a commit.

Timing uses a predetermined sample count and median. If variability exceeds the
fixed threshold, permit exactly one confirmation capture; do not repeat until
the number looks good. If that remains noisy, use `measurement_pending`. Run a
post-candidate known-good control: drift also yields `measurement_pending`.
Keep the candidate, commit, and compact receipt, then remeasure later. Device
identifiers are evidence values, never protocol constants.

## Retained artifacts

Each `experiments/NN/` directory contains:

- `report.md`: hypothesis, expected result, change, evidence, observation,
  retain/revert decision, postmortem, and next experiment.
- `evidence.json`: immutable prompt/task/provenance hashes, session, tested and
  committed candidate hashes, decision, and controller receipt hash.
- `results.json`: the compact controller receipt and timing summary.
- `commands.jsonl`: commands from structured Codex events, exit status, bounded
  redacted output excerpt, and full-output hash.
- `sources.json`: query, stable locator, source content hash, summary, and
  influence; an empty list requires an explicit reason.

Reverted experiments additionally retain `tested_candidate.py` and its
manifest. Never commit raw multi-gigabyte profiler trees. Retain remote paths,
durable handles, hashes, and compact in-place analysis instead. Reports contain
concise rationale and evidence, not private chain-of-thought.

## Commands

Run one real agent (Docker is the supported default):

```sh
python scripts/audited_experiment.py \
  --repo /isolated/agent-1 --prompt prompts/audited-three-experiment.md \
  --task /campaign/TASK.md --run-id campaign-01 --agent-id agent-1 \
  --rounds 4 \
  --controller '/path/to/controller --profile treatment-a' \
  --auth-home /path/to/codex-home
```

Resume a blocked run by repeating the command with `--resume`. The controller
must recover or re-observe retained handles from its own durable state.

Run the real three-agent no-NPU acceptance battery:

```sh
python scripts/run_audited_toy_battery.py /new/battery/root \
  --auth-home /path/to/codex-home
```

It creates three isolated repositories, launches three agents concurrently,
and requires three experiment commits from each. It deliberately uses a local
deterministic controller so it tests agent/session/evidence integration without
remote or device noise. It does invoke real agents and is therefore separate
from the fast test suite.

Validate a completed branch offline:

```sh
python scripts/validate_audited_experiment.py /isolated/agent-1 --base main
```

The verifier checks the unmerged branch, exact history, immutable seed,
material changes, candidate/manifest/receipt hashes, command and source
evidence, persistent session, revert archives, and unchanged prior evidence.
Compare performance only after all branches pass this verification.
