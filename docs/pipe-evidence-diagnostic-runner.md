# Supplied-evidence diagnostic runner

This offline runner separates repeated thought, pipe information and incremental
frozen-skill guidance. It does not execute or optimize kernels. Existing packet
exports contain six public task bundles and three baseline pipe captures each;
the private answer key is never copied into a bundle or agent mount.

```sh
python scripts/run_pipe_evidence_diagnostic.py \
  --packets /tmp/exported-packets --skill /tmp/frozen-profiling-skill \
  --model-json /tmp/explicit-model.json --output /tmp/new-diagnostic \
  --forbidden /absolute/orchestration-workspace
```

The explicit model JSON requires `name` and `reasoning_effort`. The coordinator
must verify the skill directory against its claimed revision; file hashes are
recorded here, not a fabricated Git provenance claim. Budget defaults to 240
seconds per fresh context, including tool waiting. At most three trajectories
run concurrently. Diagnostic deadline metadata reserves the final 30 seconds
for committing, matching the prompt; ordinary optimization launchers retain
their existing 120-second default. Output must not overlap either input tree.
Every task has timing, pipes and pipes-skill arms; all 18
first stages receive exactly the same prompt and no pipes or skill. Their 18
second stages receive another identical prompt and only their own committed
first answer plus condition-specific evidence. Skill is visible only to the
pipes-skill second stage. No conversation is resumed.

Agents write and commit the answer schema described in the frozen prompt.
Inputs are read-only, including proposed candidates and the sealed prior answer.
The runner checks source bytes, answer schema and exact committed answer bytes.
Commands, events, partial files and commits remain in each stage directory.
The neutral `experiment-tool help` explains the offline interface; attempts to
time, profile or run anything fail explicitly without remote dispatch.

An existing output is rejected unless `--resume` is explicit and all input,
model, prompt and runtime identities match. Resume only launches an absent
stage after revalidating the retained source, committed answer and exact HEAD.
The frozen skill must retain exactly its original files and file types.
Existing stage directories without
terminal results are left for inspection, never relaunched; failed stages are
retained and not replaced. Changing runtime sources during execution blocks
subsequent dispatches. The process exits nonzero unless all 18 trajectories
produce valid answers. Final `results.json` is execution status, not a skill
usefulness score: arm-anonymous assessment and the prospective rubric remain
separate coordinator work.

Functional tests (no model calls):

```sh
pytest -q tests/test_pipe_evidence_diagnostic.py
```

They exercise all 36 stage boundaries with a Git-backed fake launcher and three
real Bubblewrap visibility/read-only checks when the local runtime is available.
