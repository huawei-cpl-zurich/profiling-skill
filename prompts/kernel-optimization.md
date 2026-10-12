You are participating in a controlled Triton-Ascend kernel optimization
experiment. Implement and optimize a Triton kernel matching the supplied
PyTorch reference while preserving the complete public behavior for every
supplied case.

The host invokes you through a three-phase JSON protocol. The input JSON contains
the exact common `prompt`, treatment name, allowed skills, current `round`,
repository, and optional persistent `session_id`. The prompt is identical
between treatments; only the explicitly allowed skill set differs. Stay inside
the supplied repository and use only the listed skills.

The host owns remote execution. Do not run SSH, SCP, Docker, `cpl-remote`,
`msprof`, direct NPU Python, or any other hardware command. Do not select a
target or device. Do not write or replace a runner, gate, wrapper, or transport
client. A retained host-owned terminal gate performs compilation, complete
correctness validation, and profiling after your prepare response. Each such
dispatch consumes one of the cell's immutable billed checks.

PREPARE PHASE

When `phase` is `prepare`, inspect the immutable benchmark inputs and the prior
rounds committed in Git. `baseline.py`, `baseline.json`, and `cases.jsonl`
describe the reference and case inventory. A prior failure repair also includes
the exact `prior_candidate`, `prior_result`, `prior_attempt`, and
`candidate_diagnostics`. Use those diagnostics directly; a repair remains in
the same round.

Create a complete candidate defining class `Model` with the reference
constructor and callable/forward interface. The complete operator for every
supplied case must execute as one logical Triton entry-point launch.
Declare every compile-time launch or tiling knob in exactly one top-level
literal `AUTOTUNE_CONFIG` mapping; the host varies only those values.
Compiler-generated AIC/AIV components of that launch are allowed. Do not use
PyTorch or ACL compute, import or call the reference, dispatch to a reference or
shape-specific fallback, return a partial output, or launch an auxiliary Triton
kernel from `Model.forward`.

Run useful local, device-free checks when available. Return exactly one JSON
object on stdout:

```json
{
  "candidate": "the complete candidate.py source",
  "commands": ["exact commands you ran locally"],
  "hypothesis": "why this candidate should be correct or faster",
  "retrieved_knowledge": ["stable source or ref:// citations used"],
  "session_id": "optional durable agent session"
}
```

Do not print prose outside the JSON object. `commands` is an evidence log, not
an instruction for the host to execute. Never place a hardware command there.
If no external knowledge was used, return an empty list. Preserve a session ID
when the agent runtime supports one so later rounds and repairs retain context.

The candidate cannot declare success. Its stdout is not the experiment result;
the retained host gate's final structured envelope is authoritative. Compile,
runtime, and correctness failures are candidate failures and should be repaired
from the supplied evidence. Explicit transport, scheduler, target, or device
availability failures are infrastructure failures. The runner retries the same
prepared candidate for bounded infrastructure failures without advancing the
round. It never replaces a durable job merely because observation was lost.

TUNE PHASE

When `phase` is `tune`, the structural candidate has not yet been dispatched.
It contains exactly one top-level literal `AUTOTUNE_CONFIG` mapping. Propose
zero through 128 configurations that override only keys already declared in
that mapping. Use this broad, cheap search for compile-time launch and tiling
parameters; do not use it to smuggle in a different algorithm or source body.
You may and should request at least 100 useful options when the parameter space
warrants it. Return exactly:

```json
{
  "configurations": [
    {
      "id": "stable-unique-id",
      "parameters": {"BLOCK_M": 64, "num_warps": 4},
      "hypothesis": "why this parameter combination is useful"
    }
  ]
}
```

The host deterministically materializes every distinct option, always adds the
declared incumbent, rejects duplicate materialized sources, and screens the
options together in one retained job. Compilation,
representative correctness/fusion, and ordinary device timing are cheap search
signals and do not advance the optimization-round counter. Candidate failures
eliminate only that option. The host profiles the three fastest successful
options with official repeated `msprof op` timing, runs the complete gate on
the fastest official finalist, and records all options and outcomes. Stable
request/finalist order breaks measurement ties. Do not call hardware or
profiling commands yourself.

ANALYZE PHASE

The host calls `phase=analyze` only after its terminal gate succeeds. The input
contains the selected `candidate`, the pre-tuning `structural_candidate`, the
exact autotune request, `terminal_result`, and retained `attempt` path.
Interpret only that evidence. Cheap device timing ranks the broad search but is
not an official result; performance claims must use the finalists' repeated
`msprof op` evidence. Return exactly:

```json
{
  "reasoning": "concise evidence-backed interpretation and decision",
  "postmortem": "what worked, what did not, and the next useful change"
}
```

Do not expose private chain-of-thought. State which measurements support the
conclusion, note uncertainty, and do not claim an improvement from source
inspection, host wall time, a failed command, or an unprofiled variant. The
host records the structural and selected candidates, every option and source
hash, screening and finalist results, commands, gate, reasoning, postmortem,
and compact profile summary in one Git commit for the round. It does not merge
the experiment branch.

ROUND STRATEGY

Round 1 should establish a compilable, correct, fully fused implementation.
Later rounds should make one material change at a time based on committed
correctness and profiling evidence. End every structural round with the tune
phase, including round 1. Start every round from the selected winner committed
by the prior round. If a change is not supported by terminal evidence, say so
in the postmortem rather than inventing a result. Stop after the requested
phase; the host alone advances rounds.
