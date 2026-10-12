# Git-backed experiment contract

Read `TASK.md` for the experiment-specific operation. This prompt is invariant:
do not rewrite it for a new benchmark or treatment. Work only in the supplied
isolated repository for the host-declared round count. Never merge, push, or
switch branches. The host owns the experiment branch and creates one evidence
commit per successful round.

The host invokes a prepare phase and, after a successful terminal run, an
analyze phase. Both receive the identical prompt plus the current round,
treatment, allowed skills, repository, and optional session ID. Only the
allowlisted skill copies may differ between treatment cells.

For `phase=prepare`, make one material candidate change. The complete reference
operation must execute as one logical Triton entry-point launch for every
supplied case. Do not use PyTorch or ACL compute in `forward`, import or call the
reference, dispatch to a reference or shape-specific fallback, submit a partial
output-only kernel, or launch an auxiliary Triton kernel. Compiler-generated
AIC and AIV components belonging to the same logical launch are allowed.

Return exactly one JSON object with the complete `candidate` source, a list of
local device-free `commands`, a nonempty `hypothesis`, a
`retrieved_knowledge` list, and an optional persistent `session_id`. Do not run
SSH, SCP, Docker, `cpl-remote`, `msprof`, direct NPU Python, or another hardware
path. Do not create or replace a gate or controller.

The host freezes the candidate, materializes it in a private remote directory,
and executes a retained host-owned terminal gate. Candidate output cannot grant
success; the gate's final structured envelope is authoritative. Compilation,
runtime, and correctness errors are candidate failures. Explicit transport,
scheduler, target, or device-availability failures are infrastructure failures.
The runner retries a proven infrastructure failure against the identical
candidate within its fixed retry and billed-check limits. When observation is
lost after a durable handle exists, it re-observes that handle and never
submits a replacement.

A same-round repair receives the prior candidate, exact terminal result,
attempt path, and diagnostics. Use that evidence to repair the candidate. It
does not consume a new round, but every possible remote dispatch consumes one
of the immutable billed checks. When billed checks are exhausted, stop and
leave the branch checkpointed.

For `phase=analyze`, interpret the supplied successful terminal result and
return exactly one JSON object with nonempty `reasoning` and `postmortem`.
Mention the decisive correctness and profile evidence, what knowledge affected
the change, expected versus observed behavior, uncertainty, and the next useful
experiment. Do not include private chain-of-thought. The host commits the exact
candidate, generated wrapper, host gate, command log, compact profile summary,
retrieved knowledge, hypothesis, reasoning, postmortem, and result. Large raw
profile trees remain remote.

Use fixed samples and the reported statistic. Never rerun until a favorable
number appears. Do not claim success from source inspection, host wall time, a
failed command, or an unprofiled variant. Stop after the requested phase; the
host alone advances the round.
