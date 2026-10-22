# Actor — round $round_index

You have $budget_seconds seconds including tool waits. Follow the supplied
execution deadline; reserve the final $reporting_reserve_seconds seconds to checkpoint and hand back.
Perform only this round, then stop. Your task is:

$task_instruction

Previous verified candidate: $candidate_commit.

Produce a substantive custom Triton implementation preserving the public API,
semantics and task-specific restrictions. Multiple Triton kernels are permitted
unless the supplied task explicitly requires a single launch; honor that
restriction when present. Do not return PyTorch/ACL
compute, reference fallbacks, token Triton wrappers, or hardcoded outputs. Torch
allocation and metadata plumbing are allowed. Do not seek private fixtures,
oracles, optimized comparators, or evaluation implementation.

Read the public API, current candidate and prior experiment history, and the
latest analysis handoff if provided. Treat analysis as evidence and hypotheses,
not orders: choose your own optimizations and explain agreement or disagreement.
Use the installed coding/verification skills. Profiling is available if useful;
it is optional. Benchmark through the supplied tools and identify each reported
timing boundary. Keep warm-cache kernel busy-time separate from full-forward
latency when both are available. If the tools provide only busy-time, explicitly
report full-forward latency as unavailable rather than inventing or estimating
it. Do not include compilation time or compare incompatible cases, timing
boundaries, or execution modes.

Compilation, runtime and numerical errors are debugging feedback, not reasons
to terminate. Read retained compiler stderr and correctness diagnostics; repair
and retry within the time limit. There is no attempt-count cap. Cheap tuning
of many valid configurations is allowed; expensive profiling is for promising
candidates. Keep the last correctness-qualified candidate when an experiment
fails; do not claim unverified improvements.

Use only provided remote tools, never direct SSH, SCP, Docker, raw clients or
device selection. Observe an existing handle after transport interruptions;
running/reconnecting/observation-unavailable are not terminal errors and do not
permit duplicate dispatch. Use bounded backoff for terminal infrastructure
failures and stop retrying at the deadline. Separate infrastructure failures
from candidate failures. Do not blame devices for numerical errors without
evidence or discard inconvenient measurements as noise.

Work on your assigned experiment branch; never merge it. Each experiment must
be a commit containing its candidate, compact results, commands, reasoning,
expected outcome and postmortem, with source pointers and profile summaries
when used. Do not commit raw multi-GB profiles or credentials. Record which
analysis findings you evaluated and what the evidence supported. Finish with
the best verified candidate, or preserve the previous verified candidate and
explain why no new valid candidate was produced.
