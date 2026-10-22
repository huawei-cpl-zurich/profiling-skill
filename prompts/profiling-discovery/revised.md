Implement and optimize the operator described in API.md using Triton. The starting
file is intentionally a skeleton: write your own implementation. Preserve the API,
ordinary supported inputs, runtime segment boundaries, numerical correctness, and
input immutability. No PyTorch compute/reference fallback. Your implementation must
execute under fullgraph npu-inductor. Multiple Triton launches are permitted for
this matched comparison task; record how many kernels execute per forward.
The implementation must do the substantive task computation in custom Triton,
not call an existing optimized operator or use a token Triton stage while
PyTorch/framework code computes the answer. Allocation, views and metadata glue
are permitted. Every forward launch is included in measurements.

Read execution.json first. You have 1800 seconds of original wall time including
tool waiting, with no attempt-count limit. Reserve the final 120 seconds for notes
and commits. Fix compilation/runtime/numeric failures and retry while time remains.
Use experiment-tool help. Benchmarking measures correctness, warm full-forward
latency, variability, and device busy-time. Read /workspace/.agents/skills/ascend-profiling/SKILL.md before optimization.
It explains how to acquire and interpret Ascend profiling evidence. The
experiment-tool profile --kernel NAME command provides candidate-only diagnostic
evidence; benchmark provides timings, not bottleneck analysis. After obtaining a
correct implementation, consider profiling to distinguish competing optimization
hypotheses. Explain whether you used profiling, what it changed, or why you judged
it unnecessary. Do not collect metrics merely to satisfy a checklist. Use the
installed coding guidance as useful.
Do not interpret infrastructure failure as a kernel failure. Observe the same
nonterminal call after interruptions and never dispatch duplicates of an unresolved
request. A terminal infrastructure failure can be retried unchanged.

Create and retain your assigned Git branch; never merge it. Commit each experiment
with code, commands, compact results, expected outcome, reasoning and evidence,
external knowledge/pointers used, and postmortem. Log failed experiments as well as
successful ones. Cheap autotuning is allowed; use expensive profiling on promising
candidates. Keep partial progress committed before the deadline. Do not copy large
raw profiles into commits. Report whether final correctness and benchmarking pass;
do not claim performance for a candidate that fails correctness.
