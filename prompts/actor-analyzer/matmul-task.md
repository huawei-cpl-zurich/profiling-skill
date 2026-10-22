Optimize the streaming matmul-add Model described in API.md and task.json.
Keep the Model.forward API and numerical semantics for arbitrary valid inputs,
including tail shapes. Use substantial custom Triton computation, not PyTorch
compute, reference fallbacks, token wrappers, or hardcoded outputs. Multiple
Triton kernels are allowed. Torch allocation and metadata plumbing are allowed.

Use experiment-tool help for correctness, timing and optional profiling. Observe
the original call after an interruption. Compile and numerical failures are
repairable feedback: keep debugging within the deadline, with no attempt cap.
Cheap autotuning is allowed at the end of each round; expensive profiling is for
promising candidates rather than every tuning option. Preserve the best verified
candidate. Distinguish candidate failures from confirmed infrastructure failures.

Record code, commands, compact results, expected outcome, reasoning and postmortem
in commits on your unmerged experiment branch. Do not seek private test inputs,
optimized comparator kernels or previous campaign results. Use only the public
API and supplied tools. Keep full-forward latency, device busy time, and device
active union as distinct timing boundaries; do not include compilation time.
