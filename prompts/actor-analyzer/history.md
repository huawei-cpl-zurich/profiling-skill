
The JSON example above describes the legacy common fields. For contract 2 also
set contract_version: 2 and source_sha256 from /history/context.json. The history
mount is controller-owned, read-only, and contains only this experiment.

The controller already performs the required supplied-tool profile acquisition
for each new distinct verified kernel before you start. Read required_profile in
/history/context.json and use its exact current-source original evidence path.
Do not repeat that acquisition or substitute an actor capture for a successful
controller acquisition. Its staging/checks/captures/waits consumed part of your
original budget: /workspace/execution.json still has the original role deadline.
If required acquisition failed or remains observation-required, report the
retained diagnostics and missing measurements honestly; never duplicate its
dispatch or extend the deadline.
For an unchanged kernel, cite its repeated_round retained record and explain
reuse; do not recapture merely for historical comparison. Never extend the role
deadline or recapture older kernels. Retain unavailable diagnostics honestly.

Add optimizations: a nonempty list of {status: observed|none, description,
diff_lines}. Describe observed source changes using exact lines from the supplied
source_diff in diff_lines; separate likely effects from observed changes. For
first/unchanged kernels use none with an explanation.

Add utilization: a nonempty list of {status: measured|unavailable, description,
cases: [...], kernels: [...], evidence: [original listed paths]}. Cover each
exported case/kernel; explicitly explain unsupported/missing utilization rather
than conflating activity and throughput. Use unknown labels only if attribution
is unavailable. Include nonempty ceilings with estimated or not_established
status, numerical assumptions, sources, units and missing information. Discuss
estimated distance to the appropriate roofline, or why it cannot be established.

Add comparison: {status: copy context.status, rounds: [...], selection_reason,
interpretation}. For the first round with empty history, use exactly
{"status": "first", "rounds": [], "interpretation": "No previous kernel exists"};
the controller owns the empty historical selection. Never select the current
round. Include context.previous_distinct whenever present, even if its
analysis failed (explain unavailable metrics). Optionally select one additional
older distinct kernel with a specific reason. Read each selected round's
candidate.py, analysis.json if valid, record.json and retained evidence. Compare
utilization, throughput, bottlenecks and roofline estimates as well as runtime;
explain unavailable/incompatible metrics. The controller computes numeric pairs
and deltas from retained original summaries; do not provide pairs yourself.
Devices on the same node are equivalent. Different cases, runtime, capture
settings, timing boundaries or metric semantics are not comparable. When kernel
decomposition changes, compare workload aggregates and explain why kernel-level
pairing is unavailable. Do not sum ratios. Interpretation must state uncertainty
and distinguish source changes from their hypothesized performance effects.
