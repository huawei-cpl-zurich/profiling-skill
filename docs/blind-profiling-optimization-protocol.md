# Blind profiling-guidance optimization pilot

This pilot tests whether profiling guidance helps an agent interpret evidence
while optimizing a kernel. It does not test whether the agent can repeat a
reference answer or achieve a prescribed rewrite.

## Controlled comparison

Use six contrasting starting kernels: runtime scalar/control work, indirect
memory, small matrix work, masked repeated accesses, scheduling options, and
native synchronization placement. Export only each starting implementation,
the callable contract and a candidate copy. Keep the correctness oracle,
reference outputs, historical profiles, after implementations and repository
history outside agent isolation.

Each task has two fresh contextless agents: one with the candidate profiling
skill, one without it. Both receive the same model configuration, prompt,
600-second wall budget and neutral tools. The tools provide correctness,
five-sample `msprof op` BasicInfo timing, three PipeUtilization captures,
available compiler artifacts and optional supported sanitizer analysis.
Compilation diagnostics and retained evidence are accessible to both arms.
Compiler dumps and sanitizer checks are options, not mandatory gates.

Use a clean agent state and fresh Git repository containing only the before
kernel. Mount baseline and contract read-only. The agent commits work on its
own branch and never merges it. Retain agent output, tool commands, source
snapshots, compiler errors and unfinished changes on timeout.

## Identical agent prompt

> Optimize the supplied kernel while preserving its public interface, supported inputs, numerical correctness and single-kernel execution contract. Measure the starting implementation and your final candidate using the supplied benchmark interface. Use available tools and guidance as you find useful. Record experiments, commands, results, and the evidence supporting your decisions. Explain what improved, what did not, and what remains uncertain. Commit your work on the assigned branch; do not merge it.

The scored cohort appends this exact paragraph to both arms:

> You have a 600-second wall-clock budget, including all tool waiting. Read /workspace/execution.json and note its UTC start and deadline before beginning. Reserve approximately the final 120 seconds for notes and commits; do not start new tool calls near the cutoff. Commit validated work early and update your experiment log incrementally so partial progress survives interruption. Prefer a small completed evidence-backed experiment to an unfinished search. Use experiment-tool help to learn the neutral tool interface. Every operational tool call already performs correctness checks; avoid redundant checks unless investigating a specific failure.

Freeze the exact versioned prompt with each cohort.

The initial calibration omitted these instructions. Its timeouts remain
reported as budget failures, not infrastructure failures. Do not combine
different prompt versions as a single paired comparison.

## Order and infrastructure

Run at most three agents concurrently and balance which arm starts first:

| Wave | Cells |
| --- | --- |
| 1 | task01 guidance, task02 control, task03 guidance |
| 2 | task01 control, task02 guidance, task03 control |
| 3 | task04 control, task05 guidance, task06 control |
| 4 | task04 guidance, task05 control, task06 guidance |

Freeze source revisions, skill allowlist, prompt, model and rubric before
launch. Require all before kernels to pass correctness and profiling smokes.
First run an actual contextless interface canary, not merely an isolation test.
Lease idle devices dynamically; device identity is provenance, not a fixed
part of the experiment protocol.

Every remote operation uses the installed remote-access client and named
runtime. Preserve the returned durable handle across transport interruptions.
Observe the same job rather than duplicating a live workload. Retained evidence
must distinguish candidate compilation/runtime/correctness failures from host,
transport, profiler and evidence failures. Infrastructure failures are unscored;
do not silently turn candidate or profiler failures into successful samples.
Retain every declared timing sample and its dispersion, including noisy results.

## Frozen blind rubric

Give an arm-anonymous reviewer the task, code, commands, measurements and
decision summary. Score each dimension 0 (absent or incorrect), 1 (partially
sound), or 2 (sound):

1. Measurement and provenance: real baseline/final evidence, correct selector,
   measurement definition and retained samples.
2. Interpretation and calibration: code-to-execution explanation supported by
   artifacts; distinguish activity, throughput and capacity.
3. Controlled decisions: predicted effect, meaningful change and evaluation;
   do not require any particular reference implementation.
4. Uncertainty and regressions: recognize contradictory/noisy evidence and
   correctness limits; do not invent certainty from missing fields.

Report correctness, speedup, elapsed/tool costs and actual profiling use
separately. An unchanged but correctly investigated candidate is not a reasoning
failure by definition. A faster incorrect candidate is not a performance success.
Before accepting performance, independently audit the full candidate computation
and compiled artifacts against the single-kernel and volatile-repetition
contracts. A numerical pass alone cannot validate these properties. Required
work moved to auxiliary Torch kernels or eliminated contrary to the contract
invalidates a candidate even if its final tensor matches.

Flag critical unsupported claims: activity proves saturation; scalar activity
excludes memory cost; zero wait proves no synchronization; overlapping or
cross-replay times form an exact latency decomposition; simulated timestamps
prove hardware speedup; early IR proves binary equivalence; or a negative
sanitizer result proves race freedom.

The pilot supports usefulness only with at least four of six paired reasoning
wins, no reduction in correctness success and no critical unsupported hardware
claims in the guidance arm. Otherwise report benefit unproven. Do not revise
against these exposed tasks and call a rerun independent confirmation. Six
pairs are a qualitative pilot, not a population-wide statistical claim.
