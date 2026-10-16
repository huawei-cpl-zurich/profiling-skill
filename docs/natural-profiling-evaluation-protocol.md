# Natural-use profiling evaluation protocol

This prospective evaluation asks whether profiling guidance helps an agent make
better evidence-supported optimization decisions on diagnostic kernels. It keeps
the earlier negative pilot report and original protocol unchanged. A preparation
snapshot is not evidence that this evaluation has qualified or run.

## Comparison and qualification

Use six new Triton starting kernels: three workload families, each with two
regimes. Export only the starting implementation, candidate copy and public
contract. Keep correctness oracles, private interventions, qualification profiles,
observed diagnoses, gold answers and repository history outside agent isolation.

Before any scored agent launch, qualify every task privately using the baseline
and two scoped alternative implementations. Check numerical correctness and the
single-kernel computation contract for each variant. Acquire five BasicInfo timing
samples and three pipe captures per variant in each of two independent acquisition
batches; preserve all samples, errors and failed attempts. Require reproducible
resource distinctions and intervention responses that pipe evidence helps
discriminate. Activity alone does not prove saturation, and overlapping pipes
must not be summed into a latency decomposition.

All six tasks must exhibit useful resource contrasts before the cohort starts.
If any fails qualification, report the failure and stop cohort inclusion. Do not
replace a task with an easy control, retune guidance to exposed answers, or lower
the gate to fill six slots. Freeze qualified sources, private observed diagnoses
and retained acquisition evidence before scored optimization runs. Independent
batches are separate acquisitions, not arbitrary partitions of one sample set.

## Agent conditions

Launch twelve fresh contextless agents, one guidance and one control arm for each
task. Guidance uses the unchanged skill revision
`294f76fc602f9d121cb9826bfcdc6c9e524c1bd2`; control receives no skill. Freeze the
exact explicit model and reasoning effort. Both arms receive identical ordinary
optimization prompts, public inputs, neutral tool access and a 1200-second
wall-clock budget including tool waiting, with a 120-second finishing reserve.
Generate the shared prompt with the frozen launcher's `build_prompt(1200)`.

There is no mandate to acquire pipe metrics, no required profiling collection
point, and no score for collection alone. Normal autotuning, timing and compiler
inspection remain available. Agents may choose tools as useful; timing-only
optimization is valid. Public task contracts must not disclose gold diagnoses or
prescribe an intended rewrite.

Use fresh isolated Git workspaces with declared public baseline/contract files
read-only. Agents commit on their own branches and never merge. Freeze the
first-party launcher/tools/exporter, guidance allowlist, prompt, model, protocol
and source hashes. The controller-only protocol includes this rubric and is not
mounted into either agent. At most three agents run concurrently in four waves:

1. task01 guidance, task02 control, task03 guidance.
2. task01 control, task02 guidance, task03 control.
3. task04 control, task05 guidance, task06 control.
4. task04 guidance, task05 control, task06 guidance.

Retain every arm's code, commits, command journal, samples, errors, failed attempts,
unfinished work and budget failures. Do not silently retry, discard or replace
cells. Use the managed remote-access client, named runtime and registry-derived
working directory for remote work. Select eligible devices dynamically and retain
durable handles; after an observation interruption, observe the same job.
Distinguish candidate failures from transport, host, profiler and evidence failures.

## Blind rubric and outcome

Give an arm-anonymous reviewer the public task, code, commands, measurements and
decision summary. Score four dimensions from 0 to 2: 0 absent or incorrect,
1 partially sound, 2 sound.

1. Measurement and provenance: real baseline/final evidence, correct selector,
   measurement definition and retained samples.
2. Interpretation and calibration: execution explanation supported by artifacts;
   distinguish activity, throughput and capacity.
3. Controlled decisions: predicted effect, meaningful change and evaluation;
   no prescribed reference implementation.
4. Uncertainty and regressions: acknowledge noisy or contradictory evidence and
   correctness limits; do not infer certainty from missing fields.

Usefulness requires at least four of six paired reasoning wins, no reduction in
correctness success and no critical unsupported hardware claims in guidance arms.
Audit full computation and compiled artifacts against the single-kernel contract;
a numerical pass alone cannot validate work moved to auxiliary kernels or omitted
contrary to the contract. Faster incorrect results are not performance successes.

Report voluntary profiling adoption separately from actual information use: what
evidence changed the next edit, candidate selection or explanation. Also report
correctness, speedup, elapsed time and experiment/tool counts separately. Operational
recovery alone does not establish profiling benefit. Merely collecting metrics
does not establish an evidence-driven decision. An unchanged but well-investigated
candidate need not be a reasoning failure.

Flag unsupported claims such as activity proving saturation, scalar activity
excluding memory costs, zero wait proving synchronization is absent, overlapping
or cross-replay times providing exact latency decomposition, early IR proving
binary equivalence, or a negative sanitizer result proving race freedom.

If agents do not collect pipes or do not show benefit, report that honest negative
result with the retained evidence. Do not force new collection, substitute cases
or retune guidance to rescue the outcome. Six pairs are a qualitative pilot, not
a population-wide statistical claim.
