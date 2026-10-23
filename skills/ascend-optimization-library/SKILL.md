---
name: ascend-optimization-library
description: Diagnose Ascend kernel optimization opportunities and plan guarded transformations using source-backed paired examples and retained profiler evidence.
---

# Ascend optimization library

Use this library to connect an observed source pattern and bottleneck to a
specific transformation. Each recipe is a hypothesis until its measurement
record establishes correctness and a benefit or cost for the stated target.
The current wave is under validation; do not describe its examples as verified
until a complete evidence record is linked from the relevant recipe.

Select a relevant recipe rather than loading the full collection. Start with
[scaling hoist](references/scaling-hoist.md) for an invariant row scale applied
to operands before a dot product. Additional recipes become available after
their paired examples have been acquired and reviewed.

## Use an example to make a decision

Match the mathematical contract, shapes, strides, layout, dtypes, target and
compiler before transferring an observed result. Record the exact source
pattern and mechanism that make the transformation plausible. A source
comment or measurement from another target is supporting context, not a
measurement of the current kernel.

Read the complete operation profile: preparation, conversions, GEMM, reduction,
epilogue and workspace traffic belong to its cost. A faster central kernel
can leave the complete call slower. Match kernel identities and iteration
ownership before attributing measurements to the code being considered.

Choose apply, avoid or insufficient evidence. Explain the expected gain and
its competing cost, the dispatch preconditions and numerical guards, and an
observation that would falsify the diagnosis. Preserve required outputs,
packed-sequence boundaries and input immutability. Do not relax an error
contract to keep an optimization.

For an actor, give a concrete transformation plan and guard/fallback coverage
before writing code. For an analyzer, distinguish a source-derived hypothesis
from a supported profiler diagnosis, state confidence, and identify a paired
comparison that isolates the proposed mechanism.

## Interpret measurements

Examples retain a single independent CPU oracle per semantic case. Each arm
gets one correctness call, followed by ten profiled forwards with no tensor
inspection or hidden warmups. Event durations are instrumented measurements.
Use all ten samples and their spread; effects within observed variability
remain inconclusive. Keep export, accounting, kernel identity and PMU
availability separate from the presence of timing samples.

Pipe activity percentages are time fractions, not achieved bandwidth or
fraction of peak throughput. Only estimate roofline distance when work, traffic,
elapsed time and a target-supported capacity denominator are available with
explicit units and assumptions. Missing counters remain unavailable. A high
activity fraction alone does not establish saturation or overlap.

The project pair runner is `scripts/optimization_pair.py` at repository root;
it accepts a language adapter, case JSON, output directory, oracle cache and
provenance JSON. Read the adapter's source and the recipe's retained evidence
before using it. Measurement and remote execution follow the available
`ascend-profiling` and `remote-access` skills; this library does not authorize
additional launches or remote actions.
