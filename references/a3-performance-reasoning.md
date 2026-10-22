# A3: connect code, execution and performance

Use this guide to form explanations, not to select a predetermined rewrite.
The observations motivating it are A3-specific; compiler versions and layouts
can change the result. Separate architectural expectations, actual emitted
operations, measured activity and unresolved hypotheses.

## Which resource can execute the work?

| Resource | Architectural role and interpretation boundary |
| --- | --- |
| Scalar | Address calculations, loop/predicate/control instructions; scalarized indirect GM loads can also dominate. High scalar activity need not be arithmetic overhead. |
| Vector | Vector arithmetic, conversions, masking and reductions; some local copies also use Vector. Scalar-looking source can introduce vector work. |
| Cube | Matrix multiply/accumulate work. Confirm the lowered matrix operation or suitable runtime evidence; `tl.dot` is not a universal single-pipe guarantee. |
| MTE1 | L1 to L0A/L0B matrix operand movement. Confirm actual address spaces and lowered copies. |
| MTE2 | DMA input paths such as GM to L1/UB. A tensor `tl.load`, especially an indexed gather, need not become an efficient MTE2 transfer. |
| MTE3 | Applicable outbound DMA paths, notably UB to GM. Not every store or matrix-result write is an MTE3 operation. |
| FIXPIPE | Applicable matrix-result movement/conversion, including observed CC/L0C to GM lowering. Verify that path instead of attributing output to an incidental MTE3 field. |

These architectural roles do not update the pinned compiler inventory's
`direct`/`inferred`/`unknown` statuses. The inventory documents particular
reviewed operations, not every compiler version. See [compiler attribution](triton-pipe-attribution.md)
and [export semantics](a2-a3-msprof-op.md) for those separate evidence layers.

## Activity, throughput and time answer different questions

The useful objective is efficient useful work with avoidable stalls and gaps
removed, not making an arbitrary utilization percentage large. A resource
approaching a validated capacity can identify the next limit, but does not
prove that the algorithm, traffic or layout is already optimal.

- Pipe activity divides active cycles by a task/row window. It does not say
  whether active transfers were efficient or peak bandwidth was reached.
  Near-full MTE2 activity can coexist with very different latencies.
- Compare absolute active time alongside fractions. A faster kernel may keep
  the same dominant fraction, or increase it because total time shrank.
- Keep matched core/sub-core rows separate from whole-device coverage. One
  reported block is not necessarily representative or the critical block.
- Pipe intervals can overlap. Do not sum active times into a latency
  decomposition, or multiply a median ratio by an independently measured
  median task time. Separate replays cannot establish simultaneous overlap.
- A bandwidth claim needs validated bytes, units and its exact active-time or
  whole-task denominator. Tiny intervals can yield misleadingly large active
  bandwidth. Missing fields remain unavailable, not zero.

Keep three quantities distinct: useful bytes from the algorithm divided by
independent kernel latency; raw reported transfer amount/bandwidth with its
exported units and row scope; and activity or active bandwidth within a pipe's
active window. Higher reported transfer bandwidth can accompany more wasted
traffic and lower useful throughput. Matching a counter's arithmetic formula
does not independently validate its physical byte semantics. Do not relabel
`KB` as `KiB` or call a raw amount exact HBM traffic without product/version
evidence; leave unknown byte factors unknown.

Use ordinary timing to decide whether a change helped and instrumented
evidence to explain why. A change in counters without a reliable timing
change is a mechanism observation, not an established optimization gain.

## Source constructs can hide a different execution path

**Indirect loads:** a tensor gather can lower to copying indices followed by
a scalar load loop. Runtime indices that happen to be contiguous do not
necessarily enable a direct-copy lowering. High scalar activity with low MTE2
activity can therefore be a memory problem. Consider locality, instruction
path and access semantics separately; do not assume one universal remedy.

**Masks:** placing a predicate on `tl.load` does not guarantee fewer physical
bytes. A compiler may copy the valid contiguous slice, then apply vector masks.
Lower MTE2 time alone cannot prove traffic reduction; additional indexing or
vector work may outweigh it. Inspect emitted copies and masks if that
distinction matters to the explanation.

The converse matters too: retaining a dense copy or adding mask operations
does not prove the candidate will be slower. Source-level predication can help
without an established physical-traffic reduction. Inspect structure to reject
unsupported causal stories, then measure the candidate; static operation counts
are neither executed instruction counts nor a latency model. An observed gain
with an unresolved cause is a valid result.

**Scalar decisions:** extracting a scalar or broadcasting a decision may add
scalar/vector conversions, conditional work and cross-pipe set/wait
dependencies. Scalar-looking source is not automatically cheaper. Correlate
both scalar and vector costs, while preserving the same useful work.

**Scheduling knobs:** equal TTIR or adapter IR does not prove tuning options
were inert. Buffer rotation and synchronization can appear later. Change
knobs independently when attributing their effects; a combined change cannot
identify each knob's contribution. Autotuning finds candidates; selected
candidate profiling explains mechanisms without needing expensive profiling
for every option.

Extra buffers and set/wait sites demonstrate a compiler transformation, not
achieved overlap, fewer dynamic waits or a speedup. Likewise, fewer loop sites
or iterations can alter transfer sizes, resource pressure and setup costs.
Without matched timing, keep candidate ranking tentative even when the
structural interpretation is well supported.

**Overlap:** source load/compute order does not prove independence or realized
overlap. Runtime aliasing, live ranges and dependencies matter. A source-level
phase label is not a source map or a common hardware clock.

## Synchronization and small matrix work

Start with native numerical correctness, matched hardware timing and supported
pipe counters; a simulator is optional, not an acquisition gate. Preserve the
exporter's AIC/AIV, block/sub-block and unit scope. For the A3 fields documented
in the [profiler's metric definitions](https://raw.githubusercontent.com/Ascend/msopprof/master/docs/en/user_guide/msopprof_performance_data.md):

| Field family | What it measures, not what its name might suggest |
| --- | --- |
| `ai*_scalar_wait_time(us)` | Intra-core wait blockage within Scalar; not every destination pipeline's waiting. |
| `ai*_scalar_wait_id*_time(us)` | Per-ID inter-core Scalar blockage; separate from the intra-core scalar-wait field. |
| `*_scalar_*_stall_time(us)` for an instruction queue (IQ) | Scalar issue blocked by a full named engine's queue; not the duration of a particular source wait. |
| `ai*_scalar_wait_ib_time(us)` | Scalar waiting for instruction-cache supply through the instruction buffer (IB); not an inter-core synchronization counter. |

Use the installed product/version definitions and actual available columns,
not the wildcard alone, to classify a field. Missing or `NA` remains unavailable.
Zero scalar intra-core wait does not rule out synchronization, per-ID blocking
or destination-pipeline waits. Moving a wait need not move every wait-like field.
These aggregates are neither source-PC traces nor a shared clock; do not add
them into an exact synchronization-cost budget.

Keep a reported ID distinct from a source flag ID. Matching names/numbers alone
do not prove association. A controlled flag-renumbering experiment, preserving
useful work, placement and otherwise matched compilation, can test whether the
exported field follows that flag on the observed build. Even a confirmed
association does not establish a particular instruction PC or a temporal phase.
Retain source intent separately from verified emitted flags; an undecodable
binary does not establish machine-level placement.

Try moving a wait only across genuinely independent work, without moving a
consumer before its producer is ready or allowing premature buffer reuse.
Preserve aliases, lifetimes, participant counts and cross-core ordering. Keep
paired signals/waits and framework event ownership: reserve distinct intra-core
event IDs for simultaneously outstanding same-type events and release owned IDs
after their matching waits. Cross-core flags have a separate namespace. A large
counter is not permission to remove a required dependency.

Tiny timings can be noisy even with complete captures. Keep task duration,
per-core counter times and ordinary stream-event intervals distinct; the latter
can include host-submission gaps. Retain dispersion and label an unresolved
comparison inconclusive rather than ranking tiny differences precisely.
Standalone load/compute/store controls can expose qualitative costs, but change
launch, cache and compiler context. Their sums or maxima are not exact bounds
on a combined kernel's latency or proof of overlap.

Equal Cube-active time can coexist with different whole-kernel durations.
For small matrix work, compare input movement, output/FIXPIPE, layout and
scalar/setup costs before claiming different matrix arithmetic efficiency.
Small one-program controls do not establish device-wide Cube saturation.
Do not subtract overlapping pipe times to assign an exact overhead budget.

## Make the conclusion falsifiable

Record the candidate identity, shapes/layouts, exact selector, controlled
change, predicted observable effect, actual timing and diagnostic results.
Explain competing causes and the next discriminating observation when needed.
Distinguish observed correlation from compiler-confirmed mechanism and from
critical-path proof. Retain all declared samples and their dispersion; a noisy
result may support a hypothesis but not a precise speedup. Do not blame a
device merely because compilation, correctness or timings surprised you.

State whether correctness was actually checked and where its result is retained.
Successful compilation/capture and plausible source establish neither executed
correctness nor a recorded numerical check. When candidate timings or checks
were not provided, say so; do not fill those gaps from provenance or confidence
in the implementation. Separate a prediction's direction from its explanation:
a correct latency prediction does not validate the proposed mechanism.
