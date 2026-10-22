# A3: connect code, execution and performance

Use this guide to choose a discriminating measurement, not a predetermined
rewrite. Keep four layers separate: architectural path, emitted operation,
reported counter, and observed effect on independently measured task latency.
A3 observations do not establish A2 or A5 behavior.

## Follow addresses and instructions, not source load/store names

GM is global device memory, UB is Vector local storage, L1/cbuf stages matrix
operands, L0A/L0B feed Cube, and L0C/CC holds matrix accumulators. `PIPE_M` names
the matrix/Cube pipe in compiler IR; `PIPE_FIX` names result movement/conversion.
A memory-space name describes storage, not automatically the executing pipe.

| Common operation / path | Expected resource | What to verify before attribution |
| --- | --- | --- |
| Contiguous GM → UB input | MTE2 | Lowered `load`/DMA addressing, transfer dimensions, valid slice and stride. An indirect source load can take a different path. |
| GM → L1/cbuf matrix input, including ND→NZ packing | MTE2 | `nd2nz` or copy destination in cbuf; distinguish requested useful elements from packed/strided physical transfer. |
| GM → L0A/L0B where the emitted instruction supports it | MTE2 | These channels are listed by A2/A3 MemoryDetail; do not assume the compiler bypasses L1 merely because the architecture exposes them. |
| L1/cbuf → L0A/L0B operands | MTE1 | Address spaces and operand-copy/matrix macro expansion. `mmadL1` can include this movement internally. |
| L0A × L0B → L0C/CC accumulation | Cube (`PIPE_M`) | Lowered matrix multiply, dtype and tile. A matrix macro may include several resources rather than one Cube instruction. |
| L0C/CC → GM | FIXPIPE | `fixpipe`, conversion mode, layout, destination address and exact result dtype. A source `tl.store(dot(...))` need not use MTE3. |
| L0C/CC → L1 | FIXPIPE | Same result-movement family with L1 destination, not an input MTE1 copy. |
| UB → GM ordinary Vector output | MTE3 | Lowered `store`/copy, contiguous versus scalarized route, conversion and supported atomic mode. |
| L1 → GM through NZ→ND conversion | MTE3 | `nz2nd` is a separate outbound conversion path; MTE3 is not limited to UB stores. |
| UB → UB lowered `CopyOp` | Vector (`PIPE_V`) in the pinned mapping | Confirm it is the reviewed UB-to-UB CopyOp; local copies consume work even though they do not reach GM. |
| Arithmetic, masks, conversions, reductions and other local layout operations | Vector for the pinned Vector base operation, with specialization-dependent exceptions | Inspect the actual operation: scalar-looking expressions can introduce Vector conversions or scalar loops; do not generalize a CopyOp mapping to every layout operation. |
| Indexed GM reads lowered into per-element address/load loops | Scalar can dominate | Actual scalar loads, loop trip count, index movement and instruction path. Runtime identity indices do not make the compiled operation a contiguous DMA. |
| Scalarized or irregular output writes | Emitted store instruction determines the resource | A source store is not a universal MTE3 assertion. Check scalar versus DMA writes, address space, supported scatter/atomic form and any intermediate UB/workspace conversion. |
| Result conversion/activation/scale before final output | Vector and/or FIXPIPE plus movement | Follow the accumulator into UB/workspace and final destination. A fused epilogue can introduce cross-core handoff and several pipes. |
| Flags, barriers and core handoffs | Synchronization, not a data-transfer pipe | Producer, consumer, event/flag, address and lifetime; naming MTE2 in a wait does not make the wait MTE2 activity. |

Pinned compiler evidence is AscendNPU-IR
`af5499b3b9f3dbab50b2834bcfff5da5c2a1d920`:
`bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td:1–620` defines
Load/MTE2, Store/MTE3, Fixpipe/FIX, ND2NZ/MTE2 and NZ2ND/MTE3;
`HIVMMacroOps.td:1–300` defines compound matrix operations and sync operands.
The [official multi-buffer guide](https://ascendnpu-ir.gitcode.com/en/developer_guide/features/multi_buffer.html)
explains the corresponding L1, L0C, UB and workspace producer/consumer chains.
These mappings supplement the narrower [query inventory](triton-pipe-attribution.md);
they do not silently change its reviewed `direct`/`inferred`/`unknown` statuses.

## What did the profiler actually measure?

Read [the field glossary and calculations](a2-a3-msprof-op.md#field-glossary-and-calculations)
before deciding that a large fraction is a bottleneck. Compare the same
block/sub-block/core type across candidates, retaining both absolute time and
ratio. Cover all rows rather than choosing a convenient core. Rows with different
start times have no common exported clock.

Activity answers “how much of this row's task-cycle window contained the counted
work?” Capacity answers “how much validated useful work could that resource
sustain?” They differ. High MTE2 activity with poor bytes per active microsecond
suggests inefficient transfers; high scalar activity with low MTE2 can be an
indirect-memory instruction path. A faster candidate can have a larger dominant
fraction simply because its denominator shrank.

Do not sum overlapping active times into a latency decomposition. Never multiply
a ratio median from one replay by a latency median from another to manufacture
active time. Use same-row exported times, with their capture identity. A scalar
wait counter of zero rules out only that counted event in that capture; queue,
inter-core, instruction-fetch and UB stalls are distinct. Follow the
[wait/stall decision table](a2-a3-msprof-op.md#diagnose-waits-and-stalls), then
inspect [generated synchronization](a3-synchronization-inspection.md).

## Source constructs can hide a different path

**Indirect accesses.** First establish whether the address is affine, a
compile-time index or a runtime-loaded index. A gather can copy indices and then
iterate scalar loads even when values happen to form an identity permutation.
Compare a semantically equivalent affine implementation only when the index
contract permits it. For genuinely arbitrary indices, test locality, index
representation and coalescing rather than replacing them with assumed order.

**Masks and sparse work.** A masked load can copy a contiguous valid slice and
apply predicates afterward. Count useful bytes separately from transferred
bytes. Sparse predicates do not establish sparse physical traffic. Moving
predicates can add scalar loops, vector selection or output writes; inspect all
those paths and check full correctness.

**Scalar decisions.** Scalar extraction/broadcast can add conversions and
cross-pipe handoffs. High scalar activity is not necessarily a user-written loop,
and unrolling can enlarge code and worsen instruction-fetch stalls. Locate the
emitted loop before choosing a remedy.

**Buffering and scheduling.** Equal early IR can lead to different buffer
addresses, rotating flags and final scheduling. Compare compiler arguments and
late memory/sync IR. More buffers permit overlap but increase local-memory
footprint; changed static flag counts do not measure dynamic waiting. A compiler
can fall back after resource pressure, so record what it emitted, not just the
requested option.

**Small matrix work.** Equal Cube time with different task latency points to
movement, FIXPIPE/epilogue, setup, handoffs or scheduling as competing causes.
A one-program matrix probe is not a device-wide capacity test. Use multiple
matrix sizes and enough programs/K tiles before extrapolating to a steady state.

## Turn the gap into the next action

Use [architectural and empirical envelopes](a3-performance-envelope.md) to name
the limiting resource, the useful-work definition, and what remains unmeasured.
A candidate near an empirical envelope can still improve by reducing bytes,
reusing operands or changing the algorithm. A candidate far below it needs an
explanation at the actual shape and dispatch scale, not an arbitrary utilization
goal. [Paired examples](a3-paired-examples.md) show how to bind source changes,
late IR, counters and ordinary timing without conflating them.

Record a prediction before changing code: which counter/path should change and
why should ordinary timing improve? If timing and counters disagree, retain the
result, competing causes, and the next measurement. Correctness, profiler
availability and speedup are separate outcomes.
