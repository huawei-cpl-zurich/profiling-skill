# A3 performance envelopes: quantify the gap, then choose an action

Use architectural bounds and empirical reference curves under explicit
assumptions, rather than a utilization score. Keep an architectural limit
separate from an empirical reference measured on
the same product, runtime, data type, shape regime, access contract, cache
regime and device conditions. A contiguous-copy result is not an attainable
target for arbitrary gathers or a mandatory strided layout unless the workload
contract permits changing that access path.
Neither establishes that a particular kernel has saturated a resource.

## Architectural lower bounds

Define useful work first. For dense `M×K` times `K×N`, conventional useful
FLOPs are `F = 2MNK`; state whether that convention includes an epilogue.
With validated dtype-specific whole-device compute capacity `P FLOP/s`,
`t_compute ≥ F/P`. Do not apply an fp16 dense capacity to fp32, sparse or mixed
operations. Core count, frequency and whether enough programs occupy the device
are part of the bound. One small tile cannot assume whole-device compute rate.

For a validated traffic lower bound `Q` bytes and compatible bandwidth `B B/s`,
`t_memory ≥ Q/B`. A simple GEMM reading fp16 A and B once and writing fp32 C
has logical minimum `Q = 2MK + 2KN + 4MN` bytes. Actual tiling, duplicated
reads, packing, padding and epilogues can increase traffic; caches can change
which hierarchy services it. A copy of N fp32 elements has `8N` useful bytes
including read and write; an identity-index gather adds `4N` logical index
bytes. A stride-2 copy has the same useful count but a larger accessed span.
Span, useful bytes and physical bus traffic are different quantities.

An optimistic roofline lower bound is
`max(F/P, Q/B)` and arithmetic intensity is `I = F/Q` FLOP/byte. It assumes
sufficient overlap and parallelism. It excludes launch/startup, pipeline fill
and drain, scalar addressing, synchronization, output conversion and other
resources; do not predict exact latency from that maximum. If stages must
serialize, a dependency-aware sum or measured stage envelope is more relevant,
but overlapping pipe times are not such a sum.

Do not hardcode an A3 peak from another SKU. Obtain a product/dtype/frequency
specification and cite it, or leave the architectural number unavailable. The
bundled activity model deliberately supplies no A2/A3 capacity denominator.

## Build a measured reference envelope

Use `examples/a3_paired_probes.py` for reproducible reference controls, with
BasicInfo as the timing method and exact exported selectors discovered first.
Sweep copy counts, for example `65536`, `1048576`, `16777216`, and GEMM square
sizes `256`, `512`, `1024`. Record allocations, source/compiled identity,
requested stages, all timing samples and correctness. Reference scripts require
Torch-NPU and a compatible Triton-Ascend installation; they do not install tools.

A measured reference is a demonstrated attainable target, **not a proven upper
bound on hardware capacity**. A better implementation may exceed it. Try to
match or exceed the best comparable reference rather than treating its rate
as a hardware ceiling.

For each size, retain median and dispersion and calculate:

- Useful copy throughput: `(8N)/(median_us × 1000)` decimal GB/s. Include
  index bytes only when explicitly reporting indexed logical traffic; never
  relabel logical traffic as measured DRAM traffic.
- Useful GEMM rate: `(2MKN)/(median_us × 10^6)` TFLOP/s. Keep tile, dtype,
  dimensions and requested/emitted buffering alongside the result.
- Candidate/reference latency ratio: `t_candidate/t_reference`, and the
  descriptive reference gap `1 − t_reference/t_candidate` when the reference
  is faster. This compares implementations; it is not a proven removable time.
  If the candidate is faster, report that improvement directly; do not force
  its gap to a positive number or declare an impossible capacity violation.

Example arithmetic only: if a size-matched copy moves `8 MiB` of useful data
in `20 μs`, useful whole-task throughput is `419.43 GB/s`. If another measured
implementation of that same work takes `30 μs`, its latency is `1.5×` the
reference and its descriptive gap is `33.3%`. It is incorrect to call this
“33.3% unused MTE2 capacity”: the reference includes output and other overhead.

Keep cache regime explicit. Repeated warmup/replay of one allocation can measure
cache-resident behavior; a large working set or rotated inputs is a different
experiment. Small sizes expose startup costs, middle sizes expose the transition,
and large sizes can show a throughput plateau. A maximum observed rate among
several sizes is not a matched small-shape target. Avoid extrapolating a noisy
single sample or a Pipe/Memory/Conflict-instrumented duration into an
envelope. Independent BasicInfo timing is the declared measurement contract
used here; it is distinct from those diagnostic captures.

## Read the gap as a decision table

| Evidence | Useful next action | What would distinguish it |
| --- | --- | --- |
| Both candidate and reference slow at tiny sizes, close at large sizes | Reduce dispatch/startup or fuse useful work if semantics permit | Size curve and program count; separate device-task from host launch time. |
| Low whole-task bandwidth, high MTE2 active fraction, low active bandwidth | Improve transaction shape, alignment, contiguous spans, reuse or packing | Late transfer instructions, validated active bandwidth and physical-traffic evidence. |
| Low whole-task bandwidth but reasonable active bandwidth | Find gaps, serialization, insufficient programs or load imbalance | All core rows, wait/stall categories, grid and buffer lifetimes. |
| High scalar time, scalarized runtime gather, low DMA activity | Use an affine path when the contract guarantees it; otherwise improve locality/index representation | Emitted address/load loop and a correctness-preserving direct/gather pair. |
| Low GEMM rate with small grid or short K loop | Increase useful tile/program parallelism or amortize setup while respecting memory | Sweep shape/K tile count; compare Cube active time to movement/FIXPIPE and task time. |
| Equal useful work, more buffers, less task time | Check what moved off the critical chain | Real rotating buffers/flags, matched pipe times and queue/wait counters; overlap remains inferred without a common-clock trace. |
| More buffers increase latency or fail allocation | Reduce resource pressure, revisit tile size or stages | Emitted extents/fallback, program residency, startup/drain and ordinary timing. |
| High measured useful throughput near a matched reference | Reduce required bytes/FLOPs or improve reuse, rather than chase a fraction | Algorithmic work accounting plus correctness and fresh timing. |

Declare the conclusion's strength: observed timing difference; observed
counter/path difference; compiler-confirmed transformation; or validated
capacity bound. If available evidence supports only the first two, propose the
next discriminating capture rather than inventing a saturation threshold.
