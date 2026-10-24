# Interval DMA for short, wide A tiles — E12

This source-backed example changes only the A GM-to-L1 copy method. Both arms
compute the complete FP16 GEMM with FP32 accumulation and FP16 output. A, B and
C use dense, unpadded row-major ND storage. No conversion prologue is omitted
from either arm. The L1/L0 tiles remain `128×256×256` / `128×256×64`, with the
same pingpong policy, stages, unit flag, B movement and scheduler.

```cpp
struct IntervalCopy : DefaultCopy {
    using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<Arch::AtlasA2,AType>;
};
using BlockMmadType = Gemm::Block::BlockMmad<Policy,L1Tile,L0Tile,
    AType,BType,CType,void,IntervalCopy>;
```

The default arm inherits `DefaultCopy::CopyGmToL1A`. The treatment issues one
strided copy per valid A row into the same zN destination. This is instruction
selection for an unchanged representation; it is distinct from GM repacking,
padding, preloading or tile tuning. The specialization is specifically **half
RowMajor to zN on AtlasA2**. It does not establish A5 or other dtype support.

Expected benefit: source documentation identifies inefficient ND2NZ movement
for very few rows, especially M below8. Expected cost: the explicit row loop
increases DMA issue requests for larger M. The after arm deliberately remains
unconditional inside this constrained experiment so that this cost hypothesis
can be tested; a production dispatch should use a measured crossover.

| Recognize in code | Look for in retained profiles | Tradeoff to check |
| --- | --- | --- |
| A tiles have few valid rows and substantial K | Matched whole-GEMM duration and MTE2 composition | ND2NZ replacement may help even when several row requests are issued |
| Tile contains many valid A rows | Increased duration or Cube bubbles alongside copy activity | Per-row issue overhead can outweigh conversion savings |
| K/N or physical storage are incompatible | Reject before device execution | Do not silently pad, reinterpret storage, or benchmark a different contract |

`copy_plan` reports logical source-call opportunities only. It is not a count
of compiled DMA instructions, physical transferred bytes, cache traffic or
roofline distance. MTE2 active ratios alone cannot establish any of those.
Inspect generated instructions when available and retain unsupported metrics
as unavailable. Compare the complete device-operation span including all tasks
and gaps; do not present profiled duration as unprofiled application latency.

The CPU gate exercises the actual configuration header through a host shared
library, arm mismatch rejection, M tails, N/K alignment, the independent oracle
and mutation/error handling. It **does not compile or execute Ascend code**.
The four frozen A3 case hypotheses remain unqualified until device compilation,
one correctness call and ten profiled forwards per arm/case succeed. Numerical
tolerances remain `atol=rtol=1e-3`. No warmup or measured I/O inspection is used.

Build prerequisite: a filtered dependency closure pinned to Catlass
`40aced927fdaaa0f3eaa9c6f5c5d859e06392f55`, CANN Ascend C headers, `bisheng` for
`dav-c220`, and a runtime exposing the native bridge. The compile helper binds
the additional configuration header into each binary's source manifest. Paths
and runtime selection belong in ignored profiles. This example preserves the
existing native bridge and does not modify low-level synchronization APIs.

Source citations (CANN Open Software License2.0; see `CANN_LICENSE`):

- `ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp:L286-L314`
- `ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#docs/en/2_Design/01_kernel_design/04_matmul_summary.md:L452-L485`
- `ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/06_optimized_matmul/optimized_matmul.cpp:L43-L110`
- `ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/block/block_mmad.hpp:L19-L29`
