# E12: Select Interval DMA for the A operand

Change only the native Catlass A GM-to-L1 copy primitive to an interval-copy implementation.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/catlass/interval_adapter.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```cpp
// Before, in a TileCopy derived from the same default copy policy:
using CopyGmToL1A = typename DefaultCopy::CopyGmToL1A;
// After; B/C movement, geometry, staging and scheduling stay identical:
using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<
    Arch::AtlasA2, AType>;
```

## Contract and recognition

Atlas A2/A3 architecture 2201; row-major FP16 A/B/C, aligned N and K multiples of 16,
positive M/N/K, with deliberate M tails. The adapter admits dimensions at most 32768, not
proof every shape fits or performs well. Keep L1 tile 128×256×256, L0 tile 128×256×64,
ping-pong stages and identity swizzle fixed. Compiled explicit L1 reservation is 393216
bytes and workspace zero; verify actual compiled resource and numerical contracts.

Recognize the copy-policy alias rather than inferring a different GEMM from a kernel name.
The copy implementation may change DMA command/setup behavior for row intervals; partial
rows and copy granularity matter. Compare compiled copy lowering and complete GEMM span at
identical geometry. No packing or compilation is inside the profiled forwards.

## What the gallery establishes

Both captures are inconclusive, including small M. The noisy small-M means do not
establish a crossover or reliable benefit. Unchanged DMA lowering or no reproducible
full-span gain falsifies a timing recommendation; supported activity fields alone cannot
establish fewer transferred bytes.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e12-gallery-many-rows | 6.474 ± 1.829 / 6.616 ± 0.6352 | 0.9785×; inconclusive |
| e12-gallery-small-m | 33.438 ± 12.93 / 32.868 ± 15.04 | 1.017×; inconclusive |

Source mechanism: [pinned source 1](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#docs/en/2_Design/01_kernel_design/04_matmul_summary.md:L452-L485),
[pinned source 2](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp:L286-L314),
[pinned source 3](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/06_optimized_matmul/optimized_matmul.cpp:L43-L61).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
