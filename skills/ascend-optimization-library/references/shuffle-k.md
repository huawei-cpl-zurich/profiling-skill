# Stagger the per-core K starting tile

Status: source-backed example, awaiting A3 correctness and ten-forward captures.
Native family11 is in `examples/optimization-library/catlass/preload.cpp`.

Recognize a large K traversal where cores concurrently start at the same
operand region. A conflict or cache-contention explanation needs source and
profile support; high MTE activity alone does not identify that cause.

Both arms use the same `OptimizedMatmul<void,void,...>`, scheduler,
128x256x256 tiles and FP16 operands/output. Change only
`MmadAtlasA2Preload<true,false>` to `<true,true>`. The policy changes
startTileIdx from zero to blockIdx and visits
(startTileIdx+kLoopIdx)%ceil(K/256). Each core still visits every K tile.
This is K accumulation order, distinct from output-tile program ordering.

Test2048x2048x4096 against512x512x256. With sixteen K tiles, staggered starts
may reduce conflicting accesses; modulo/index work and changed accumulation
order are competing costs. With one K tile both arms visit the same tile;
compiler elimination can remove the overhead, so do not promise a slowdown
or invent a tradeoff from equal paths. The unchanged numerical contract is
FP32 CPU accumulation then FP16 output,1e-3 absolute/relative tolerance.
Rounding differences must be measured; bitwise equivalence is not required.

Compare complete-launch ten-sample distributions and supported memory/pipe
observations. A benefit restricted to the multi-tile case is consistent with
the hypothesis; equal timing, an absent contention signal or a one-tile
speedup demands another explanation. Pipe ratios are neither bandwidth nor
distance to roofline. Both policies reserve384KiB L1 in this assembly.

Sources: [rotation implementation](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/block/block_mmad_preload.hpp:L150-L174),
[design rationale](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#docs/en/2_Design/01_kernel_design/04_matmul_summary.md:L430-L453).
