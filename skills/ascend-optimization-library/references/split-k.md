# SplitK and complete partial reduction

Status: source-backed example; hardware correctness and profiles pending.

When M/N output tiles leave most Cube cores idle and K is long, split K into
independent FP32 partial matrices, then reduce every partial before converting
the complete result to FP16. In `parallelism.cpp`, family06 changes BasicMatmul
into `SplitkMatmul<BlockMmad,void,SplitkGemmIdentityBlockSwizzle<3,0>,ReduceAdd>`
with `SplitkReduceAdd<AtlasA2,float,half,2,16384>`. The source heuristic chooses
the factor; the adapter records the actual value and workspace bytes.

Compare whole-call latency, per-kernel activity and workspace traffic. Extra
partial writes, AIV reduction and cross-core synchronization can cost more than
extra parallelism saves. With abundant M/N tasks the factor may be one while
the mixed reduction still runs. The negative case therefore remains meaningful
without forcing artificial extra slices. Changed summation order must pass the
same FP32 CPU accumulation → FP16 oracle, rtol=atol=1e-3. A factor above one
alone is not evidence of a speedup or reduced bandwidth pressure.

Gallery: 64×64×16384 versus 2048×2048×256; held-outs use 96×96×12288 and
1536×1536×384. The complete mixed launch includes reduction and its barrier.
Workspace is allocated before capture and its capacity cost is retained.

Pinned [kernel and heuristic](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/splitk_matmul.hpp:L142-L276),
[reduction](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/splitk_matmul.hpp:L23-L139).
No recipe timing or roofline claim is verified yet.
