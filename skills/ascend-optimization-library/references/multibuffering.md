# Operand multibuffering

Status: source-backed example; hardware correctness and profiles pending.

Look for repeated K-stage transfer/compute dependence in a kernel with enough
K work to amortize fill and drain. In `parallelism.cpp`, family07 holds tile
geometry, types, scheduler and BasicMatmul fixed while changing
`MmadPingpong<AtlasA2,true,false,1,false,S,S,S,S>` from S=1 to S=2.
L1 tiles are 128×256×256 and L0 tiles 128×256×64 in both arms.

Two operand buffers consume more L1/L0 capacity and may worsen a short K case.
Use the retained complete-call samples and supported timeline evidence to test
whether transfer gaps shrink. Pipe activity ratios alone do not demonstrate
simultaneous overlap. A crossover is inconclusive if it is within sample
variability; do not call additional buffering universally beneficial.

Gallery: 2048×2048×4096 versus 256×256×256. Held-outs: 1536×2304×3072 versus
192×320×192. FP16 input/output and FP32 accumulation retain rtol=atol=1e-3.
The adapter records the stage count and actual compiled source/binary identity.

Pinned [stage policy](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/dispatch_policy.hpp:L351-L365),
[dispatch and capacity checks](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/block/block_mmad_pingpong.hpp:L24-L139).
