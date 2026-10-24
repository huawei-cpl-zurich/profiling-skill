# Retain a full A tile in L1

Status: source-backed example, awaiting A3 correctness and ten-forward captures.
Native family09 is in `examples/optimization-library/catlass/preload.cpp`.

Recognize multiple output tiles that use the same A rows on a given core.
A large N and M<=128 make this explicit for the example's fixed scheduler.
The before path reloads A via `BasicMatmul`/ping-pong. The after path uses
`MatmulFullLoadA`/`MmadAtlasA2FullLoadA<true>` and suppresses later A loads
when the per-core logical A offset matches its previous task. Tiles and
scheduler stay fixed. Global memory savings remain a measurement claim;
a changed pipe ratio is not a byte counter.

The admission check is128*K*2+256*256*2*2<=512KiB with aligned FP16 row-major
inputs and M<=128. The policy reserves a fixed256KiB A partition plus two
128KiB B buffers, even when the actual A tile is smaller. This512KiB reserved
footprint exceeds the384KiB ping-pong footprint. Reject unsupported capacity
before preparation; a failed build or wrong answer is not a tradeoff case.

Test128x32768x256 (128outputtasks, repeated A offsets per core) against
128x256x256 (one task, no per-core reuse). The held-out K384 cases must also
satisfy capacity. Expected benefit is fewer redundant A loads; competing
costs are the larger fixed L1 reservation and reuse bookkeeping. In the short
case the retained tile may add no useful work. Report actual ten-forward
whole-launch duration, resource configuration and supported traffic evidence.
No repeated A offset or no duration/traffic improvement weakens the diagnosis.
Both paths retain the same FP32 CPU oracle followed by FP16 conversion,
1e-3 absolute/relative tolerance, and immutable inputs.

Sources: [capacity and policy](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/25_matmul_full_loadA/matmul_full_loadA.cpp:L87-L110),
[per-core offset reuse](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/matmul_full_loadA.hpp:L122-L154).
