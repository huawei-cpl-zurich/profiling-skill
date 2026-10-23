# Retain a full A tile in L1

Status: both listed native gallery pairs passed A3 correctness and ten-forward
profile/identity collection.
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

Test 128x32768x256 (128outputtasks, repeated A offsets per core) against
128x256x256 (one task, no per-core reuse). Expected benefit is fewer redundant A loads; competing
costs are the larger fixed L1 reservation and reuse bookkeeping. In the short
case the retained tile may add no useful work. Report actual ten-forward
whole-launch duration, resource configuration and supported traffic evidence.
No repeated A offset or no duration/traffic improvement weakens the diagnosis.
Both paths retain the same FP32 CPU oracle followed by FP16 conversion,
1e-3 absolute/relative tolerance, and immutable inputs.

Sources: [capacity and policy](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/25_matmul_full_loadA/matmul_full_loadA.cpp:L87-L110),
[per-core offset reuse](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/matmul_full_loadA.hpp:L122-L154).

## Code format

```cpp
// Before: A is loaded for each output task.
using BeforePolicy = Gemm::MmadAtlasA2Pingpong<true>;
using BeforeKernel = Gemm::Kernel::BasicMatmul<BeforeBlockMmad, void, Scheduler>;
// After: BlockMmad uses the full-A policy and the matching kernel API.
using AfterPolicy = Gemm::MmadAtlasA2FullLoadA<true>;
using AfterKernel = Gemm::Kernel::MatmulFullLoadA<AfterBlockMmad, void, Scheduler>;
```

Instantiate each `BlockMmad` with its corresponding policy, types and tiles.
The snippets show assembly choices; the linked file contains complete types.
Use the complete [native source](../../../examples/optimization-library/catlass/preload.cpp),
family `09`, and `preload_adapter:factory`. Admit capacity and the per-core A
reuse condition before choosing the resident path.


## What the gallery establishes

Neither listed case established a timing gain at either boundary. The after
assembly reserved 524,288 L1 bytes versus 393,216 before: a confirmed capacity
cost without a demonstrated speed benefit here. Repeated A offsets make reuse
possible, but do not prove that the original loads were the limiting cost.
Treat residency as an unproved performance hypothesis for these regimes.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| catlass / benefit | 15.7 ± 0.9159 / 15.322 ± 3.357 | 1.025×; inconclusive | 0.270156 ± 0.4029 / 0.172478 ± 0.1913 | inconclusive |
| catlass / cost | 4.494 ± 0.9108 / 4.492 ± 0.8899 | 1×; inconclusive | 0.239518 ± 0.1523 / 0.21912 ± 0.2523 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aic_mte2_ratio` | `aic_mac_ratio` | `cube_utilization(%)` |
| --- | ---: | ---: | ---: |
| catlass/benefit/before/`wave1_resident_before` | 0.6194 | 0.6062 | 86.769 |
| catlass/benefit/after/`wave1_resident_after` | 0.5147 | 0.6463 | 84.508 |
| catlass/cost/before/`wave1_resident_before` | 0.0455 | 0.0551 | 30.006 |
| catlass/cost/after/`wave1_resident_after` | 0.0431 | 0.0539 | 30.145 |

Observed resource/numerical costs and limits:
- **catlass/benefit**: l1_reserved_bytes: 393216 -> 524288.
- **catlass/cost**: l1_reserved_bytes: 393216 -> 524288.
