# Fuse a GEMM add epilogue

Status: both Triton and both native gallery pairs passed A3 correctness and
ten-forward profile/identity collection. The example is
`examples/optimization-library/triton/gemm.py`, family `02`, adapter `gemm:Adapter`.
Its shared CPU contract lives in the neighboring `gemm_contract.py` so native
examples can reuse the same oracle cache instead of computing a new baseline.

The [pinned native block-epilogue assembly](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/03_matmul_add/matmul_add.cpp:L111-L120)
motivates this independently written Triton pair. It is not evidence that our
implementation or target improves.

Recognize a GEMM result immediately consumed by an elementwise add, with no
other consumer of the intermediate. Before:

```python
intermediate = allocate_fp32(M, N)  # Allocation belongs to every before call.
gemm_fp16_inputs_fp32_store(A, B, intermediate)
add_fp32(intermediate, Z, output)
```

After, add Z while the accumulator is still FP32:

```python
acc = dot_fp16_accumulate_fp32(A, B)
acc += load_fp32(Z)
store_fp32(output, acc)
```

This removes a launch and the caller-visible FP32 intermediate allocation.
It also lengthens the accumulator lifetime and adds an epilogue load to the
source kernel. The compiler can still stage Cube results through implicit GM
workspace before Vector processing; this source fusion does not prove that the
physical intermediate write/read disappeared. A compute-heavy long-K GEMM may gain little while paying resource or scheduling
cost. Compare complete calls and workspace bytes, not just the matrix kernel.
The gallery tests 2048×2048×256 and 256×256×8192. These test benefit and
competing cost rather than define a crossover threshold.

Both arms use FP16 A/B, FP32 Z[M,N], and FP32 output. The unfused arm writes an
FP32 intermediate; no FP16 rounding is inserted before the add. Both must pass
the same FP32 CPU matmul-plus-add oracle at rtol=atol=1e-3. Reject fusion if
the intermediate is separately required or a layout/numerical contract differs.

In the profile, establish two expected device kernels per iteration before and
one after, ten owned iterations each, plus successful numerical checks. This
supports the launch-count change. Claim reduced physical GM traffic only when
an available counter establishes it; logical eliminated bytes are an estimate.
Missing PMU fields remain unavailable. Falsifiers include unchanged complete-call
time within variability, a slower fused call, or extra compiler-generated work.

Read each timing boundary below. A regression or an inconclusive caller-event
comparison must not be replaced by a selected faster sample.

## Native Catlass C++ format

`examples/optimization-library/catlass/mixed.cpp`, FAMILY=2, keeps the same
FP16 operands and FP32 accumulation/output contract. The before arm launches
BasicMatmul into an FP32 workspace and a separate AIV add; the after arm
launches MatmulEpilogue. Construct its public Params with separate Z and output
pointers: the upstream convenience Arguments aliases the addend and output.
The addend remains immutable and no reset/copy is hidden outside timing.

This format fuses the launch and cross-core scheduling; it **retains** the
M*N*4-byte explicit intermediate GM workspace, its store and its AIV read.
The Triton format removes the caller-visible allocation but can still have
compiler-created GM staging. Neither explicit-workspace size alone certifies
physical traffic savings. Both native arms preallocate that workspace before
the measured call. Their
measurements include its device traffic and the standalone add or mixed-kernel
synchronization, but cannot claim an allocation saving. The Triton before arm,
in contrast, allocates its intermediate in each call. Inspect real kernel/task
identities and all per-forward tasks before attributing a gain to launch fusion. High K or a small
output may make synchronization/UB residency dominate the saved launch.

Native factory: `mixed_adapter:epilogue_factory`; the oracle is the exact shared
`gemm_contract.py` provider used by Triton. Source: [MatmulEpilogue public Params
and workspace](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/matmul_epilogue.hpp:L51-L107).
The standalone native add must use the compiled execution context, not infer
AIV-only behavior from a source hint. In the tested mixed 1:2 context, sibling
AIV subblocks must share a logical output-tile owner before `BlockAdd` splits
that tile internally:

```cpp
const auto owner = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
// Both sibling subblocks receive the same logical tile.
// BlockAdd then applies its own subcontext partition.
```

Assigning a different logical tile to each sibling and then splitting again
left half the output unwritten in a rejected baseline attempt. The corrected
gallery baseline and fused arm passed the same oracle. Resource metadata binds
logical mixed blocks to the observed Cube/Vector topology; preserve that
relationship when porting the launch.

No correctness or latency claim follows from compilation alone.

Runnable sources: [Triton GEMM/add](../../../examples/optimization-library/triton/gemm.py),
`gemm:Adapter`; [native mixed assembly](../../../examples/optimization-library/catlass/mixed.cpp),
`mixed_adapter:epilogue_factory`. Both use the same
[CPU provider](../../../examples/optimization-library/triton/gemm_contract.py).


## What the gallery establishes

Both Triton regimes improved device span, so the long-K case did not demonstrate
a regression. Its caller-event comparisons remained inconclusive. Neither native
regime established a device-span gain in these captures. The native format has
its own table and workspace boundary; do not transfer the Triton
format's caller-visible allocation change to a mixed kernel that still uses GM
workspace. A required intermediate consumer or a different conversion order
would invalidate this particular fusion.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| triton / benefit | 103.472 ± 10.54 / 78.6245 ± 1.257 | 1.316×; gain | 0.316694 ± 0.08611 / 0.246434 ± 0.1198 | inconclusive |
| catlass / benefit | 36.25 ± 6.975 / 31.242 ± 5.299 | 1.16×; inconclusive | 0.184558 ± 0.2222 / 0.137796 ± 0.07743 | inconclusive |
| triton / cost | 87.1421 ± 11.85 / 57.5509 ± 1.084 | 1.514×; gain | 0.334488 ± 0.1524 / 0.208334 ± 0.09668 | inconclusive |
| catlass / cost | 54.44 ± 4.259 / 52.336 ± 1.34 | 1.04×; inconclusive | 0.189488 ± 0.1125 / 0.161666 ± 0.1537 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aic_scalar_ratio` | `aiv_vec_ratio` | `aic_mte2_ratio` |
| --- | ---: | ---: | ---: |
| triton/benefit/before/`gallery_add` | 0 | 0.1933 | 0 |
| triton/benefit/before/`gallery_gemm` | 0.1671 | 0 | 0.9383 |
| triton/benefit/after/`gallery_gemm` | 0.6827 | 0.0338 | 0.9367 |
| catlass/benefit/before/`wave1_add_matmul` | 0.5214 | 0.1218 | 0.5336 |
| catlass/benefit/before/`wave1_add_vector` | 0.207 | 0.1427 | 0.021 |
| catlass/benefit/after/`wave1_add_fused` | 0.8592 | 0.0754 | 0.554 |
| triton/cost/before/`gallery_add` | 0 | 0.0447 | 0 |
| triton/cost/before/`gallery_gemm` | 0.1972 | 0 | 0.9729 |
| triton/cost/after/`gallery_gemm` | 0.2278 | 0.002 | 0.9442 |
| catlass/cost/before/`wave1_add_matmul` | 0.6236 | 0.1217 | 0.5601 |
| catlass/cost/before/`wave1_add_vector` | 0.2065 | 0.0123 | 0.021 |
| catlass/cost/after/`wave1_add_fused` | 0.7103 | 0.0046 | 0.3934 |

Observed resource/numerical costs and limits:
- **triton/benefit**: adapter-explicit workspace allocation 16777216->0 bytes.
- **catlass/benefit**: no additional resource/numerical cost identified by the retained fields.
- **triton/cost**: adapter-explicit workspace allocation 262144->0 bytes; cost recipe did not produce a device-span regression in this capture.
- **catlass/cost**: no additional resource/numerical cost identified by the retained fields.
