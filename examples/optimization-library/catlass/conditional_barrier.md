# W006 L05: stock conditional Cube barrier

This experiment compares the pinned CATLASS no-bias `TileMmad` helper with a
conservative copy that always executes its final `PIPE_M` barrier. AFTER is
unmodified stock behavior. BEFORE replaces only the first overload's final
predicate with `true`; it does not call stock and append a second barrier.
The bias overload and the architecture-specific float branches remain intact.
The unexplained stock threshold is neither tuned nor interpreted here.

The source pin is `40aced927fdaaa0f3eaa9c6f5c5d859e06392f55`, under the
CANN Open Software License Agreement Version 2.0; see `CANN_LICENSE`.
Dependencies must be a verified, broker-filtered include closure, including
`small_matmul.hpp`, with the pinned `manifest.json`. The compiler validates
all file hashes before copying the closure into separate build dependencies.
`barrier-patch.json` records the original manifest/header/no-bias-body hashes
and the compiled header hash. Original dependencies remain unchanged.

Both arms use the same native SmallMatmul block, explicit AIC annotation,
architecture 2201, STAGES=2, unit flags disabled, row-major FP16 inputs/output,
and FP32 accumulation. Input copies, producer events, accumulator updates,
Fixpipe completion and terminal barriers are identical. Inputs are represented
FP16 values in [-0.25,0.25]; the independent CPU FP64 oracle uses those exact
represented inputs, with atol=0.002 and rtol=0.002. Inputs remain immutable.
Outputs begin as NaN and have sixteen FP16 canary values on each side.

Each case has a distinct compiled tile resource. Actual helper dimensions are
64x64, 32x64, 64x80 and 32x48 for the four main cases, with L1 K=256 and L0 K=64.
These cover 16, 8, 20 and 6 native C0 fragments respectively. The correctness
boundary guard uses 32x80 (10 fragments). The second guard executes two distinct
64x64x192 GEMMs through one composite API, with separately drawn inputs and
outputs and multiple K updates in each GEMM. Its one cached expected payload
contains both FP64 results. Guards have no captures.

Compile all ten resources before running any correctness calls:

```bash
python examples/optimization-library/catlass/conditional_barrier_compile.py \
  --catlass FILTERED_DEPENDENCY --output FRESH_BUILD_DIRECTORY
```

The selected runtime supplies the compiler and CANN environment. The output
directory must not exist. For the adapter, set
`CATLASS_CONDITIONAL_BARRIER_DIRECTORY` to that build directory and
`CATLASS_CUBE_CORES` to the queried physical Cube count. Use
`conditional_barrier_adapter:factory` through the repository's optimization
pair interface. The specification must include `m`, `n`, `k`, `tile_m`,
`tile_n` and an integer seed; `m/n` must equal the declared compiled tile.
An optional `consecutive_distinct_calls=2` is accepted only for guard cases.
Private binary loader images keep each prepared case's mutable C ABI shape
state separate. Preparation performs allocation and layout audits; forwards
only launch the prebuilt native operation and return its output.

Before math, inspect each actual binary's embedded kernel metadata and bind
its engine type to the selected SDK's enum definitions. This collection
qualified kind 2 and `__CCE_KernelMetaData=1` as AIC only; generic profiler
`AI_CORE` labels alone do not establish placement. The retained generated SDK
preprocessor enum proof comes from the final AFTER compilation per tile.
BEFORE and AFTER binaries, exact compiler argv, full dependency hashes and
source identities remain independently retained.

The frozen matrix is `experiments/optimization-wave006/L05.json`. Collection
creates six CPU fixture payloads once, checks each arm once per case, and
captures exactly ten full API forwards for each arm of the four main pairs.
No warmup, autotuning, extra timing forwards or profiled tensor inspection is
part of this protocol. Native errors and bounded watchdog failures stop the
collection rather than continuing to another case.

`conditional_barrier-evidence.json` records the collected results. These are
profiler-active device operation spans with all ten samples and supported raw
PMU rows. They do not establish ordinary latency, physical memory traffic,
roofline behavior, scalar instruction counts or a general recommendation to
elide barriers. Equal, slower and uncertain outcomes are retained.

Collected profiler-active device spans (microseconds, mean ± sample SD):

| Actual tile / K | Conservative BEFORE | Stock AFTER |
| --- | ---: | ---: |
| 64x64 / 256 | 3.084 ± 1.142 | 2.704 ± 0.040 |
| 32x64 / 256 | 2.550 ± 0.102 | 2.538 ± 0.073 |
| 64x80 / 192 | 2.820 ± 0.081 | 2.882 ± 0.059 |
| 32x48 / 192 | 2.436 ± 0.080 | 2.484 ± 0.059 |

There is no consistent diagnostic span improvement. The first BEFORE capture
contains a 6.32µs first sample, retained in the evidence; the heldout 64x80
stock arm is slightly slower. No samples were removed or additional captures
used to obtain a preferred sign. Both guards passed numerical and canary gates.
