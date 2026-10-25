# Share rotary trigonometry across Q and K heads

This prepared FP32 forward component compares two educational head-parallel
kernels (Q then K) with the pinned source's single sequence program, which
computes trigonometry once and loops over Q heads then K heads. The difference
includes launch consolidation, trig reuse and redistribution of head programs
into loops. Small sequence/head workloads may lose parallelism. This is not a
vendor baseline or pure trigonometric instruction comparison.

The source is _fused_rotary_emb_kernel in index/fused_rotary_emb.py at
7acb9c40a6ca827e885d902b7eaee5d45457b230. The copied kernel retains Huawei and
DeepLink attribution and [license notices](../examples/optimization-library/triton/rotary_trig_NOTICE).
Both arms use BLOCK32, half-D feature blocks, four warps and identical compiler
flags and pointwise operation order. Masked loads specify other=0 in both.
No alternate trig implementation, tile tuning or backward kernel is included.

Inputs are owned contiguous Q/K in BSHD layout, FP32 finite dyadics in [-1,1],
int64 positions in [0,2048], and fixed FP32 inverse frequency (j+1)/1024.
D is 64 or 128 and scaling is 2 or 4. Position stride two uses an actual [B,2S]
backing and [:,::2] CPU/GPU view. The adapter copies the entire backing and
preserves that view, including in input-only probes. Private Q/K outputs retain
input strides and are reused in both arms; allocation is excluded.

The [prospectively frozen v2 matrix](../experiments/optimization-wave009/R03.json)
contains four main pairs and two guards. An independent CPU FP64 provider
checks every Q and K output with fixed atol=3e-5, rtol=2e-5. Paired FP32 bits
and maximum pair errors are secondary diagnostics. Zero-position arithmetic
may alter zero signs; input-bit identity is not a correctness requirement.
Output tuple arity, shape, dtype, finiteness, private halos, nonaliasing and
full physical input backing immutability are checked after correctness calls.

The public standalone rotary_trig_cli.py is exploratory and admits the case
before accessing the common cache. The controlled collector additionally
compiles all cases before making six scientific CPU payloads, gates all twelve
numeric checks before profiling, and holds both host export locks throughout
capture and synchronous analysis. It profiles ten complete Q/K APIs per arm
for each main pair; guards have no captures, with no warmups or extra timings.

An initial source-porting error added an unsupported other argument to stores.
The input-only compile screen rejected it before CPU payload generation or
kernel execution. That failed source and archive remain retained. The corrected
version removes the argument from stores, preserving zero-filled masked loads,
source arithmetic, cases, tolerances and flags. It repeats compile qualification;
there were no earlier scientific checks to replay.

The corrected native A3 collection passed all twelve primary gates. All six
paired output bit comparisons also matched in this run; that observation does
not strengthen the primary numerical contract. Five NONFROZEN functional tests
passed locally and on A3. The frozen coverage/layout audit passed.

Mean profiled complete Q/K operation spans, microseconds (ten samples per arm):

| Case | Two head-parallel kernels | Shared sequence program | Before/after |
|---|---:|---:|---:|
| 01 | 159.563 | 72.315 | 2.207 |
| 02 | 97.450 | 11.640 | 8.372 |
| 03 | 111.088 | 62.183 | 1.786 |
| 04 | 96.244 | 22.720 | 4.236 |

The [retained evidence](data/optimization-w009-rotary-trig-sharing.json) includes
all samples, task durations, gaps, supported raw PMUs, binary/resource hashes,
six fixture bindings, and verified manifests for both the failed and successful
versions. The successful API count is eighty, comprising 120 kernel tasks:
two per baseline API and one per source API. These are instrumented prepared
component spans, including launch gaps; they do not establish ordinary latency,
physical traffic, occupancy or a pure trigonometric instruction reduction.

The standalone exploratory entry point accepts a flat or nested case:

    python examples/optimization-library/triton/rotary_trig_cli.py \
      --case case.json --cache oracle-cache --output result \
      --provenance provenance.json
