# Resident multi-problem GEMM (F09)

This pair isolates deterministic cyclic tile ownership across independent FP16
GEMMs with fixed N/K and varying M. It does not split K or merge partial sums.
The before API builds consumed prefix descriptors and launches a GEMM for each
nonempty group. The after API builds the same descriptors and launches twenty
resident programs; each traverses group descriptors and computes its assigned
whole tiles at a stride of twenty across the union of tiles. Group index selects
B even when M is zero. Concatenated row prefixes make output regions disjoint.

Both arms use the same 16x64x64 tile, ascending K order, FP16 operands bounded
by .125 and FP32 accumulation/output. Raw A/B and group row counts are public
inputs. Preparation transfers them and preallocates output/descriptor storage;
every API forward constructs descriptors on device and computes all outputs.
Descriptor construction is included in expected kernels and operation span.
CPU correctness uses independent FP64 matrix products rounded to FP32, with
absolute and relative tolerance .003. No output reading occurs during capture.

The four cases and seeds are frozen in `resident_gemm_cases.json`: many small
groups, one larger group, irregular groups with interleaved zeros, and a larger
single heldout group. Two correctness-only guards reject decreasing prefixes
and exercise interleaved empty groups. These guards perform zero captures.
All kernel resources must compile before the first correctness forward; the
collection then checks each arm once and captures ten complete API forwards
per arm with no warmup or autotuning. Kernel task counts differ from API calls:
F09-01 before has one descriptor and 32 GEMM launches per forward.

The primary comparison uses the complete device operation span from first
owned task start through last task end, including all interlaunch gaps. Active
task union and idle gaps are separate. Profiler event times are instrumented
diagnostics. Exported per-task PMU values do not establish physical traffic,
whole-device utilization or a roofline claim.

Source provenance: independent Triton implementation of the plain-half grouped-M
scheduling mechanism in
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-transformer#gmm/grouped_matmul/op_kernel/grouped_matmul_kernel.h:L300-L332`,
with dispatch at `#gmm/grouped_matmul/op_kernel/grouped_matmul.cpp:L603-L627`
and A2/A3 registration at `#gmm/grouped_matmul/op_host/grouped_matmul_def.cpp:L168-L169`.
The Catlass grouped descriptor/resource loop is corroborating context:
`ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/grouped_matmul.hpp:L150-L209`.
These are filtered broker citations, not claims that this pair executes those
native libraries.

## A3 collection

On the selected healthy idle Ascend910B4-1, all twelve arm/case compile
qualifications passed before scientific execution. Four paired cases passed
correctness and completed exactly ten profiled API forwards per arm. Both
guards passed with zero captures. The 80 profiled API calls produced 700 device
tasks: descriptor construction plus per-group or resident GEMM tasks.

| Case | Before span mean (us) | After span mean (us) | Before / after |
| --- | ---: | ---: | ---: |
| F09-01, 32 tiny groups | 3998.488 | 178.924 | 22.35 |
| F09-02, single group | 177.900 | 182.754 | 0.97 |
| F09-03, zero/tail groups | 3405.784 | 187.132 | 18.20 |
| F09-04, larger single group | 200.544 | 205.982 | 0.97 |

These are instrumented device spans, including descriptor-to-GEMM and
inter-GEMM gaps. They support launch aggregation in this API/toolchain, rather
than an uninstrumented latency claim. F09-01 before has a 312.16 us first span
and approximately 4.4 ms later spans; all samples and their large variance are
retained without trimming. The single-group costs are small relative to the
observed variation, so they do not establish a reliable slowdown. Active task
union and idle gaps, all individual samples, per-task PMU fields and compilation
hashes are in [the evidence record](data/optimization-w003-resident-gemm.json).

Source v1 used uninitialized output storage, checked every output against the
oracle, and produced all performance captures. Final source v2 changes only
preparation to fill the output with NaN. A separate A3 correctness-only
qualification checked the same six cached fixtures exactly once per arm and
verified binary identities match v1; it produced zero additional captures or
timing calls. This establishes sentinel overwrite coverage separately from
v1 profiling. Neither the kernel bodies nor the API run body changed.
