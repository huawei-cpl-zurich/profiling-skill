# Sinkhorn runtime iteration binding

E06 compares the qualified MindSpeedOps source-style eight-lane Sinkhorn forward
at parent `776f3e05ab940ea58e8263c59631cf0f5e101a06`. The before kernel takes
`SINKHORN_ITERS: tl.constexpr` and uses `tl.static_range`. The after kernel uses
a runtime integer, `do_not_specialize`, and ordinary `range`. All initialization,
FP32 row/column arithmetic, entrywise epsilon, initial column normalization,
GROUP32, flags, separate pre/post launch, output strides `(32,8,1)`, and positive
zero padding remain identical. The licensed parent and its CPU reference are
unchanged. This is a source-backed operator application, with no new unique
mechanism claim.

The public `sinkhorn_runtime_loop:Adapter` and standalone
`examples/optimization-library/triton/sinkhorn_runtime_loop_cli.py` are generic
exploratory one-case interfaces. The CLI labels report provenance
`qualification="exploratory/unqualified"`, `pre_oracle_pair_screen=false`, and
`paired_codegen_screen_before_device_math=false`.
Each prepared arm caches basic compile-only qualification before its first
public launch, retains selected artifact identities, and performs no further
compile warmups on subsequent calls. This per-arm compatibility check does not
provide the controlled study's pre-oracle all-pair resource/code-noop screen.
The CLI accepts `--case`, `--output`, `--cache`, `--provenance`, and optionally
`--validate-only`; host admission works from an external working directory.
Owned contiguous finite CPU FP32 mixes `[BS,24]`, scale `[3]`, and base `[24]`
retain the parent value bounds. BS must be positive, iterations integer 1..20,
and epsilon finite 1e-6..1e-3. Flat/nested specs accept role and expected metadata.
Unsupported counts fail before fixture cache creation.

The exact frozen cases are in `experiments/optimization-wave007/E06.json`.
Four main pairs have ten profiled complete API calls per arm after one correctness
call. Guards have no captures. Compound guard GA uses one input and count 4 then
count 20 with separate output buffers; each outer call returns two tuples of
three tensors. Its CPU provider computes pre/post once and takes two snapshots
from one FP64 normalization trajectory. Across the study there are six cached
payloads, seven component expectations, twelve outer correctness calls, fourteen
complete computation components, and eighty profiled full API calls. GB verifies
count 1 performs zero loop repetitions and preserves epsilon 1e-3 on equal logits.

In the retained controlled collection, actual code/resource qualification
preceded CPU oracle generation and scientific forwards. The pinned controller is
`collect-v1.py`, SHA256
`710ab4645c27e016614274878897785d98045185b609eeaed79b45fe5ab057cb`,
retained in the verified archive and its `artifact_manifest` entry in the compact
evidence. That controller defines the study-wide ordering; the generic CLI's
ordinary once-correctness/ten-profile runner does not reproduce that qualification.
A broader qualified public CLI is outside this PR's scope. A compiler rejection of runtime 3D state ends the study as unsupported;
an effective code noop ends it without invented performance evidence. Required
compile calls retain wall times and artifact identities; warm cache lookups are
not presented as cold compilation latency. GA checks actual runtime artifact
hashes at both counts without an additional empirical forward.

All three outputs retain the fixed `atol=rtol=3e-6` FP64 gate. Validation checks
layout, canaries, immutable inputs, and positive-zero padding. Crosspath output
bits are recorded read-only and do not add a bitwise pass requirement. Prepared
output allocation and input movement occur once; output buffers are reused.
Profiled operation spans include both kernels and their gap, and exclude that
allocation. They do not establish ordinary application latency, physical memory
traffic, occupancy, or a roofline.

## A3 result

The retained job `remote:gz-a3:job:b560d17d-31f1-4936-b338-0b213bad9ddb`
completed with exit 0 on healthy idle physical 6, logical 0, using gz-a3
`py311-torch`. All six cases passed. Six payloads and seven expectations were
created once after compilation; durable entry counters verify all twelve outer
correctness calls, fourteen computation components, and eighty profile calls.
All guards had zero captures. The four E06 functional tests passed on A3, and
the ten parent/child functional tests passed locally.

Runtime count 4 and count 20 in GA both used compiled COMB hash
`b1ac4d9121b18daf064c8730bb1c11f35856373673db43756bf6f40dff0e5c5a`.
All six cases had different comb `.text` bytes across arms; pre/post binaries
were identical. Static20 comb code was 6,668 bytes versus runtime 2,248;
static 1 was 1,804 versus 2,248; static 4 was 2,640 versus 2,248.
GB epsilon specialization gave 1,792 versus 2,236 bytes. No instruction counts
are inferred from these byte sizes.

The required compile phase made 28 compile-only API calls covering 12 unique
artifact hashes. A fresh stage cache was initially absent. For the first
aligned-input comb variant, static20 compile wall time was 9.503 s and runtime
was 6.823 s; required subsequent runtime lookups reused the same identity.
These are single observed compile API wall times, not replicated cold-compiler
statistics. The artifact API did not expose a cache-hit flag; null values remain
explicit, with previously seen hashes reported separately. Imported Triton was
3.2.0 and the triton-ascend distribution 3.2.2; installed triton distribution
metadata reported 3.5.0, which is retained separately from the imported version.
A bounded read-only dependency query confirmed selected CANN 9.1.0,
bishengir-compile 1.2.0 (LLVM 19.1.7), and bisheng clang 15.0.5; exact
executable and SDK version-file hashes are retained in the evidence.

| Case | Before mean span (µs) | After mean span (µs) | Before / after |
| --- | ---: | ---: | ---: |
| gallery BS1024, iters20 | 193.362 | 185.824 | 1.041 |
| gallery BS32, iters1 | 172.150 | 185.218 | 0.929 |
| heldout BS257, iters20 | 183.418 | 189.402 | 0.968 |
| heldout BS33, iters4 | 177.882 | 174.212 | 1.021 |

The span results are mixed and noisy, with no uniform benefit. Every sample,
including the 90.08 µs after sample in the final pair, is retained. No case,
tolerance, source, flags, or repetition counts were retuned. Read-only bit
comparisons happened to match on all six cases; numerical correctness did not
require crosspath bit equality.

[Compact evidence](data/optimization-wave007-E06.json) retains every sample,
raw supported PMU field, source/fixture/binary binding, compiler metadata,
compile wall time, and code section identity. The 369-file archive is 1,374,957
bytes, SHA256 `bae105de0ed911614abefd79e481c9442751647c6a66ab5728ae07e1f6fb60dd`.
All manifest hashes verified after transfer. Full vendor reports and the fresh
compiler cache remain retained remotely. Evidence extraction performed zero
new computations or captures.


## Public interface scope correction

The public one-case CLI now labels exploratory qualification explicitly, and
its prepared Adapter compiles each arm once before its first public launch.
Selected compilation artifacts remain available in report metadata; later
forwards reuse the prepared state without further warmups. This basic per-arm
check does not supply the pinned collector's full study qualification.

This host-only correction preserves the original JIT kernels, CPU provider,
six cached scientific payloads, binary identities, and profiled results.
The evidence records original scientific and publication host hashes separately.
Functional tests verify compile/launch ordering, cached compilation, admission
before the generic runner, and exploratory report provenance. All eleven
parent/child tests passed locally and on the CPU-only A3 host gate
`remote:gz-a3:job:23362740-ba33-43b4-8fb1-33941a16b8cf` (82.596 seconds,
exit 0). No scientific compilation, device forward, oracle payload, or capture
was repeated.
