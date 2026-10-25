R02 studies a prepared FP32 pack-then-unpack component from Triton-Ascend-Kernels
7acb9c40a6ca827e885d902b7eaee5d45457b230. The before arm copies `_pack_seq_kernel`
and the last active `_unpack_seq_triton_kernel` (the earlier duplicate is shadowed).
The MIT license and complete third-party notice accompany the kernels.

Both arms use BT64, BD64, the source three-dimensional grid, four warps, and
common num_stages=1, multibuffer=False, enable_fp_fusion=False. These are controlled
common settings rather than the vendor public stage-2 wrapper configuration.
Both arms specify other=0 for inactive loads; no inactive values reach stores.
The two-kernel before API serially sums preceding lengths in every program.
The three-kernel after API computes B+1 exclusive int32 prefix entries on the GPU
on every call, then uses them in otherwise identical pack and unpack kernels.
Both packed output and roundtrip output are returned; padding remains a separate
source store before the valid-value store.

The admitted domain is contiguous CPU FP32 X[N,D], int32 lengths[B], B<=256,
Lmax<=128, D<=256, lengths in [0,Lmax], actual max=Lmax, 0<N=sum(lengths)<=32768.
X is finite q/64 in [-2,2]. CPU sum/max/validation, device transfers, allocations,
and compiler qualification occur outside both prepared APIs. Outputs and prefix
workspace are reused. No fresh vendor host API or allocator latency is measured.
Input-only probes cannot run; compilation of every component commits atomically.

Address audit: all active X/roundtrip offsets are <=N*D-1; active packed offsets
are <=B*Lmax*D-1. Every program forms padded offsets even when its mask is false.
With T=ceil(Lmax/64)*64 and F=ceil(D/64)*64, their upper bounds are
(N+T-1)*D+F-1 for X/roundtrip and ((B-1)*Lmax+T-1)*D+F-1 for packed.
Both are below 2^24 and signed int32 range throughout the admitted domain.
Nonnegative lengths make all prefix sums <=32768; prefix stores cover exactly
[0,B] in B+1 allocated elements. The guarded buffers include 32 sentinels on each
side; no masked-pointer assertion substitutes for the active-address proof.

The independent CPU provider uses a serial integer prefix and explicit row
copies. Correctness requires both output shapes/dtypes, every FP32 bit including
padding -9.5 and signed zero, unchanged inputs, nonaliasing, and intact canaries.
The after prefix is compared once to the same cached CPU expected prefix during
its numerical validation, without an extra device execution.

The unchanged source passed native BZ-A3-2 qualification on physical0/NPU0/chip0,
observed 9382 Ascend910V1 with 24 Cube and 48 vector cores, imported Triton3.2.0,
triton-ascend3.2.2, Torch2.10.0+cpu/Torch-NPU2.10.0, and CANN9.1. All twelve
input-only compile cells (thirty component binaries, including text/allocated/
relocation sections and resource metadata) completed before the six scientific
CPU expected payloads were generated once. All twelve numerical calls passed
before eighty profiled complete API forwards in eight captures; guards had no
captures. Both outputs, padding and signed-zero bits, internal prefix, input
immutability, canaries, and paired bits passed. Full six-case coverage audit passed.

The added scan did not improve these instrumented complete API measurements:

| B,Lmax,D | Before/after operation-span mean (us) | Before/after API event mean (ms) |
| --- | --- | --- |
| 4,32,64 | 84.312 / 191.962 | 0.382244 / 0.513614 |
| 32,64,128 | 89.190 / 207.042 | 0.412184 / 0.531150 |
| 128,64,256 | 104.744 / 232.865 | 0.414616 / 0.615276 |
| 256,128,256 | 126.689 / 240.443 | 0.425054 / 0.526858 |

Both series include all ten samples and first-call variation. Operation spans
include gaps between device tasks; API events were measured with profiling active.
These are diagnostic prepared-component measurements, with no ordinary latency,
physical traffic, bandwidth, occupancy, roofline, or full-model claim. No cases,
tiles, flags, or tolerances were retuned after the negative result.

An earlier staging attempt failed before imports/compilation/oracles/NPU work
because its relative upload did not place the archive in the required registry
working directory. Its empty source directory, failure logs, and cleanup proof
were retained. A unique stage with an absolute verified archive destination
performed the first scientific acquisition on the identical source. Successful
cases were not repeated. After terminal completion, the device lease and both
export locks were confirmed free, with no live workload process and clean device
health/process checks.

The verified compact receipt binds 563 files to archive SHA256
7738f00c9467d0df39acac5dc773508f82d3b6dc1ee82b46a199a62ab7645571.
It retains original source, compiler IR/cache/binaries/resources, fixtures,
all samples and raw supported PMU columns; full vendor trees remain remote.
Four functional CPU tests also passed on the native A3 target using nonfrozen
cases and seeds above190000. The standalone public CLI is exploratory and does
not implement the frozen all-case scientific collector.

The single frozen-head review wave identified a host allocation bug under changed
Torch global defaults. Publication now explicitly allocates both outputs as FP32.
Nonfrozen CPU/mock regressions under float64 and float16 defaults pass and restore
the caller default afterward. All five functional tests passed a native A3
CPU-only followup gate, with no real compilation, scientific oracle generation,
NPU numerical calls or profiles. R02-publication.json separates original measured
adapter hashes from the updated host hash; provider, JIT, fixtures, binaries and
all measured results remain unchanged. GitHub Codex reviewed the original
measured head with no findings; no recursive review wave was requested.
