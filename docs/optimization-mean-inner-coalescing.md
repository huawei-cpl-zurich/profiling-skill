# Unqualified middle-axis mean inner-lane coalescing study

The frozen first paired numerical case FAILED bit equivalence: 18 of 64 outputs
differed. Both arms passed the fixed FP64 tolerance; that does not satisfy the
source-tree preservation claim. No profiles were collected. Default numerical
Adapter.make_case/factory/prepare and CLI paths now reject before CPU oracle
generation, backend access or input transfer. Only explicit input-only compile
inspection is available; its run() rejects. This is a negative study, with no
qualified numerical or performance example. Later cases were not attempted.


This component compares one prepared FP32 middle-axis reduction over logical
`[M,N,K]` inputs. The original pinned `mean_kernel` executes one program per
output. The candidate executes one `[1024,8]` rectangle per eight adjacent K
lanes, reduces only axis0, and retains serial 1024-element chunk accumulation
and final division. Common flags are stage1, multibufferFalse, fusionFalse.
The imported `mean_inner_source.py` and derived `mean_inner_kernels.py`
carry Apache-2.0 SPDX identifiers; their original Huawei/vLLM copyright
notices remain. The [complete Apache 2.0 license](../examples/optimization-library/triton/mean_inner_LICENSE-APACHE)
includes all nine sections and its appendix. The separately retained
[upstream repository MIT notice](../examples/optimization-library/triton/mean_inner_LICENSE)
does not relicense those Apache-2.0 files. The
[third-party notice excerpt](../examples/optimization-library/triton/mean_inner_THIRD_PARTY_NOTICE)
is explicitly labeled as the filtered upstream excerpt, which ends mid-section;
the complete Apache license is supplied separately. This is an operator
application, with no new-mechanism claim.

Candidate dispatch requires K>1 and actual input K stride1. K1 and logical MNK
views of MKN backing use the original source kernel in both arms. Guard views
retain actual strides; preparation does not materialize a contiguous copy.
Each arm owns immutable inputs and a private contiguous FP32 MK output with
canaries. Preparation and output allocation are outside both prepared APIs;
outputs are reused. One kernel is charged in each API. The candidate's reduced
program count and larger input tile are part of the tradeoff, so coalescing
alone is not a measured physical-traffic or occupancy claim.

The frozen matrix and inventory are in `experiments/optimization-wave009/R07*`.
Compilation/resource/ELF screens for all twelve cells precede all scientific
CPU expected generation. Six CPU FP64 means are generated once and cached by
provider/spec identity. Correctness executes once per arm/case, checking fixed
atol2e-5/rtol2e-5, immutable input bits, output shape/dtype/strides/private
storage, canaries, duplicate-row consistency, and paired FP32 bits including
signed zeros. The sole correctness readback retains mismatch coordinates,
actual/expected values, nonfinite counts, and paired bit diagnostics. A failed
bit gate is retained without relaxing tolerances or replaying the case.

All twelve numerical gates must pass before four main pairs receive exactly
ten prepared complete-API profiling forwards per arm. Guards receive no
profiles. There are no warmups, autotuning, unprofiled timing runs, per-profile
input checks, or extra correctness launches. Profiler operation spans and
active-event samples are diagnostic; they do not establish ordinary latency,
physical memory traffic, universal deterministic batch partitioning, or a
public multidimensional mean implementation.

`mean_inner.Adapter` exposes compile inspection for this unqualified study. `compile_probe`
returns an input-only item whose `run()` rejects before launch. Compilation is
transactional and cached only after a real executable binary is present.
The standalone CLI requires --inspect-compile and accepts strict JSON case files,
including normal trailing newlines; default numerical execution rejects before
case/cache IO. It does not replace the frozen all-case controller.

The original host screen stopped before math because it compared Python
argument names as ABI. All four main executable texts differed (1692 to1940
bytes); positional runtime types, actual argument order, ELF ABI, loader
metadata, allocated data and semantic relocations matched. Guard full-ELF
identity was expected. The host-only continuation normalized ordered runtime
types and directly loaded exactly the retained binaries, with all hashes
verified, without JIT warmup or recompilation. The first continuation had a
host binder initialization-order error before math; its receipts remain. The
corrected continuation generated exactly one scientific CPU fixture and ran
each first-case arm once, then stopped at the bit gate. No source, flags,
tile, cases, tolerance or reduction order was retuned. Original attempted
adapter and controller hashes are distinct from publication host guards.
