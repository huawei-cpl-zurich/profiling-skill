# SROTM unit-stride DMA

This prepared Ascend C experiment compares the pinned ops-blas scalar CopyIn/CopyOut with a derivative that uses DataCopyPad for unit stride. Both preserve the source vector formulas, coefficient order, fixed eight launched blocks, active-core selection, and full 192 KiB UB allocation. Stride two follows the original scalar path. This applies a known DMA mechanism; it introduces no new algorithm.

Every measured API first runs a common native reset kernel, copying both immutable original vectors to private working vectors, then runs SROTM on the current Torch ACL stream. Output storage and tiling are allocated once and reused. These are instrumented prepared device operation spans, excluding allocation, host BLAS dispatch, and autograd latency.


| n | Prospective label | Before mean µs | After mean µs | Before / after |
|---:|---|---:|---:|---:|
| 127 | cost risk | 40.675334 | 42.025559 | 0.9679× |
| 4096 | cost risk | 149.580861 | 39.043258 | 3.8312× |
| 65536 | benefit | 466.244673 | 41.063128 | 11.3543× |
| 1048579 | benefit | 7039.643303 | 52.702659 | 133.5728× |

The four main cases and two guards were frozen before implementation. Inputs are bounded exact dyadic FP32 values; an independent FP64 provider supplies exact cast-to-FP32 expected outputs. Nonzero output bits must match, zero values compare numerically, and zero signs are reported separately. All twelve numerical gates pass, including full-storage stride gaps, canaries, input immutability, and nonaliasing. Paired output bits also match in all six cases.

The qualified build uses the actual installed FindASC recipe matching the source CMake language and architecture: ASC, Release `-O3`, `--npu-arch=dav-2201`, and documented explicit auto-sync with diagnostic logging. Both Cube and Vec device objects, host ABI exports, full section hashes, registration metadata, actual commands, SDK module hashes, and inserted synchronization logs are retained. The Cube object has an empty text section; all 160 profiler tasks report `AI_VECTOR_CORE`. Compiler-managed scratch remains unknown.

Earlier attempts remain in the evidence. A legacy CCE/O2 extraction produced one finite incorrect BEFORE result and no profiles. Its build-mode parity with the source was unqualified. The corrected ASC build is a source build-contract repair, with unchanged native C++, source kernel, provider, cases, formulas, and tolerances. Mode, optimization level, and SDK defaults changed together; this does not isolate auto-sync as the cause of the old failure. Metadata-selector, diagnostic-option placement, and missing compile_commands reader failures each occurred before scientific math and are retained separately.

The successful collection reused the original six hash-verified cached CPU payloads and generated zero additional expected payloads. All twelve input-only ABI/code/resource probes preceded cache loading; twelve new numerical gates preceded eighty APIs across eight captures. Guards received no profiles. Each capture contains all ten samples, with no warmup, autotuning, extra timing, outlier removal, or scientific replay. Both host export locks cover capture and synchronous analysis, and 600-second watchdogs bound startup, execution, validation, and profiling.

The adapter rejects unsupported dtype, layout, aliasing, stride, flag, nonfinite, and value domains before compilation or device transfers. Prepared handles bind their device, restore the caller's device, and expose idempotent cleanup. The standalone CLI is exploratory; the controlled frozen collector supplies the complete qualification protocol. Ten functional NONFROZEN host/mock tests also pass on the selected A3 runtime.

See [the portable evidence](data/optimization-w010-native-srotm-dma.json) for every sample, raw PMU/task data, fixture/compiler/source bindings, failure history, archive manifest, and release proof. Task durations do not establish physical traffic, instruction counts, occupancy, or a roofline model.

DMA transfers charge exactly `dataCount * sizeof(float)` valid bytes per vector. CopyIn padding covers only the rounded vector-compute tail; CopyOut writes only valid bytes. Both original vectors enter separate local buffers before either global vector is overwritten. The stride-two guard retains scalar transfer behavior. The host tiling ABI is 80 bytes with explicit coefficient positions; the independent provider uses `[flag,h11,h21,h12,h22]`.
