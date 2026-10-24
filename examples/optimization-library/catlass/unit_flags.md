# Cube-to-Fixpipe completion flags (L02)

This native 2201 study instantiates the pinned CATLASS `SmallMatmul` and
`BlockMmad` body twice. The only policy difference is
`MmadAtlasA2Small<2, false, false>` versus
`MmadAtlasA2Small<2, true, false>`. Both retain the same small-kernel control,
row-major task order, producer-buffer events, buffering, movement, FP32
accumulation, FP16 output, and terminal barrier. The upstream true branch
uses intermediate Mmad `0b10`, final Mmad `0b11`, and Fixpipe `0b11`.

Eligibility requires positive dimensions aligned to 16, K at most 256,
dense row-major FP16 ND operands/output, and an output grid no larger than
the actual queried Cube-core count. The adapter rejects other inputs;
`SmallMatmul::CanImplement` alone does not enforce these restrictions.
A/B/C strides are respectively `(K,1)`, `(N,1)`, `(N,1)` with zero offsets.
`NativeArm` audits physical ND format and addressed storage before launch.
Both arms receive identical represented inputs in [-0.25, 0.25].
The CPU oracle computes those FP16 values in FP64 once per cached fixture;
comparison uses fixed `atol=2e-3, rtol=2e-3`, requires finite output, checks
immutable inputs, and verifies complete overwrite of a NaN output sentinel.

Compile the entire resource pair before forwarding, using the pinned,
manifested dependency closure and the selected runtime's compiler:

```sh
python examples/optimization-library/catlass/unit_flags_compile.py \
  --catlass "$FILTERED_CATLASS" --output "$BUILD_DIRECTORY"
```

Set `CATLASS_UNIT_FLAGS_DIRECTORY` to that build directory and
`CATLASS_CUBE_CORES` from `torch.npu.get_device_properties(0).cube_core_num`,
then use `unit_flags_adapter:factory` with `scripts/optimization_pair.py`.
The frozen matrix is `experiments/optimization-wave006/L02.json`;
its uppercase dimensions map to the adapter's lowercase `m,n,k` fields.
Each main arm receives one correctness call and exactly ten complete API
profile calls. Guards receive correctness only. The second guard is one
composite API executing two independently drawn GEMMs with separate inputs
and output buffers; its cached expected value is a tuple of two CPU oracles.
It therefore counts as two component GEMMs per correctness API call.

Profiler-active operation spans are diagnostic. They do not establish
ordinary latency, physical traffic, whole-device utilization, or a roofline.
No benefit sign is assumed and no post-result tuning is part of this study.
The exact source pin and prospective cases are retained in the frozen matrix;
target-specific collection receipts and all failed versions are retained in
the central task record and compact evidence, outside this source tree.

The qualified GZ-A3 run used a healthy idle physical device 1 (logical 0),
CANN 9.1, and 20 queried Cube cores. All six cases passed both arms;
maximum absolute error was below 4.89e-4. The corrected run made 12
correctness APIs (14 component GEMMs), eight captures and 80 profile APIs;
it reused the six cached oracle payloads with zero new oracle calls.
All 80 retained task rows report AI_CORE, Block Num 20, Mix Block Num 0.

| Frozen case | Before span mean ± sample SD (µs) | After span mean ± sample SD (µs) |
| --- | ---: | ---: |
| W006-L02-01 | 3.512 ± 1.782 | 2.712 ± 0.236 |
| W006-L02-02 | 5.524 ± 0.126 | 5.296 ± 0.133 |
| W006-L02-03 | 2.494 ± 0.102 | 2.364 ± 0.059 |
| W006-L02-04 | 5.402 ± 0.160 | 5.300 ± 0.180 |

Each cell includes all ten retained samples. The first before sample in
case 01 is 8.58 µs and remains in the mean; no samples were discarded.
These profiler-active observations show modest reductions in this capture,
with a large first-case outlier, and do not establish ordinary latency gains.
The [portable evidence](unit_flags-evidence.json) includes every sample,
per-task PMU columns, fixture/source/compiler/binary hashes, physical layout
audits and exact retained job handles. Full raw trees and compact archives
remain in the central evidence collection.

An earlier common wrapper omitted the upstream `__attribute__((aic))` and
failed its first before correctness call with an AIVector internal-buffer
error, before any capture. That failed version and its one entered API are
retained. Both corrected arms use the exact upstream AIC wrapper. Generated
binary metadata and version-matched compiler enums, followed by all 80
actual task rows, qualify the corrected AIC route. The failure is not
classified as a hardware fault.
