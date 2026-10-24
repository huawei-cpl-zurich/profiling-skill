# MHC post bmm2 epilogue fusion

This is a source-backed arch32 forward-inference application of epilogue fusion.
It does not establish a new optimization mechanism or cover backward, recompute,
autograd, arch35, or other dtypes. The immutable six-case matrix is
`experiments/optimization-wave007/E02.json`.

The before API performs GPU transpose of the user's residual coefficients,
bmm1, bmm2, and addition. The after API performs the same transpose and bmm1,
then bmm2 with an addition epilogue: four components become three. No
input-dependent preparation is moved outside either API. The common bmm1
stores FP32 products. bmm2 preserves the vendor's explicit nested FMA order,
including the separate final product, and rounds its result to FP32 before
addition. Both arms use `enable_fp_fusion=False`, `num_stages=1`, and
`multibuffer=False`; explicit FMAs remain enabled.

The source is MindSpeedOps commit
`4efe2b675c2d51517c3fdd14c721020b8d41e241`, under Apache 2.0, with the complete
license in `examples/optimization-library/triton/MHC_POST_LICENSE`. The selected
source bodies have no additional file copyright header. Filtered source receipts:

- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-mindspeed-4efe2b675c2d#mindspeed_ops/arch32/triton/mhc/post_bmm1.py`
- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-mindspeed-4efe2b675c2d#mindspeed_ops/arch32/triton/mhc/post_bmm2.py`
- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-mindspeed-4efe2b675c2d#mindspeed_ops/arch32/triton/mhc/post_add.py`

The API receipt in `mindspeed_ops/api/triton/mhc_post.py` establishes the user
coefficient transpose. The implementation is source-style, with a one-pass GPU
transpose and contiguous portable addition materialization. It is not an
unmodified vendor pipeline. The extension `extract_slice` import is probed before
compilation. Only an unavailable extension selects the common sixteen scalar
coefficient loads; both arms use the same probe result.

Inputs are owned, contiguous, finite FP32 buffers in [-1,1], with positive BS
and power-of-two D up to 4096. The independent FP64 oracle applies the user
transpose and casts once. The numerical limit is fixed at atol=rtol=3e-6.
Exact output bits must also match between the two once-only correctness calls.
The collector checks every declared case before any capture. A bit mismatch
stops qualification without changing cases, tolerances, flags, or arithmetic.
Private device buffers have canaries and are checked for aliases and input
mutation. Unsupported parameters are rejected before fixture-cache access.

The measurement contract requests six CPU oracle payloads generated once,
one correctness forward per arm/case, and ten profiled full-API forwards per
arm for each of four main pairs. The two guards have no captures. There are
no warmups, autotuning, or extra ordinary timing forwards. Compilation is
screened for all components before device math, retaining `.text`, relocations,
full ELF and resources. Profile operation spans are diagnostics; they do not
establish ordinary latency, DRAM traffic, occupancy, or roofline position.

The generic standalone runner supports this adapter from an external cwd by
putting `examples/optimization-library/triton` on `PYTHONPATH` and selecting
`--adapter mhc_post:factory`. Use `--correctness-only` for host-admitted device
correctness without capture. `role=guard` also selects correctness-only mode.

The retained GZ-A3 run completed on physical die 0 (logical 0), Ascend910B4-1,
with the extension `extract_slice` available and used. Every case passed the
fixed FP64 and paired-bit gates before profiling. The maximum observed absolute
FP64-oracle error was 4.76837158203125e-7. Input bits and all buffer canaries
were preserved. Each main arm captured ten forwards; guards captured zero.
Common transpose and bmm1 code matched exactly, and bmm2 code differed in
`.text` for all six pairs. This sampled result does not prove universal
compiler rounding behavior for other inputs.

Mean profiled operation spans, including the common API components:

| Frozen case | Before (µs) | After (µs) | Before / after |
| --- | ---: | ---: | ---: |
| Gallery benefit, BS64 D256 | 517.836 | 379.382 | 1.365 |
| Gallery cost, BS1 D64 | 472.482 | 331.388 | 1.426 |
| Heldout benefit, BS129 D1024 | 516.918 | 339.416 | 1.523 |
| Heldout cost, BS1 D4096 | 486.522 | 330.902 | 1.470 |

Scenario names are prospective labels; both designated cost cases improved
under this instrumentation. All ten samples and raw supported PMU fields,
source/oracle/binary bindings, and archive digest are retained in
`docs/data/optimization-wave007-mhc-post.json`. Raw PMU rows use the recorded
column order without converting field semantics. The verified compact archive
contains 551 artifacts (5,876,066 bytes); full vendor reports remain remote.
