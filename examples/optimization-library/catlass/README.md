This is the native C++ boundary and first tile-balance pair for wave 001. The
before kernel uses L1/L0 tiles 128×256×256 / 128×256×64; after uses
256×128×256 / 256×128×64. Both compute FP16 GEMM with FP32 accumulation and FP16 output. A and C are
row-major; separate binaries support row-major and column-major B. On the observed 20 Cube-core device, the gallery
benefit shape 1024×640×6144 schedules 24 before tiles and 20 after tiles.

The implementation adapts the basic example and headers pinned at
`40aced927fdaaa0f3eaa9c6f5c5d859e06392f55`:
[filtered source](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/00_basic_matmul/basic_matmul.cpp).
The original copyright and license notice remains in the adapted source;
`CANN_LICENSE` retains the license text. Dependency headers are supplied through
the filtered source bundle and checksum manifest, rather than vendored here.

Run `compile_tile.py --catlass FILTERED_BUNDLE --output BUILD_DIRECTORY`
inside the selected named runtime. The compiler and CANN include directories
are derived from that runtime; compilation performs no device initialization or
forward. `compile_pair(..., source=CPP_FILE, variants={name: [compiler_defines]})`
also supports other adapters using the same C ABI. By default the helper builds
`row-before`, `row-after`, `column-before`, and `column-after` variants. Output directories must be
new; builds retain exact argv, diagnostics, source/compiler/binary digests.

`tile_adapter:adapter` uses the shared `triton/gemm_contract.py` CPU oracle and
the pair runner's correctness-once/ten-profiled-forwards protocol. Configure
`CATLASS_NATIVE_DIRECTORY` and the observed `CATLASS_CUBE_CORES`; the case uses
`family`, `m`, `n`, `k`, `seed`, and `b_layout`. The adapter checks the source-bound native B-layout getter against each case
before permitting a column-major input. Column inputs must have dense physical
storage, offset zero and 2D stride `(1, K)`; preparation records shape, stride,
physical storage and NPU ND format2. Default bridge calls still require
contiguous tensors. Native launches use existing Torch tensors and the current
Torch-NPU stream through a dispatcher-visible operator.

The retained first A3 run passed both correctness checks and captured exactly
ten forwards per arm. Its original parser rejected native CANN flow accounting;
strict replay subsequently qualified the existing trace through genuine
HostToDevice flow IDs and Node launch thread identities, with no extra forwards. No kernel identity is inferred from task
ordering or device execution time.

Native source-bound [FullLoadA residency pair](L1_A_RESIDENCY.md) retains the selected CATLASS header closure, A3 correctness results, and complete prepared-operation profiling evidence.
