# A3 small Cube controls

`benchmarks/a3_profile_small_matmul.py --manifest` lists six M×N×K shapes
for each implementation: squares 16, 32, 64, 128, and 16×64×32 / 64×16×32.
Both use FP16 inputs and FP32 accumulation, one program/block and one dot.
These are pipeline-composition probes, not a competitive GEMM benchmark.

Native inputs are packed **on the host**, outside the measured kernel:
A is ZZ (row-major fractals, row-major within each 16×16 fractal), B is ZN
(row-major fractals, column-major within each fractal). GM→L1 `DataCopy` and
non-transposing L1→L0 `LoadData` preserve those layouts. One low-level `Mmad`
produces NZ (column-major fractals, row-major within each fractal), exported
unchanged using `Fixpipe<..., CFG_NZ>` and unpacked on the CPU.
The layout equations are executable as `offset` in the Python driver and
independently used by the native host verifier. No high-level Matmul API is used.

Triton takes ordinary contiguous row-major inputs and uses one `tl.dot`.
Its compiler can add layout conversions and different schedules: **native and
Triton times do not isolate Cube arithmetic throughput**. All input values
are nonuniform signed multiples of 1/8, with distinct A/B salts; every output
is checked exactly against an independently computed CPU dot. This avoids
all-ones inputs hiding transpose/fractal-order bugs. Initialization/reference
calculation use the CPU, not extra NPU arithmetic kernels.

First run a numerical pilot, then capture:

```bash
python benchmarks/a3_profile_small_matmul.py --output pilot --correctness-only
python benchmarks/a3_profile_small_matmul.py --output capture --case native/16x16x16
python benchmarks/a3_profile_small_matmul.py --output triton --case triton/16x16x16
```

Use the selected A3 runtime and shared idle-device lease. A numerical failure
stops the batch, leaving command/stdout/stderr and summary evidence. Profiling
reuses `a3_memory_batch.run_case`: discovery first, exact exported selector,
five BasicInfo timing captures and three PipeUtilization captures in one
lease. Retain all samples and dispersion; >5% spread is low confidence, not
permission to remove slow samples. Never sum overlapping pipe durations or
pool Cube and Vector ratios. Tiny one-block launches need not saturate the chip.

Native compilation retains source/binary hashes, exact arguments, full compiler
diagnostics and temporary objects. The installed objdump may not decode the
device ISA; `available=false` is not instruction verification. Triton retains
the actual launched kernel's string-valued `compiled.asm` artifacts without an
extra launch. Source/API mapping and numerical layout proof are distinct from
late-machine-instruction verification, and reconstructed IR is not old-run proof.
