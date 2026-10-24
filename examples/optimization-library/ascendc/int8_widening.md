The N01 pair independently extracts the exact signed INT8 widening mechanism from
`cann/ops-sparse` commit `bfc7c19d83ce246b03c65dc916ed4753e5467cf3`,
`sparse/spmv/arch22/kernels/spmv_kernel.h:267–302`. Its real SpMV callsite
at lines 727–734 enables INT8→half→float; upstream SpMV tests are not an
exhaustive isolated conversion test. The adapted C++ retains its Huawei and
CANN Open Software License Agreement 2.0 notice; its full license text is
[retained here](../catlass/CANN_LICENSE). This standalone format does
not claim to reproduce SpMV or introduce a new unique mechanism.

The public interface accepts immutable contiguous signed INT8 `[N]`, returning
nonalias FP32 `[N]` without scale, dequantization or rounding relaxation.
`empty_result` returns a distinct FP32 empty object without native launch.
Both arms stage each tile from GM on every call, reserve 4096 bytes of INT8,
8192 bytes of half and 16384 bytes of FP32 UB, and use the same 4096-element
grid, copy tails, and cross-pipe ownership barriers. The source scalar loop
and two Cast stages use the same compiler flags. The half reservation on the
scalar arm is experimental isolation, not a production memory requirement.

The frozen six-payload matrix is `experiments/optimization-wave006/N01.json`.
The CPU oracle converts signed Python integers directly to FP32, and validation
compares FP32 bits, input immutability and both input/output GM canaries.
The GB guard combines 4097-element tail conversion with a distinct empty result;
its component count is one widening launch and zero empty launches per arm.
The four main pairs each use exactly ten profiled complete API calls per arm,
with no warmup or separate timing pass. Guards have no captures. Operation
spans and pipe fields are profiler diagnostics, not ordinary latency, physical
traffic, instruction counts or roofline measurements. Actual SDK kind3/vector
entry and absent Cube entry must be audited before numerical execution.
