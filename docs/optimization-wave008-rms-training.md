# RMSNorm saved statistics versus recomputation

This independent study compares the full prepared RMSNorm forward/backward
computation using cached row statistics with recomputing those statistics in
the backward kernel. The source is `triton-ascend-ops` revision
`7acb9c40a6ca827e885d902b7eaee5d45457b230`, with MIT and third-party notices
preserved. The source's supported `row_mode=True` is common to both arms and
selects its single-row forward kernel on grid `rows`. This is an explicit
component configuration; the default public wrapper chooses blocked forward
for some large-row shapes.

Every prepared API launches four kernels: source-style forward, explicit common
zero-dX, blocked BR1 backward on the driver's actual vector-core count, and the
same final partial-dW reduction. Both arms use Gemma casting mode 1, offset 0/1,
epsilon 1e-6, and the original FP32 arithmetic order. There is no row batching
optimization in this study. Portable Triton zero/reduction helpers adapt the
source wrapper's torch operations on both sides. The outputs are `(Y, DX, DW)`
for the caller's supplied arbitrary finite quantized `DY`.

For `D<=1024`, the candidate omits the forward RSTD store and computes
`rsqrt(sum(X*X)/D + eps)` from the already-loaded X in backward. For `D>1024`,
both forward and backward retain exactly the cached-stat configuration. A
private statistics buffer is allocated in preparation on both arms; the
recompute path neither writes nor reads it. Allocation latency is excluded,
and no extra statistics instrumentation/store or GPU forward is introduced.
Internal RSTD values are neither observed nor claimed bitwise equal.

An independent CPU provider computes full RMSNorm Y and derivatives in FP64
once per case from quantized 1/128 FP32 inputs in [-1,1]. Fixed gates are Y
`atol=2e-5, rtol=2e-5`, DX `atol=1e-4, rtol=1e-5`, and DW
`atol=5e-4, rtol=3e-5`. Paired Y output bits must match. DX/DW are checked
numerically because recomputation may lower its reduction differently.
Canaries, shape/dtype/layout, storage separation and input immutability are
checked outside captures. The adapter clears paired-output state when reused
for a new pair, including reversed arm order.

The public first launch qualifies all four components transactionally before
any execution. The separate input-only probe protocol allocates uninitialized
device inputs and outputs, without a scientific fixture/oracle or fake expected
payload. Its run method rejects execution before compilation or launch; callers
use compile_only to obtain artifacts. Controlled collection must compile/resource/codegen-screen all frozen
cells before generating the six CPU payloads, then validate all twelve APIs
before the eight ten-call captures. The standalone CLI is exploratory and does
not claim those complete collector gates.

The native collection passed all six numerical and paired-Y-bit gates. All 48
logical component probe cells qualified, yielding 28 unique artifact hashes;
these counts do not assert an external compiler process invocation count. Four
recompute cases emitted distinct selected code. The D2048 heldout control and
wide guard retained identical complete four-kernel artifacts across arms.
The independent coverage inventory marks the heldout control as not requiring
the optimized route; the separate measurement receipt still requires and
retains all ten profiled API calls per arm for that control.

| Shape rows×D | Route after | Summed work before → after (µs) | Complete API span before → after (µs) |
| --- | --- | ---: | ---: |
| 32768×32 | Recompute | 1450.743 → 1547.039 | 1450.815 → 1547.113 |
| 256×1024 | Recompute | 35.947 → 33.397 | 281.710 → 284.578 |
| 16384×64 | Recompute | 675.340 → 753.975 | 675.414 → 754.047 |
| 33×2048 | Cached control | 17.838 → 16.246 | 279.280 → 288.708 |

These are all-ten-sample profiled means, including first samples. No main case
showed a lower mean complete API span. The identical-code control's differing
measurements illustrate capture variability rather than an optimization.
Task work, launch gaps and event samples remain separately retained. These
results establish neither ordinary latency nor physical memory traffic or
occupancy, and provide no automatic unique-mechanism claim.

An initial host-controller check assumed 40 vector cores from the prior GZ
observation. It stopped before real compilation, frozen fixtures or scientific
calls. Before any data, the authoritative v2 contract clarified use of the
selected target's actual driver properties, observed as 48 vector/24 Cube
cores. The adapter, JIT kernels and CPU provider already queried correctly and
were unchanged. Both the zero-science failed attempt and the completed version
are retained in T08-versions.json and verified compact archives.

The completed acquisition generated six CPU payloads once, validated twelve
full APIs before profiling, and captured eighty four-kernel APIs in eight
captures; the guards captured zero. Every sample, fixed-gate error, paired-Y-bit
result, source/fixture/binary/trace hash, selected-code receipt and supported
raw PMU field is retained. The original coverage audit passes.

Before publication, the host adapter's probe run method gained an explicit
rejection guard, with nonfrozen mocked counters proving no compilation or
launch before or after compile_only. Four host tests and a CPU-only A3 gate
pass. T08-publication.json distinguishes measured and published host source
hashes. The JIT, CPU provider, six scientific payloads, binaries and measurements
remain unchanged; no scientific replay accompanies this host fix.
