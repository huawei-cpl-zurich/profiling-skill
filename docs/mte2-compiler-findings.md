# Triton expressions versus actual compiler consumers on A3

## What the first four captures establish

Compiler dumps reveal material differences hidden by Triton source, but different
stages answer different questions. Source predication need not reduce transfers;
identical early adapter IR need not mean identical buffering. These observations
are mechanisms to test against independent profiling, not performance claims.

| Controlled change | Early/source appearance | Actual later compiler evidence |
|---|---|---|
| Dense load then select → predicated load | Only every fourth value is required | Both contain contiguous GM→UB load calls and vector selection; predicated version adds Boolean operations |
| Tensor offsets modulo4 | Compact tensor arithmetic | Per-element UB integer load/remainder/store loop in HIVM and LLVM input |
| Matmul num_stages1 →2 | Byte-identical adapter IR | Auto-multibuffer compiler flag changes; later IR contains two input buffers per operand and alternating event IDs |

The [compact observation data](data/mte2-compiler-observations.json) binds each
excerpt to an exact artifact SHA256, byte length and original line. Artifact
labels identify files in the retained coordinator archive, not compiler stage
filenames. Actual manifest input arguments establish stage provenance.

## Acquisition and reproducibility

The four controls are `mask-before`, `mask-after`, `buffering-before`, and
`buffering-after` from `mte2_compiler_controls.py`. All passed correctness and
reference execution, each compiling a single selected Triton workload. CPU
initialization/reference are outside that kernel. Capture uses the managed
`mlir-triton-dump` helper via an isolated copy of runtime Python/compiler tools;
shared tools are not replaced. See [capture instructions](mte2-compiler-evidence.md).

Actual workload source SHA256 is
`619fb00eb600cb50aa8d3e7988274cb49c1c93ca54f734c0981ede49e4d14843`,
present in task commit112ff94 and unchanged by subsequent inspector fixes.
Observed target is `Ascend910_9382`, Triton module3.2.0. Actual `hivmc --version`
stdout reports `hivmc 0.3.0 (923475c46003 2026-07-29)`, Release build. These
captures do not establish a CANN or msprof version, nor an A5 result.

Actual wrapped compiler invocations retain adapter input, late HIVM input,
device LLVM input and host LLVM input. Capability queries are recorded separately
from actual compilation. Forced compilation/debug acquisition is separate from
timing; none of the following static observations is a speedup measurement.

Only compact text/JSON was retrieved:186481 compressed bytes for four sets of
31 selected files and an index, archive SHA256
`59d38b866d0afb6076b59b6a96458dd77453e887cec04aa7833abb1a8f907eb3`.
Full source-bound compiler artifacts, invocation/error logs and index remain
retained outside Git; no executable, runtime clone or raw profile was published.

## Masking does not imply sparse movement

Both variants use1048576 fp32 elements, BLOCK1024, and produce zero except at
every fourth element. Before loads a dense tile then applies `tl.where`. After
passes the selection predicate into `tl.load`. Their output semantics match.

In the actual late HIVM inputs:

- Before `mask-before/002-captured-input`, line122 constructs a1024-element GM
  view with stride1. Line137 calls `load_gm_to_ubuf_1d_float`; line147 calls
  `vsel_vs_1d_float`.
- After `mask-after/005-captured-input`, line126 constructs the same contiguous
  view. Line153 calls the same load-helper family; line171 calls vector select.
  Lines145 and162 add `vnot_1d_bool` and `vand_1d_bool` respectively.
- LLVM consumer input retains load/select calls: before lines376/381, after
  lines452/459. Helper parameters differ; physical traffic is not measured by
  a helper name or inferred from these static call sites.

The supported conclusion is that this source-level predication still lowers to
contiguous load plus selection, not an established fourfold reduction in physical
GM traffic. The narrower lesson is to inspect the lowered transfer/mask path
before predicting that sparse source predicates save bandwidth. Independent
memory counters require correct units, matching kernel and scope to test that.

## Tensor expressions can create scalar work

Both mask variants contain an unexpected scalar-looking mask-construction loop.
Before late HIVM lines101–104 and after lines105–108 iterate1024 elements, load
UB integer offsets, calculate `arith.remsi` modulo4, and store UB integer values.
Before LLVM input lines345–348 preserve `load i32`, `srem i32 ...,4`, `store i32`.

Consequently, high scalar activity must not automatically be attributed to a
user-written outer loop or fixed by unrolling it. Even this tensor expression
produces a per-element scalar path at the captured stage. Conversely, those
LLVM operations are still inputs to final bisheng `-O2`; final optimization may
simplify remainder arithmetic. Neither exact dynamic scalar instruction count
nor the share of latency caused by that loop is established here.

## Early IR misses a real buffering change

The matmul inputs and tiling are unchanged: M128,N128,K512,BM32,BN64,BK32, fp16
inputs and fp32 accumulation. The only explicit option change is
`num_stages=1` → `num_stages=2`.

Both actual adapter inputs are9375 bytes with SHA256
`c7ec840e3042a5335795a958bb24c7ccc76c632d47839b0421dd9766a615ae07`.
Nevertheless, real compiler argv changes `--enable-auto-multi-buffer=False` to
`--enable-auto-multi-buffer=True`, and the later HIVM/LLVM inputs differ:

- Before HIVM has lhs/rhs cbuf bases0/2048, with2048/4096-byte layouts. After
  lines43–46 have four bases2048/8192/0/6144: two buffers per operand. The input
  storage extent represented by these layouts grows6144→12288 bytes.
- After map6 at line7 is modulo2. Lines84/89 select rotating buffer bases;
  lines98/100 select alternating event IDs. These are compiler-generated
  buffer/event transformations rather than changes visible in early adapter IR.
- Both variants retain GM→cbuf `nd2nz_half`, `mma_tile_half_to_float`, and
  result FIXPIPE conversion calls. The static HIVM set/wait flag sites increase
  from9/9 to11/11. Static site counts are not executed event counts or wait time.
- After LLVM input lines136–139 initialize additional flags,194/202 contain
  modulo2 buffer rotation, and272/295 use `llvm.hivm.WAIT.FLAG.REG`.

This establishes changed buffer/event machinery, not useful hardware overlap.
Extra buffers create an opportunity to pipeline movement and compute; independent
timing and profile evidence must decide whether that opportunity benefits this
small matmul or whether overhead/another resource dominates.

## Controlled memory: compiled path versus runtime locality

The completed controlled memory matrix adds an important separation. Its source
retains a direct load or loads runtime indices before loading selected values.
The compact data's `controlled_memory` section binds10 actual first-correctness
adapter artifacts to their SHA256 and line excerpts; each was checked against
the retained586-file curation manifest, archive SHA256
`906f91a99215756f17ebf2d6b784255c0ccff897e950b83fbc7c971a59e154b6`.
These artifacts are original successful workload outputs, not reconstructions.

For all six fixed-count nonunit strides2/4/8/16/32/64, adapter line12 creates a
256-element input view with that stride; line24 copies the strided view to a
dense destination. Direct stride1 instead has a stride1 view/copy at lines10/22.
No per-element scalar input-load loop is visible in these direct adapter inputs.
That does not establish the final DMA sequence: `memref.copy` is an intermediate
operation, not a measured transfer count or a guaranteed hardware pipe binding.

Runtime-indexed contiguous, local32 permutation and global permutation all have
**byte-identical5355-byte adapter IR**, SHA256
`4d12d93035f7b95bfa3407532b7f94af71b0a1620a025c93188f0f0153dd8b79`.
Line25 copies a contiguous integer index tile. Lines28–35 then loop over up to256
elements, extract an index and perform a one-element input `memref.load` at
line32. This is distinct from the direct contiguous input copy, even when runtime
index values describe exactly those same contiguous input addresses.

Observed BasicInfo medians in the same indexed comparison are30.099µs direct,
481.470µs indexed-contiguous,483.090µs local32 permutation and2378.992µs global
permutation. Thus a compiler-path difference is consistent with much of the
direct-versus-indexed penalty; additional index traffic/control also differs.
But the global-versus-contiguous indexed slowdown is **not explained by an
adapter-code change**: the code bytes are identical and runtime indices change.
Address locality is a plausible factor, not an independently isolated cache,
HBM or final instruction-level mechanism. Identical IR is a useful negative
result: compiler dumps cannot explain every runtime performance difference.

The fixed-count stride measurements also resist simplistic explanation.
Stride2 has median882.742µs and stride64 median264.455µs for the same useful
input count. Every nonunit stride reports a GM→UB amount equal to32× useful
input bytes under the empirical KiB conversion, while row-median MTE2 activity
stays approximately99.5–99.9%. Higher busy fraction therefore does not mean
higher useful throughput, and equal reported transfer amount does not predict
equal latency. Counter units/scopes remain as qualified by the memory report;
this is not established physical HBM traffic.

The retained adapter artifacts do **not** establish why stride2 is slower than
stride64 or why the reported amount is32×. Fixed transaction granularity,
overlapping small requests, cache behavior and channel mapping are hypotheses
for future discriminating measurements, not findings. No later HIVM/LLVM for
these exact memory matrix specializations was retained, and no recapture is
claimed in this report. Do not convert the ratio into an asserted128-byte
hardware transaction size solely by multiplying it by the4-byte element size.

## What to teach, and what remains unproven

Use source→adapter→actual late consumer inputs when source reasoning conflicts
with profiling. Verify options in actual compiler argv, and inspect later IR
when investigating synchronization/buffering. Do not stop at an unchanged
adapter dump or convert a generic `memref.copy` into a particular pipe claim.

Actual device LLVM here is **before final bisheng optimization**, not final ISA,
an execution timeline or proof that an operation dominates. Hardware performance,
physical bandwidth and the benefit to agents require the separate qualified
measurements and evidence-ablation assessment; they are not supplied by these
four compiler captures alone. Null gains and source-level optimizations that
leave the important lowering unchanged are useful results too.
