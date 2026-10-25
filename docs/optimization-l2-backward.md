# Provided-state L2 normalization backward

This component uses the D>512 backward branch from Triton-Ascend-Kernels
revision `7acb9c40a6ca827e885d902b7eaee5d45457b230`. The MIT/Huawei license and
FLA attribution are retained. The baseline expression is
`DY*R - sum(DY*Y)*Y*R`; the candidate changes only its association to
`(DY - sum(DY*Y)*Y)*R`. Addresses, masks, row grid, power-of-two D tile,
source num_warps4, reduction, and common compiler flags remain equal.

The bounded interface accepts separate contiguous FP32 Y[rows,D],
RSTD[rows], and DY[rows,D], with rows1..512 and D513..4096. Inputs must be
finite, RSTD positive and at most1000.001, |Y|<=1.000001, |DY|<=1. The
prepared output DX has separate storage and prefix/suffix canaries. Input
shape, dtype, layout, value bounds and nonaliasing are checked before backend
imports. Forward calls reuse the prepared output and perform no input IO.

The fixture provides saved state generated on CPU: dyadic X in[-1,1],
FP64 r=1/sqrt(sum(X²)+1e-6) cast to FP32 R, then Y=(X*R32) cast to FP32.
There is no hidden GPU forward, and this is not a claim of the source
forward's exact saved state. The independent CPU FP64 expected derivative
uses the actual provided Y32/R32. Every element must be finite and meet
fixed atol2e-6/rtol2e-5. Pair bit differences are reported without requiring
bit equality. Guard cases include zero, parallel-gradient and cancellation
rows. D64 is rejected on the host; no blocked-source tail correctness or
fake expected fixture is claimed.

The controlled collector compiles input-only probes for all five numeric
cases before generating scientific oracle payloads or launching math. It
retains actual ELF text, relocations, resource metadata, and compiler IR.
Effective compiler identity can terminate the study as a no-op with zero
numeric or profile forwards; differing whole ELF files or entry names alone
are insufficient evidence. No optimization flags force a distinction.

If the binary screen distinguishes the candidates, all five paired numeric
gates must pass before exactly ten complete API profile calls per arm on
four main cases. Guards have zero captures. The D64 host scenario checks
each arm once before cache/backend/oracle access. There are five cached
numeric fixtures and one host-only scenario. No warmups, autotuning, extra
timing, profile input inspection, or data-driven case/tolerance changes are
permitted.

Public exploratory CLI from any working directory:

```sh
python /absolute/repo/examples/optimization-library/triton/l2_backward.py \
  --case case.json --output results --cache oracle-cache \
  --provenance provenance.json --correctness-only
```

Cases accept flat `{"rows":2,"D":517,"seed":911}` or nested parameters.
The CLI rejects unsupported shapes before cache lookup and declares its
exploratory scope; it does not run the controlled all-case screening order.
First launch qualifies and caches actual executable text before math.

The authoritative six-case inventory includes a legitimate host denial
that the generic route auditor cannot represent. Preserve its original
expected audit failure as a tool limitation; report a separate five-numeric
subset audit without inventing a reference device route. Host rejection is
its own observable result. An audit pass is not evidence for unsupported D64
kernel behavior.

Native BZ validation must report its observed9382/Ascend910V1 product;
GZ910B4 labels are not inherited. Profile event and operation-span samples
are instrumented diagnostics, without ordinary latency, allocator, physical
traffic, roofline, occupancy, full normalization-forward, or training-speed
claims. Raw supported PMU fields and all ten samples are retained.

The retained native acquisition passes all five paired numerical gates. The
largest observed absolute error against the independent expected output is
3.73e-9. ELF text and relocation screens distinguish every candidate binary.
The diagnostic operation-span means are:

| rows × D | Before (µs) | After (µs) |
|---|---:|---:|
| 128 × 1024 | 7.488 | 7.690 |
| 1 × 1024 | 1.632 | 1.592 |
| 512 × 2048 | 15.778 | 16.240 |
| 33 × 4096 | 4.982 | 4.710 |

Two means increase and two decrease; the samples establish no consistent
diagnostic improvement. All ten samples, including first-call costs and
variation, are retained in `T09-results.json`, with raw supported PMU rows in
`T09-metrics.json`. No ordinary-latency benefit is claimed.

The first native runtime attempt on device2 failed during lazy initialization
on its first input transfer, before candidate compilation, scientific oracle
generation, or numerical/profile forwards. Error107020 and the vendor AICPU
timeout are retained as an unresolved runtime failure. Its two successful
host rejection checks were reused by exact receipt hash and matrix/spec/arm
binding in the authorized device0 acquisition; they were not repeated. Both
attempts use identical operator source, oracle provider, cases, tolerances,
and compilation parameters. The earlier failure remains separate evidence.

Publication interface correction: compile probes expose compilation, metadata
and binary identity, while their `run()` always rejects execution. Ordinary
prepared items retain normal execution. `T09-publication.json` separates the
original measured adapter hash from this host-only interface revision; JIT,
oracle provider, binaries, fixtures and scientific results are unchanged.
