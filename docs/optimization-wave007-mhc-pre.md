# Exclusive MHC pre-gradient stores

This is a bounded, source-backed application of exclusive output ownership to
MindSpeed-Ops arch32 MHC pre backward. It does not claim a new optimization
mechanism or measure the unmodified vendor pipeline.

The source is `mindspeed_ops/arch32/triton/mhc/pre_bmm.py`, pinned to
`4efe2b675c2d51517c3fdd14c721020b8d41e241`, from
[MindSpeed-Ops](https://gitcode.com/Ascend/MindSpeed-Ops).
The filtered original source digest is
`2c231ad92773b9fb803c3aac8c257b72108934f348d383766b4ec2fc881325d6`.
The repository Apache 2.0 license is retained in `MHC_LICENSE`; the selected
source has no additional per-file notice. The filtered API/README maps arch32
to A2/A3. This mapping does not establish compatibility with an installed
compiler: all six frozen cases must compile before numerical execution.

The public adaptation accepts owned contiguous CPU FP32 fixtures `H[BS,4]`,
`X[BS,4,D]`, `dY[BS,D]`, then moves them to the selected NPU. It returns
`(dH[BS,4], dX[BS,4,D])` in private buffers. BS must be positive, D must be a
power of two no greater than 4096, and tile_D must be a power of two no greater
than D. Empty and unsupported cases fail before fixture caching.

Both arms use source GROUP=1, the same dX products, four dot reductions, grid,
strides, and compiler flags. The baseline initializes dH to positive zero
inside **every API invocation** and performs four atomic adds. To provide a
stable kernel identity, a one-pass Triton zero kernel implements the source
`torch.zeros` device work. This portability adaptation is explicit; it is not
a claim about the speed of the original vendor allocation pipeline.

Only an after invocation with tile_D=D uses exclusive stores and omits zeroing.
The grid is then `(BS,1)`. Program b owns exactly offsets `4*b+{0,1,2,3}` in a
new contiguous dH allocation, whose storage aliases neither inputs nor dX.
GROUP=1 and the lack of any other nontrivial program axis prove those sets are
disjoint and exhaustive. With multiple D programs, both arms use the exact
same zero-plus-atomic fallback; no performance capture is taken for that guard.

The CPU oracle evaluates dX products and dH dot products independently in FP64
and casts once to FP32. Both outputs use fixed atol=rtol=2e-5. Numerical
agreement does not promise bitwise signed-zero identity: adding a reduction to
positive zero and directly storing it can produce different zero signs. The
zero/cancellation guard retains observed zero bit patterns and input/canary
checks. The six frozen fixture payloads are generated once and cached; the
provider file is separate from kernel portability changes.

For an external working directory, pass absolute paths to the standalone CLI:

```bash
python /path/to/mhc_pre_cli.py --case case.json --output result \
  --cache cpu-fixtures --provenance provenance.json --validate-only
```

Omit `--validate-only` to run the generic pair protocol: one correctness call
per arm, then exactly ten profiled full API calls per arm for a main case.
Guard cases run correctness only. The diagnostic operation span includes both
baseline components, and is not ordinary latency, physical DRAM traffic,
roofline evidence, or occupancy proof. All raw supported profiler fields and
ten samples must be retained, including slower or uncertain outcomes.

Both arms passed all six frozen numerical/input/canary gates on gz-a3 physical
die 6 (logical 0), Ascend910B4-1, CANN 9.1.0, imported Triton 3.2.0,
Triton-Ascend distribution 3.2.2, Torch 2.10.0+cpu / Torch-NPU 2.10.0.
All six pairs compiled before numerical work; actual backward `.text` differs
for five exclusive pairs, and both full fallback component binaries match.
No decoded instruction-count claim is made.

| Case | Span before / after (µs, mean ± SD) | Component busy sum before / after (µs) |
|---|---:|---:|
| W007-E01-01 | 204.028 ± 7.302 / 15.902 ± 0.334 | 20.294 / 15.902 |
| W007-E01-02 | 195.550 ± 20.520 / 2.312 ± 0.120 | 3.666 / 2.312 |
| W007-E01-03 | 218.142 ± 11.659 / 23.898 ± 0.963 | 30.946 / 23.898 |
| W007-E01-04 | 191.758 ± 12.855 / 3.446 ± 0.121 | 4.496 / 3.446 |

Each arm retains ten samples. The large span differences include the gap
between baseline zero and backward launches; the component busy sums show a
much smaller difference. Neither establishes ordinary application latency or
the isolated cost of the four atomic instructions. Common compiler flags were
`num_stages=1`, `multibuffer=False`, `enable_fp_fusion=False`; the compiler
naturally blacklists AutoBlockify for atomic kernels, with no forced setting.

GA observed all 12 dH zeros positive before, and four negative zeros after.
Both dX outputs contained 512 zeros, 256 negative. This is an observed bitwise
difference within the fixed numerical contract, not an undocumented equality.
Maximum dH error across all cases was 7.62939453125e-06; dX error was zero.

The run used six once-generated CPU fixtures, 12 correctness API calls (19
component kernels), 80 profiled API calls (120 component kernels), eight
captures, and no guard captures, warmups, autotuning or extra timing calls.
A 600-second C watchdog covered run through synchronized validation and
capture through export; sibling mmap counters preserve entered API counts.
The exact die lease and two shared export locks bounded concurrency.

[All samples, raw supported PMU rows, source/compiler/binary/fixture identities,
guard zero bits and verified compact manifest](data/optimization-wave007-E01.json)
are retained. The 331-file compact archive was verified locally against every
hash; full vendor reports remain remotely retained. Collection completed exit0
at `remote:gz-a3:job:5373251d-f941-4605-9bc0-4f49b4beee38`.

The profiling adaptation allocates private output storage once per prepared
arm and reuses it. Interior output storage is uninitialized; every baseline
and fallback invocation runs zeroing, and every exclusive invocation writes
all dH and dX elements. This setup is part of the bounded adaptation scope.
