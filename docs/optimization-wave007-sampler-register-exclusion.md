# Wave007 T03: zero the recovered-token draft probability in registers

Both arms passed exact token-ID and input-restoration checks on all six frozen
cases. Mean profiled operation spans improved modestly in three main cases and
regressed in the large-vocabulary single-token case. This is a source-backed
operator application of an existing mechanism, with no mechanism-novelty claim.

| Frozen case | Before span µs, mean ± sample SD | After span µs, mean ± sample SD | Before/after |
|---|---:|---:|---:|
| W007-T03-01 | 8.768 ± 0.688 | 8.072 ± 0.274 | 1.086 |
| W007-T03-02 | 16.940 ± 0.223 | 17.180 ± 0.215 | 0.986 |
| W007-T03-03 | 6.908 ± 0.216 | 6.738 ± 0.291 | 1.025 |
| W007-T03-04 | 3.394 ± 0.052 | 3.358 ± 0.073 | 1.011 |

Each arm has ten retained samples. These instrumented operation spans are
diagnostic; they do not establish ordinary application latency, physical memory
traffic, occupancy, or roofline position. Small differences overlap sample
variation and are retained without retuning cases or selecting extra replays.

The Huawei/vLLM source is pinned to `7acb9c40a6ca827e885d902b7eaee5d45457b230`;
its attribution and MIT license are retained beside the implementation. The
baseline temporarily loads, zeroes, and restores the draft probability in global
memory. The candidate instead zeroes the loaded draft element with `tl.where`.
Both keep 4096-element chunks, target/q division, lowest-index `tl.argmax`, strict
`>` earliest-chunk merging, grid, and common compiler flags. The draft-probability
branch remains identical. Zeroing preserves a score of zero: an allzero row can
select the draft ID at a zero-score tie. This is not categorical ID exclusion.

The original source could not compile because installed Triton lacks dynamic
`tl.get_element`. That stage generated one cached CPU payload and entered zero
device forwards. The allowed common lowering replaces only the selected maximum
value access with `tl.max` in both arms; `tl.argmax` and tie semantics remain.
The unchanged CPU-provider file allowed reuse of that payload and generation of
the remaining five exactly once. The failed source and compiler error remain
retained. No optimizer disabling was used to manufacture an arm difference.

All twelve kernels compiled before mathematical launches. Actual selected ELF
`.text` bytes differ for the five no-draft pairs; neither arm has relocation
sections affecting `.text`. The unchanged draft-probability guard has identical
`.text`, as intended. Full binary hashes, all compiler intermediates, compiler
resource metadata and exact fixture bindings are retained. No decoded instruction
or store-count claim is made.

The successful gz-a3 run used healthy idle physical die 6, logical device 0,
Ascend910B4-1, CANN 9.1.0, imported Triton 3.2.0, Triton-Ascend 3.2.2,
Torch 2.10.0+cpu with Torch-NPU 2.10.0. Distribution metadata separately reports
Triton 3.5.0. Both arms used `num_stages=1`, `multibuffer=False`, and
`enable_fp_fusion=False`; the compiler's actual metadata is retained.

Six CPU fixture/oracle payloads used normalized integer target weights and
independent positive powers-of-two q. Each arm ran correctness once per case,
then ten full-API forwards per main case. Totals: 12 correctness API/kernel calls,
80 profiled API/kernel calls, eight captures, no guard captures, no runtime
warmups, no autotuning, and no extra timing forwards. Validation checked exact
integer output, valid IDs, output canaries, and bitwise input restoration outside
captures. The 4097 guard covers a cross-chunk tie, endpoint IDs, allzero scores,
and an empty request; the other guard checks the unchanged draft-probability path.

A 600-second C watchdog and sibling mmap API counter bound correctness through
validation's CPU synchronization and each capture through synchronous export.
Capture/export held one of the exact shared two-slot locks; the device lease
covered fresh bounded health and all work. The first read-only compact collector
failed on a malformed glob; the fixed collector reused unchanged evidence and
made no new forwards, oracle calls, or captures.

[Samples, raw supported PMU rows, source/compiler/binary identities, six fixture
bindings and the verified compact manifest](data/optimization-wave007-T03.json)
retain all ten samples and unavailable metrics. Full vendor trees remain on the
target. Successful collection handle:
`remote:gz-a3:job:8b7bd830-9213-4477-801c-d1beb99b2b59` (exit 0).

The public CLI accepts both flat and nested case parameters from any working
directory and rejects unsupported metadata before cache access:

```sh
python examples/optimization-library/triton/sampler_cli.py \
  --case CASE.json --output NEW_OUTPUT --cache CACHE --provenance PROVENANCE.json
```

Add `--validate-only` for host metadata validation without the NPU backend or
cache access. The supported envelope is owned, nonaliasing contiguous CPU fixture
storage copied to exclusive device rows: FP32 target/q and optional draft
probabilities, positive finite q, int32 valid draft IDs and cumulative endpoints
matching nonnegative request lengths. Unit tests use separate seeds/shapes and
exercise scalar oracle behavior, invalid inputs/ownership, cache tampering,
compile-only launch accounting, adapter reuse, and the external-cwd public CLI.
The CLI was added separately after collection; measured adapter, kernel and oracle
source bytes remained unchanged.
