# GDN output: share Q and accumulate state/local contributions in one kernel

This is a qualified prepared **output component**, not complete GDN. Both arms take the same Q, K, corrected `v_new`, precomputed chunk-start state H, cumulative gates and chunk offsets. Upstream recurrence, inverse, wrapper transposes, metadata creation, allocation and initial copies are outside both prepared APIs. The educational baseline computes two FP32 contributions and combines them; the candidate uses the pinned source kernel. This is a known fusion technique applied to another source operator, with no new-mechanism claim.

Source: Triton-Ascend kernels `7acb9c40a6ca827e885d902b7eaee5d45457b230`, `attention/fla/chunk_o.py`. Original device code, helpers, caller, state producer, whole-GDN test and notices are retained under `examples/optimization-library/triton/gdn_output/vendor/`. The selected helper closure is complete. The upstream test covers whole GDN; it does not independently validate this standalone component. That evidence gap remains recorded as miss `66528e96848abc52dc01b8e6b416504a`.

## Optimization and preserved numerical boundaries

The source computes state `Q @ H` and local causal attention `Q @ K.T`, sharing Q inside the tile. It applies gates, casts the attention scores to FP16 **before** their multiplication by V, scales each contribution separately and writes one FP16 output. The causal mask includes the diagonal. `safe_exp(x)` returns `exp(x)` for `x <= 0` and zero otherwise; substituting a positive exponential or clamp changes the operator.

Conceptual before, with all three kernels charged every API:

```python
state_fp32 = state_kernel(Q, H, g, scale)  # QH * exp(g) * scale
local_fp32 = local_kernel(Q, K, V, g, scale)  # gate/mask, score FP16, then AV * scale
combine_kernel(state_fp32, local_fp32, output_fp16)
```

Conceptual source candidate:

```python
state = tl.dot(q, h)                       # FP16 inputs, FP32 accumulation
scores = tl.dot(q, k_transposed)
state = state * tl.exp(g)[:, None]
scores = scores * safe_exp(g[:, None] - g[None, :])
scores = tl.where(row >= column, scores, 0)
result = state * scale + tl.dot(scores.to(v.dtype), v) * scale
tl.store(output, result.to(output.dtype.element_ty))
```

The generated candidate body matches the original source after function renaming and exposing the existing heuristic values as fixed constexpr arguments. Both arms use `BT=64`, `BK=BV=128`, four warps, two stages and explicit `enable_fp_fusion=False`. The last flag defines the same separate FP32 multiply/add boundaries in both arms; the upstream wrapper's default flag is not claimed identical. No timing-based tuning occurred.

Apply this pattern when independent producers consume the same tile and their outputs only feed an immediate combination. Look for materialized FP32 producer buffers, repeated Q reads and three correlated kernels per operation. Expected benefits are fewer intermediate stores/loads and launches. Larger live state can increase compiler-managed storage or reduce scheduling flexibility; the actual compiler must accept the configuration. Per-pipe PMUs and resource fields retain their own meanings. Logical byte avoidance does not establish physical HBM traffic or occupancy.

## Admitted component

Only one nonempty packed sequence is admitted: B=1, H=Hg=8, K=V=128, T=1..2048, `cu_seqlens=[0,T]`, scale=0.5 and nonnull cumulative gates. Inputs Q/K/corrected V are dense FP16 `[1,T,8,128]`; state is dense FP16 `[1,ceil(T/64),8,128,128]`; gates are physically head-major FP32 `[1,8,T]`; chunk offsets are int64 `[0,ceil(T/64)]`. Outputs and intermediate buffers are private and reused, with every valid cell overwritten each call. Inputs are immutable and storage does not alias. Typed 128-byte canaries protect each allocation.

General batches, grouped heads, multiple sequences, missing gates, dense mode, arbitrary layouts and other dimensions are rejected before backend access. This does not repair or qualify upstream wrapper behavior outside the component. State and corrected V are component inputs, so their construction cannot be counted as a gain.

The independent CPU oracle uses FP64 products over represented inputs and explicitly preserves the FP32, gate, causal, score-FP16, scale and final-output boundaries. Primary correctness uses fixed `atol=2e-4`, `rtol=2e-3` against that oracle and between arms. All-bit and zero-sign comparisons are secondary. Q/K/V/H fixtures are dyadic, gates and state are nontrivial, and the same oracle payload is cached once per case.

## Verified measurements

GZ-A3-2 physical device 0, masked logical device 0, actual Ascend910_9362 with 20 Cube/40 vector cores; Python 3.12.13, Torch 2.10.0, Torch-NPU 2.10.0.post4, Triton-Ascend 3.2.2, CANN 9.1.0. The source-independent fixed grid is `(1,8)`. Original BZ planning was replaced prospectively by a GZ-only host receipt before any compilation or scientific data. BZ had no scientific calls.

All 12 input-only compile/typed-IR/ELF screens passed before generating six once-only CPU payloads. All 12 correctness gates passed before 80 complete prepared API calls in eight captures. T=33/127 guards received correctness checks and zero profiles. There were no device warmups, timed autotuning, additional timings, profiling I/O checks, outlier removal or scientific replay.

Each cell below is a mean of all ten samples, in microseconds:

| T | Prospective role | Task sum before | Task sum after | Task speedup | Operation span before | Operation span after | Span speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | Cost probe | 15.989675 | 8.025900 | 1.992× | 243.743050 | 8.025900 | 30.370× |
| 64 | Cost probe | 18.355675 | 8.005850 | 2.293× | 236.701225 | 8.005850 | 29.566× |
| 128 | Benefit | 21.533550 | 9.865800 | 2.183× | 237.811300 | 9.865800 | 24.105× |
| 2048 | Benefit | 127.157400 | 70.948600 | 1.792× | 274.644375 | 70.948600 | 3.871× |

Task sum equals active union here. Before-operation gaps were 227.753375, 218.345550, 216.277750 and 147.486975 µs, respectively; the candidate has one task and zero inter-task gaps. The large span ratios therefore include removed profiled submission gaps. They are **not ordinary application latency or 30× compute speedups**. The two intended-benefit cases also reduce actual charged task sums by 2.183× and 1.792×. Cost probes improved too; their designation was a prospective risk, not a promised slowdown. These ten-sample means do not establish statistical significance or physical-memory/roofline claims.

The captures account for all 160 actual device tasks: 120 `MIX_AIC` and 40 `AI_VECTOR_CORE`. Baseline dispatches three tasks per API, candidate one; vector work is retained, not dropped. All six paired outputs were bit-identical as a secondary observation. Independent-oracle maximum absolute error was at most 6.103515625e-5, within the frozen tolerance, with zero mismatches and intact inputs/canaries.

Full means, samples, typed compiler/ELF bindings, correctness, fixture hashes and source provenance are in `experiments/optimization-wave011/P06-result.json`. All raw per-task PMU rows and task objects are in `P06-profile-details.json.gz`, digest-bound by the result. Original traces, cache payloads, compiler artifacts and reports remain in the verified 11,952,285-byte compact archive, SHA256 `93a02193d9e12a053672f4c8dbe21ce1431b5f57432786e3ce4de405d274a441`.

The first read-only archive attempt incorrectly treated container PGID=0 as unique ownership and stopped. The corrected reader uses exact owner-PID absence, unique source/output paths, device/export leases and fresh healthy/process-empty device checks; PGID=0 is explicitly unusable for isolation. It changed no scientific source or data. The device was released healthy, all 24 prepared/probe objects closed, and no reset or scientific rerun occurred.

## Functional interface

`gdn_output.factory()` provides the existing `optimization_pair.run_pair` adapter interface. Add `examples/optimization-library/triton/gdn_output` to `PYTHONPATH` and use the generic pair harness with a supported case spec, cache, output and provenance. Input-only probes compile without launching; public prepared items have idempotent `close()`. The complete capture, including event recording, profiling, synchronization and export, stays on the prepared device and explicitly requests `PipeUtilization`.

Five CPU/mock tests exercise the typed oracle and domain, probe/compile failures, real generic pair cleanup and caller-device restoration, full capture metrics/device scope and the canonical evidence consumer. These tests use small nonfrozen cases and never run scientific NPU kernels. Source files retain Apache-2.0 and copied FLA MIT attribution; complete Apache/MIT and third-party notices accompany them.
