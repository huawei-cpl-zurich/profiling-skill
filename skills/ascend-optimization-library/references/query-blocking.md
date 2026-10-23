# Query blocking and K/V reuse

Status: both listed Triton gallery pairs passed A3 correctness and ten-forward
profile/identity collection. Their measured behavior appears below.

Use this mechanism when source programs process one query row each and reload
the same K/V sequence across nearby queries. A query tile reuses a loaded K/V
tile for several rows and exposes matrix operations to Cube. The live score
and output accumulators grow with the query tile, so short queries and large
head dimensions can lose to padding or resource pressure.

Pinned source context: [historical online attention update and UB slicing](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-historical-scope-9319a71c74a6#ascend/examples/tutorials/04-fused-attention.py:L111-L138).
The tutorial's fixed scheduling stride and floor-divided sequence bounds are
not portable scheduling rules. Our example independently uses a complete
query grid and masked query/key tails.

The paired code is `examples/optimization-library/triton/attention.py`:

```python
# Before: one query row per program, vector reductions over each K/V tile.
score = sum(key * query, axis=D) / sqrt(D)
# After: BM query rows share K/V; matrix products replace row reductions.
score = dot(query_tile, key_tile) / sqrt(D)
```

Both implementations keep FP32 online maxima, denominators and accumulators.
Both round probabilities to FP16 for the weighted-value product while retaining
an FP32 denominator. The before implementation uses vector reductions and the
after uses Cube dot products: this is a query-blocking transformation that also
changes the execution primitive and floating-point accumulation order, not a
pure cache-scheduling experiment. Both must pass the same FP32 CPU attention
oracle with rtol=atol=1e-3. Inputs are contiguous FP16, output FP32, noncausal,
B=1, H=4/8, D=64/128, Q/K=1..1536. No reference fallback is hidden in either arm.

Test recipes are long square attention (Q=K=1024,D=64,H=8) and short irregular
attention (Q=17,K=65,D=128,H=8). The second case includes partial query and key
tiles. Its disadvantage is a hypothesis, not an answer key. The adapter accepts
`{b,h,q,k,d,seed}` and exports exact live JIT kernel names and compiler artifact
hashes; preparation performs no forward.

Inspect complete-call duration, per-kernel task ownership, emitted vector/Cube
work, supported movement metrics and compiler resource reports. Fewer source
loads do not prove fewer physical reads. Higher Cube activity alone does not
prove bandwidth saturation or roofline proximity. This example retains no
attention-matrix intermediate in either arm; do not attribute a speedup to
removing such an intermediate.

Apply only when query reuse and resource capacity plausibly outweigh the extra
live state/padded work. Avoid or narrow the dispatch if short/tail cases regress
or numerical margins fail. A paired result showing equal or worse total latency
with no supported traffic/resource benefit falsifies a confident performance
recommendation for that regime. Retain inconclusive results when variability
covers the observed difference.

Runnable [attention.py](../../../examples/optimization-library/triton/attention.py)
uses `attention:factory`, with its independent
[CPU contract](../../../examples/optimization-library/triton/attention_contract.py).


## What the gallery establishes

Both the long and short/tail regimes improved device span. The proposed short
counter-regime did not establish a query-blocking penalty on this implementation.
The long case also improved the instrumented caller interval, while the short
case's caller comparison was inconclusive. Larger live state and padded work
remain applicability/resource costs, not measured losses to invent.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| 03-triton-gallery-benefit | 4537.93 ± 2.714 / 1455.64 ± 9.375 | 3.117×; gain | 4.57082 ± 0.1008 / 1.50654 ± 0.1576 | gain |
| 03-triton-gallery-cost | 20.184 ± 0.2372 / 10.9 ± 0.2451 | 1.852×; gain | 0.337766 ± 0.2815 / 0.224594 ± 0.08376 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aic_mac_ratio` | `aiv_vec_ratio` | `aiv_mte2_ratio` |
| --- | ---: | ---: | ---: |
| 03-triton-gallery-benefit/before/`wave001_attention_row` | 0 | 0.7379 | 0.1325 |
| 03-triton-gallery-benefit/after/`wave001_attention_query_tile` | 0.028 | 0.1069 | 0.0715 |
| 03-triton-gallery-cost/before/`wave001_attention_row` | 0 | 0.6576 | 0.1508 |
| 03-triton-gallery-cost/after/`wave001_attention_query_tile` | 0.0252 | 0.111 | 0.0602 |

Observed resource/numerical costs and limits:
- **03-triton-gallery-benefit**: no additional resource/numerical cost identified by the retained fields.
- **03-triton-gallery-cost**: cost recipe did not produce a device-span regression in this capture.
