# Query blocking and K/V reuse

Status: source-derived experiment, not yet A3 verified. No measured speedup or
counter claim is available until paired artifacts are attached.

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
