# E05 selected-block traversal: frozen A3 qualification

Source commit `ff6aeb7455906b25efd2ca8241318d188dde529c`, measured on one healthy idle A3 die. The [portable evidence](optimization-expansion-sparse-evidence.json) includes frozen input specs, all timing samples, correctness results, raw PMU rows, and source/compiled-binary/trace hashes. Same query16/key32, GQA semantics, FP16 inputs/FP32 output and CPU FP32 oracle in both variants. Before visits all key blocks and masks; after visits supplied selected IDs, skipping arbitrary interior -1 holes. Supplied selection construction is excluded identically.

8 correctness calls, 8 ten-forward captures, 80 profiled forwards, 4 cached CPU oracle payloads; zero warmups/autotuning/extra replays. All immutable-input correctness checks passed at fixed rtol=atol=1e-3. Compiler identity stable across each capture. Each iteration has exactly one selected_attention device task; span therefore also equals complete-operation task duration.

## Primary complete-operation device spans

| Frozen case | Before mean ± sample SD (us) | After mean ± sample SD (us) | Before/after |
|---|---:|---:|---:|
|E05-triton-gallery-benefit|17060.744 ± 9.012|157.978 ± 3.237|107.994x|
|E05-triton-gallery-cost|867.072 ± 5.247|453.372 ± 2.276|1.912x|
|E05-triton-heldout-benefit|4050.932 ± 7.673|122.520 ± 0.476|33.063x|
|E05-triton-heldout-cost|949.702 ± 14.114|442.452 ± 1.323|2.146x|

Ten raw samples for every arm are in the portable evidence. These are profiler-instrumented device spans; no uninstrumented latency claim.

## Separate profiler-instrumented caller events

| Case | Before mean ± sample SD (ms) | After mean ± sample SD (ms) |
|---|---:|---:|
|E05-triton-gallery-benefit|17.121226 ± 0.183091|0.461274 ± 0.176634|
|E05-triton-gallery-cost|0.939154 ± 0.219632|0.645148 ± 0.252944|
|E05-triton-heldout-benefit|4.118234 ± 0.199315|0.471718 ± 0.217235|
|E05-triton-heldout-cost|1.015848 ± 0.187031|0.655892 ± 0.269110|

## Interpretation and limitation

The low-density cases support the structural-skip benefit. Both preregistered nearly dense cost cases also improve here: a slowdown/tradeoff was **not demonstrated**. They show smaller gains, not negative evidence fabricated into a loss. No case was changed or repeated. This family is therefore runtime-qualified for these four points, but does not yet satisfy a paired observed-regression requirement. The before arm still pays selected membership reduction on each dense block, which can outweigh indirect sparse overhead even at high density.

Supported same-capture PMU rows include aic/aiv total cycles and per-pipe reported time/ratio, and cube_utilization(%). All 80 device tasks were identity-joined to raw CSV rows by device/stream/task/name. These remain raw vendor metrics; no whole-device utilization or physical traffic claim. No verified product bandwidth/FLOP ceiling was collected, so distance to roofline is unavailable.

Hardware reported: Ascend910B4-1, 20 Cube cores, 40 Vector cores, 62,420 MB total memory and 168 MB L2 cache.

Versions: Torch2.10.0+cpu, Torch-NPU2.10.0, Triton3.2.0; profiler Torch-NPU Level1 PipeUtilization. Exact compiled npubin/source hashes, full raw PMU CSV records, 80 device-window samples, caller events and trace hashes are retained in receipt. The original vendor reports remain retained separately; their trace hashes bind the published evidence.

## Reproducing the operation

The adapter is `sparse_pairs:factory` in [sparse_pairs.py](../examples/optimization-library/triton/sparse_pairs.py); its independent CPU oracle and input contract are in [sparse_contract.py](../examples/optimization-library/triton/sparse_contract.py). Use the supplied case specs with the `optimization_pair` runner: prepare the immutable fixture once, run correctness once per arm, then capture exactly ten forwards per arm. Preparation transfers inputs and allocates output; `run` contains one complete launch and no input/output checks.

Q is contiguous `[HQ,Q,D]`, K/V are contiguous `[HK,K,D]`, and `HQ` is divisible by `HK`. The supplied int32 selection tensor is `[HQ,ceil(Q/16),SLOTS]`; its nonnegative block IDs must be unique and in range. `-1` holes may occur anywhere and never terminate traversal. Valid blocks may appear in any order. Empty lists return finite zero output; query/key tails are masked. Every fixture explicitly contains an empty tile. CPU interpreter tests additionally exercise interior holes, arbitrary order, GQA, tails and input immutability. FP16 probability conversion before the value dot is identical between arms; the oracle uses dense CPU FP32 softmax.

Choose selected traversal when a supplied sparse worklist removes substantial key-block work. Inspect the dense loop, repeated membership reduction and masked dot operations in code; in the profiler compare complete-operation spans and the same-capture per-pipe rows. The after arm retains per-slot loads, dynamic checks, indirect block addresses and online reduction state. These are possible overheads, especially for dense lists or small work, but none of these four frozen cases demonstrates a net slowdown. Do not infer whole-device utilization, physical bytes, a roofline distance or a universal crossover from these results.

The source motivation is the pinned NSA sparse-attention selected-block traversal cited in the kernel module. This experiment deliberately supports arbitrary interior holes rather than inheriting a valid-prefix assumption. It is a scoped noncausal GQA operation, not an NSA drop-in replacement. Index construction is outside this supplied-index API; a wider application must include that cost when comparing end-to-end alternatives. The active optimization skill and its first-study guidance remain unchanged.
