# Supplied-map MoE row movement

This study compares scalar row movement with a fixed four-row tile on A2/A3.
It covers the `sort_chunks_by_map` primitive with an already supplied int32
bijection. It does not measure map construction, prefix sums, sorting, or
end-to-end MoE performance. CPU chunk metadata generates an API input; its
cost is outside this movement-only experiment.

The source is MindSpeedOps commit
`4efe2b675c2d51517c3fdd14c721020b8d41e241`, specifically
`mindspeed_ops/arch32/triton/sort_chunks_by_idx.py` and
`mindspeed_ops/api/triton/sort_chunks_by_idx.py`. Huawei and NVIDIA notices
are preserved in `moe_rows_kernel.py`; `MOE_ROWS_LICENSE` contains Apache 2.0.
Frozen source receipt hashes and the four prospective cases plus two guards
are recorded in `experiments/optimization-wave007/E03.json`.

Forward movement writes `Y[row_map[i], :] = X[i, :]`; inverse movement reads
`Y[i, :] = X[row_map[i], :]`. Optional probabilities move once per token.
Both arms use the queried AIV core count, the same contiguous row partition,
and a power-of-two bound on each program's row loop. The tiled arm masks
its four rows to that same partition. It does not steal another program's
rows, retune the grid, or shrink the tile when compilation fails.

The source-style arm preloads the map and uses `get_element` when its real
Triton extension compiles. The actual extension import is attempted even
when the name is absent from `triton.language`. If compilation rejects that
lowering, the retained compiler failure explains selection of a
**portable source-style scalar map load baseline**. Direct scalar map loads
change map traffic; results from that fallback are not unmodified vendor
performance. Compiler optimizations remain enabled in either case.

Inputs are owned contiguous CPU FP32 `x[N,D]`, int32 `row_map[N]`, and
optional FP32 `probs[N]`. Admission requires a bijection, `0 <= N < 2**24`,
power-of-two `D <= 4096`, and int32-safe element offsets. Validation checks
exact output arity, shapes, dtypes, device, private output storage, canaries,
input immutability, and FP32 bits, including signed zero and NaN payloads.
N=0 allocates empty outputs and automatically uses correctness-only mode.

The public adapter supports flat parameters or a `parameters` object, with
`seed` and optional `role` metadata at the top level. For example, put this
in a case JSON file:

```json
{"seed": 914, "parameters": {"N": 25, "D": 32,
 "forward": true, "with_probs": false, "chunks": 6,
 "map_kind": "shuffled_chunks"}}
```

With the adapter directory on `PYTHONPATH`, call the shared runner from any
working directory:

```sh
python /checkout/scripts/optimization_pair.py --adapter moe_rows:Adapter \
  --case case.json --provenance provenance.json --cache cpu-cache --output result
```

Scientific collection uses six cached independent CPU expected payloads,
one correctness call per arm/case, and exactly ten profiled complete API
calls per arm for each of four main cases. The two frozen guards have no
captures. Compilation and actual ELF text/relocation comparison happen
before device correctness. Identical generated code, resource rejection,
unsupported behavior, and partial captures remain outcomes. Profiled
operation spans are diagnostic measurements, not ordinary latency or
physical memory traffic, roofline, or occupancy evidence.

Outputs are private buffers allocated during preparation and reused equally
by both arms. A measured API invocation is one complete **prepared movement**
invocation. The original vendor Python wrapper allocates fresh outputs each
call; its allocator cost is excluded here. Device operation spans do not
establish unmodified vendor Python API latency. All input-dependent movement
and optional probability movement are included in each measured invocation.

The public runner qualifies compilation before its first movement launch.
If an importable `get_element` extension cannot lower, it selects the
portable scalar-map specialization before executing movement. Successful
qualification is cached for later calls. This host adapter correction was
validated with CPU doubles; the retained scientific run used a supported
extension. Its measured adapter hash remains separate from the published
host adapter hash; JIT kernels, fixtures, and scientific measurements are
unchanged.
