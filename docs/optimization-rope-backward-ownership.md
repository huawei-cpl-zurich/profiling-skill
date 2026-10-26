# RoPE backward: partial atomic reductions versus full-group ownership

This is an **unqualified, compile-stopped study**. The full-head H=256 candidate exceeds the compiler's UB capacity. No scientific CPU payloads, numerical checks, or profiling calls ran. There are no measured speedups or profiler conclusions.

The paired code is derived from MindSpeed-Ops `4efe2b675c2d51517c3fdd14c721020b8d41e241`, `mindspeed_ops/arch32/triton/inplace_rope.py`. Exact source, public autograd context, helper and Apache-2.0 license are retained beside the example. The source is indexed at `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-mindspeed-scope-4efe2b675c2d#mindspeed_ops/arch32/triton/inplace_rope.py`.

The prepared component consumes `dy`, the **saved pre-rotation** contiguous `x_rope`, and per-token cosine/sine coefficient arrays. It returns all of `dx`, FP32 `dcos`, and FP32 `dsin`; the nonrotary `dx` prefix is copied in both arms. This comparison excludes forward rotation, autograd, saving inputs, allocations and initial transfers consistently. It does not claim full public-operator or product-dispatch validation.

The source arm uses `BLOCK_ROWS=min(H,64)`, reduces each partial head group, and atomically adds into `dcos` and `dsin`. A separate kernel zeros both coefficient gradients **on every prepared call**. The candidate uses `BLOCK_ROWS=H` and replaces the two atomic additions with direct stores, so one program owns each complete coefficient group. All other source backward arguments and arithmetic remain unchanged. Both use four warps, two stages and the explicitly common `enable_fp_fusion=False` policy. The source wrapper normally chooses block rows through a hardware helper; this experiment supplies the documented explicit parameter and makes no helper-parity claim.

The intended optimization removes a reset launch and partial atomic contributions. Its cost is a larger live tile: more head rows are loaded and retained for coefficient-gradient reduction. In future qualified evidence, two charged source tasks versus one owned task and the supported raw pipeline metrics would help assess the hypothesis. Those are prospective expectations here, not observed runtime results or physical-memory/occupancy claims.

| Frozen case | Role | B | S | H | Half rotary dimension | Nonrotary prefix | Compile result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | Cost probe | 1 | 1 | 8 | 32 | 64 | Both arms passed |
| 02 | Cost probe | 1 | 64 | 64 | 32 | 64 | Both arms passed |
| 03 | Intended benefit | 1 | 64 | 128 | 16 | 64 | Both arms passed |
| 04 | Intended benefit | 2 | 128 | 256 | 16 | 64 | Source passed; owned arm failed UB capacity |
| G1 | Guard | 1 | 3 | 32 | 8 | 128 | Unattempted |
| G2 | Guard | 2 | 5 | 128 | 16 | 192 | Unattempted |

The compiler reports **2,491,392 bits required versus 1,572,864 bits available**, equivalent to **311,424 B versus 196,608 B**. Its own automatic code-motion/multibuffer fallback also failed. This is a compiler resource failure, not infrastructure or a numerical mismatch. Smaller cases compiled only; their correctness and performance remain unknown. The frozen matrix was not reduced or retuned after the failure.

All twelve input-only compilation/type/code/resource cells were scheduled before any scientific CPU reference generation. Seven passed, one failed, and four were unattempted. The eleven successful logical component artifacts retain typed pointer and atomic/store IR, executable ELF sections, resources and hashes. All twelve correctness counters remain zero, and no scientific oracle or profile directories exist.

The independent CPU oracle uses represented dyadic inputs `q/16`, integer `q` in `[-4,4]`. Products have denominator 256; a two-product gradient numerator is bounded by 32 and the largest full-head sum by 8,192, below FP32's exact-integer bound. `dx` is returned as FP16 and coefficient gradients as FP32. Exact nonzero values and numeric zeros are the primary intended check; signed-zero bits are secondary. This certificate supports a future numerical gate; it does not replace the unattempted NPU checks.

Public `factory()`, `Adapter` numerical methods, module fixture generation and default compile probe fail closed before provider/backend/cache/transfers. Explicit `compile_probe(..., inspect_compile=True)` supports input-only inspection whose returned `run()` refuses execution. The private measured adapter remains for functional mocks and retained experimental provenance, not an advertised qualification bypass.

Six CPU/mock tests passed on GZ-A3 after publication quarantine, covering independent integer formulas, admission before backend access, compile-probe/cache failure, all three output lifetimes, reusable resets, real generic-pair cleanup, caller-device restoration and the complete capture guard with explicit `PipeUtilization`. No real compiler, NPU, frozen references or profiling ran in that gate. Its first host-only attempt stopped before tests because Torch tried to auto-load the blocked NPU backend; the corrected gate disables backend autoload explicitly. The scientific compiler failure was not replayed.

`experiments/optimization-wave011/P07-result.json` retains the outcome and all seven successful compile proofs. `P07-compile-diagnostics.json.gz` preserves the full compiler error/IR, resource and runtime bindings, original source proof, release receipt and all 169 verified compact-artifact hashes. `P07-public-quarantine.json` binds measured and publication host versions. The device and both host export locks were verified released healthy before the next study was handed off. All transport used the installed remote-access client, and all work stayed on GZ-A3 as requested.
