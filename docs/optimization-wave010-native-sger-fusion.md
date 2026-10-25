# Prepared rank-one fusion: native SGER

This source-backed example contrasts an educational split rank-one materialization plus matrix addition with the pinned CANN SGER device kernel. Both prepared calls include the same full original-A reset. The split arm charges reset, product and addition; the fused arm charges reset and SGER. Allocation, CPU tiling and H2D setup are outside the prepared call. This is a component experiment, not the vendor host API or ordinary application latency.

The vendor device source and tiling header are preserved byte for byte from ops-blas `c2332551586763f8c2d27621f8bc119bd1c3a301`, under CANN Open Software License Agreement 2.0. Exact filtered provenance is in `experiments/optimization-wave010/P05.json`. The original host's `GetAivCoreCount` helper could not be recovered (query `57a0a776fe9b44ad9000def7f6367f50`, miss `d18c5047faf1e893c64242b444b20886`). The explicitly chosen prepared policy fixes 48 AIV partitions in both arms and requires the actual runtime to report 48 vector cores. No equivalence to the missing source host helper is claimed.

The original device code shortens a column interval only on the last launched block. Some otherwise valid matrix dimensions would therefore overrun the final column when the last active block precedes the final launched block. The prepared interface rejects those partitions before building or moving tensors. The six prospective shapes have bounded, disjoint column ownership; the source was not repaired or retuned.

Inputs are separate CPU-owned contiguous FP32 arrays: full column-major A storage `[lda*n]`, X `[m]`, Y `[n]`, unit strides. The result contains all A storage, including unchanged row padding. A, X and Y values are integer/16 within [-4,4], alpha is 0.5 or -0.25. Every intermediate and result is exactly representable in FP32 under this bounded dyadic contract. An independent FP64 outer product modifies only active rows of an A copy; correctness checks nonzero bit identity, numeric zero identity, full padding, input immutability and private halos. Zero-sign differences are secondary diagnostics.

The unchanged source allocates 192 KiB across its queues. Its temporary queue has template depth 1 and buffer count 2; this source declaration is retained. Compiler-managed scratch is unknown unless reported. The compiler uses the actual SDK ASC frontend, Release `-O3`, `--npu-arch=dav-2201` and explicit automatic synchronization for both arms. The retained build command, compiler/SDK hashes, host library, Cube and Vec objects, registrations and text sections bind the actual program.

Cases 01/02 are prospective cost-risk probes; 03/04 predict a lower mean complete profiled operation span and task time from eliminating the product buffer round trip and one launch. A cost label does not promise a slowdown. Collection performs twelve input-only compile/ABI checks, six once-only frozen CPU fixtures, twelve once-only correctness calls, then ten complete prepared calls per arm and main case. Guards are never profiled. There are no warmups, extra timings, profiler I/O inspections or retries of scientific calls.

All launches, validation and the entire profiler/event/synchronization scope use the captured prepared device and restore the caller's ambient device. Prepared handles support idempotent close and reject use after close. CPU/C-ABI/mock tests cover resets, exact results and padding, invalid domain/layouts, source-column overrun, runtime core mismatch before build, compile-only behavior, atomic build publication, changed caller device during capture and normal/error cleanup.

Results: first-attempt BZ-A3 qualification completed successfully: all twelve correctness calls passed exactly before 80 profiling calls, eight captures, and 200 explicitly accounted `AI_VECTOR_CORE` tasks. All 368 compact files and 20 original source bindings were hash verified. All twelve prepared handles closed; no owned live processes remained, device and both export locks were released, and fresh device health was clean. Profiled operation spans include gaps between charged kernels and must be reported alongside summed task time, active union and gaps. They do not establish ordinary latency, physical HBM savings, occupancy, distance to roofline or an actor advantage.


| m × n; lda | Prospective label | Before span µs | After span µs | Span ratio | Before task sum µs | After task sum µs | Task-sum ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| 128 × 48; 128 | cost | 51.292934 | 31.051630 | 1.6519× | 10.145795 | 6.405870 | 1.5838× |
| 512 × 240; 520 | cost | 54.764809 | 37.637073 | 1.4551× | 19.375612 | 14.631703 | 1.3242× |
| 1024 × 480; 1024 | benefit | 57.502770 | 50.675198 | 1.1347× | 30.755386 | 24.087517 | 1.2768× |
| 2049 × 96; 2056 | benefit | 51.923020 | 37.047180 | 1.4015× | 12.947738 | 9.117817 | 1.4200× |

The four-case geometric mean of profiled span ratios is **1.3983×**. Both intended-benefit cases have lower mean spans and task sums; both cost probes also have lower means. All ten samples are retained, including the large first samples and high span variance. Task sums show a device-time improvement as well as fewer launch gaps; this does not establish ordinary latency or statistical significance. The common ABI allocates product scratch in both arms, so this experiment does not measure allocated-workspace reduction.

Raw samples, individual task and PMU rows, active unions, gaps, compilation/source identities and archive bindings are in [P05-result.json](../experiments/optimization-wave010/P05-result.json). Compact archive SHA-256: `0bce19b03609c85a706387c74fa2e57c8e34bc3e0905be300c3361d4ae7929fb`.
