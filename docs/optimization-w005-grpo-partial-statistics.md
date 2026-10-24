# GRPO policy sufficient statistics

This forward-only pair replaces a materialized clipped-policy tensor with tile
sums. It is another operator application of sufficient statistics, adding no
new unique-technique claim. Both implementations preserve the retrieved source's
unusual objective: **one global policy sum is broadcast into every output row**.
This is not ordinary per-token PPO, and the pair makes no backward claim.

For FP32 log probabilities `logp`, `old`, `ref` of shape `[T,V]` and advantages
`adv[T]`, define:

```python
ratio = exp(logp - old)
p = maximum(-adv[:, None] * ratio,
            -adv[:, None] * clamp(ratio, 0.8, 1.2))
k = ref - logp
kl = k                         # kl
kl = exp(k) - k - 1             # unbias
kl = k * k / 2                 # mse
loss = (p.sum() + kl_coef * kl.sum(axis=1)) / (T * V)
```

All inputs are immutable, contiguous FP32, finite and in `[-1,1]`. Differences
therefore lie in `[-2,2]`. Outputs are separate FP32 `[T]`. The coefficient is
0.1 for the four main cases; the zero-coefficient guard still retains the policy
term. NaNs, infinities, noncontiguous inputs, other dtypes and unsupported KL
names are rejected by the standalone input contract.

The code format is:

```python
# BEFORE: full policy producer, same tile reduction, common final reducer
produce_policy_and_kl_tiles(logp, old, ref, adv, policy_TV, kl_tiles)
reduce_policy_tiles(policy_TV, policy_tiles)
finish_global_policy_and_row_kl(policy_tiles, kl_tiles, loss)

# AFTER: keep only the sufficient tile statistics
produce_policy_and_kl_tiles(logp, old, ref, adv, policy_tiles, kl_tiles)
finish_global_policy_and_row_kl(policy_tiles, kl_tiles, loss)
```

Both arms use 256-column tiles, the same row-major final policy reduction tree,
the same KL tile reductions and arithmetic association, and disabled FP fusion.
The shared final kernel redundantly reads the global policy partials for each
output row; this is an explicit small educational contrast, not an optimized
vendor baseline. The upstream row-wide scheduling is independently adapted to
bounded tiles for the measured backend. Every producer, reduction and finish
launch occurs inside the measured API. Input copies, output and scratch
allocations occur before the API in both arms; each scratch value used by a
forward is overwritten during that forward.

The logical scratch model is `8*T*ceil(V/256)` bytes for policy and KL partials,
plus `4*T*V` bytes for BEFORE's full policy tensor. This is not measured peak
allocation: both arms are prepared together. The expected opportunity is avoiding
one full policy write/read and a reduction launch. The competing cost is reducing
within the producer and retaining reduction state there; the common final global
reduction and its repeated reads can dominate. A compiler may also make the
intended contrast less useful. No improvement sign is promised by the prospective
benefit/cost labels.

Recognize this opportunity when a full elementwise policy tensor has only a sum
consumer, while row-wise KL and one scalar policy statistic are the required
outputs. Check exact clipping, advantage signs, KL convention and the global
broadcast before applying it. In a profile, compare complete producer-to-finish
spans, task counts and supported original vector/MTE fields. More producer
resources or an unchanged dominant final reducer can falsify the benefit
hypothesis. PMU pipe activity does not measure physical traffic, bandwidth or
roofline distance; those are not estimated here.

The independent CPU FP64 oracle evaluates the represented FP32 inputs. Every row
must meet `atol=2e-5, rtol=2e-4`. The sign/clip guard includes negative, zero and
positive advantages, nearest represented log clip boundaries and differing row
KL. The tail guard uses `[5,129]` and `kl_coef=0`. These are correctness-only.
The frozen matrix is `experiments/optimization-wave005/T04.json`, SHA256
`7ac50712b597de16718dccb07ddb2598e2990d4cea564c75c1ebf2557e5a4483`.
Six CPU payloads are cached once, each side is checked once per case, then each
main arm executes exactly ten profiled full API forwards. There are no warmups,
autotuning, extra timing passes or input/output inspections during those forwards.

The primary timing boundary is the **profiled device-operation span** from the
first producer task to the final output task, including interkernel gaps. Active
task union and noncompute gap are separate fields; a gap can include caller
scheduling and is not proof of all-device idle. Instrumented caller events are
reported separately. Neither boundary is ordinary unprofiled application latency.

Collection is in progress; no device performance result is claimed yet.

## Provenance

Independent implementation follows actual Triton body/interface/tests from
`triton-ascend-kernels` commit `7acb9c40a6ca827e885d902b7eaee5d45457b230`:

- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/loss/grpo_loss.py:L15-L96`
- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/loss/grpo_loss.py:L207-L245`
- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#tests/loss/test_grpo_loss.py:L8-L50`

Huawei and DeepLink notices and the MIT license are retained beside the source.
Upstream tests print some `allclose` results instead of asserting them; this
pair supplies independent assertions. Source presence alone does not qualify
910B4-1/CANN9.1/Triton3.2 execution, which requires the recorded device gate.
