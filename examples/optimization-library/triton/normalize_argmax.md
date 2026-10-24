# Common normalization before argmax

This pair illustrates a source-documented removal in
`triton-ascend-kernels/src/triton_ascend_kernels/postprocess/reject_sample_recovered_tokens.py`
at commit `7acb9c40a6ca827e885d902b7eaee5d45457b230` (MIT), originally adapted
from vLLM's rejection sampler. The source's comment explicitly omits
`prob / sum(prob)` before argmax. The small pair excludes the source's
mutating no-draft-probabilities branch and retains its ascending-tile winner
selection. Stable filtered source citation:
`ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/postprocess/reject_sample_recovered_tokens.py:L65-L95`.

The public demonstration takes contiguous FP32 `target`, `draft`, `q`
with shape `[rows,V]` and returns int32 lowest-index argmax `[rows]`.
Both arms compute `r=max(target-draft,0)`. BEFORE selects the maximum of
`(r/sum(r))/q`, and AFTER selects the maximum of `r/q`. Zero residual rows
use denominator one, returning zero. Both arms use the same tile, launch
options, reductions and strict ascending-tile merge. BEFORE additionally
traverses the full row for its normalization sum; no residual is cached.
Input device copies and output allocation happen in preparation, while
all normalization, division and argmax work happens in every forward.

The frozen cases construct probability distributions: target is `1/V`
everywhere, draft is zero on a seeded half of columns and `2/V` elsewhere.
Both sum to one, and the residual sum is exactly one half. The seeded
unique winner has `q=1/2`; other entries have `q=1`. Nonzero scores are
normal powers of two, so normalization is an exact exponent shift in FP32
and cannot change ordering. The two guards cover zero residual and tied
maxima. This does not establish invariance for arbitrary FP32 probabilities,
near ties, overflow or underflow. There is no profiled admission check.

Run CPU checks with `python -m pytest -q tests/test_optimization_normalize_argmax.py`.
The paired adapter is `normalize_argmax:Adapter` with its directory on
`PYTHONPATH`; `normalize_argmax_cases.json` contains the immutable six cases.
Compile every case and both arms before any device forwards. `warmup` here
is Triton's compile-only entry, not a device warmup. The scientific protocol
uses one correctness call and ten profiled calls per arm for the four
non-guard cases, one correctness call per arm and zero profiles for guards.
Device operation span including stage gaps is the primary profile comparison;
profiled durations are diagnostic, not application latency.

## Upstream MIT license

Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
Copyright contributors to the vLLM project.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
