# Bounded row residency for log-softmax

This source-backed example compares the pinned streaming log-softmax implementation
with a resident reduction for contiguous finite FP32 rows. The source is
`triton-ascend-kernels` commit `7acb9c40a6ca827e885d902b7eaee5d45457b230`,
`deterministic/log_softmax_batch_invariant.py`; the Huawei and SGLang notices and
MIT license accompany the adapted implementation.

The before arm retains the source's three passes with BLOCK2048. Both arms use
the actual queried AIV core count, the same contiguous row assignment, and a
sequential row loop per core. The after arm retains one power-of-two row at
widths up to4096 and performs max, exp/sum, and output from that row. Widths
above4096 use the identical streaming function in both arms. The reduction tree
can differ, so paired arms promise the fixed numerical gate, not identical bits.
Within each arm, the selected row must produce identical output bits in batch
contexts1,7,and32; the guard validates all context outputs against FP64 as well.

`logsoftmax_contract.make_case` accepts flat or `parameters`-nested positive
`rows` and `width`, plus a nonnegative `seed`. Inputs are CPU FP32, contiguous,
finite and within[-20,20]. Its independent FP64 stable log-softmax reference is
cast to FP32 once per component, with fixed atol=rtol=3e-5. The six frozen study
payloads include one compound tuple; unit tests use other seeds and shapes.
`logsoftmax_residency.factory()` supplies the optimization-pair adapter. Put the
example directory on PYTHONPATH when loading that adapter from another working
directory. Compilation uses num_stages=1, multibuffer=False and
enable_fp_fusion=False, with no autotuning or warmup launches.

Compile/codegen/resource evidence precedes scientific execution. Retain
compiler-noops, unsupported specializations, numerical failures and slower
results. The frozen study is `experiments/optimization-wave007/T02.json`.
The study profiles ten complete forwards per arm on its four main pairs;
guards perform correctness only. The guard with three contexts has three
component kernel launches per correctness API call. Profiled operation spans
are diagnostic; they do not establish ordinary latency, physical GM traffic,
roofline bounds or occupancy. This is a new operator application of an existing
residency/reduction-fusion mechanism, with no unique-mechanism claim.

The retained A3 study passed all twelve arm/case accuracy checks (maximum absolute
error3.814697265625e-6). Both guards passed; the larger-width guard emitted an
identical full binary. Diagnostic operation-span means and sample SDs are:

| Rows × width | Before mean ± SD (µs) | After mean ± SD (µs) |
|---|---:|---:|
|32 ×4096|7.216 ±0.612|5.210 ±0.274|
|1 ×31|2.536 ±0.493|2.314 ±0.154|
|80 ×2049|10.098 ±0.319|6.516 ±0.132|
|2 ×65|4.358 ±0.303|3.936 ±0.235|

All ten samples, raw supported PMU fields, fixtures, binaries and report hashes
are bound in `experiments/optimization-wave007/T02-evidence.json` and its compact
archive receipt. There were six cached payloads, twelve correctness API calls
(sixteen component launches), eighty profiled API/component calls, eight
captures, and no guard captures. The compiler reported resource metadata and
accepted all specializations; its shared-memory fields are not interpreted as
measured UB use or occupancy.

The full-binary codegen screen preceded science. A stronger actual executable
section comparison ran afterward, without new compilation or forwards, and
confirmed distinct code for resident cases. The transport job reported canceled
while its controller survived and completed; durable entry counters and six
complete result records establish the retained counts. A second controller
failed its shared device lease before generating any fixture or executing any
forward. No science was repeated. The controller exit code is unavailable after
that transport cancellation, and the audit ordering remains explicit in the
receipt.
