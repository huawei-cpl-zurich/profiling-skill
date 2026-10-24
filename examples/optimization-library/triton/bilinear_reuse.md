# Reuse bilinear geometry across channels

This is an application of invariant/coefficient reuse, adding no automatic new
mechanism count. The pair is an independent implementation of the documented
bilinear equation and coordinate interface in `cann/ops-cv` commit
`1e859a524a93d0a94b563cacaa067db6abbfd37c`, under CANN Open Software License
Agreement 2.0. The upstream resize device body was not retrieved. Its README
lists A2/A3, while the retrieved registration targets newer products; this
example establishes its own bounded 910B execution evidence.

Source context: `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-cv-curated-text#image/resize_bilinear_v2/README.md:L15-L32`,
coordinate attributes at `#image/resize_bilinear_v2/README.md:L67-L106`, interface
at `#image/resize_bilinear_v2/op_host/resize_bilinear_v2_def.cpp:L19-L60`, and
actual ACLNN example at `#image/resize_bilinear_v2/examples/test_aclnn_resize.cpp:L1-L130`.
Sharing geometry across channels is an inference from channel independence,
not a claim about the unseen native implementation.

## Code format and contract

`bilinear_reuse.py` contains two one-kernel full forwards. The baseline assigns
256 flattened output elements per program; every channel lane computes source
coordinates and interpolation weights. The candidate computes four spatial
positions per program and broadcasts those coordinates/weights across a
power-of-two channel tile. It includes all four masked pixel gathers and all
geometry inside the call. This necessarily changes tiling, especially for one
or two channels and non-power-of-two channel counts; it isolates a format
comparison rather than a claim of identical scheduling or a best vendor baseline.

Both use contiguous FP32 NHWC input in `[-1,1]`, immutable input, separate
FP32 output, and `align_corners=False`, `half_pixel_centers=False`, no antialias.
Coordinates are FP32 `output_index * FP32(input_size/output_size)`; lower and
upper neighbors clamp to the input border. Horizontal interpolation precedes
vertical interpolation with the same association and FP fusion disabled.
Other coordinate modes fail at preparation before NPU import. `bilinear_contract.py`
provides the nested-JSON fixture CLI and an independent FP64 four-corner oracle;
fixed gates are `atol=rtol=1e-5`. Resource support beyond the recorded shapes is
not inferred from the gallery.

## Recognition and competing cost

Look for geometry, floor, clamp and weight arithmetic repeated over a channel
axis whose coordinates do not depend on channels. An analyzer should compare
compiled scalar/vector resources and task spans; the compiler may already
eliminate common work. A candidate can reduce geometry arithmetic while losing
parallelism, increasing live channel state, or paying for padded channel lanes.
Four irregular gathers can dominate both forms. One-channel inputs expose the
scheduling cost without channel reuse. Do not infer reduced physical traffic
or bandwidth from source-level repeated expressions or pipe percentages.

Each valid side is numerically checked once, then runs exactly ten profiled
full API forwards without tensor inspection, warmup or autotuning. The one-pixel
border guard has no captures; unsupported-coordinate rejection is checked on
the host for both arms. Inputs and output buffers are prepared outside the
forward; no input-dependent coordinate tables are precomputed there. Profiled
device-operation span is diagnostic and distinct from ordinary application
latency. Active union is retained separately; the noncompute remainder is not
whole-device idle time. Roofline distance is not estimated without supported
capacity and traffic denominators.
