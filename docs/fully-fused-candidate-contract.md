# Fully fused candidate contract

New fully fused campaigns use `profiling-skill/candidate-kernel/v2`. The
manifest declares the exact exported `kernel_name`, its Triton `entrypoint`,
and this exact fusion object:

```json
{"schema_version":1,"mode":"single-logical-launch","complete_operator":true}
```

The selected kernel must be the entry point or its compiler-emitted
`_mix_aic`/`_mix_aiv` component. Historical v1 manifests remain historical
evidence; they do not assert full fusion.

## Trusted runtime boundary

After correctness over the complete case inventory and before timing, the
trusted controller profiles one isolated candidate forward for every case. It
must classify the complete compute-op stream into `triton`, `torch`, or `acl`,
assign the same `launch_id` to compiler-generated components of one logical
Triton invocation, identify their entry point, and bind the returned output to
its producing launch with `output_launch_id`.

The benchmark backend recognizes v2 manifests and runs a full-case correctness
job before every profile request. The trusted benchmark runner wraps the
declared Triton entry point, records each launch and its tensor arguments,
observes Torch/ACL tensor dispatch, and binds returned output storage to the
launch that received it. Both production A3 clients propagate this evidence.
The backend invokes `scripts/fully_fused_contract.py` over every configured
case before dispatching the timing job or exposing any profile timing.
The gate
accepts exactly one logical Triton launch per case, including a mixed AIC/AIV
family. It rejects zero or multiple launches, Torch/ACL compute, another
entrypoint, a selector outside the declared family, output bypass, and missing
or duplicate cases. A rejection is a candidate failure and must not receive a
performance result.

The evidence producer is trusted harness code, not agent-controlled output.
Do not infer operator origin or output provenance from candidate declarations.
Do not use a selector-filtered replay as the complete compute-op stream: that
would hide framework fallback and auxiliary launches.
