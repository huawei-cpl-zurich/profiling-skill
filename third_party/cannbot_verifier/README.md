# Pinned cann bot correctness verifier

These Python files and LICENSE are copied byte-for-byte from
https://gitcode.com/gcw_85ZVbQ4M/cannbot-skills_5171 at commit
`4f03c3c2677c1a486b06b3cb04607271f645cc31`, directory
`ops/triton-op-verifier/scripts`. `provenance.json` records the source and SHA256
of each dependency. Preserve the license and upstream copyright notices.

The benchmark adapter calls `verify.compare` and its input-dtype inference.
It preserves the upstream default: ecosystem precision criteria first, with
NPU Benchmark criteria for dtypes outside that table. NaN/Inf checks, integer
and quantization handling, non-compute binary equality, Chinese diagnostics and
structured accuracy metrics are upstream behavior. The other copied modules
are the verifier's import dependency closure; the benchmark does not invoke the
standalone verifier's CLI, hidden-state probe, or remote/device preflight.

Update the complete dependency closure and provenance together when intentionally
changing policy; private profile caches bind this identity and refuse old policy.
