# Native ShapeInfo code-generation screen (W006 L03)

The selected CANN 9.1 / native 2201 compiler expands `K_MAX_SHAPE_DIM` to 8
by default. Defining it as 0 before all AscendC headers changes that setting,
but this unused-metadata SmallMatmul produces identical device code and
allocation records. This candidate is a **compile no-op**, with no measured
latency or physical stack reduction.

`shape_metadata.cpp` is independent of `unit_flags.cpp`, which already forces
zero metadata. Both arms use the same source, kernel name, native SmallMatmul,
128×256×256 tile, L0 K64, two stages, false unit flags, AIC annotation and
explicit pointer/shape host ABI. No AscendC descriptor crosses translation-unit
boundaries. `shape_metadata_abi.h` contains portable host validation and each
loader image owns its state.

Compile with the qualified pinned filtered Catlass closure and runtime-provided
compiler/CANN environment:

```bash
python examples/optimization-library/catlass/shape_metadata_compile.py \
  --catlass /path/to/filtered-catlass --output /new/evidence/directory
```

The compiler screen retains expanded sources, device objects, shared libraries,
compiler commands/identity, dependency/source hashes, extracted sections and
`screen.json`. It checks actual expanded dimensions and compares executable,
data, relocation and allocation-note records. Host library hashes and metadata
strings alone do not establish a device change. Raw stack allocation records
are compiler evidence, not certified physical peak usage.

`--request-shape-info` rejects before dependency or output access. Applications
requiring ShapeInfo must use a separate SDK-default implementation. This pair
never invokes ShapeInfo APIs. Both independent translation units expose opaque
pointers and explicit dimensions; their AscendC class layouts may differ.

The frozen four prospective numerical/profile cases remain **unattempted**.
There are zero scientific oracles, zero NPU API forwards and zero profiles.
If another qualified compiler produces a real device change, the screen reports
`physical-change-requires-runtime`; that result requires the frozen scientific
protocol before any correctness or performance claim. The screen itself never
creates fixtures or launches a device.

Functional tests cover metadata selection and public CLI rejection, host pointer
ABI and independent state, actual ELF code/resource changes, evidence retention,
relative dependency paths and working-directory recovery:

```bash
python -m unittest discover -s tests -p test_optimization_shape_metadata.py -v
```

The outcome and retained artifact bindings are in
[shape_metadata-evidence.json](shape_metadata-evidence.json). The A3 compiler
screen establishes code-generation equivalence for this qualified source and
compiler, not numerical runtime correctness or behavior on another product.
