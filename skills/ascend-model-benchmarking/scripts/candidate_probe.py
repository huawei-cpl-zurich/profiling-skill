"""Private controller probe; return only the explicitly constructed public result.

Run in a process isolated from the agent. ``evidence_dir`` is private, never an
agent-mounted directory. This is a diagnostic boundary, not a Python sandbox.
Transport, process deadlines, device leases and retries belong to the caller.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import traceback

from model_execution import CapabilityError, GraphCompiler, ModelExecutionError, prepare_model


def _chain(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def _compiler_diagnostic(error, candidate_file):
    """Only release a real Triton error whose AST source belongs to candidate.

    Only the compiler's own message and candidate-owned source are released, not
    its formatted traceback, wrapper messages or arbitrary nested exceptions.
    Call exclusively from candidate-preparation, never from private validation.
    """
    try:
        errors = importlib.import_module("triton.compiler.errors")
        compiler_type = errors.CompilationError
        source = Path(candidate_file).read_text()
    except (ImportError, AttributeError, OSError):
        return None
    for item in _chain(error):
        if not isinstance(item, compiler_type):
            continue
        fragment = getattr(item, "src", None)
        node = getattr(item, "node", None)
        if not isinstance(fragment, str):
            continue
        if not isinstance(node, ast.AST):
            # Current Triton stores line/column, not the original node object.
            try:
                node = next(part for part in ast.walk(ast.parse(fragment))
                            if getattr(part, "lineno", None) == getattr(item, "line", None)
                            and getattr(part, "col_offset", None) == getattr(item, "column", None))
            except (SyntaxError, StopIteration):
                continue
        # Exact whole-line source ownership; no traceback-frame string matching.
        if not fragment.strip() or source.count(fragment) != 1:
            continue
        offset = source.index(fragment)
        if offset and source[offset - 1] != "\n":
            continue
        line_in_fragment = getattr(node, "lineno", 0)
        lines = fragment.splitlines()
        if not 1 <= line_in_fragment <= len(lines):
            continue
        message = getattr(item, "error_message", getattr(item, "error_msg", ""))
        if not isinstance(message, str) or not message or len(message) > 4096:
            continue
        return {"kind": "triton-compilation", "message": message,
                "file": Path(candidate_file).name,
                "line": source[:offset].count("\n") + line_in_fragment,
                "source_line": lines[line_in_fragment - 1]}
    return None


def _frontend_diagnostic(error, candidate_file):
    """Triton dependency discovery can raise raw AttributeError before compilation.

    Authenticate the real visitor and its AST, then release only a reconstructed
    missing-language-member message owned by one expression in the candidate.
    """
    try:
        jit = importlib.import_module('triton.runtime.jit')
        language = importlib.import_module('triton.language')
        source = Path(candidate_file).read_text()
        tree = ast.parse(source)
    except (ImportError, AttributeError, OSError, SyntaxError):
        return None
    aliases = {alias.asname or alias.name for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names
               if alias.name == 'triton.language'}
    for item in _chain(error):
        if not isinstance(item, AttributeError) or getattr(item, 'obj', None) is not language:
            continue
        trace = item.__traceback__
        while trace:
            frame = trace.tb_frame
            node = frame.f_locals.get('node')
            if frame.f_globals.get('__name__') == 'triton.runtime.jit' \
                    and frame.f_code.co_name == 'visit_Attribute' \
                    and isinstance(frame.f_locals.get('self'), jit.DependencyFinder) \
                    and isinstance(node, ast.Attribute) and getattr(item, 'name', None) == node.attr:
                matches = [part for part in ast.walk(tree) if isinstance(part, ast.Attribute)
                    and isinstance(part.value, ast.Name) and part.value.id in aliases
                    and ast.dump(part, include_attributes=False) == ast.dump(node, include_attributes=False)]
                if len(matches) == 1:
                    line = matches[0].lineno
                    return dict(kind='triton-frontend', file=Path(candidate_file).name, line=line,
                        source_line=source.splitlines()[line - 1],
                        message="module 'triton.language' has no attribute '" + node.attr + "'")
            trace = trace.tb_next
    return None


def _numeric_summary(actual, expected):
    """Single-tensor aggregate diagnostics; never serialize values or reprs."""
    torch = importlib.import_module("torch")
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        return None
    result = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
              "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype),
              "actual_finite": bool(torch.isfinite(actual).all().item()),
              "expected_finite": bool(torch.isfinite(expected).all().item())}
    if actual.shape == expected.shape and actual.numel():
        a, b = actual.detach().to(torch.float64), expected.detach().to(torch.float64)
        difference = (a - b).abs()
        relative = difference / b.abs().clamp_min(torch.finfo(torch.float64).eps)
        # JSON has no portable NaN/Infinity. Nonfinite is already explicit above.
        for name, value in (("max_abs", difference.max()), ("max_relative", relative.max())):
            result[name] = float(value.item()) if bool(torch.isfinite(value).item()) else None
    return result


def _resource_diagnostic(error):
    try:
        resource_type = importlib.import_module("triton.runtime.errors").OutOfResources
    except (ImportError, AttributeError):
        return None
    resources = {"shared memory", "registers", "tensor memory", "local memory",
                 "ub", "ubuffer", "unified buffer", "l1", "l0a", "l0b", "l0c"}
    for item in _chain(error):
        if not isinstance(item, resource_type):
            continue
        required, limit = getattr(item, "required", None), getattr(item, "limit", None)
        name = getattr(item, "name", None)
        if (type(required) is int and type(limit) is int and required >= 0 and limit >= 0
                and isinstance(name, str) and name.lower() in resources):
            return {"kind": "triton-resource-limit", "resource": name.lower(),
                    "required": required, "limit": limit}
    return None


def probe_candidate(factory, case_provider, validator, *, candidate_file,
                    evidence_dir, backend, reference=None, synchronize=lambda: None,
                    compiler=None):
    """Compile and validate private cases, retaining errors without leaking them.

    Provider yields ``(args, kwargs)``. Validator receives output,args,kwargs.
    Optional trusted reference(args,kwargs) returns an independently computed
    tensor, used *only* for aggregate diagnostics after validator failure. A
    validator failure is not itself proof of a candidate defect: classify it as
    correctness failure only when independently computed tensor aggregates differ.
    Compiler injection exists for functional CPU tests; production passes none.
    No retry/attempt counter is consumed here and no timings are claimed.
    """
    private = Path(evidence_dir)
    private.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = {"state": "failed", "category": "private-controller-error"}
    stage = "private-case-provider"
    captured = {}
    evidence = []
    try:
        for args, kwargs in case_provider():
            captured.clear()
            stage = "candidate-preparation"

            def validate(output):
                nonlocal stage
                captured["output"] = output
                stage = "private-validator"
                validator(output, args, kwargs)

            selected = compiler or GraphCompiler(backend_name=backend)
            if selected.backend_name != backend:
                raise ValueError("injected compiler does not match explicit backend")
            # Cold eager diagnostic preflight exposes frontend errors that some
            # graph backends suppress. It never substitutes for compiled proof.
            stage = "candidate-eager-preflight"
            torch = importlib.import_module("torch")
            leaves, _ = torch.utils._pytree.tree_flatten((args, kwargs))
            snapshots = [(item, item.detach().clone()) for item in leaves
                         if isinstance(item, torch.Tensor)]
            with selected.torch.inference_mode():
                model = factory()
                if hasattr(model, "eval"):
                    model.eval()
                output = model(*args, **kwargs)
                synchronize()
            validate(output)
            stage = "candidate-input-mutation"
            for original, snapshot in snapshots:
                torch.testing.assert_close(original, snapshot, rtol=0, atol=0, equal_nan=True)
            stage = "candidate-preparation"
            prepared = prepare_model(lambda: model, args, kwargs, validate,
                                     compiler=selected, synchronize=synchronize)
            captured["evidence"] = prepared.evidence
            evidence.append(prepared.evidence)
            stage = "private-case-provider"
        if "evidence" not in captured:
            raise ValueError("private case provider yielded no cases")
        (private / "compiled-evidence.json").write_text(json.dumps(evidence, indent=2))
        return {"state": "passed", "category": "validated-compiled-candidate",
                "backend": backend, "fullgraph": True, "eager_preflight": True,
                "compile_count": sum(item["compile_count"] for item in evidence),
                "compiled_executions": sum(item["compiled_executions"] for item in evidence)}
    except Exception as error:
        retained = traceback.format_exc()
        (private / "traceback.txt").write_text(retained)
        digest = hashlib.sha256(retained.encode()).hexdigest()
        result.update(private_evidence_sha256=digest, phase=stage)
        if stage == "candidate-preparation" and isinstance(error, CapabilityError):
            result["category"] = "capability-unavailable"
        elif stage in ("candidate-eager-preflight", "candidate-input-mutation") or (stage == "candidate-preparation" and isinstance(error, ModelExecutionError)):
            result["category"] = "candidate-compile-or-runtime-error"
            detail = (_compiler_diagnostic(error, candidate_file)
                      or _frontend_diagnostic(error, candidate_file) or _resource_diagnostic(error))
            if detail:
                result["diagnostic"] = detail
            else:
                result["diagnostic_unavailable"] = "no authenticated candidate-owned Triton AST diagnostic; full compiler/runtime evidence retained privately"
        elif stage == "private-validator" and reference is not None:
            try:
                summary = _numeric_summary(captured["output"], reference(args, kwargs))
                if summary is not None:
                    result["numeric"] = summary
                    if (summary["actual_shape"] != summary["expected_shape"] or
                            summary["actual_dtype"] != summary["expected_dtype"] or
                            not summary["actual_finite"] or summary.get("max_abs", 0) != 0):
                        result["category"] = "candidate-correctness-error"
            except Exception:
                (private / "reference-diagnostic-error.txt").write_text(traceback.format_exc())
        (private / "public-result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
