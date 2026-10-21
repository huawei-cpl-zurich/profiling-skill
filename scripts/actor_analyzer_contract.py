"""Small, transport-neutral contracts for alternating candidate and analysis roles.

Validation establishes attribution and structure, not the truth of an agent's
reasoning. The controller must verify retained evidence and correctness itself.
"""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import re
from string import Template

ACTOR_BUDGET_SECONDS = 1200
ANALYZER_BUDGET_SECONDS = 600
DEFAULT_ROUNDS = 4
PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "actor-analyzer"
COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _texts(value, label, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label} must be a list{' with entries' if nonempty else ''}")
    for item in value:
        _text(item, label)
    return value


def _number(value, label, *, positive=False):
    try:
        finite = isinstance(value, (float, int)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not finite or value < 0 or (positive and value == 0)):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _objects(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    for item in value:
        _object(item, label)
    return value


def build_prompt(role, budget_seconds, round_index, task_instruction, candidate_commit=None):
    """Render a versioned role prompt; task text is data, never a template."""
    if role not in {"actor", "analyzer"}:
        raise ValueError("role must be actor or analyzer")
    for label, value in (("budget_seconds", budget_seconds), ("round_index", round_index)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    _text(task_instruction, "task_instruction")
    if candidate_commit is not None and (not isinstance(candidate_commit, str)
                                          or not COMMIT.fullmatch(candidate_commit)):
        raise ValueError("candidate_commit must be a full Git object ID")
    if role == "analyzer" and candidate_commit is None:
        raise ValueError("analyzer requires a frozen candidate_commit")
    return Template((PROMPTS / f"{role}.md").read_text()).substitute(
        budget_seconds=budget_seconds, round_index=round_index,
        task_instruction=task_instruction,
        candidate_commit=candidate_commit or "No previous verified candidate")


def roofline(compute_ops, traffic_bytes, peak_ops_per_second,
             bandwidth_bytes_per_second, measured_us=None):
    """Ideal lower bound, not a prediction; caller supplies defensible ceilings.

    Peak compute must match product, dtype, operation, and counting convention.
    Traffic and bandwidth must describe the same memory boundary. ideal_fraction
    is deliberately uncapped: values above one expose inconsistent assumptions.
    """
    _number(compute_ops, "compute_ops")
    _number(traffic_bytes, "traffic_bytes")
    _number(peak_ops_per_second, "peak_ops_per_second", positive=True)
    _number(bandwidth_bytes_per_second, "bandwidth_bytes_per_second", positive=True)
    compute = compute_ops / peak_ops_per_second * 1e6
    memory = traffic_bytes / bandwidth_bytes_per_second * 1e6
    _number(compute, "compute_limit_us")
    _number(memory, "memory_limit_us")
    result = {"compute_limit_us": compute, "memory_limit_us": memory,
              "lower_bound_us": max(compute, memory)}
    if measured_us is not None:
        _number(measured_us, "measured_us", positive=True)
        result["ideal_fraction"] = result["lower_bound_us"] / measured_us
        _number(result["ideal_fraction"], "ideal_fraction")
    return result


def validate_analysis(report, expected_commit):
    """Reject unsupported attribution/structure; return the original report.

    Evidence paths are relative to the analyzer output directory. The controller
    checks existence, containment including symlinks, provenance, and actual
    profiler acquisition separately. Extra fields are preserved for provenance.
    """
    _object(report, "report")
    if not isinstance(expected_commit, str) or not COMMIT.fullmatch(expected_commit):
        raise ValueError("expected_commit must be a full Git object ID")
    if report.get("candidate_commit") != expected_commit:
        raise ValueError("analysis must identify the frozen candidate commit")
    profiling = _object(report.get("profiling"), "profiling")
    if profiling.get("status") not in {"acquired", "unavailable"}:
        raise ValueError("profiling.status must be acquired or unavailable")
    if profiling["status"] == "unavailable":
        _text(profiling.get("reason"), "profiling.reason")
    paths = set()
    has_profile = False
    for item in _objects(report.get("evidence"), "evidence"):
        path = _text(item.get("path"), "evidence.path")
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in path or path == ".":
            raise ValueError("evidence.path must be a contained relative file path")
        if path in paths:
            raise ValueError("evidence paths must be unique")
        paths.add(path)
        if item.get("kind") not in {"profile", "timing", "compiler"}:
            raise ValueError("evidence.kind must be profile, timing, or compiler")
        _texts(item.get("kernels"), "evidence.kernels", nonempty=True)
        _texts(item.get("cases"), "evidence.cases", nonempty=True)
        has_profile |= item["kind"] == "profile"
    if has_profile != (profiling["status"] == "acquired"):
        raise ValueError("profiling status must agree with retained profile evidence")
    for field in ("observations", "bottlenecks"):
        for claim in _objects(report.get(field), field):
            _text(claim.get("finding"), f"{field}.finding")
            references = _texts(claim.get("evidence"), f"{field}.evidence", nonempty=True)
            if not set(references) <= paths:
                raise ValueError(f"{field} refers to unlisted evidence")
            if claim.get("confidence") not in {"high", "medium", "low"}:
                raise ValueError(f"{field}.confidence must be high, medium, or low")
    for ceiling in _objects(report.get("ceilings"), "ceilings"):
        _text(ceiling.get("resource"), "ceilings.resource")
        _text(ceiling.get("basis"), "ceilings.basis")
        estimated = ceiling.get("status") == "estimated"
        if ceiling.get("status") not in {"estimated", "not_established"}:
            raise ValueError("ceilings.status must be estimated or not_established")
        _texts(ceiling.get("assumptions"), "ceilings.assumptions", nonempty=estimated)
        if estimated:
            _number(ceiling.get("lower_bound_us"), "ceilings.lower_bound_us", positive=True)
            if "measured_us" in ceiling:
                _number(ceiling["measured_us"], "ceilings.measured_us", positive=True)
        elif "lower_bound_us" in ceiling or "measured_us" in ceiling:
            raise ValueError("not_established ceilings cannot carry numeric estimates")
    _texts(report.get("unknowns"), "unknowns")
    return report
