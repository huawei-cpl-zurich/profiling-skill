#!/usr/bin/env python3
"""Prepare, gate, and summarize the fully-fused A3/A5 campaign.

This module is intentionally not a second scheduler.  It is the fail-closed
campaign boundary around ``audited_campaign``: archive and matrix identities
are checked before launch, known-good product gates are checked before any
candidate cell is admitted, and terminal evidence is reduced to reviewable
tables only after all branch histories have been independently verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import tarfile
from pathlib import Path
from typing import Callable


PRODUCTS = ("a3", "a5")
TASKS = ("matmul", "gdn", "bsa")
TREATMENTS = (
    "cannbot-all", "cannbot-new-profiler",
    "guarded-new-profiler", "guarded-old-profiler",
)
MATMUL_FULL_CASES = tuple(range(10))
ARCHIVE_SCHEMA = "profiling-skill/pre-campaign-archive/v1"
PREPARED_SCHEMA = "profiling-skill/fully-fused-campaign-prepared/v1"
GATE_SCHEMA = "profiling-skill/fully-fused-matmul-gates/v1"
ADMISSION_SCHEMA = "profiling-skill/fully-fused-campaign-admission/v1"
REPORT_SCHEMA = "profiling-skill/fully-fused-campaign-report/v1"
RECEIPT_DIGEST_FIELDS = {
    ARCHIVE_SCHEMA: "seal_sha256", PREPARED_SCHEMA: "prepared_sha256",
    GATE_SCHEMA: "gate_sha256", REPORT_SCHEMA: "report_sha256",
}
VALIDATION_BUNDLE_PATHS = (
    "scripts", "tests", "docs", "benchmarks", "prompts", "references",
    "experiments", "SKILL.md",
)


class ExecutionError(RuntimeError):
    """A campaign invariant is absent or has drifted."""


def document_sha256(document: dict) -> str:
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def receipt_sha256(document: dict, digest_field: str | None = None) -> str:
    expected = RECEIPT_DIGEST_FIELDS.get(document.get("schema"))
    if expected is None or digest_field not in {None, expected}:
        raise ExecutionError("receipt schema or digest field is unsupported")
    return document_sha256({key: value for key, value in document.items()
                            if key != expected})


def _verify_receipt_digest(document: object, schema: str) -> dict:
    field = RECEIPT_DIGEST_FIELDS[schema]
    if (not isinstance(document, dict) or document.get("schema") != schema
            or not _digest(document.get(field))
            or document[field] != receipt_sha256(document)):
        label = {
            ARCHIVE_SCHEMA: "archive seal", PREPARED_SCHEMA: "prepared",
            GATE_SCHEMA: "gate", REPORT_SCHEMA: "report",
        }[schema]
        raise ExecutionError(f"{label} receipt digest is invalid")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def deterministic_tree_digest(root: Path) -> str:
    """Match the deterministic tar hash recorded by the archive procedure."""
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ExecutionError(f"archived root is unavailable: {root}")
    # The retained archive records the path relative to the orchestration
    # workspace (including ``.agent-state/runtime``), so that path is part of
    # the authenticated tar stream.  Non-workspace callers use the leaf name.
    workspace = next(
        (parent for parent in (root, *root.parents)
         if (parent / ".agent-state").is_dir()),
        root.parent,
    )
    archive_name = str(root.relative_to(workspace)) if workspace != root.parent else root.name
    command = [
        "tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0",
        "--numeric-owner", "-cf", "-", archive_name,
    ]
    digest = hashlib.sha256()
    try:
        process = subprocess.Popen(
            command, cwd=workspace, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(chunk)
        _, stderr = process.communicate()
    except OSError as error:
        raise ExecutionError(f"cannot hash archived root: {error}") from error
    if process.returncode:
        raise ExecutionError(
            f"cannot hash archived root: {stderr.decode(errors='replace').strip()}"
        )
    return digest.hexdigest()


def build_validation_bundle(repository: Path, destination: Path) -> Path:
    """Build the self-contained functional-test bundle used by remote gates."""
    repository = repository.resolve()
    missing = [name for name in VALIDATION_BUNDLE_PATHS
               if not (repository / name).exists()]
    if missing:
        raise ExecutionError(
            "validation bundle inputs are incomplete: " + ", ".join(missing)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as archive:
        for name in VALIDATION_BUNDLE_PATHS:
            archive.add(repository / name, arcname=name, recursive=True)
    return destination


def verify_archive_seal(seal: dict) -> dict:
    """Recompute every zero-copy root and completion-file binding."""
    try:
        _verify_receipt_digest(seal, ARCHIVE_SCHEMA)
    except ExecutionError as error:
        raise ExecutionError("archive seal is malformed or unauthenticated") from error
    if not isinstance(seal, dict):
        raise ExecutionError("archive seal is malformed or unauthenticated")
    roots, completions = seal.get("roots"), seal.get("completion_files")
    if not isinstance(roots, list) or not roots:
        raise ExecutionError("archive seal contains no frozen roots")
    if not isinstance(completions, list) or not completions:
        raise ExecutionError("archive seal contains no completion files")
    seen: set[Path] = set()
    for binding in roots:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ExecutionError("archive root binding is malformed")
        path = Path(binding["path"])
        if not path.is_absolute() or path in seen or not _digest(binding["sha256"]):
            raise ExecutionError("archive root binding is not unique and pinned")
        seen.add(path)
        if deterministic_tree_digest(path) != binding["sha256"]:
            raise ExecutionError(f"archived root drifted: {path}")
    for binding in completions:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ExecutionError("archive completion binding is malformed")
        path = Path(binding["path"])
        if (not path.is_absolute() or not path.is_file() or path.is_symlink()
                or not _digest(binding["sha256"])
                or _sha256(path) != binding["sha256"]):
            raise ExecutionError(f"archive completion file drifted: {path}")
    return {
        "schema": ARCHIVE_SCHEMA, "seal_sha256": seal["seal_sha256"],
        "root_count": len(roots), "completion_file_count": len(completions),
    }


def _validate_rankings(rankings: dict) -> dict:
    if (not isinstance(rankings, dict) or set(rankings) != set(PRODUCTS)
            or any(not isinstance(rankings[product], dict)
                   or set(rankings[product]) != set(TASKS)
                   for product in PRODUCTS)):
        raise ExecutionError("product-ranked case rankings must cover both products and all tasks")
    normalized: dict[str, dict[str, list[int]]] = {}
    for product in PRODUCTS:
        normalized[product] = {}
        for task in TASKS:
            cases = rankings[product][task]
            if (not isinstance(cases, list) or not cases
                    or len(cases) != len(set(cases))
                    or any(type(case) is not int or case < 0 for case in cases)):
                raise ExecutionError(f"rankings for {product}/{task} are invalid")
            normalized[product][task] = list(cases)
    return normalized


def prepare_campaign(manifest: dict, rankings: dict, *, archive_attestation: str) -> dict:
    """Bind the exact matrix, branch lineage, archive, and ranked cases."""
    dimensions = manifest.get("dimensions") if isinstance(manifest, dict) else None
    cells = manifest.get("cells") if isinstance(manifest, dict) else None
    if (not isinstance(dimensions, dict) or not isinstance(cells, list)
            or dimensions.get("products") != list(PRODUCTS)
            or dimensions.get("tasks") != list(TASKS)
            or dimensions.get("treatments") != list(TREATMENTS)
            or dimensions.get("round_count") != 4 or len(cells) != 24):
        raise ExecutionError("campaign requires the exact 24-cell product/task/treatment matrix")
    expected = {(product, task, treatment) for product in PRODUCTS
                for task in TASKS for treatment in TREATMENTS}
    actual = {(cell.get("product"), cell.get("task"), cell.get("treatment"))
              for cell in cells if isinstance(cell, dict)}
    if actual != expected:
        raise ExecutionError("campaign requires the exact 24-cell product/task/treatment matrix")
    if any(cell.get("round_count") != 4 for cell in cells):
        raise ExecutionError("every campaign cell requires exactly four rounds")
    branches = [cell.get("branch") for cell in cells]
    if (any(not isinstance(branch, str) or not branch.startswith("experiment/")
            for branch in branches) or len(set(branches)) != 24):
        raise ExecutionError("every campaign cell requires a unique unmerged branch")
    if not _digest(archive_attestation):
        raise ExecutionError("archive attestation must be a SHA-256 digest")
    ranked = _validate_rankings(rankings)
    prepared = {
        "schema": PREPARED_SCHEMA, "run_id": manifest.get("run_id"),
        "archive_attestation": archive_attestation,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "cell_count": 24, "rounds_per_cell": 4,
        "expected_round_commits": 96, "branches": branches,
        "ranked_cases": ranked,
        "ranked_cases_sha256": document_sha256({"ranked_cases": ranked}),
    }
    prepared["prepared_sha256"] = receipt_sha256(prepared)
    return prepared


def verify_prepared_receipt(value: object) -> dict:
    receipt = _verify_receipt_digest(value, PREPARED_SCHEMA)
    branches = receipt.get("branches")
    if (receipt.get("cell_count") != 24 or receipt.get("rounds_per_cell") != 4
            or receipt.get("expected_round_commits") != 96
            or not _digest(receipt.get("archive_attestation"))
            or not _digest(receipt.get("manifest_sha256"))
            or not isinstance(receipt.get("run_id"), str) or not receipt["run_id"]
            or not isinstance(branches, list) or len(branches) != 24
            or len(set(branches)) != 24
            or any(not isinstance(branch, str)
                   or not branch.startswith(f"experiment/{receipt.get('run_id')}/")
                   for branch in branches)):
        raise ExecutionError("prepared campaign receipt contract is invalid")
    _validate_rankings(receipt.get("ranked_cases"))
    if receipt.get("ranked_cases_sha256") != document_sha256({
        "ranked_cases": receipt["ranked_cases"],
    }):
        raise ExecutionError("prepared campaign ranked cases digest is invalid")
    return receipt


def validate_matmul_gate(product: str, ranked_cases: list[int], receipt: dict) -> dict:
    """Validate one known-good fused matmul check before candidate dispatch."""
    if receipt.get("status") == "infrastructure_error":
        handle = receipt.get("handle")
        suffix = f"; observe retained handle {handle}" if handle else ""
        raise ExecutionError(f"{product} known-good gate is infrastructure pending{suffix}")
    if (receipt.get("status") != "ok" or receipt.get("product") != product
            or receipt.get("task") != "matmul"
            or not isinstance(receipt.get("handle"), str) or not receipt["handle"]):
        raise ExecutionError(f"{product} known-good matmul gate failed")
    samples, median = receipt.get("samples_us"), receipt.get("median_us")
    if (not isinstance(samples, list) or len(samples) != 3
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) or value <= 0 for value in samples)
            or not isinstance(median, (int, float)) or isinstance(median, bool)
            or not math.isclose(statistics.median(samples), median, rel_tol=1e-12)):
        raise ExecutionError(f"{product} known-good gate lacks deterministic timing")
    fusion = receipt.get("fusion_gate")
    cases = fusion.get("cases") if isinstance(fusion, dict) else None
    if (not isinstance(fusion, dict) or cases != list(MATMUL_FULL_CASES)
            or any(case not in cases for case in ranked_cases)
            or fusion.get("logical_launches_per_case") != 1
            or not fusion.get("entrypoint") or not fusion.get("kernel_name")):
        raise ExecutionError(f"{product} known-good matmul is not fully fused")
    return {
        "product": product, "handle": receipt["handle"],
        "median_us": float(median), "samples_us": [float(value) for value in samples],
        "variability_ratio": receipt.get("variability_ratio"),
        "fusion_gate": fusion,
    }


def run_matmul_gates(rankings: dict, probe: Callable[[str, list[int]], dict], *,
                     prepared_sha256: str) -> dict:
    ranked = _validate_rankings(rankings)
    if not _digest(prepared_sha256):
        raise ExecutionError("matmul gates require a prepared receipt digest")
    results = [validate_matmul_gate(
        product, ranked[product]["matmul"], probe(product, ranked[product]["matmul"]),
    ) for product in PRODUCTS]
    receipt = {"schema": GATE_SCHEMA, "status": "passed",
               "prepared_sha256": prepared_sha256, "products": results}
    receipt["gate_sha256"] = receipt_sha256(receipt)
    return receipt


def verify_gate_receipt(value: object, rankings: dict, *, prepared_sha256: str) -> dict:
    receipt = _verify_receipt_digest(value, GATE_SCHEMA)
    if (receipt.get("status") != "passed"
            or receipt.get("prepared_sha256") != prepared_sha256
            or not isinstance(receipt.get("products"), list)
            or len(receipt["products"]) != len(PRODUCTS)):
        raise ExecutionError("matmul gate receipt is not bound to the prepared campaign")
    ranked = _validate_rankings(rankings)
    by_product = {row.get("product"): row for row in receipt["products"]
                  if isinstance(row, dict)}
    if set(by_product) != set(PRODUCTS):
        raise ExecutionError("matmul gate receipt does not cover both products")
    for product in PRODUCTS:
        row = by_product[product]
        validate_matmul_gate(product, ranked[product]["matmul"], {
            "status": "ok", "product": product, "task": "matmul", **row,
        })
    return receipt


def campaign_admission_identity(archive: object, prepared: object,
                                gates: object) -> dict:
    """Build the terminal identity in the one-way campaign hash chain.

    Admission is deliberately not manifest provenance: the prepared receipt
    authenticates the already-frozen manifest and the gate receipt
    authenticates preparation.  The pinned runtime config is created last and
    authenticates this terminal identity, so no hash depends on itself.
    """
    archive_result = verify_archive_seal(archive)
    prepared_receipt = verify_prepared_receipt(prepared)
    gate_receipt = verify_gate_receipt(
        gates, prepared_receipt["ranked_cases"],
        prepared_sha256=prepared_receipt["prepared_sha256"],
    )
    return validated_campaign_admission_identity(
        archive_result, prepared_receipt, gate_receipt,
    )


def validated_campaign_admission_identity(archive_result: dict,
                                           prepared_receipt: dict,
                                           gate_receipt: dict) -> dict:
    """Derive identity from receipts already validated in the same operation."""
    if prepared_receipt["archive_attestation"] != archive_result["seal_sha256"]:
        raise ExecutionError("prepared campaign is not bound to the archive seal")
    return {
        "schema": ADMISSION_SCHEMA,
        "manifest_sha256": prepared_receipt["manifest_sha256"],
        "archive_seal_sha256": archive_result["seal_sha256"],
        "prepared_sha256": prepared_receipt["prepared_sha256"],
        "gate_sha256": gate_receipt["gate_sha256"],
    }


def _failure_outcome(row: dict) -> str:
    if row.get("status") != "candidate_failed":
        return str(row.get("status"))
    failure = row.get("failure") or row.get("candidate_errors")
    if isinstance(failure, list) and failure:
        kind = failure[-1].get("failure_type") if isinstance(failure[-1], dict) else None
        return f"candidate_failed:{kind or 'candidate_error'}"
    if isinstance(failure, dict):
        return f"candidate_failed:{failure.get('failure_type', 'candidate_error')}"
    return "candidate_failed:candidate_error"


def compact_report(manifest: dict, report: dict) -> dict:
    """Validate terminal lineage and construct product-specific review tables."""
    cells = report.get("cells") if isinstance(report, dict) else None
    if not isinstance(report, dict) or report.get("schema_version") != 2:
        raise ExecutionError("terminal report schema is unsupported")
    if report.get("run_id") != manifest.get("run_id"):
        raise ExecutionError("terminal report run identity does not match the manifest")
    if report.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ExecutionError("terminal report manifest identity does not match")
    if not isinstance(cells, list) or len(cells) != 24:
        raise ExecutionError("terminal report must contain all 24 cells")
    expected = {cell["cell_id"]: cell for cell in manifest["cells"]}
    if {row.get("cell_id") for row in cells} != set(expected):
        raise ExecutionError("terminal report cells do not match the frozen manifest")
    tables = {product: [] for product in PRODUCTS}
    commit_count = 0
    for row in cells:
        frozen = expected[row["cell_id"]]
        if any(row.get(field) != frozen[field]
               for field in ("product", "task", "treatment", "branch")):
            raise ExecutionError("terminal report branch lineage does not match the manifest")
        commits = row.get("commits")
        if (not isinstance(commits, list) or len(commits) != 4
                or len(set(commits)) != 4
                or any(not isinstance(commit, str) or len(commit) != 40
                       or any(character not in "0123456789abcdef" for character in commit)
                       for commit in commits)):
            raise ExecutionError(f"{row['cell_id']} lacks four unique round commits")
        commit_count += len(commits)
        fusion = row.get("fusion_gate")
        fusion_text = ("1 logical launch/case" if isinstance(fusion, dict)
                       and fusion.get("logical_launches_per_case") == 1 else None)
        if (row.get("status") == "complete"
                and isinstance(row.get("comparison_median_us"), (int, float))
                and fusion_text is None):
            raise ExecutionError(
                f"{row['cell_id']} has timing without a successful fusion proof"
            )
        tables[row["product"]].append({
            "task": row["task"], "treatment": row["treatment"],
            "outcome": _failure_outcome(row),
            "raw_baseline_us": row.get("baseline_median_us"),
            "raw_candidate_us": row.get("comparison_median_us"),
            "speedup": row.get("speedup_vs_baseline"),
            "variability_ratio": row.get("variability_ratio"),
            "fusion": fusion_text, "profiler_evidence": row.get("profiler_evidence", []),
            "bottleneck": row.get("bottleneck"), "best_round": row.get("best_round"),
            "branch": row["branch"], "commits": commits,
        })
    if commit_count != 96:
        raise ExecutionError("terminal report does not contain exactly 96 round commits")
    result = {
        "schema": REPORT_SCHEMA, "run_id": report.get("run_id"),
        "manifest_sha256": report.get("manifest_sha256"),
        "summary": report.get("summary"), "round_commit_count": commit_count,
        "infrastructure_exclusions": len(report.get("discarded_infrastructure_attempts", [])),
        "tables": tables,
    }
    result["report_sha256"] = receipt_sha256(result)
    return result


def verify_compact_report(value: object) -> dict:
    receipt = _verify_receipt_digest(value, REPORT_SCHEMA)
    tables = receipt.get("tables")
    if (not isinstance(receipt.get("run_id"), str) or not receipt["run_id"]
            or not _digest(receipt.get("manifest_sha256"))
            or receipt.get("round_commit_count") != 96
            or not isinstance(receipt.get("infrastructure_exclusions"), int)
            or not isinstance(tables, dict) or set(tables) != set(PRODUCTS)
            or any(not isinstance(tables[product], list)
                   or len(tables[product]) != 12 for product in PRODUCTS)):
        raise ExecutionError("compact report receipt contract is invalid")
    commits = []
    for product in PRODUCTS:
        for row in tables[product]:
            row_commits = row.get("commits") if isinstance(row, dict) else None
            if (not isinstance(row_commits, list) or len(row_commits) != 4
                    or len(set(row_commits)) != 4
                    or any(not isinstance(commit, str) or len(commit) != 40
                           or any(character not in "0123456789abcdef"
                                  for character in commit)
                           for commit in row_commits)
                    or (row.get("outcome") == "complete"
                        and isinstance(row.get("raw_candidate_us"), (int, float))
                        and row.get("fusion") != "1 logical launch/case")):
                raise ExecutionError("compact report row contract is invalid")
            commits.extend(row_commits)
    if len(commits) != 96:
        raise ExecutionError("compact report lineage is incomplete")
    return receipt


def render_markdown(report: dict) -> str:
    lines = ["# Fully fused campaign results", ""]
    for product in PRODUCTS:
        lines += [f"## {product.upper()}", "", (
            "| Product | Task | Treatment | Outcome | Baseline us | Candidate us | "
            "Speedup | Variability | Fusion | Bottleneck | Best round | "
            "Profiler evidence | Branch | Round commits |"
        ), ("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | "
             "---: | --- | --- | --- |")]
        for row in report["tables"][product]:
            def value(name: str) -> str:
                item = row.get(name)
                return "-" if item is None else str(item)
            lines.append(
                f"| {product} | {row['task']} | {row['treatment']} | {row['outcome']} | "
                f"{value('raw_baseline_us')} | {value('raw_candidate_us')} | "
                f"{value('speedup')} | {value('variability_ratio')} | "
                f"{value('fusion')} | {value('bottleneck')} | {value('best_round')} | "
                f"{', '.join(row['profiler_evidence']) or '-'} | `{row['branch']}` | "
                f"{', '.join(f'`{commit}`' for commit in row['commits'])} |"
            )
        lines.append("")
    lines.append(
        f"Infrastructure attempts excluded: {report['infrastructure_exclusions']}; "
        f"round commits retained: {report['round_commit_count']}."
    )
    return "\n".join(lines) + "\n"


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutionError(f"JSON document must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    archive = commands.add_parser("verify-archive")
    archive.add_argument("--seal", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--rankings", type=Path, required=True)
    prepare.add_argument("--archive-seal", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--rankings", type=Path, required=True)
    gate.add_argument("--prepared", type=Path, required=True)
    gate.add_argument("--a3-receipt", type=Path, required=True)
    gate.add_argument("--a5-receipt", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--campaign-report", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-archive":
            result = verify_archive_seal(_load(args.seal))
        elif args.command == "prepare":
            seal = _load(args.archive_seal)
            attestation = verify_archive_seal(seal)
            manifest = _load(args.manifest)
            # Use the full scheduler validator at the CLI boundary.  Keeping
            # the pure preparation function independently testable lets dry
            # fixtures exercise its matrix invariants without fabricating all
            # scheduler provenance fields.
            try:
                from audited_campaign import CampaignError, verify_manifest
                verify_manifest(manifest)
            except (ImportError, CampaignError) as error:
                raise ExecutionError(f"campaign manifest is invalid: {error}") from error
            result = prepare_campaign(
                manifest, _load(args.rankings),
                archive_attestation=attestation["seal_sha256"],
            )
        elif args.command == "gate":
            receipts = {"a3": _load(args.a3_receipt), "a5": _load(args.a5_receipt)}
            prepared = verify_prepared_receipt(_load(args.prepared))
            result = run_matmul_gates(
                _load(args.rankings), lambda product, _: receipts[product],
                prepared_sha256=prepared["prepared_sha256"],
            )
        else:
            result = compact_report(_load(args.manifest), _load(args.campaign_report))
            if args.markdown:
                args.markdown.parent.mkdir(parents=True, exist_ok=True)
                args.markdown.write_text(render_markdown(result))
    except ExecutionError as error:
        parser.error(str(error))
    if args.command == "verify-archive":
        print(json.dumps(result, sort_keys=True))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
