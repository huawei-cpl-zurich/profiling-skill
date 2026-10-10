import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fully_fused_contract", ROOT / "scripts" / "fully_fused_contract.py"
)
contract = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(contract)


def manifest():
    return {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "complete_kernel_mix_aiv",
        "entrypoint": "complete_kernel",
        "fusion": {
            "schema_version": 1,
            "mode": "single-logical-launch",
            "complete_operator": True,
        },
    }


def case_evidence(case, components=("aic", "aiv")):
    return {
        "schema": "profiling-skill/fusion-evidence/v1",
        "case": case,
        "output_launch_id": "launch-0",
        "operators": [
            {
                "name": f"complete_kernel_mix_{component}",
                "origin": "triton",
                "entrypoint": "complete_kernel",
                "launch_id": "launch-0",
                "component": component,
            }
            for component in components
        ],
    }


def test_accepts_one_complete_mixed_core_launch_over_every_case():
    parsed = contract.validate_manifest(manifest())
    result = contract.validate_fusion_evidence(
        parsed, [case_evidence(3), case_evidence(8)], expected_cases=[3, 8]
    )
    assert result == {
        "entrypoint": "complete_kernel",
        "kernel_name": "complete_kernel_mix_aiv",
        "cases": [3, 8],
        "logical_launches_per_case": 1,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(operators=[]), "exactly one logical Triton launch"),
        (
            lambda row: row["operators"].append(
                {"name": "other", "origin": "triton", "entrypoint": "other",
                 "launch_id": "launch-1", "component": "aiv"}
            ),
            "exactly one logical Triton launch",
        ),
        (
            lambda row: row["operators"].append(
                {"name": "aten::matmul", "origin": "torch", "launch_id": "framework-1"}
            ),
            "Torch/ACL compute",
        ),
        (
            lambda row: row["operators"].append(
                {"name": "aclnnIndexPut", "origin": "acl", "launch_id": "framework-2"}
            ),
            "Torch/ACL compute",
        ),
        (
            lambda row: row["operators"][0].update(entrypoint="output_cast"),
            "unrelated entrypoint",
        ),
        (
            lambda row: row.update(output_launch_id="reference-output"),
            "output is not produced",
        ),
    ],
)
def test_rejects_bypass_fallback_partial_or_auxiliary_launches(mutate, message):
    evidence = case_evidence(3, components=("aiv",))
    mutate(evidence)
    with pytest.raises(contract.FusionContractError, match=message):
        contract.validate_fusion_evidence(manifest(), [evidence], expected_cases=[3])


def test_rejects_missing_duplicate_or_unrequested_case_evidence():
    with pytest.raises(contract.FusionContractError, match="case coverage"):
        contract.validate_fusion_evidence(
            manifest(), [case_evidence(3)], expected_cases=[3, 8]
        )
    with pytest.raises(contract.FusionContractError, match="case coverage"):
        contract.validate_fusion_evidence(
            manifest(), [case_evidence(3), case_evidence(3)], expected_cases=[3]
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(schema="profiling-skill/candidate-kernel/v1"),
        lambda value: value.pop("entrypoint"),
        lambda value: value.update(entrypoint="different"),
        lambda value: value["fusion"].update(complete_operator=False),
        lambda value: value["fusion"].update(mode="multi-launch"),
        lambda value: value["fusion"].update(extra=True),
    ],
)
def test_manifest_requires_exact_versioned_full_fusion_declaration(change):
    value = manifest()
    change(value)
    with pytest.raises(contract.FusionContractError):
        contract.validate_manifest(value)


def test_cli_validates_a_complete_fixture_and_emits_compact_result(tmp_path, capsys):
    manifest_path = tmp_path / "candidate.manifest.json"
    evidence_path = tmp_path / "fusion-evidence.json"
    manifest_path.write_text(json.dumps(manifest()))
    evidence_path.write_text(json.dumps({"cases": [case_evidence(3), case_evidence(8)]}))
    assert contract.main([
        "--manifest", str(manifest_path), "--evidence", str(evidence_path),
        "--expected-cases", "3,8",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
