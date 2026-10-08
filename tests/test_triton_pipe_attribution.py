from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "query_triton_pipe_attribution.py"
DATA = ROOT / "references" / "triton-pipe-attribution.json"
PROBE = ROOT / "benchmarks" / "triton_pipe_probe.py"


def run_cli(*arguments: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--mapping", str(DATA), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def run_inventory_cli(tmp_path: Path, inventory: dict, *arguments: str):
    path = tmp_path / "inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--mapping", str(path), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def load_query_module():
    spec = importlib.util.spec_from_file_location("triton_pipe_attribution_query", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_direct_copy_lookup_preserves_compiler_provenance_and_scope():
    run = run_cli("--construct", "gm_to_l1_copy", "--product", "a3")

    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == "direct"
    assert result["compiler_pipe"] == "PIPE_MTE2"
    assert result["profiler_pipe"] == "MTE2"
    assert result["profiler_pipe_status"] == "inferred"
    assert result["products"] == ["a3"]
    assert result["compiler_revisions"] == {
        "ascendnpuir": "af5499b3b9f3dbab50b2834bcfff5da5c2a1d920",
        "triton_ascend": "23ac2717c0a38ba962cbd4a0425fc069e7ae104d",
    }
    assert result["citations"]
    assert all(item.startswith("ref://") for item in result["citations"])


def test_all_source_proven_direct_mappings_are_queryable():
    expected = {
        "ub_to_ub_copy": "PIPE_V",
        "l0c_to_gm_copy": "PIPE_FIX",
        "gm_to_l1_copy": "PIPE_MTE2",
        "vbrc_to_l1": "PIPE_MTE2",
        "vbrc_to_ub": "PIPE_V",
    }

    for construct, pipe in expected.items():
        run = run_cli("--construct", construct, "--product", "a3")
        assert run.returncode == 0, run.stderr
        result = json.loads(run.stdout)
        assert result["status"] == "direct"
        assert result["compiler_pipe"] == pipe


def test_unresolved_construct_returns_unknown_without_likely_pipe():
    run = run_cli("--construct", "triton_dot", "--product", "a3")

    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == "unknown"
    assert result["compiler_pipe"] is None
    assert result["profiler_pipe"] is None
    assert result["why_unknown"]
    assert "microbenchmark" in result["next_evidence"]


def test_original_triton_op_is_not_promoted_from_lowering_sequence():
    run = run_cli("--construct", "triton_load", "--product", "a3")

    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == "unknown"
    assert result["lowering_path"][0]["stage"] == "ttir-adapter"
    assert result["lowering_path"][-1]["stage"] == "hivm-scheduling"
    assert all(step["status"] == "direct" for step in result["lowering_path"])


def test_a2_and_a5_stay_unknown_without_product_validation():
    for product in ("a2", "a5"):
        run = run_cli("--construct", "gm_to_l1_copy", "--product", product)
        assert run.returncode == 0, run.stderr
        result = json.loads(run.stdout)
        assert result["status"] == "unknown"
        assert result["compiler_pipe"] is None
        assert result["profiler_pipe"] is None


def test_construct_lookup_requires_explicit_product():
    run = run_cli("--construct", "gm_to_l1_copy")

    assert run.returncode == 2
    assert run.stdout == ""
    assert "--product is required with --construct" in run.stderr


def test_product_is_rejected_for_product_neutral_actions():
    for action in ("--list", "--validate"):
        run = run_cli(action, "--product", "a3")
        assert run.returncode == 2
        assert run.stdout == ""
        assert "--product is only valid with --construct" in run.stderr


def test_inventory_lists_direct_inferred_and_unknown_claims():
    run = run_cli("--list")

    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["schema_version"] == 1
    assert result["counts"]["direct"] == 5
    assert result["counts"]["unknown"] >= 8
    assert result["counts"]["inferred"] >= 5
    assert "triton_reduction" in result["constructs"]


def test_validate_binds_exact_inventory_identity():
    inventory = json.loads(DATA.read_text())
    run = run_cli("--validate")

    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {
        "status": "valid",
        "inventory_id": inventory["inventory_id"],
        "compiler_revisions": inventory["compiler_revisions"],
        "runtime_scope": inventory["runtime_scope"],
    }


def test_unknown_selector_fails_clearly_without_fabricating_a_mapping():
    run = run_cli("--construct", "not_in_inventory", "--product", "a3")

    assert run.returncode == 3
    result = json.loads(run.stdout)
    assert result == {
        "construct": "not_in_inventory",
        "status": "unknown",
        "why_unknown": "construct is not present in the reviewed attribution inventory",
    }


def test_validator_rejects_unknown_claim_with_a_pipe(tmp_path: Path):
    inventory = json.loads(DATA.read_text()) if DATA.exists() else {
        "schema_version": 1,
        "compiler_revisions": {
            "triton_ascend": "23ac2717c0a38ba962cbd4a0425fc069e7ae104d",
            "ascendnpuir": "af5499b3b9f3dbab50b2834bcfff5da5c2a1d920",
        },
        "lowering_path": [],
        "mappings": [],
    }
    inventory["mappings"] = [{
        "construct": "bad_unknown",
        "status": "unknown",
        "compiler_pipe": "PIPE_V",
        "profiler_pipe": None,
        "products": ["a3"],
        "citations": ["ref://example/source#file:L1-L2"],
        "why_unknown": "not established",
        "next_evidence": "microbenchmark",
    }]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(inventory))

    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--mapping", str(path), "--validate"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2
    assert "unknown mapping must not name a pipe" in run.stderr


def test_validator_rejects_invalid_runtime_scopes(tmp_path: Path):
    baseline = json.loads(DATA.read_text())
    invalid_scopes = (
        None,
        {"live_validated_products": [], "unvalidated_products": ["a2", "a3", "a5"]},
        {"live_validated_products": ["a3"], "unvalidated_products": ["a3", "a5"]},
        {"live_validated_products": ["a3"], "unvalidated_products": ["a2", "a9"]},
        {"live_validated_products": ["a3"], "unvalidated_products": ["a2"]},
    )
    for index, scope in enumerate(invalid_scopes):
        inventory = json.loads(json.dumps(baseline))
        if scope is None:
            inventory.pop("runtime_scope")
        else:
            inventory["runtime_scope"] = scope
        run = run_inventory_cli(tmp_path / str(index), inventory, "--validate")
        assert run.returncode == 2
        assert "runtime_scope" in run.stderr


def test_validator_rejects_direct_a5_mapping_outside_live_scope(tmp_path: Path):
    inventory = json.loads(DATA.read_text())
    inventory["mappings"][0]["products"].append("a5")

    run = run_inventory_cli(tmp_path, inventory, "--validate")

    assert run.returncode == 2
    assert "live_validated_products" in run.stderr


def test_query_fails_closed_for_direct_a5_edit_even_before_validation():
    module = load_query_module()
    inventory = json.loads(DATA.read_text())
    inventory["mappings"][0]["products"].append("a5")

    result = module.query(inventory, "ub_to_ub_copy", "a5")

    assert result["status"] == "unknown"
    assert result["compiler_pipe"] is None
    assert result["profiler_pipe"] is None


def test_query_fails_closed_without_a_product_even_before_cli_validation():
    module = load_query_module()
    inventory = json.loads(DATA.read_text())

    result = module.query(inventory, "gm_to_l1_copy", None)

    assert result["status"] == "unknown"
    assert result["compiler_pipe"] is None


def test_validator_requires_nonempty_inventory_identity(tmp_path: Path):
    for identity in (None, ""):
        inventory = json.loads(DATA.read_text())
        inventory["inventory_id"] = identity
        run = run_inventory_cli(tmp_path / str(identity), inventory, "--validate")
        assert run.returncode == 2
        assert "inventory_id must be a non-empty string" in run.stderr


def test_probe_contract_exposes_single_purpose_capture_cases():
    run = subprocess.run(
        [sys.executable, str(PROBE), "--contract"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    contract = json.loads(run.stdout)
    assert contract["schema_version"] == 1
    assert contract["seed"] == 20261008
    assert [case["name"] for case in contract["cases"]] == [
        "copy",
        "elementwise",
        "dot",
    ]
    assert all(case["expected_status"] == "unknown" for case in contract["cases"])
    assert len({case["kernel_name"] for case in contract["cases"]}) == 3
    assert [case["constructs"] for case in contract["cases"]] == [
        ["triton_load", "triton_store"],
        ["triton_load", "triton_elementwise", "triton_store"],
        ["triton_load", "triton_dot", "triton_store"],
    ]
    for case in contract["cases"]:
        for construct in case["constructs"]:
            lookup = run_cli("--construct", construct, "--product", "a3")
            assert lookup.returncode == 0, lookup.stderr
            assert json.loads(lookup.stdout)["status"] == case["expected_status"]
