import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bsa_candidate_profiler", ROOT / "scripts" / "bsa_candidate_profiler.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def option(identifier: str, source: str, **parameters: int) -> dict:
    return {
        "id": identifier,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "parameters": parameters,
        "hypothesis": f"try {identifier}",
    }


def campaign_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "campaign"
    treatments = (
        "a3-bsa-cannbot-all",
        "a3-bsa-cannbot-new-profiler",
        "a3-bsa-guarded-new-profiler",
        "a3-bsa-guarded-old-profiler",
    )
    shared = option("same", "SOURCE_A", BLOCK_N=64, num_warps=4)
    for treatment in treatments:
        request = {
            "schema": "simple-experiment-autotune-request/v1",
            "optimization_round": 1,
            "expected_declared_kernel_name": "block_sparse_attention_kernel",
            "options": [shared],
        }
        write_json(root / treatment / "attempts" / "1" / "autotune-request.json", request)
    # A retry is an occurrence, but maps to the same exact execution.
    write_json(
        root / treatments[1] / "attempts" / "2" / "autotune-request.json",
        {
            "schema": "simple-experiment-autotune-request/v1",
            "optimization_round": 1,
            "expected_declared_kernel_name": "block_sparse_attention_kernel",
            "options": [shared, option("other", "SOURCE_B", BLOCK_N=128)],
        },
    )
    write_json(
        root / treatments[1] / "attempts" / "2" / "result.json",
        {"outcome": "candidate_failure", "diagnostics": "compile failed"},
    )
    return root


def test_inventory_preserves_occurrences_and_maps_exact_duplicates(tmp_path: Path) -> None:
    manifest = MODULE.inventory_campaign(
        campaign_fixture(tmp_path),
        environment={"product": "a3", "runtime": "py311-torch", "suite": "bsa-v1"},
    )

    assert manifest["summary"] == {
        "treatments": 4,
        "occurrences": 6,
        "recoverable_occurrences": 6,
        "unrecoverable_occurrences": 0,
        "unique_executions": 2,
    }
    assert set(manifest["treatments"]) == set(MODULE.TREATMENTS)
    assert manifest["treatments"][MODULE.TREATMENTS[0]]["unique_executions"] == 1
    assert manifest["treatments"][MODULE.TREATMENTS[1]]["unique_executions"] == 2
    shared = [x for x in manifest["executions"] if x["source"] == "SOURCE_A"]
    assert len(shared) == 1
    assert len(shared[0]["occurrence_ids"]) == 5
    failed_attempt = next(
        item for item in manifest["occurrences"]
        if item["treatment"] == MODULE.TREATMENTS[1] and item["attempt"] == "2"
    )
    assert failed_attempt["artifact_outcome"] == "candidate_failure"


def test_inventory_retains_hash_only_candidate_as_unrecoverable(tmp_path: Path) -> None:
    root = campaign_fixture(tmp_path)
    request = json.loads(
        (root / MODULE.TREATMENTS[0] / "attempts/1/autotune-request.json").read_text()
    )
    request["options"].append(
        {"id": "lost", "source_sha256": "a" * 64, "parameters": {"BLOCK_N": 256}}
    )
    write_json(root / MODULE.TREATMENTS[0] / "attempts/1/autotune-request.json", request)

    manifest = MODULE.inventory_campaign(root, environment={"product": "a3"})

    lost = next(x for x in manifest["occurrences"] if x["option_id"] == "lost")
    assert lost["recoverability"] == "source_unavailable"
    assert manifest["summary"]["unrecoverable_occurrences"] == 1


def make_executor(path: Path, bodies: list[dict]) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import argparse,json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--response');a=p.parse_args()\n"
        f"bodies={bodies!r}\n"
        "state=Path(a.response).with_suffix('.count')\n"
        "n=int(state.read_text()) if state.exists() else 0\n"
        "state.write_text(str(n+1))\n"
        "Path(a.response).write_text(json.dumps(bodies[min(n,len(bodies)-1)]))\n"
    )
    path.chmod(0o755)


def one_execution() -> dict:
    source = "SOURCE"
    return {
        "schema": "profiling-skill/bsa-candidate-manifest/v1",
        "environment": {"product": "a3"},
        "executions": [{
            "execution_id": "exec-1",
            "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "parameters": {"BLOCK_N": 64},
            "declared_kernel_name": "block_sparse_attention_kernel",
            "occurrence_ids": ["occ-1"],
        }],
        "occurrences": [{"occurrence_id": "occ-1", "execution_id": "exec-1"}],
    }


def many_executions(count: int = 5) -> dict:
    manifest = one_execution()
    manifest["executions"] = []
    manifest["occurrences"] = []
    for index in range(count):
        source = f"SOURCE_{index}"
        execution_id = f"exec-{index}"
        manifest["executions"].append({
            "execution_id": execution_id,
            "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "parameters": {"BLOCK_N": 32 * (index + 1)},
            "declared_kernel_name": "block_sparse_attention_kernel",
            "occurrence_ids": [f"occ-{index}"],
        })
        manifest["occurrences"].append({
            "occurrence_id": f"occ-{index}", "execution_id": execution_id,
        })
    return manifest


def test_runner_discovers_kernel_and_accepts_stable_three_samples(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [
        {"status": "ok", "kernel_names": ["block_sparse_attention_kernel"],
         "basic_info": {"rows": 1}, "evidence_sha256": "1" * 64},
        {"status": "ok", "duration_us": 100.0, "pipe_utilization": {"aiv_vec_ratio": 0.7},
         "evidence_sha256": "2" * 64},
        {"status": "ok", "duration_us": 102.0, "pipe_utilization": {"aiv_vec_ratio": 0.7},
         "evidence_sha256": "3" * 64},
        {"status": "ok", "duration_us": 98.0, "pipe_utilization": {"aiv_vec_ratio": 0.7},
         "evidence_sha256": "4" * 64},
    ])
    output = tmp_path / "results.json"

    result = MODULE.run_campaign(
        one_execution(), output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=None,
    )

    execution = result["executions"]["exec-1"]
    assert execution["status"] == "completed"
    assert execution["kernel_name"] == "block_sparse_attention_kernel"
    assert execution["cases"]["47"]["median_us"] == 100.0
    assert execution["cases"]["47"]["relative_range"] == 0.04
    assert len(execution["cases"]["47"]["samples"]) == 3


def test_runner_retries_noise_to_find_tightest_stable_window(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [
        {"status": "ok", "kernel_names": ["actual_kernel"], "basic_info": {},
         "evidence_sha256": "1" * 64},
        *[
            {"status": "ok", "duration_us": value, "pipe_utilization": {"p": value},
             "evidence_sha256": str(index) * 64}
            for index, value in enumerate([100.0, 130.0, 101.0, 99.0], start=2)
        ],
    ])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None,
    )

    case = result["executions"]["exec-1"]["cases"]["47"]
    assert [x["duration_us"] for x in case["samples"]] == [100.0, 130.0, 101.0, 99.0]
    assert case["accepted_sample_indexes"] == [0, 2, 3]
    assert case["median_us"] == 100.0


def test_runner_classifies_candidate_failure_and_keeps_diagnostics(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [{
        "status": "compile_error", "diagnostics": "line 9: unsupported layout",
        "evidence_sha256": "a" * 64,
    }])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None,
    )

    execution = result["executions"]["exec-1"]
    assert execution["status"] == "candidate_failure"
    assert execution["failure"]["kind"] == "compile_error"
    assert "unsupported layout" in execution["failure"]["diagnostics"]


def test_runner_preserves_infrastructure_failure_for_resume(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [
        {"status": "infrastructure_error", "diagnostics": "device busy"},
        {"status": "ok", "kernel_names": ["block_sparse_attention_kernel"],
         "basic_info": {}, "evidence_sha256": "1" * 64},
    ])
    output = tmp_path / "results.json"
    manifest = one_execution()

    first = MODULE.run_campaign(
        manifest, output, executor, cases=[47], stable_samples=3, max_samples=7,
        max_actions=None,
    )
    assert first["executions"]["exec-1"]["status"] == "retryable_infrastructure"
    assert first["executions"]["exec-1"]["infrastructure_failures"][0]["diagnostics"] == "device busy"

    second = MODULE.run_campaign(
        manifest, output, executor, cases=[47], stable_samples=3, max_samples=7,
        max_actions=1,
    )
    assert second["executions"]["exec-1"]["status"] == "running"
    assert second["executions"]["exec-1"]["kernel_name"] == "block_sparse_attention_kernel"


def test_runner_marks_instability_only_after_capped_valid_samples(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [
        {"status": "ok", "kernel_names": ["block_sparse_attention_kernel"],
         "basic_info": {}, "evidence_sha256": "1" * 64},
        *[
            {"status": "ok", "duration_us": value, "pipe_utilization": {"p": value},
             "evidence_sha256": f"{index:x}" * 64}
            for index, value in enumerate([100, 120, 140, 160, 180, 200, 220], start=2)
        ],
    ])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None,
    )

    execution = result["executions"]["exec-1"]
    assert execution["status"] == "measurement_unstable"
    assert len(execution["cases"]["47"]["samples"]) == 7
    assert execution["failure"]["category"] == "infrastructure"


def test_production_mode_rejects_success_without_remote_provenance(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [{
        "status": "ok", "kernel_names": ["block_sparse_attention_kernel"],
        "basic_info": {}, "evidence_sha256": "1" * 64,
    }])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None, require_provenance=True,
    )

    execution = result["executions"]["exec-1"]
    assert execution["status"] == "retryable_infrastructure"
    assert execution["infrastructure_failures"][0]["kind"] == "invalid_provenance"


def test_sharding_is_deterministic_exhaustive_and_does_not_change_identity(tmp_path: Path) -> None:
    manifest = many_executions(11)
    original_ids = [item["execution_id"] for item in manifest["executions"]]
    executor = tmp_path / "unused.py"
    executor.write_text("raise SystemExit('must not execute')\n")
    executor.chmod(0o755)
    selected = []

    for index in range(4):
        result = MODULE.run_campaign(
            manifest, tmp_path / f"shard-{index}.json", executor, cases=[47],
            stable_samples=3, max_samples=7, max_actions=0,
            shard_count=4, shard_index=index,
        )
        assert result["shard"] == {
            "algorithm": "sha256-mod-v1", "count": 4, "index": index,
        }
        assert {item["status"] for item in result["executions"].values()} == {"pending"}
        selected.extend(result["executions"])

    assert sorted(selected) == sorted(original_ids)
    assert len(selected) == len(set(selected))
    assert [item["execution_id"] for item in manifest["executions"]] == original_ids


def test_merge_combines_disjoint_shards_without_losing_checkpoint_data(tmp_path: Path) -> None:
    manifest = many_executions(7)
    executor = tmp_path / "unused.py"
    executor.write_text("raise SystemExit('must not execute')\n")
    executor.chmod(0o755)
    paths = []
    for index in range(3):
        path = tmp_path / f"shard-{index}.json"
        MODULE.run_campaign(
            manifest, path, executor, cases=[47], stable_samples=3,
            max_samples=7, max_actions=0, shard_count=3, shard_index=index,
        )
        paths.append(path)
    first = json.loads(paths[0].read_text())
    first_execution = next(iter(first["executions"].values()))
    first_execution["infrastructure_failures"].append({
        "kind": "device_error", "provenance": {"handle": "remote:x:job:y"},
    })
    write_json(paths[0], first)

    merged = MODULE.merge_results(manifest, paths)

    assert merged["missing_shards"] == []
    assert merged["missing_execution_ids"] == []
    assert sorted(merged["executions"]) == sorted(
        item["execution_id"] for item in manifest["executions"]
    )
    retained = merged["executions"][next(iter(first["executions"]))]
    assert retained["infrastructure_failures"][0]["kind"] == "device_error"


def test_merge_reports_missing_shards_and_executions(tmp_path: Path) -> None:
    manifest = many_executions(9)
    executor = tmp_path / "unused.py"
    executor.write_text("raise SystemExit('must not execute')\n")
    executor.chmod(0o755)
    path = tmp_path / "shard-1.json"
    result = MODULE.run_campaign(
        manifest, path, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=0, shard_count=3, shard_index=1,
    )

    merged = MODULE.merge_results(manifest, [path])

    assert merged["missing_shards"] == [0, 2]
    assert sorted(merged["missing_execution_ids"]) == sorted(
        set(item["execution_id"] for item in manifest["executions"])
        - set(result["executions"])
    )


def test_merge_rejects_overlapping_or_mismatched_shards(tmp_path: Path) -> None:
    manifest = many_executions(4)
    executor = tmp_path / "unused.py"
    executor.write_text("raise SystemExit('must not execute')\n")
    executor.chmod(0o755)
    path = tmp_path / "shard.json"
    MODULE.run_campaign(
        manifest, path, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=0, shard_count=2, shard_index=0,
    )
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(path.read_bytes())

    try:
        MODULE.merge_results(manifest, [path, duplicate])
    except MODULE.InvalidArtifact as exc:
        assert "overlapping shard index" in str(exc)
    else:
        raise AssertionError("overlap was accepted")

    mismatch = json.loads(path.read_text())
    mismatch["manifest_sha256"] = "0" * 64
    write_json(duplicate, mismatch)
    try:
        MODULE.merge_results(manifest, [duplicate])
    except MODULE.InvalidArtifact as exc:
        assert "manifest mismatch" in str(exc)
    else:
        raise AssertionError("manifest mismatch was accepted")

    mismatch = json.loads(path.read_text())
    mismatch["run_contract"]["max_samples"] = 99
    write_json(duplicate, mismatch)
    try:
        MODULE.merge_results(manifest, [path, duplicate])
    except MODULE.InvalidArtifact as exc:
        assert "run contract mismatch" in str(exc)
    else:
        raise AssertionError("contract mismatch was accepted")


def test_extract_features_covers_bsa_code_and_launch_dimensions() -> None:
    source = """
AUTOTUNE_CONFIG = {"BLOCK_M": 16, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2}
for hh in range(0, H):
    rank += (hh < h)
for start in range(0, TOTAL_K, BLOCK_N):
    keep = valid & predicate
    has = tl.sum(keep.to(tl.int32), axis=0) > 0
    if has:
        kv = tl.load(k + offsets, mask=keep[:, None], other=0.0)
score = tl.sum(kv * qv[None, :], axis=1)
grid = (triton.cdiv(tq, BLOCK_M), h)
"""
    features = MODULE.extract_features(
        source, {"BLOCK_M": 16, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2}
    )

    assert features == {
        "loop_bound": "total_packed_k",
        "head_rank_strategy": "linear_scan",
        "block_n": 128,
        "whole_tile_skip": True,
        "scalar_block_decision": False,
        "kv_mask_style": "final_keep",
        "num_warps": 4,
        "num_stages": 2,
        "reduction_strategy": "row_tl_sum",
        "query_tiling": "block_m",
        "block_m": 16,
        "grid_strategy": "query_tiles_by_head",
    }


def test_inventory_emits_only_provable_lineage_and_feature_deltas(tmp_path: Path) -> None:
    root = campaign_fixture(tmp_path)
    treatment = MODULE.TREATMENTS[0]
    incumbent = option("incumbent", "for start in range(0, TOTAL_K, 64):\n pass", BLOCK_N=64)
    child = option(
        "local-bound", "for start in tl.range(0, sk, 128):\n pass",
        BLOCK_N=128, num_warps=4,
    )
    write_json(root / treatment / "attempts/1/autotune-request.json", {
        "schema": "simple-experiment-autotune-request/v1",
        "optimization_round": 1,
        "incumbent_id": "incumbent",
        "expected_declared_kernel_name": "block_sparse_attention_kernel",
        "options": [incumbent, child],
    })
    write_json(root / treatment / "attempts/1/result.json", {"outcome": "success"})
    write_json(root / treatment / "repo/rounds/01/autotune-result.json", {
        "winner_id": "local-bound", "winner_source_sha256": child["source_sha256"],
    })
    write_json(root / treatment / "attempts/2/autotune-request.json", {
        "schema": "simple-experiment-autotune-request/v1",
        "optimization_round": 2,
        "incumbent_id": "incumbent",
        "expected_declared_kernel_name": "block_sparse_attention_kernel",
        "options": [{**child, "id": "incumbent"}],
    })

    manifest = MODULE.inventory_campaign(root, environment={"product": "a3"})
    occurrences = {item["occurrence_id"]: item for item in manifest["occurrences"]}
    round1_incumbent = next(
        item for item in occurrences.values()
        if item["treatment"] == treatment and item["round"] == 1
        and item["option_id"] == "incumbent"
    )
    round1_child = next(
        item for item in occurrences.values()
        if item["treatment"] == treatment and item["round"] == 1
        and item["option_id"] == "local-bound"
    )
    round2_incumbent = next(
        item for item in occurrences.values()
        if item["treatment"] == treatment and item["round"] == 2
    )

    assert round1_child["parent_occurrence_id"] == round1_incumbent["occurrence_id"]
    assert round1_child["lineage_note"] == "request-declared incumbent"
    assert round1_child["parent_feature_deltas"]["loop_bound"] == {
        "before": "total_packed_k", "after": "per_sequence_k",
    }
    assert round2_incumbent["parent_occurrence_id"] == round1_child["occurrence_id"]
    assert round2_incumbent["lineage_note"] == "prior round winner hash"
    assert round2_incumbent["parent_feature_deltas"] == {}
    # Other treatments have no evidence linking their round-one incumbent.
    unlinked = next(
        item for item in occurrences.values()
        if item["treatment"] == MODULE.TREATMENTS[2]
    )
    assert "parent_occurrence_id" not in unlinked
