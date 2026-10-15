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


def starter(source: str = "SOURCE_A", **parameters: int) -> dict:
    return {
        "source": source,
        "parameters": parameters or {"BLOCK_N": 64, "num_warps": 4},
        "declared_kernel_name": "_bsa_reference_fwd",
    }


def test_inventory_preserves_occurrences_and_maps_exact_duplicates(tmp_path: Path) -> None:
    manifest = MODULE.inventory_campaign(
        campaign_fixture(tmp_path),
        environment={"product": "a3", "runtime": "py311-torch", "suite": "bsa-v1"},
        starter=starter(),
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

    manifest = MODULE.inventory_campaign(
        root, environment={"product": "a3"}, starter=starter()
    )

    lost = next(x for x in manifest["occurrences"] if x["option_id"] == "lost")
    assert lost["recoverability"] == "source_unavailable"
    assert manifest["summary"]["unrecoverable_occurrences"] == 1


def test_inventory_identifies_one_global_starter_without_consumer_guessing(
    tmp_path: Path,
) -> None:
    manifest = MODULE.inventory_campaign(
        campaign_fixture(tmp_path), environment={"product": "a3"}, starter=starter()
    )

    occurrences = {
        item["occurrence_id"]: item for item in manifest["occurrences"]
    }
    starter_occurrence = occurrences[manifest["starter_occurrence_id"]]
    assert starter_occurrence["lineage_role"] == "starter"
    assert starter_occurrence["source_sha256"] == hashlib.sha256(b"SOURCE_A").hexdigest()
    assert sum(
        item.get("lineage_role") == "starter" for item in manifest["occurrences"]
    ) == 1


def test_inventory_adds_synthetic_starter_execution_when_outside_options(
    tmp_path: Path,
) -> None:
    manifest = MODULE.inventory_campaign(
        campaign_fixture(tmp_path), environment={"product": "a3"},
        starter=starter("FROZEN_STARTER", BLOCK_N=32),
    )

    occurrence = next(
        item for item in manifest["occurrences"]
        if item["occurrence_id"] == manifest["starter_occurrence_id"]
    )
    execution = next(
        item for item in manifest["executions"]
        if item["execution_id"] == occurrence["execution_id"]
    )
    assert occurrence["lineage_role"] == "starter"
    assert occurrence["treatment"] is None
    assert execution["declared_kernel_name"] == "_bsa_reference_fwd"
    assert execution["source"] == "FROZEN_STARTER"


def make_executor(path: Path, bodies: list[dict]) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import argparse,json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--response');a=p.parse_args()\n"
        f"bodies={bodies!r}\n"
        "state=Path(a.response).with_suffix('.count')\n"
        "requests=Path(a.response).with_suffix('.requests')\n"
        "history=json.loads(requests.read_text()) if requests.exists() else []\n"
        "history.append(json.loads(Path(a.request).read_text()));requests.write_text(json.dumps(history))\n"
        "n=int(state.read_text()) if state.exists() else 0\n"
        "state.write_text(str(n+1))\n"
        "Path(a.response).write_text(json.dumps(bodies[min(n,len(bodies)-1)]))\n"
    )
    path.chmod(0o755)


def one_execution() -> dict:
    source = "SOURCE"
    return {
        "schema": "profiling-skill/bsa-candidate-manifest/v1",
        "environment": {
            "product": "a3",
            "target": "gz-a3",
            "runtime": "py311-torch",
            "runtime_activation": "named-runtime",
        },
        "executions": [{
            "execution_id": "exec-1",
            "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "parameters": {"BLOCK_N": 64},
            "declared_kernel_name": "block_sparse_attention_kernel",
            "occurrence_ids": ["occ-1"],
        }],
        "starter_occurrence_id": "occ-1",
        "occurrences": [{
            "occurrence_id": "occ-1", "execution_id": "exec-1",
            "lineage_role": "starter",
        }],
    }


def gz_provenance(handle: str = "remote:gz-a3:job:valid") -> dict:
    return {
        "target": "gz-a3",
        "runtime": "py311-torch",
        "runtime_activation": "named-runtime",
        "physical_device": 0,
        "command": [
            "cpl-remote", "run", "gz-a3", "--runtime", "py311-torch",
        ],
        "handle": handle,
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
    manifest["starter_occurrence_id"] = "occ-0"
    manifest["occurrences"][0]["lineage_role"] = "starter"
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


def test_runner_uses_first_stable_consecutive_window_without_cherry_picking(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [
        {"status": "ok", "kernel_names": ["actual_kernel"], "basic_info": {},
         "evidence_sha256": "1" * 64},
        *[
            {"status": "ok", "duration_us": value, "pipe_utilization": {"p": value},
             "evidence_sha256": str(index) * 64}
            for index, value in enumerate([100.0, 130.0, 101.0, 99.0, 100.0], start=2)
        ],
    ])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None,
    )

    case = result["executions"]["exec-1"]["cases"]["47"]
    assert [x["duration_us"] for x in case["samples"]] == [100.0, 130.0, 101.0, 99.0, 100.0]
    assert case["accepted_sample_indexes"] == [2, 3, 4]
    assert case["median_us"] == 100.0


def test_run_contract_and_executor_request_bind_target_and_runtime_activation(
    tmp_path: Path,
) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [{
        "status": "infrastructure_error", "diagnostics": "stop after request",
    }])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=1,
    )

    expected = {
        "target": "gz-a3",
        "runtime": "py311-torch",
        "runtime_activation": "named-runtime",
    }
    assert result["run_contract"]["execution_environment"] == expected
    assert result["run_contract"]["executor_identity"] == {
        "campaign_revision": MODULE.DIRECT_API_CAMPAIGN_REVISION,
        "executable_sha256": hashlib.sha256(executor.read_bytes()).hexdigest(),
    }
    requests = json.loads(
        (tmp_path / "results.protocol" / "response.requests").read_text()
    )
    assert requests[0]["execution_environment"] == expected


def test_runner_rejects_bz_checkpoint_reuse_as_gz(tmp_path: Path) -> None:
    executor = tmp_path / "unused.py"
    executor.write_text("raise SystemExit('must not execute')\n")
    executor.chmod(0o755)
    bz_manifest = one_execution()
    bz_manifest["environment"].update({
        "target": "bz-a3-1", "runtime_activation": "named-runtime",
    })
    output = tmp_path / "results.json"
    MODULE.run_campaign(
        bz_manifest, output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=0,
    )

    gz_manifest = one_execution()
    try:
        MODULE.run_campaign(
            gz_manifest, output, executor, cases=[47], stable_samples=3,
            max_samples=7, max_actions=0,
        )
    except MODULE.InvalidArtifact as exc:
        assert "identity mismatch" in str(exc)
    else:
        raise AssertionError("BZ checkpoint was reused for a GZ campaign")

    try:
        MODULE.merge_results(gz_manifest, [output])
    except MODULE.InvalidArtifact as exc:
        assert "manifest mismatch" in str(exc)
    else:
        raise AssertionError("BZ checkpoint was merged into a GZ campaign")


def test_runner_rejects_missing_or_inconsistent_execution_environment(
    tmp_path: Path,
) -> None:
    executor = tmp_path / "unused.py"
    executor.write_text("raise SystemExit('must not execute')\n")
    executor.chmod(0o755)
    missing = one_execution()
    del missing["environment"]["runtime_activation"]
    inconsistent = one_execution()
    inconsistent["environment"]["runtime_activation"] = "target-default"

    for manifest in (missing, inconsistent):
        try:
            MODULE.run_campaign(
                manifest, tmp_path / f"{id(manifest)}.json", executor,
                cases=[47], stable_samples=3, max_samples=7, max_actions=0,
            )
        except MODULE.InvalidArtifact as exc:
            assert "execution environment" in str(exc)
        else:
            raise AssertionError("invalid execution environment was accepted")


def test_runner_profiles_all_discovered_components_and_aggregates_max_duration(
    tmp_path: Path,
) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [
        {
            "status": "ok",
            "components": [
                {"selector": "block_sparse_attention_kernel_aic", "kind": "aic"},
                {"selector": "block_sparse_attention_kernel_aiv", "kind": "aiv"},
            ],
            "basic_info": {"rows": 2},
            "evidence_sha256": "1" * 64,
        },
        *[
            {
                "status": "ok", "duration_us": duration,
                "pipe_utilization": {"rows": [{"selector": selector}]},
                "evidence_sha256": str(index) * 64,
            }
            for index, (selector, duration) in enumerate([
                ("aic", 80.0), ("aiv", 100.0),
                ("aic", 81.0), ("aiv", 102.0),
                ("aic", 79.0), ("aiv", 98.0),
            ], start=2)
        ],
    ])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None,
    )

    execution = result["executions"]["exec-1"]
    assert execution["selectors"] == [
        "block_sparse_attention_kernel_aic",
        "block_sparse_attention_kernel_aiv",
    ]
    samples = execution["cases"]["47"]["samples"]
    assert [sample["duration_us"] for sample in samples] == [100.0, 102.0, 98.0]
    assert samples[0]["duration_aggregation"] == "max_component_duration"
    assert set(samples[0]["components"]) == set(execution["selectors"])
    requests = json.loads(
        (tmp_path / "results.protocol" / "response.requests").read_text()
    )
    assert [request["kernel_name"] for request in requests[1:]] == [
        "block_sparse_attention_kernel_aic",
        "block_sparse_attention_kernel_aiv",
    ] * 3


def test_runner_resumes_partial_multi_component_sample_without_recapturing(
    tmp_path: Path,
) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [
        {"status": "ok", "components": [{"selector": "aic"}, {"selector": "aiv"}],
         "basic_info": {}, "evidence_sha256": "1" * 64},
        {"status": "ok", "duration_us": 80.0, "pipe_utilization": {},
         "evidence_sha256": "2" * 64},
        {"status": "infrastructure_error", "diagnostics": "profiler unavailable",
         "terminal": True},
        {"status": "ok", "duration_us": 100.0, "pipe_utilization": {},
         "evidence_sha256": "3" * 64},
    ])
    output = tmp_path / "results.json"

    first = MODULE.run_campaign(
        one_execution(), output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=None,
    )
    partial = first["executions"]["exec-1"]["cases"]["47"]["in_progress_sample"]
    assert set(partial["components"]) == {"aic"}

    second = MODULE.run_campaign(
        one_execution(), output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=1,
    )
    sample = second["executions"]["exec-1"]["cases"]["47"]["samples"][0]
    assert set(sample["components"]) == {"aic", "aiv"}
    assert sample["duration_us"] == 100.0


def test_runner_rejects_legacy_checkpoint_with_migration_message(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    write_json(output, {"schema": "profiling-skill/bsa-candidate-results/v1"})
    executor = tmp_path / "unused.py"
    executor.write_text("raise SystemExit(1)\n")
    executor.chmod(0o755)

    try:
        MODULE.run_campaign(
            one_execution(), output, executor, cases=[47], stable_samples=3,
            max_samples=7, max_actions=0,
        )
    except MODULE.InvalidArtifact as exc:
        assert "legacy v1 checkpoint" in str(exc)
        assert "regenerate" in str(exc)
    else:
        raise AssertionError("legacy checkpoint was accepted")


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


def test_production_rejects_wrong_target_compile_failure_as_infrastructure(
    tmp_path: Path,
) -> None:
    executor = tmp_path / "executor.py"
    wrong = {
        **gz_provenance("remote:bz-a3-1:job:wrong"),
        "target": "bz-a3-1",
        "runtime_activation": "named-runtime",
    }
    make_executor(executor, [{
        "status": "compile_error", "diagnostics": "candidate did not compile",
        "terminal": True, "provenance": wrong,
    }])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=1, require_provenance=True,
    )

    execution = result["executions"]["exec-1"]
    assert execution["status"] == "retryable_infrastructure"
    assert "failure" not in execution
    assert "retry" not in execution
    assert execution["infrastructure_failures"][-1]["kind"] == "invalid_provenance"
    assert execution["infrastructure_failures"][-1]["rejected_provenance"] == wrong


def test_production_accepts_matching_candidate_failure_provenance(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [{
        "status": "compile_error", "diagnostics": "candidate did not compile",
        "terminal": True, "provenance": gz_provenance(),
    }])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=1, require_provenance=True,
    )

    execution = result["executions"]["exec-1"]
    assert execution["status"] == "candidate_failure"
    assert execution["failure"]["kind"] == "compile_error"


def test_production_rejects_wrong_target_capture_candidate_failures(
    tmp_path: Path,
) -> None:
    wrong = {
        **gz_provenance("remote:bz-a3-1:job:wrong"),
        "target": "bz-a3-1",
        "runtime_activation": "named-runtime",
    }
    for status in ("runtime_error", "correctness_error"):
        case_dir = tmp_path / status
        case_dir.mkdir()
        executor = case_dir / "executor.py"
        make_executor(executor, [
            {
                "status": "ok", "kernel_names": ["block_sparse_attention_kernel"],
                "basic_info": {}, "evidence_sha256": "1" * 64,
                "provenance": gz_provenance("remote:gz-a3:job:discovery"),
            },
            {
                "status": status, "diagnostics": "candidate launch failed",
                "terminal": True, "provenance": wrong,
            },
        ])

        result = MODULE.run_campaign(
            one_execution(), case_dir / "results.json", executor, cases=[47],
            stable_samples=3, max_samples=7, max_actions=2, require_provenance=True,
        )

        execution = result["executions"]["exec-1"]
        assert execution["status"] == "retryable_infrastructure"
        assert "failure" not in execution
        assert "retry" not in execution
        assert execution["infrastructure_failures"][-1]["kind"] == "invalid_provenance"
        assert execution["infrastructure_failures"][-1]["rejected_provenance"] == wrong


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


def test_runner_accepts_first_stable_window_ending_at_sample_cap(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [
        {"status": "ok", "kernel_names": ["block_sparse_attention_kernel"],
         "basic_info": {}, "evidence_sha256": "1" * 64},
        *[
            {"status": "ok", "duration_us": value, "pipe_utilization": {"p": value},
             "evidence_sha256": f"{index:x}" * 64}
            for index, value in enumerate(
                [100, 130, 170, 220, 200, 201, 199], start=2
            )
        ],
    ])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None,
    )

    execution = result["executions"]["exec-1"]
    assert execution["status"] == "completed"
    case = execution["cases"]["47"]
    assert len(case["samples"]) == 7
    assert case["accepted_sample_indexes"] == [4, 5, 6]
    assert case["median_us"] == 200.0


def test_terminal_malformed_basic_info_requests_a_new_attempt(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    handle = "remote:gz-a3:job:malformed-basic"
    make_executor(executor, [{
        "status": "ok",
        "kernel_names": ["block_sparse_attention_kernel"],
        "basic_info": 42,
        "evidence_sha256": "1" * 64,
        "terminal": True,
        "provenance": gz_provenance(handle),
    }])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None, require_provenance=True,
    )

    state = result["executions"]["exec-1"]
    assert state["status"] == "retryable_infrastructure"
    assert state["retry"] == {"mode": "new_attempt", "prior_handle": handle}
    assert state["infrastructure_failures"][-1]["kind"] == "invalid_basic_info"


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


def test_production_mode_rejects_wrong_target_runtime_and_invalid_digest(
    tmp_path: Path,
) -> None:
    executor = tmp_path / "executor.py"
    provenance = {
        "target": "bz-a5", "runtime": "wrong-runtime", "physical_device": 0,
        "command": ["run"], "handle": "remote:bz-a5:job:wrong",
    }
    make_executor(executor, [{
        "status": "ok", "kernel_names": ["block_sparse_attention_kernel"],
        "basic_info": {}, "evidence_sha256": "not-a-digest",
        "provenance": provenance,
    }])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None, require_provenance=True,
    )

    failure = result["executions"]["exec-1"]["infrastructure_failures"][0]
    assert failure["kind"] == "invalid_provenance"
    assert "A3 target" in failure["diagnostics"]

    provenance.update({
        "target": "gz-a3", "runtime": "py311-torch",
        "runtime_activation": "named-runtime",
        "handle": "remote:gz-a3:job:right",
    })
    make_executor(executor, [{
        "status": "ok", "kernel_names": ["block_sparse_attention_kernel"],
        "basic_info": {}, "evidence_sha256": "not-a-digest",
        "provenance": provenance,
    }])
    manifest = one_execution()
    manifest["environment"]["runtime"] = "py311-torch"
    digest_result = MODULE.run_campaign(
        manifest, tmp_path / "digest-results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None, require_provenance=True,
    )
    digest_failure = digest_result["executions"]["exec-1"]["infrastructure_failures"][0]
    assert digest_failure["kind"] == "invalid_evidence_digest"


def test_terminal_malformed_success_requests_a_new_attempt(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    handle = "remote:gz-a3:job:malformed-terminal"
    make_executor(executor, [{
        "status": "ok",
        "kernel_names": ["block_sparse_attention_kernel"],
        "basic_info": {},
        "evidence_sha256": "not-a-digest",
        "terminal": True,
        "provenance": gz_provenance(handle),
    }])

    result = MODULE.run_campaign(
        one_execution(), tmp_path / "results.json", executor, cases=[47],
        stable_samples=3, max_samples=7, max_actions=None, require_provenance=True,
    )

    state = result["executions"]["exec-1"]
    assert state["status"] == "retryable_infrastructure"
    assert state["retry"] == {"mode": "new_attempt", "prior_handle": handle}


def test_resume_rejects_executor_or_campaign_revision_drift(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [{"status": "infrastructure_error", "diagnostics": "stop"}])
    output = tmp_path / "results.json"
    MODULE.run_campaign(
        one_execution(), output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=1, campaign_revision="campaign-one",
    )

    executor.write_text(executor.read_text() + "\n# changed harness\n")
    try:
        MODULE.run_campaign(
            one_execution(), output, executor, cases=[47], stable_samples=3,
            max_samples=7, max_actions=1, campaign_revision="campaign-one",
        )
    except MODULE.InvalidArtifact as exc:
        assert "run contract mismatch" in str(exc)
    else:
        raise AssertionError("executor drift was accepted")

    make_executor(executor, [{"status": "infrastructure_error", "diagnostics": "stop"}])
    try:
        MODULE.run_campaign(
            one_execution(), output, executor, cases=[47], stable_samples=3,
            max_samples=7, max_actions=1, campaign_revision="campaign-two",
        )
    except MODULE.InvalidArtifact as exc:
        assert "run contract mismatch" in str(exc)
    else:
        raise AssertionError("campaign revision drift was accepted")


def test_merge_rejects_executor_identity_drift(tmp_path: Path) -> None:
    manifest = many_executions(2)
    executor = tmp_path / "executor.py"
    make_executor(executor, [{"status": "infrastructure_error", "diagnostics": "stop"}])
    first = tmp_path / "first.json"
    MODULE.run_campaign(
        manifest, first, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=0,
        campaign_revision="campaign-one",
    )
    executor.write_text(executor.read_text() + "\n# changed harness\n")
    second = tmp_path / "second.json"
    MODULE.run_campaign(
        manifest, second, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=0,
        campaign_revision="campaign-one",
    )

    try:
        MODULE.merge_results(manifest, [first, second])
    except MODULE.InvalidArtifact as exc:
        assert "run contract mismatch" in str(exc)
    else:
        raise AssertionError("shards from different executors were merged")


def test_retry_request_observes_nonterminal_handle_and_replaces_terminal_attempt(
    tmp_path: Path,
) -> None:
    executor = tmp_path / "executor.py"
    handle = "remote:gz-a3:job:still-running"
    make_executor(executor, [
        {"status": "infrastructure_error", "diagnostics": "observer disconnected",
         "terminal": False, "provenance": {"handle": handle}},
        {"status": "infrastructure_error", "diagnostics": "job failed",
         "terminal": True, "provenance": {"handle": handle}},
        {"status": "ok", "kernel_names": ["block_sparse_attention_kernel"],
         "evidence_sha256": "a" * 64},
    ])
    output = tmp_path / "results.json"
    manifest = one_execution()

    MODULE.run_campaign(
        manifest, output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=None,
    )
    MODULE.run_campaign(
        manifest, output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=None,
    )
    MODULE.run_campaign(
        manifest, output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=1,
    )

    requests = json.loads(
        (tmp_path / "results.protocol" / "response.requests").read_text()
    )
    assert requests[1]["retry"] == {"mode": "observe", "handle": handle}
    assert requests[2]["retry"] == {"mode": "new_attempt", "prior_handle": handle}


def test_acquisition_budget_stops_new_attempts_but_never_retained_observation(tmp_path):
    executor = tmp_path / "executor.py"
    make_executor(executor, [{"status":"infrastructure_error", "terminal":True,
                              "diagnostics":"duration unavailable"}]*3)
    output = tmp_path / "results.json"
    for _ in range(5):
        result = MODULE.run_campaign(one_execution(), output, executor, cases=[47],
                                     stable_samples=3, max_samples=7, max_actions=None)
    state = next(iter(result["executions"].values()))
    assert state["status"] == "measurement_unstable"
    assert state["failure"]["kind"] == "acquisition_exhausted"
    requests = json.loads((tmp_path / "results.protocol/response.requests").read_text())
    assert len(requests) == 2
    state["retry"] = {"mode":"observe", "handle":"remote:gz-a3:job:alive"}
    assert MODULE._acquisition_exhausted(state) is False


def test_acquisition_success_resets_failure_budget():
    state = {"infrastructure_failures":[{"terminal":True,"diagnostics":"old"}]*3}
    MODULE._acquisition_succeeded(state)
    assert MODULE._acquisition_exhausted(state) is False
    state["infrastructure_failures"].extend(
        {"terminal":True,"diagnostics":str(index)} for index in range(3))
    assert MODULE._acquisition_exhausted(state) is True


def test_distinct_terminal_infra_failures_allow_only_two_retries(tmp_path):
    executor = tmp_path / "executor.py"
    make_executor(executor, [{"status":"infrastructure_error", "terminal":True,
                              "diagnostics":f"transport failure {index}"}
                             for index in range(4)])
    output = tmp_path / "results.json"
    for _ in range(6):
        result = MODULE.run_campaign(one_execution(), output, executor, cases=[43],
                                     stable_samples=3, max_samples=7, max_actions=None)
    state = next(iter(result["executions"].values()))
    assert state["failure"]["kind"] == "acquisition_exhausted"
    assert len(state["infrastructure_failures"]) == 3


def test_retry_rejects_wrong_target_failure_handle(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [{
        "status": "infrastructure_error",
        "diagnostics": "observer disconnected",
        "terminal": False,
        "provenance": {
            "target": "bz-a3-1",
            "runtime": "py311-torch",
            "runtime_activation": "named-runtime",
            "handle": "remote:bz-a3-1:job:wrong-target",
        },
    }])
    output = tmp_path / "results.json"

    result = MODULE.run_campaign(
        one_execution(), output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=1,
    )

    state = result["executions"]["exec-1"]
    assert state["status"] == "retryable_infrastructure"
    assert state["infrastructure_failures"][-1]["kind"] == "invalid_provenance"
    assert "retry" not in state


def test_retry_rejects_inconsistent_failure_provenance(tmp_path: Path) -> None:
    executor = tmp_path / "executor.py"
    make_executor(executor, [{
        "status": "infrastructure_error",
        "diagnostics": "observer disconnected",
        "terminal": False,
        "provenance": {
            "target": "gz-a3",
            "runtime": "py311-torch",
            "runtime_activation": "target-default",
            "handle": "remote:gz-a3:job:right-target",
        },
    }])
    output = tmp_path / "results.json"

    result = MODULE.run_campaign(
        one_execution(), output, executor, cases=[47], stable_samples=3,
        max_samples=7, max_actions=1,
    )

    state = result["executions"]["exec-1"]
    assert state["infrastructure_failures"][-1]["kind"] == "invalid_provenance"
    assert "retry" not in state


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

    manifest = MODULE.inventory_campaign(
        root, environment={"product": "a3"}, starter=starter()
    )
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
