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
