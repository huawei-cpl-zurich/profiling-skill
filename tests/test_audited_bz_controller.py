import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load("audited_bz_controller")
contract = load("audited_contract")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "candidate.py").write_text("VALUE = 1\n")
    (repo / "candidate.manifest.json").write_text(
        '{"schema":"profiling-skill/candidate-kernel/v1","kernel_name":"kernel"}\n'
    )
    return repo, digest(repo / "candidate.py"), digest(repo / "candidate.manifest.json")


class FakeBackend:
    def __init__(self, overrides=None):
        self.requests = []
        self.overrides = list(overrides or [])
        self.serial = 0

    def __call__(self, request):
        self.requests.append(json.loads(json.dumps(request)))
        self.serial += 1
        if self.overrides:
            override = self.overrides.pop(0)
            if override is not None:
                return override
        handle = f"bz-a3-1:job-{self.serial}"
        if request["action"] == "calibrate":
            return {"status": "ok", "handle": handle, "latency_us": 10.0,
                    "artifacts": {"remote_profile_evidence": "/remote/calibration.json"}}
        if request["action"] == "check":
            return {"status": "ok", "handle": handle, "passed": True}
        rows = [
            {"case": case, "samples_us": [10.0 + index, 20.0 + index, 40.0 + index],
             "median_us": 20.0 + index}
            for index, case in enumerate(request["cases"])
        ]
        return {"status": "ok", "handle": handle, "cases": rows,
                "artifacts": {"remote_profile_evidence": "/remote/evidence.json"}}


def make_controller(repo: Path, backend, **options):
    settings = {"variability_threshold": 2.0, "request_budget": 24, **options}
    return controller.AuditedBzController(
        repo=repo,
        state_dir=repo.parent / "state",
        benchmark="matmul",
        backend=backend,
        devices=[{"id": "bz-a3-1/device-0", "device": 0}],
        development_cases=[7, 8, 9],
        all_cases=list(range(10)),
        round_count=4,
        **settings,
    )


def test_success_runs_controls_full_final_check_and_three_profile_repetitions(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend()
    receipt = make_controller(repo, backend).run(4, candidate_hash, manifest_hash)

    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
    assert [request["action"] for request in backend.requests] == [
        "calibrate", "check", "profile", "calibrate",
    ]
    assert backend.requests[1]["scope"] == "full"
    assert backend.requests[1]["cases"] == list(range(10))
    assert backend.requests[2]["repeats"] == 3
    assert receipt["case_results"][0]["case"] == 7
    expected = [
        math.exp(sum(math.log(value) for value in sample) / 3)
        for sample in zip([10.0, 20.0, 40.0], [11.0, 21.0, 41.0], [12.0, 22.0, 42.0])
    ]
    assert receipt["samples_us"] == expected
    assert receipt["policy"]["operations_consumed"] == 4
    assert receipt["compact_artifacts"] == ["/remote/evidence.json"]


def test_compile_error_is_counted_without_profile_or_post_control(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend([
        None,
        {"status": "compile_error", "failure_type": "compile_error",
         "diagnostics": "bad tl.load", "handle": "bz-a3-1:compile"},
    ])
    receipt = make_controller(repo, backend).run(1, candidate_hash, manifest_hash)

    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
    assert receipt["status"] == "candidate_error"
    assert receipt["failure_type"] == "compile_error"
    assert receipt["policy"]["post_control"] == "not_run"
    assert receipt["policy"]["operations_consumed"] == 2
    assert [request["action"] for request in backend.requests] == ["calibrate", "check"]


def test_infrastructure_handle_is_checkpointed_and_observed_without_budget_charge(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend([
        None,
        {"status": "infrastructure_error", "failure_type": "observer_error",
         "diagnostics": "observer disconnected", "handle": "bz-a3-1:retained"},
        {"status": "ok", "handle": "bz-a3-1:retained", "passed": True},
    ])
    adapter = make_controller(repo, backend)
    interrupted = adapter.run(1, candidate_hash, manifest_hash)

    assert interrupted["status"] == "infrastructure_error"
    assert interrupted["terminal"] is False
    assert interrupted["handle"] == "bz-a3-1:retained"
    resumed = adapter.run(
        1, candidate_hash, manifest_hash, observe_handle="bz-a3-1:retained"
    )
    contract.validate_controller_receipt(resumed, candidate_hash, manifest_hash)
    assert backend.requests[1] == backend.requests[2]
    assert resumed["policy"]["operations_consumed"] == 4
    assert resumed["policy"]["infrastructure_attempts"] == 1


def test_noisy_profile_gets_exactly_one_confirmation_then_measurement_pending(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend()
    receipt = make_controller(
        repo, backend, variability_threshold=0.01
    ).run(1, candidate_hash, manifest_hash)

    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
    assert receipt["status"] == "measurement_pending"
    assert receipt["policy"]["confirmation_count"] == 1
    assert [request["action"] for request in backend.requests] == [
        "calibrate", "check", "profile", "profile", "calibrate",
    ]


def test_measurement_pending_can_remeasure_without_rechecking_candidate(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend()
    adapter = make_controller(repo, backend, variability_threshold=0.01)
    pending = adapter.run(1, candidate_hash, manifest_hash)
    backend.overrides.extend([
        {"status": "ok", "handle": "bz-a3-1:profile-stable", "cases": [
            {"case": case, "samples_us": [10.0, 10.0, 10.0], "median_us": 10.0}
            for case in [7, 8, 9]
        ]},
        {"status": "ok", "handle": "bz-a3-1:post-stable", "latency_us": 10.0},
    ])

    receipt = adapter.run(
        1, candidate_hash, manifest_hash, remeasure_handle=pending["handle"]
    )

    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
    assert receipt["status"] == "ok"
    assert [request["action"] for request in backend.requests[-2:]] == [
        "profile", "calibrate",
    ]
    assert receipt["policy"]["operations_consumed"] == 7


def test_backend_operation_budget_is_shared_across_branch_rounds(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend()
    adapter = make_controller(repo, backend, request_budget=7)
    first = adapter.run(1, candidate_hash, manifest_hash)
    second = adapter.run(2, candidate_hash, manifest_hash)

    assert first["status"] == "ok"
    assert second["status"] == "infrastructure_error"
    assert "budget exhausted" in second["reason"]
    assert len(backend.requests) == 7


def test_cli_matches_command_controller_arguments(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = tmp_path / "backend.py"
    backend.write_text(
        """import json, sys
request=json.load(sys.stdin)
action=request['action']
handle='bz-a3-1:'+action
if action == 'calibrate': result={'status':'ok','handle':handle,'latency_us':10.0}
elif action == 'check': result={'status':'ok','handle':handle,'passed':True}
else:
 rows=[{'case':c,'samples_us':[10.0,10.0,10.0],'median_us':10.0} for c in request['cases']]
 result={'status':'ok','handle':handle,'cases':rows}
print(json.dumps(result))
"""
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema": "profiling-skill/audited-bz-controller-config/v1",
        "benchmark": "matmul", "round_count": 4, "request_budget": 24,
        "profile_repeats": 3, "variability_threshold": 0.25,
        "control_drift_threshold": 0.2,
        "devices": [{"id": "bz-a3-1/device-0", "device": 0}],
        "development_cases": [7, 8, 9], "all_cases": list(range(10)),
        "backend_command": [sys.executable, str(backend)],
    }))
    run = subprocess.run([
        sys.executable, str(ROOT / "scripts/audited_bz_controller.py"),
        "--config", str(config), "--state-dir", str(tmp_path / "state"),
        "--experiment", "1", "--candidate-sha256", candidate_hash,
        "--manifest-sha256", manifest_hash,
    ], cwd=repo, text=True, capture_output=True, check=False)

    assert run.returncode == 0, run.stderr
    receipt = json.loads(run.stdout)
    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
