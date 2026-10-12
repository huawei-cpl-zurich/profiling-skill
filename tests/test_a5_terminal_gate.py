import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]


def load(path, name):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


gate = load(ROOT / "scripts/a5_terminal_gate.py", "a5_terminal_gate_test")
builder = load(ROOT / "scripts/build_a5_gate_bundle.py", "build_a5_gate_bundle_test")
runner = load(ROOT / "scripts/simple_experiment_runner.py", "simple_runner_a5_gate_test")


def declaration(task):
    return builder.manifest(task)


def prepared_paths(tmp_path, task):
    harness = tmp_path / "harness"
    (harness / "scripts").mkdir(parents=True)
    (harness / f"benchmarks/{task}").mkdir(parents=True)
    for name in ("a3_benchmark_runner.py", "fully_fused_contract.py"):
        (harness / "scripts" / name).write_text("fixture\n")
    for name in ("baseline.py", "cases.jsonl"):
        (harness / f"benchmarks/{task}/{name}").write_text("fixture\n")
    candidate = tmp_path / "candidate.py"
    candidate.write_text("candidate\n")
    manifest = tmp_path / "candidate.manifest.json"
    manifest.write_text(json.dumps(declaration(task)))
    return harness, candidate, manifest


def fake_modules(task, status="ok"):
    spec = gate.SPECS[task]
    calls = []
    evidence = [
        {"schema": "profiling-skill/fusion-evidence/v1", "case": case,
         "output_launch_id": "launch", "operators": [{
             "name": spec["kernel_name"], "origin": "triton",
             "entrypoint": spec["entrypoint"], "launch_id": "launch",
             "component": "aic",
         }]}
        for case in spec["cases"]
    ]

    def execute(job):
        calls.append(job)
        failed = status != "ok" and len(calls) == 1
        return {"status": status if failed else "ok", "passed": not failed,
                "diagnostics": "compile failed" if failed else "",
                "fusion_evidence": evidence if len(calls) == 1 else []}

    class FusionError(ValueError):
        pass

    fusion = SimpleNamespace(
        FusionContractError=FusionError,
        validate_manifest=lambda value: value,
        validate_fusion_evidence=lambda value, rows, expected_cases: {
            "entrypoint": value["entrypoint"], "kernel_name": value["kernel_name"],
            "cases": list(expected_cases), "logical_launches_per_case": 1,
        },
    )
    return SimpleNamespace(execute=execute), fusion, calls


def test_gate_checks_every_case_before_ranked_outer_profile_launches(tmp_path, monkeypatch):
    root, candidate, manifest = prepared_paths(tmp_path, "gdn")
    run, fusion, calls = fake_modules("gdn")
    modules = iter((run, fusion))
    monkeypatch.setattr(gate, "load", lambda *_: next(modules))

    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, manifest, "gdn", repeats=3,
    )

    assert outcome == "success" and diagnostics == ""
    assert calls[0]["scope"] == "full" and calls[0]["cases"] == list(range(50))
    assert [job["cases"][0] for job in calls[1:]] == [40, 49, 47, 46, 45] * 3
    assert summary["expected_sample_count"] == 15
    assert summary["fusion"]["logical_launches_per_case"] == 1


def test_gate_counts_compile_failure_and_never_enters_profile_phase(tmp_path, monkeypatch):
    root, candidate, manifest = prepared_paths(tmp_path, "bsa")
    run, fusion, calls = fake_modules("bsa", status="compile_error")
    modules = iter((run, fusion))
    monkeypatch.setattr(gate, "load", lambda *_: next(modules))

    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, manifest, "bsa", repeats=3,
    )

    assert outcome == "candidate_failure" and diagnostics == "compile failed"
    assert summary == {} and len(calls) == 1


def test_builder_is_deterministic_and_emits_three_bound_configs(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    one = builder.generate(ROOT, first, wrapper_root=tmp_path / "wrappers",
                           state_root=tmp_path / "state")
    two = builder.generate(ROOT, second, wrapper_root=tmp_path / "wrappers",
                           state_root=tmp_path / "other-state")
    first_tar, second_tar = first / "a5-terminal-gates.tar", second / "a5-terminal-gates.tar"

    assert first_tar.read_bytes() == second_tar.read_bytes()
    digest = hashlib.sha256(first_tar.read_bytes()).hexdigest()
    assert one["bundle"]["archive_sha256"] == digest == two["bundle"]["archive_sha256"]
    assert set(one["tasks"]) == {"gdn", "bsa", "matmul"}
    assert one["tasks"]["gdn"]["remote_transport"]["warm_up"] == 50
    assert one["tasks"]["matmul"]["remote_transport"]["launch_count"] == 9
    with tarfile.open(first_tar) as archive:
        assert set(archive.getnames()) == set(builder.FILES) | {"bundle-manifest.json"}
    assert digest in (first / "bsa.sh").read_text()


class FakeWrappers:
    def __init__(self, profile_stdout):
        self.commands, self.profile_stdout = [], profile_stdout

    def __call__(self, command, **_kwargs):
        self.commands.append(command)
        name = Path(command[0]).name
        if name == "triton.sh":
            return subprocess.CompletedProcess(command, 0, "ready\n", "")
        if name == "select-device.sh":
            return subprocess.CompletedProcess(command, 0, "BZ_A5_DEVICE=6\n", "")
        if name == "scp.sh":
            return subprocess.CompletedProcess(command, 0, "BZ_A5_TRANSFER_STATE=completed\n", "")
        assert name == "profile-msprof.sh"
        return subprocess.CompletedProcess(command, 0, self.profile_stdout, "")


def wrapper_root(tmp_path):
    root = tmp_path / "wrappers"
    root.mkdir()
    for name in ("triton.sh", "select-device.sh", "scp.sh", "profile-msprof.sh"):
        path = root / name
        path.write_text("#!/bin/sh\n")
        path.chmod(0o700)
    return root


def test_approved_wrapper_stages_digest_bundle_and_profiles_ranked_launches(tmp_path):
    generated = builder.generate(ROOT, tmp_path / "generated",
                                 wrapper_root=wrapper_root(tmp_path),
                                 state_root=tmp_path / "state")
    config = dict(generated["tasks"]["gdn"]["remote_transport"])
    config["kernel_name"] = gate.SPECS["gdn"]["kernel_name"]
    config["candidate_manifest_path"] = generated["tasks"]["gdn"]["candidate_manifest_path"]
    envelope = {"schema": runner.RESULT_SCHEMA, "outcome": "success",
                "profile_summary": {"benchmark": "gdn"}}
    encoded = base64.b64encode(json.dumps(envelope).encode()).decode()
    csv_text = "Kernel Name,Task Duration(us)\n" + "".join(
        f"chunk_gated_delta_rule_kernel,{value}.0\n" for value in range(1, 16)
    )
    stdout = (f"MSPROF_PROFILE_RESULT_BASE64={encoded}\n"
              f"MSPROF_PROFILE_CSV_BASE64={base64.b64encode(csv_text.encode()).decode()}\n")
    invoke = FakeWrappers(stdout)
    remote = runner.BzA5WrapperRemote(config, invoke=invoke)
    request = {"target": "bz-a5", "campaign": "c", "cell": "a5-gdn", "round": 2,
               "remote_workspace": "/remote/work"}
    assert remote.preflight(request)["physical_device"] == 6
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\necho ok\n")

    result = remote.observe(remote.submit(request, script))

    uploads = [command for command in invoke.commands if Path(command[0]).name == "scp.sh"]
    assert len(uploads) == 2 and config["bundle_sha256"] in uploads[0][-1]
    receipt = json.loads(next((tmp_path / "state/gdn").glob("*.json")).read_text())
    assert receipt["bundle_sha256"] == config["bundle_sha256"]
    assert len(receipt["manifest_sha256"]) == 64
    profile = next(command for command in invoke.commands
                   if Path(command[0]).name == "profile-msprof.sh")
    assert profile[profile.index("--warm-up") + 1] == "50"
    assert profile[profile.index("--launch-count") + 1] == "15"
    assert result["outcome"] == "success"
    assert result["profile_summary"]["sample_count"] == 15
    assert [row["case"] for row in result["profile_summary"]["cases"]] == [40, 49, 47, 46, 45]


def test_wrong_profile_cardinality_is_infrastructure_not_candidate_failure(tmp_path):
    generated = builder.generate(ROOT, tmp_path / "generated",
                                 wrapper_root=wrapper_root(tmp_path),
                                 state_root=tmp_path / "state")
    config = dict(generated["tasks"]["matmul"]["remote_transport"])
    config["kernel_name"] = gate.SPECS["matmul"]["kernel_name"]
    config["candidate_manifest_path"] = generated["tasks"]["matmul"]["candidate_manifest_path"]
    envelope = base64.b64encode(json.dumps({
        "schema": runner.RESULT_SCHEMA, "outcome": "success",
    }).encode()).decode()
    csv_text = "Kernel Name,Task Duration(us)\nstreaming_matmul_add_kernel_mix_aic,1.0\n"
    invoke = FakeWrappers(
        f"MSPROF_PROFILE_RESULT_BASE64={envelope}\n"
        f"MSPROF_PROFILE_CSV_BASE64={base64.b64encode(csv_text.encode()).decode()}\n"
    )
    remote = runner.BzA5WrapperRemote(config, invoke=invoke)
    request = {"target": "bz-a5", "campaign": "c", "cell": "a5-matmul", "round": 1,
               "remote_workspace": "/remote/work"}
    remote.preflight(request)
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\necho ok\n")

    result = remote.observe(remote.submit(request, script))

    assert result["outcome"] == "infrastructure_failure"
    assert "exact ranked launch count" in result["diagnostics"]
