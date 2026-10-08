#!/usr/bin/env python3
"""Run a resumable, blinded profiling-skill behavioral battery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence


MANIFEST_SCHEMA = "profiling-skill/behavioral-gate-manifest/v1"
REPORT_SCHEMA = "profiling-skill/behavioral-gate-report/v1"
COUNTED_FAILURES = {
    "compile", "runtime", "profiler_command", "evidence", "interpretation"
}
INFRA_FAILURES = {"transport", "device_busy", "target_unavailable", "observer"}


class GateError(RuntimeError):
    pass


class ExecutorFailure(RuntimeError):
    def __init__(self, failure_type: str, message: str):
        super().__init__(message)
        self.failure_type = failure_type


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.is_dir():
        raise GateError(f"skill directory is unavailable: {path}")
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise GateError(f"skill contains a symlink: {item}")
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode() + b"\0")
            digest.update(f"{stat.S_IMODE(item.stat().st_mode):o}".encode() + b"\0")
            digest.update(item.read_bytes() + b"\0")
    return digest.hexdigest()


def _load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write("\n")
        temporary = Path(output.name)
    os.replace(temporary, path)


def _safe_copy_tree(source: Path, destination: Path) -> None:
    tree_digest(source)  # Reject links before copying any part of the skill.
    shutil.copytree(source, destination)


class SubprocessExecutor:
    """Adapter for an external agent launcher using request/output JSON files."""

    def __init__(self, command: Sequence[str], timeout: int = 900):
        if not command:
            raise GateError("executor command must not be empty")
        self.command = list(command)
        self.timeout = timeout

    def __call__(self, request: dict, workspace: Path, skill_root: Path) -> dict:
        request_path = workspace / "request.json"
        output_path = workspace / "answer.json"
        serializable = {key: value for key, value in request.items()
                        if key not in {"prompt_bytes", "_executor_environment"}}
        serializable["workspace"] = str(workspace)
        serializable["skill_root"] = str(skill_root)
        _write_json(request_path, serializable)
        environment = {
            "PATH": os.environ.get("PATH", ""), "HOME": str(workspace / ".home"),
            "CODEX_HOME": str(workspace / ".codex"),
            "AGENTS_HOME": str(workspace / ".agents-home"),
            "PROFILE_GATE_REQUEST": str(request_path),
            "PROFILE_GATE_OUTPUT": str(output_path),
            **request.get("_executor_environment", {}),
        }
        (workspace / ".home").mkdir()
        (workspace / ".codex").mkdir()
        (workspace / ".agents-home").mkdir()
        try:
            completed = subprocess.run(
                self.command, cwd=workspace, env=environment, text=True,
                capture_output=True, timeout=self.timeout, check=False,
            )
        except subprocess.TimeoutExpired as error:
            failure = ("interpretation" if request["mode"] == "interpretation"
                       else "evidence")
            raise ExecutorFailure(failure, "agent executor timed out") from error
        except OSError as error:
            raise ExecutorFailure("evidence", "agent executor unavailable") from error
        if completed.returncode or not output_path.is_file():
            raise ExecutorFailure(
                "interpretation" if request["mode"] == "interpretation" else "evidence",
                f"agent executor did not produce an answer (exit {completed.returncode})"
            )
        try:
            answer = _load_object(output_path)
        except GateError as error:
            failure = ("interpretation" if request["mode"] == "interpretation"
                       else "evidence")
            raise ExecutorFailure(failure, "agent answer is invalid") from error
        answer["executor"] = {
            "exit_code": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        }
        return answer


class Battery:
    def __init__(self, manifest_path: Path, run_root: Path,
                 executor: Callable[[dict, Path, Path], dict], *,
                 stop_after_attempts: int | None = None,
                 max_infra_attempts: int = 3):
        self.manifest_path = manifest_path.resolve()
        self.run_root = run_root.resolve()
        self.executor = executor
        self.stop_after_attempts = stop_after_attempts
        self.max_infra_attempts = max_infra_attempts
        self.manifest = self._validate_manifest(_load_object(self.manifest_path))
        self.state_path = self.run_root / "state.json"
        self.report_path = self.run_root / "report.json"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.state = (_load_object(self.state_path) if self.state_path.exists()
                      else {"schema": REPORT_SCHEMA, "records": []})
        if self.state.get("manifest_sha256") not in (None, file_digest(self.manifest_path)):
            raise GateError("resume manifest identity changed")
        self.state["manifest_sha256"] = file_digest(self.manifest_path)
        self.started = 0

    def _validate_manifest(self, manifest: dict) -> dict:
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise GateError("unsupported manifest schema")
        prompt = Path(manifest.get("prompt", "")).resolve()
        if not prompt.is_file() or manifest.get("prompt_sha256") != file_digest(prompt):
            raise GateError("prompt hash mismatch")
        skills = manifest.get("skills")
        if not isinstance(skills, dict) or set(skills) != {"current", "candidate"}:
            raise GateError("current and candidate skills are required")
        for arm, entry in skills.items():
            if not isinstance(entry, dict):
                raise GateError(f"invalid {arm} skill entry")
            path = Path(entry.get("path", "")).resolve()
            if entry.get("sha256") != tree_digest(path):
                raise GateError(f"{arm} skill hash mismatch")
            entry["path"] = str(path)
        cases = manifest.get("cases")
        if not isinstance(cases, list) or len(cases) != 4:
            raise GateError("exactly four frozen interpretation cases are required")
        seen = set()
        for case in cases:
            if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
                    or case["id"] in seen or case.get("product") not in {"a3", "a5"}):
                raise GateError("invalid or duplicate interpretation case")
            seen.add(case["id"])
            evidence = Path(case.get("evidence", "")).resolve()
            if not evidence.is_file() or case.get("evidence_sha256") != file_digest(evidence):
                raise GateError(f"evidence hash mismatch: {case.get('id')}")
            case["evidence"] = str(evidence)
            if (not isinstance(case.get("required_conclusions"), list)
                    or not all(isinstance(x, str) and x for x in case["required_conclusions"])
                    or not isinstance(case.get("capacity_supported_resources"), list)
                    or not all(isinstance(x, str) and x
                               for x in case["capacity_supported_resources"])):
                raise GateError(f"invalid scoring contract: {case['id']}")
        if Counter(case["product"] for case in cases) != {"a3": 2, "a5": 2}:
            raise GateError("frozen cases must contain two A3 and two A5 cases")
        acquisition = manifest.get("acquisition", {})
        interpretation = manifest.get("interpretation", {})
        if (acquisition.get("products") != ["a3", "a5"]
                or acquisition.get("complete_agents") != 3
                or not isinstance(acquisition.get("max_agent_candidates"), int)
                or acquisition["max_agent_candidates"] < 3
                or interpretation.get("agents_per_arm") != 3):
            raise GateError("battery cardinality must be 3 acquisition agents and 3 per arm")
        runtime_inputs = manifest.get("runtime_inputs", [])
        if not isinstance(runtime_inputs, list):
            raise GateError("runtime_inputs must be a list")
        names, environments = set(), set()
        reserved_environment = {
            "HOME", "PATH", "CODEX_HOME", "AGENTS_HOME",
            "PROFILE_GATE_REQUEST", "PROFILE_GATE_OUTPUT",
        }
        for item in runtime_inputs:
            if (not isinstance(item, dict)
                    or not isinstance(item.get("name"), str)
                    or not re.fullmatch(r"[A-Za-z0-9_.-]+", item["name"])
                    or item["name"] in {".", ".."}
                    or not isinstance(item.get("environment"), str)
                    or not re.fullmatch(r"[A-Z][A-Z0-9_]+", item["environment"])
                    or item["environment"] in reserved_environment):
                raise GateError("invalid scoped runtime input")
            source = Path(item.get("path", "")).resolve()
            if (not source.is_file() or source.is_symlink()
                    or item.get("sha256") != file_digest(source)):
                raise GateError(f"runtime input hash mismatch: {item.get('name')}")
            if item["name"] in names or item["environment"] in environments:
                raise GateError("duplicate scoped runtime input")
            names.add(item["name"]); environments.add(item["environment"])
            item["path"] = str(source)
        manifest["prompt"] = str(prompt)
        return manifest

    @property
    def records(self) -> list[dict]:
        return self.state["records"]

    def _terminal(self, unit: str) -> dict | None:
        return next((record for record in reversed(self.records)
                     if record["unit"] == unit
                     and record["classification"] != "discarded_infrastructure"), None)

    def _infra_count(self, unit: str) -> int:
        return sum(record["unit"] == unit
                   and record["classification"] == "discarded_infrastructure"
                   for record in self.records)

    def _acquisition_next(self) -> tuple[str, dict] | None:
        complete = 0
        maximum = self.manifest["acquisition"]["max_agent_candidates"]
        for agent in range(1, maximum + 1):
            prefix = f"acquisition/agent-{agent}"
            rows = [self._terminal(f"{prefix}/{product}")
                    for product in ("a3", "a5")]
            if any(row and row["classification"] == "counted_failure" for row in rows):
                continue
            if all(row and row["classification"] == "success" for row in rows):
                complete += 1
                if complete == 3:
                    return None
                continue
            product = "a3" if rows[0] is None else "a5"
            unit = f"{prefix}/{product}"
            return unit, {"mode": "acquisition", "agent_id": f"agent-{agent}",
                          "product": product}
        raise GateError("acquisition candidate budget cannot yield three complete agents")

    def _interpretation_next(self) -> tuple[str, dict] | None:
        for arm in ("current", "candidate"):
            for agent in range(1, 4):
                for case in self.manifest["cases"]:
                    unit = f"interpretation/{arm}/agent-{agent}/{case['id']}"
                    if self._terminal(unit) is None:
                        return unit, {"mode": "interpretation", "arm": arm,
                                      "agent_id": f"agent-{agent}", "case": case}
        return None

    def _workspace(self, unit: str, attempt: int, arm: str
                   ) -> tuple[Path, Path, dict[str, str]]:
        workspace = self.run_root / "workspaces" / unit / f"attempt-{attempt}"
        workspace.mkdir(parents=True, exist_ok=False)
        skill_root = workspace / ".agents/skills/ascend-profiling"
        skill_root.parent.mkdir(parents=True)
        _safe_copy_tree(Path(self.manifest["skills"][arm]["path"]), skill_root)
        shutil.copy2(self.manifest["prompt"], workspace / "prompt.md")
        runtime_environment = {}
        runtime_root = workspace / ".runtime-inputs"
        for item in self.manifest.get("runtime_inputs", []):
            runtime_root.mkdir(exist_ok=True)
            destination = runtime_root / item["name"]
            shutil.copy2(item["path"], destination)
            destination.chmod(0o400)
            runtime_environment[item["environment"]] = str(destination)
        return workspace, skill_root, runtime_environment

    def _request(self, spec: dict, workspace: Path,
                 runtime_environment: dict[str, str]) -> dict:
        prompt_bytes = Path(self.manifest["prompt"]).read_bytes()
        request = {
            "schema": "profiling-skill/behavioral-gate-request/v1",
            "mode": spec["mode"], "agent_id": spec["agent_id"],
            "prompt_sha256": self.manifest["prompt_sha256"],
            "prompt_bytes": prompt_bytes, "workspace": str(workspace),
            "_executor_environment": runtime_environment,
        }
        if spec["mode"] == "acquisition":
            request["product"] = spec["product"]
            request["agent_request"] = {
                "mode": "acquisition", "product": spec["product"],
                "prompt_sha256": self.manifest["prompt_sha256"],
            }
        else:
            case = spec["case"]
            shutil.copy2(case["evidence"], workspace / "evidence.json")
            request.update({"case_id": case["id"], "product": case["product"],
                            "evidence_sha256": case["evidence_sha256"]})
            request["agent_request"] = {
                "mode": "interpretation", "case_id": case["id"],
                "product": case["product"],
                "prompt_sha256": self.manifest["prompt_sha256"],
                "evidence_sha256": case["evidence_sha256"],
            }
        return request

    def _classify(self, answer: dict, spec: dict) -> tuple[str, str | None]:
        if answer.get("status") == "failure":
            failure_class, failure_type = answer.get("failure_class"), answer.get("failure_type")
            documented = (isinstance(answer.get("logs"), list)
                          and bool(answer["logs"])
                          and all(isinstance(item, str) and item for item in answer["logs"]))
            if (failure_class == "infrastructure" and failure_type in INFRA_FAILURES
                    and documented):
                return "discarded_infrastructure", failure_type
            if failure_class == "counted" and failure_type in COUNTED_FAILURES:
                return "counted_failure", failure_type
            return "counted_failure", "evidence"
        if answer.get("status") != "success":
            return "counted_failure", "evidence"
        if spec["mode"] == "acquisition":
            valid = (answer.get("product") == spec["product"]
                     and isinstance(answer.get("evidence_sha256"), str)
                     and re.fullmatch(r"[0-9a-f]{64}", answer["evidence_sha256"])
                     and isinstance(answer.get("handle"), str) and answer["handle"]
                     and isinstance(answer.get("commands"), list)
                     and all(isinstance(x, str) for x in answer["commands"])
                     and isinstance(answer.get("reasoning"), str)
                     and isinstance(answer.get("logs"), list)
                     and isinstance(answer.get("evidence_file"), str))
            return ("success", None) if valid else ("counted_failure", "evidence")
        case = spec["case"]
        valid = (answer.get("case_id") == case["id"]
                 and answer.get("evidence_sha256") == case["evidence_sha256"]
                 and isinstance(answer.get("conclusions"), list)
                 and isinstance(answer.get("saturation_claims"), list)
                 and isinstance(answer.get("reasoning"), str)
                 and isinstance(answer.get("logs"), list))
        return ("success", None) if valid else ("counted_failure", "interpretation")

    def _score(self, answer: dict, case: dict) -> dict:
        required = set(case["required_conclusions"])
        conclusions = {item for item in answer.get("conclusions", [])
                       if isinstance(item, str)}
        supported = set(case["capacity_supported_resources"])
        claims = answer.get("saturation_claims", [])
        claimed = [claim.get("resource", "<missing>")
                   if isinstance(claim, dict) else "<invalid>" for claim in claims]
        unsupported = sorted({resource for resource in claimed
                              if resource not in supported})
        missing = sorted(required - conclusions)
        return {"passed": not missing and not unsupported,
                "missing": missing, "unsupported": unsupported}

    def _attempt(self, unit: str, spec: dict) -> None:
        infra_attempts = self._infra_count(unit)
        if infra_attempts >= self.max_infra_attempts:
            raise GateError(f"infrastructure retry budget exhausted: {unit}")
        attempt = 1 + sum(record["unit"] == unit for record in self.records)
        arm = spec.get("arm", "candidate")
        workspace, skill_root, runtime_environment = self._workspace(unit, attempt, arm)
        request = self._request(spec, workspace, runtime_environment)
        try:
            answer = self.executor(request, workspace, skill_root)
        except ExecutorFailure as error:
            answer = {"status": "failure", "failure_class": "counted",
                      "failure_type": error.failure_type, "logs": [str(error)]}
        finally:
            shutil.rmtree(workspace / ".runtime-inputs", ignore_errors=True)
        case = spec.get("case")
        try:
            identity_changed = (
                file_digest(workspace / "prompt.md") != self.manifest["prompt_sha256"]
                or tree_digest(skill_root) != self.manifest["skills"][arm]["sha256"]
                or (case is not None
                    and file_digest(workspace / "evidence.json") != case["evidence_sha256"])
            )
        except (OSError, GateError):
            identity_changed = True
        if identity_changed:
            answer = {"status": "failure", "failure_class": "counted",
                      "failure_type": "evidence", "logs": ["frozen input changed"]}
        if not isinstance(answer, dict):
            answer = {"status": "failure", "failure_class": "counted",
                      "failure_type": "evidence", "logs": []}
        classification, failure_type = self._classify(answer, spec)
        retained_evidence = None
        if spec["mode"] == "acquisition" and classification == "success":
            relative = Path(answer["evidence_file"])
            source = workspace / relative
            escaped = (relative.is_absolute() or ".." in relative.parts
                       or not source.exists() or source.is_symlink()
                       or not source.is_file()
                       or not source.resolve().is_relative_to(workspace.resolve()))
            if escaped or file_digest(source) != answer["evidence_sha256"]:
                classification, failure_type = "counted_failure", "evidence"
            else:
                retained = self.run_root / "evidence" / unit / f"attempt-{attempt}.json"
                retained.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, retained)
                retained.chmod(0o444)
                if file_digest(retained) != answer["evidence_sha256"]:
                    classification, failure_type = "counted_failure", "evidence"
                    retained.unlink(missing_ok=True)
                else:
                    retained_evidence = {
                        "path": retained.relative_to(self.run_root).as_posix(),
                        "sha256": answer["evidence_sha256"],
                    }
        record = {
            "unit": unit, "attempt": attempt, "mode": spec["mode"],
            "skill_sha256": self.manifest["skills"][arm]["sha256"],
            "agent_request": request["agent_request"],
            "classification": classification, "failure_type": failure_type,
            "answer": answer,
        }
        if retained_evidence is not None:
            record["retained_evidence"] = retained_evidence
        if spec["mode"] == "interpretation" and classification == "success":
            record["arm"] = arm
            record["score"] = self._score(answer, spec["case"])
        self.records.append(record)
        self.started += 1
        _write_json(self.state_path, self.state)

    def _report(self, status: str) -> dict:
        terminal_acquisition = [r for r in self.records if r["mode"] == "acquisition"
                                and r["classification"] != "discarded_infrastructure"]
        attempted_agents = {r["unit"].split("/")[1] for r in terminal_acquisition}
        completed = 0
        for agent in attempted_agents:
            rows = [r for r in terminal_acquisition if f"/{agent}/" in r["unit"]]
            if len(rows) == 2 and all(r["classification"] == "success" for r in rows):
                completed += 1
        interpretation = {}
        for arm in ("current", "candidate"):
            rows = [r for r in self.records if r.get("arm") == arm]
            interpretation[arm] = {
                "passed": sum(r.get("score", {}).get("passed") is True for r in rows),
                "total": 12 if status == "complete" else len(rows),
            }
        candidate_complete = interpretation["candidate"]["passed"] == 12
        beats = interpretation["candidate"]["passed"] > interpretation["current"]["passed"]
        report = {
            "schema": REPORT_SCHEMA, "status": status,
            "manifest_sha256": self.state["manifest_sha256"],
            "records": self.records,
            "counts": dict(Counter(r["classification"] for r in self.records)),
            "acquisition": {"complete_agents": completed,
                            "agents_attempted": len(attempted_agents)},
            "interpretation": interpretation,
            "acceptance": {"candidate_12_of_12": candidate_complete,
                           "candidate_beats_current": beats,
                           "passed": status == "complete" and completed == 3
                           and candidate_complete and beats},
        }
        _write_json(self.report_path, report)
        return report

    def run(self) -> dict:
        while True:
            if self.stop_after_attempts is not None and self.started >= self.stop_after_attempts:
                return self._report("incomplete")
            next_unit = self._acquisition_next()
            if next_unit is None:
                next_unit = self._interpretation_next()
            if next_unit is None:
                return self._report("complete")
            if self._infra_count(next_unit[0]) >= self.max_infra_attempts:
                return self._report("blocked_infrastructure")
            self._attempt(*next_unit)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--executor-json", required=True,
                        help="JSON string array for the external agent launcher")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-infra-attempts", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        command = json.loads(args.executor_json)
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise GateError("executor command must be a JSON string array")
        report = Battery(
            args.manifest, args.run_root,
            SubprocessExecutor(command, timeout=args.timeout),
            max_infra_attempts=args.max_infra_attempts,
        ).run()
    except (GateError, json.JSONDecodeError) as error:
        raise SystemExit(f"behavioral gate failed: {error}") from error
    print(json.dumps(report, sort_keys=True))
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
