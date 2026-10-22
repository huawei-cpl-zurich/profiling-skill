"""Planning-only client: no backend, transport configuration, or workload code."""
import json
import socket
import sys


def response(arguments):
    if arguments in (["help"], ["--help"], ["-h"], ["capabilities"]):
        return {"returncode": 0, "stderr": "", "stdout": json.dumps({
            "mode": "planning-only", "execution_enabled": False,
            "available": ["help", "capabilities"],
            "production_interface": {
                "benchmark": "Correctness, warm full-forward latency and busy time",
                "profile --kernel NAME": "Candidate-only diagnostic profiling evidence",
            },
        }) + "\n"}
    return {"returncode": 77, "stdout": "", "stderr":
            "Planning-only smoke: execution is disabled; no workload dispatched.\n"}


def main():
    if sys.argv[1:2] == ["serve"]:
        print(json.dumps(response(sys.argv[4:])))
        return 0
    if len(sys.argv) < 3 or sys.argv[1] != "client":
        return 2
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.connect(sys.argv[2])
        stream.sendall(json.dumps({"arguments": sys.argv[3:]}).encode() + b"\n")
        envelope = json.loads(stream.makefile("rb").readline())
    # The host helper itself always completes; its payload carries denial status.
    result = json.loads(envelope["stdout"]) if envelope["returncode"] == 0 else envelope
    print(result["stdout"], end="")
    print(result["stderr"], end="", file=sys.stderr)
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
