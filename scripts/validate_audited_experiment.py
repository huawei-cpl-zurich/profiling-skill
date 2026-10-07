#!/usr/bin/env python3
"""Validate one retained audited-experiment branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from audited_contract import AuditError
from audited_verifier import validate_branch


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--base", default="main")
    parser.add_argument(
        "--migration-proof", action="append", nargs=3, default=[],
        metavar=("ATTESTATION", "FILE_SHA256", "ATTESTATION_SHA256"),
        help="ordered pinned controller-migration proof; repeat for each transition",
    )
    args = parser.parse_args(argv)
    proofs = [{
        "schema": "profiling-skill/audited-runtime-migration-trust/v1",
        "attestation_path": str(Path(path).resolve()),
        "attestation_file_sha256": file_sha,
        "attestation_sha256": seal,
    } for path, file_sha, seal in args.migration_proof]
    try:
        result = validate_branch(args.repo, args.base, migration_proofs=proofs)
    except AuditError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
