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
    args = parser.parse_args(argv)
    try:
        result = validate_branch(args.repo, args.base)
    except AuditError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
