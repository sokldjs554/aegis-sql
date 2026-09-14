#!/usr/bin/env python3
"""Build or verify the compact evidence manifest used by the public console."""

from __future__ import annotations

import argparse
import json

from aegis_sql.config import PROJECT_ROOT
from aegis_sql.research.evidence import build_model_experiment_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the published file is stale")
    args = parser.parse_args()

    output = PROJECT_ROOT / "data" / "research" / "model_experiment_evidence.json"
    rendered = json.dumps(build_model_experiment_evidence(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("model experiment evidence is stale; run this script without --check")
        print(f"evidence current: {output.relative_to(PROJECT_ROOT)}")
        return

    output.write_text(rendered, encoding="utf-8")
    print(f"evidence written: {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
