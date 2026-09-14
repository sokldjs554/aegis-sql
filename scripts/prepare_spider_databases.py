#!/usr/bin/env python3
"""Download and safely prepare the pinned Spider SQLite database archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_sql.research.spider_data import (
    SPIDER_DB_FILENAME,
    SPIDER_DB_REPO,
    SPIDER_DB_REVISION,
    SPIDER_DB_SHA256,
    prepare_spider_archive,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare pinned Spider databases for external evaluation")
    ap.add_argument("--out", default="data/external/spider-db")
    ap.add_argument(
        "--archive",
        default="",
        help="Optional local spider_data.zip; otherwise download the pinned Hugging Face artifact",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.archive:
        archive = Path(args.archive).resolve()
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit('huggingface_hub is missing; run `pip install -e ".[hf]"`') from exc
        archive = Path(
            hf_hub_download(
                repo_id=SPIDER_DB_REPO,
                repo_type="dataset",
                filename=SPIDER_DB_FILENAME,
                revision=SPIDER_DB_REVISION,
            )
        ).resolve()

    result = prepare_spider_archive(
        archive,
        Path(args.out),
        expected_sha256=SPIDER_DB_SHA256,
        source_repo=SPIDER_DB_REPO,
        source_revision=SPIDER_DB_REVISION,
    )
    print(
        json.dumps(
            {
                "database_root": str(result.database_root),
                "manifest": str(result.manifest_path),
                "archive_sha256": result.archive_sha256,
                "source_repo": SPIDER_DB_REPO,
                "source_revision": SPIDER_DB_REVISION,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
