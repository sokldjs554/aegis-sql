"""Contract tests for reproducible Spider SQLite database bootstrapping."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_spider_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("spider_data/database/school/school.sqlite", b"sqlite-fixture")
        archive.writestr("spider_data/tables.json", "[]")
        archive.writestr("spider_data/dev.json", "[]")


def test_prepare_archive_verifies_checksum_and_returns_database_root(tmp_path: Path):
    from aegis_sql.research.spider_data import prepare_spider_archive

    archive = tmp_path / "spider_data.zip"
    _write_spider_zip(archive)
    out = tmp_path / "out"

    result = prepare_spider_archive(
        archive,
        out,
        expected_sha256=_sha256(archive),
        source_repo="fixture/spider",
        source_revision="abc123",
    )

    assert result.database_root == out / "spider_data" / "database"
    assert (result.database_root / "school" / "school.sqlite").read_bytes() == b"sqlite-fixture"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["archive_sha256"] == _sha256(archive)
    assert manifest["source_repo"] == "fixture/spider"
    assert manifest["source_revision"] == "abc123"


def test_prepare_archive_rejects_wrong_checksum(tmp_path: Path):
    from aegis_sql.research.spider_data import prepare_spider_archive

    archive = tmp_path / "spider_data.zip"
    _write_spider_zip(archive)

    with pytest.raises(ValueError, match="SHA-256"):
        prepare_spider_archive(archive, tmp_path / "out", expected_sha256="0" * 64)


def test_prepare_archive_rejects_zip_path_traversal(tmp_path: Path):
    from aegis_sql.research.spider_data import prepare_spider_archive

    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "nope")

    with pytest.raises(ValueError, match="unsafe archive member"):
        prepare_spider_archive(
            archive,
            tmp_path / "out",
            expected_sha256=_sha256(archive),
        )
    assert not (tmp_path / "escaped.txt").exists()


def test_prepare_archive_requires_expected_spider_layout(tmp_path: Path):
    from aegis_sql.research.spider_data import prepare_spider_archive

    archive = tmp_path / "wrong-layout.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("something/else.txt", "missing database")

    with pytest.raises(ValueError, match="Spider database directory"):
        prepare_spider_archive(
            archive,
            tmp_path / "out",
            expected_sha256=_sha256(archive),
        )
