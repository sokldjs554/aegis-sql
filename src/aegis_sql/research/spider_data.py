"""Reproducible preparation of the SQLite databases used by Spider evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SPIDER_DB_REPO = "HAL-9001/spider-databases"
SPIDER_DB_REVISION = "4a01bbac6520cd35b216db9e1724e5e1ada60aa4"
SPIDER_DB_FILENAME = "spider_data.zip"
SPIDER_DB_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
_MAX_UNCOMPRESSED_BYTES = 2_000_000_000


@dataclass(frozen=True, slots=True)
class SpiderDataPreparation:
    database_root: Path
    manifest_path: Path
    archive_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(root: Path, info: zipfile.ZipInfo) -> Path:
    name = PurePosixPath(info.filename)
    if name.is_absolute() or not name.parts or any(part in {"", ".", ".."} for part in name.parts):
        raise ValueError(f"unsafe archive member: {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"unsafe archive member (symlink): {info.filename!r}")
    destination = (root / Path(*name.parts)).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"unsafe archive member: {info.filename!r}") from exc
    return destination


def _extract_verified_zip(archive: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        total = sum(info.file_size for info in members)
        if total > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"archive expands to {total:,} bytes, above safety limit {_MAX_UNCOMPRESSED_BYTES:,}"
            )
        destinations = [(info, _safe_member_path(output_dir, info)) for info in members]
        for info, destination in destinations:
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def prepare_spider_archive(
    archive: str | Path,
    output_dir: str | Path,
    *,
    expected_sha256: str = SPIDER_DB_SHA256,
    source_repo: str = SPIDER_DB_REPO,
    source_revision: str = SPIDER_DB_REVISION,
) -> SpiderDataPreparation:
    """Verify, safely extract, and provenance-stamp a Spider database archive."""
    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Spider archive not found: {archive_path}")
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "Spider archive SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, got {actual_sha256.lower()}"
        )

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted_root = root / "spider_data"
    if extracted_root.exists():
        shutil.rmtree(extracted_root)
    _extract_verified_zip(archive_path, root)

    database_root = extracted_root / "database"
    if not database_root.is_dir():
        if extracted_root.exists():
            shutil.rmtree(extracted_root)
        raise ValueError(
            "Spider database directory missing after extraction; expected spider_data/database"
        )

    sqlite_files = sorted(database_root.glob("*/*.sqlite"))
    if not sqlite_files:
        shutil.rmtree(extracted_root)
        raise ValueError("Spider database directory contains no <db_id>/<db_id>.sqlite files")

    manifest_path = root / "spider-databases.manifest.json"
    manifest = {
        "source_repo": source_repo,
        "source_revision": source_revision,
        "archive_filename": archive_path.name,
        "archive_sha256": actual_sha256,
        "database_root": str(database_root),
        "sqlite_databases": len(sqlite_files),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return SpiderDataPreparation(
        database_root=database_root,
        manifest_path=manifest_path,
        archive_sha256=actual_sha256,
    )


__all__ = [
    "SPIDER_DB_FILENAME",
    "SPIDER_DB_REPO",
    "SPIDER_DB_REVISION",
    "SPIDER_DB_SHA256",
    "SpiderDataPreparation",
    "prepare_spider_archive",
    "sha256_file",
]
