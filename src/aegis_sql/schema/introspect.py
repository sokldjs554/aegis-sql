"""Database introspection into a :class:`~aegis_sql.types.SchemaGraph`.

Two things make this more than a ``PRAGMA table_info`` wrapper:

1. **Data-dictionary recovery.**  SQLite has no ``COMMENT ON COLUMN``, and in
   practice Korean enterprise schemas keep the logical (Korean) name in a DDL
   comment or a side-car dictionary.  We parse the original ``CREATE TABLE``
   text out of ``sqlite_master`` and recover the ``-- 계약상태코드`` trailing
   comments, then overlay ``data/demo/dictionary.yaml`` when present.
2. **Code-group inference.**  Columns ending in ``_CD`` are matched against the
   common-code table so that the generator knows ``CTRT_STAT_CD`` joins
   ``TB_COMM_CD`` on ``CD_GRP = 'CTRT_STAT'``.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from aegis_sql.observability.logging import get_logger
from aegis_sql.types import ColumnInfo, ForeignKey, SchemaGraph, Sensitivity, TableInfo

log = get_logger("schema.introspect")

_COL_COMMENT_RE = re.compile(
    r"^\s*(?:\"|`|\[)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\"|`|\])?\s+"
    r"(?P<rest>[^,]*?)\s*(?:,\s*)?--\s*(?P<comment>.+?)\s*$"
)
_TABLE_COMMENT_RE = re.compile(r"--\s*(?P<comment>[^\n]+)\s*\n\s*CREATE\s+TABLE\s+(?P<table>\w+)", re.IGNORECASE)

#: Common-code table conventions.  Kept configurable rather than hard-coded so
#: the same engine works against other Korean cores (TB_CODE / TC_CMMN_CD / ...).
CODE_TABLE_CANDIDATES = ("TB_COMM_CD", "TB_CODE", "TC_CMMN_CD", "COMMON_CODE")
CODE_GROUP_COLUMN = "CD_GRP"
CODE_VALUE_COLUMN = "CD"
CODE_NAME_COLUMN = "CD_NM"


class SQLiteIntrospector:
    """Reads a SQLite file into a fully-annotated :class:`SchemaGraph`."""

    def __init__(self, path: str | Path, dictionary_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.dictionary_path = Path(dictionary_path) if dictionary_path else None

    # -- public ----------------------------------------------------------- #

    def introspect(self, with_row_counts: bool = True) -> SchemaGraph:
        if not self.path.exists():
            raise FileNotFoundError(
                f"database not found: {self.path}\n"
                "  run `make demo-db` (or `python scripts/build_demo_db.py`) first."
            )
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            graph = SchemaGraph(dialect="sqlite", name=self.path.stem)
            ddl_by_table = self._read_ddl(conn)
            comments = self._parse_comments(ddl_by_table)

            for table_name in ddl_by_table:
                table = TableInfo(name=table_name, comment=comments.get(table_name, {}).get("__table__"))
                fks = self._read_foreign_keys(conn, table_name)
                fk_by_col = {fk.from_column.lower(): fk for fk in fks}
                table.foreign_keys = fks

                for row in conn.execute(f'PRAGMA table_info("{table_name}")'):
                    _, name, dtype, notnull, _default, pk = row
                    col = ColumnInfo(
                        table=table_name,
                        name=name,
                        dtype=(dtype or "TEXT").upper(),
                        nullable=not notnull,
                        is_primary_key=bool(pk),
                        comment=comments.get(table_name, {}).get(name),
                        foreign_key=fk_by_col.get(name.lower()),
                        sensitivity=Sensitivity.PUBLIC,
                    )
                    table.columns.append(col)
                    if pk:
                        table.primary_key.append(name)

                if with_row_counts:
                    try:
                        table.row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
                    except sqlite3.Error:  # pragma: no cover - defensive
                        table.row_count = -1
                graph.tables[table_name] = table

            self._apply_dictionary(graph)
            self._infer_code_groups(conn, graph)
            log.info(
                "introspected schema",
                db=self.path.name,
                tables=len(graph.tables),
                columns=len(graph.all_columns),
                fingerprint=graph.fingerprint(),
            )
            return graph
        finally:
            conn.close()

    # -- internals -------------------------------------------------------- #

    def _read_ddl(self, conn: sqlite3.Connection) -> dict[str, str]:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return {name: (sql or "") for name, sql in rows}

    def _parse_comments(self, ddl_by_table: dict[str, str]) -> dict[str, dict[str, str]]:
        """Recover ``-- 한글논리명`` trailing comments from the stored DDL."""
        out: dict[str, dict[str, str]] = {}
        source = self._source_ddl_text()
        for table in ddl_by_table:
            per_col: dict[str, str] = {}
            for line in ddl_by_table[table].splitlines():
                m = _COL_COMMENT_RE.match(line)
                if m and not line.strip().upper().startswith(("PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "CHECK")):
                    per_col[m.group("name")] = m.group("comment").strip()
            out[table] = per_col
        # Table-level comments live on the line above CREATE TABLE in the source file.
        if source:
            for m in _TABLE_COMMENT_RE.finditer(source):
                tbl = m.group("table")
                if tbl in out:
                    out[tbl]["__table__"] = m.group("comment").strip()
        return out

    def _source_ddl_text(self) -> str:
        candidate = self.path.parent / "schema.sql"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
        return ""

    def _read_foreign_keys(self, conn: sqlite3.Connection, table: str) -> list[ForeignKey]:
        fks: list[ForeignKey] = []
        for row in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            # (id, seq, table, from, to, on_update, on_delete, match)
            _id, _seq, ref_table, from_col, to_col = row[0], row[1], row[2], row[3], row[4]
            fks.append(ForeignKey(table, from_col, ref_table, to_col or from_col))
        return fks

    def _apply_dictionary(self, graph: SchemaGraph) -> None:
        """Overlay an external data dictionary (mirrors a Postgres COMMENT dump)."""
        path = self.dictionary_path or (self.path.parent / "dictionary.yaml")
        if not Path(path).exists():
            return
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for tname, payload in (data.get("tables") or {}).items():
            table = graph.table(tname)
            if not table:
                continue
            if payload.get("comment"):
                table.comment = payload["comment"]
            for cname, ccomment in (payload.get("columns") or {}).items():
                col = table.column(cname)
                if col and ccomment:
                    col.comment = ccomment

    def _infer_code_groups(self, conn: sqlite3.Connection, graph: SchemaGraph) -> None:
        """Link ``*_CD`` columns to their group in the common-code table.

        The heuristic mirrors the naming convention actually used by Korean
        insurers: ``<GROUP>_CD`` where ``<GROUP>`` is the ``CD_GRP`` value
        (``CTRT_STAT_CD`` → ``CTRT_STAT``).  We verify against the real code
        table before recording the link, so a wrong guess never reaches a prompt.
        """
        code_table = next((t for t in CODE_TABLE_CANDIDATES if graph.table(t)), None)
        if not code_table:
            return
        try:
            groups = {
                row[0]
                for row in conn.execute(f'SELECT DISTINCT "{CODE_GROUP_COLUMN}" FROM "{code_table}"')
            }
        except sqlite3.Error:  # pragma: no cover - schema without the convention
            return

        for col in graph.all_columns:
            if col.table == code_table or not col.name.upper().endswith("_CD"):
                continue
            stem = col.name.upper()[:-3]
            for candidate in (stem, stem.split("_")[-1], stem.replace("_", "")):
                if candidate in groups:
                    col.code_group = candidate
                    break

        linked = sum(1 for c in graph.all_columns if c.code_group)
        log.debug("code groups inferred", code_table=code_table, linked_columns=linked, groups=len(groups))


def introspect(path: str | Path, **kwargs: Any) -> SchemaGraph:
    """Convenience wrapper used across the codebase."""
    return SQLiteIntrospector(path).introspect(**kwargs)
