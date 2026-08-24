"""Column profiling — the ingredient that makes *value* linking possible.

Text-to-SQL fails on Korean legacy schemas not only because column names are
cryptic, but because the *values* are too: a user asks for ``"실효된 계약"`` and
the answer requires ``CTRT_STAT_CD = '02'``.  Profiling extracts representative
values per column (with the code-table label attached where one exists) and
caches them, so schema linking can match a question's literals against real
data instead of hallucinating predicates.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aegis_sql.observability.logging import get_logger
from aegis_sql.schema.introspect import CODE_GROUP_COLUMN, CODE_NAME_COLUMN, CODE_VALUE_COLUMN
from aegis_sql.types import SchemaGraph

log = get_logger("schema.profile")

_NUMERIC = {"INTEGER", "INT", "REAL", "NUMERIC", "DECIMAL", "FLOAT", "BIGINT", "DOUBLE"}


@dataclass(slots=True)
class ColumnProfile:
    table: str
    column: str
    distinct_count: int = -1
    null_ratio: float = 0.0
    min_value: str | None = None
    max_value: str | None = None
    avg_length: float = 0.0
    #: Up to ``max_values`` representative values.
    values: list[str] = field(default_factory=list)
    #: For code columns: ``{"01": "정상", ...}``.
    code_labels: dict[str, str] = field(default_factory=dict)
    #: True when the column looks like a low-cardinality categorical.
    is_categorical: bool = False
    #: True when the column holds ``YYYYMMDD`` strings (Korean legacy dates).
    is_yyyymmdd: bool = False

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(slots=True)
class SchemaProfile:
    fingerprint: str
    columns: dict[str, ColumnProfile] = field(default_factory=dict)

    def get(self, table: str, column: str) -> ColumnProfile | None:
        return self.columns.get(f"{table}.{column}")

    def to_json(self) -> str:
        return json.dumps(
            {"fingerprint": self.fingerprint, "columns": {k: asdict(v) for k, v in self.columns.items()}},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> SchemaProfile:
        data = json.loads(payload)
        prof = cls(fingerprint=data["fingerprint"])
        for key, val in data["columns"].items():
            prof.columns[key] = ColumnProfile(**val)
        return prof


class Profiler:
    def __init__(self, db_path: str | Path, sample: int = 200, max_values: int = 25) -> None:
        self.db_path = Path(db_path)
        self.sample = sample
        self.max_values = max_values

    def profile(self, schema: SchemaGraph, cache_path: str | Path | None = None) -> SchemaProfile:
        fp = schema.fingerprint()
        if cache_path:
            cached = self._load_cache(Path(cache_path), fp)
            if cached:
                log.debug("profile cache hit", fingerprint=fp)
                return cached

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            prof = SchemaProfile(fingerprint=fp)
            code_labels = self._load_code_labels(conn, schema)
            for table in schema.tables.values():
                for col in table.columns:
                    prof.columns[col.qualified] = self._profile_column(conn, table.name, col.name, col.dtype)
                    cp = prof.columns[col.qualified]
                    if col.code_group and col.code_group in code_labels:
                        cp.code_labels = code_labels[col.code_group]
                        cp.is_categorical = True
                    col.sample_values = cp.values[:10]
                    col.distinct_count = cp.distinct_count
        finally:
            conn.close()

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(prof.to_json(), encoding="utf-8")
        log.info("schema profiled", columns=len(prof.columns), fingerprint=fp)
        return prof

    # -- internals -------------------------------------------------------- #

    def _load_cache(self, path: Path, fingerprint: str) -> SchemaProfile | None:
        if not path.exists():
            return None
        try:
            prof = SchemaProfile.from_json(path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - corrupt cache
            return None
        return prof if prof.fingerprint == fingerprint else None

    def _load_code_labels(self, conn: sqlite3.Connection, schema: SchemaGraph) -> dict[str, dict[str, str]]:
        for candidate in ("TB_COMM_CD", "TB_CODE", "TC_CMMN_CD", "COMMON_CODE"):
            if not schema.table(candidate):
                continue
            try:
                rows = conn.execute(
                    f'SELECT "{CODE_GROUP_COLUMN}", "{CODE_VALUE_COLUMN}", "{CODE_NAME_COLUMN}" FROM "{candidate}"'
                ).fetchall()
            except sqlite3.Error:  # pragma: no cover
                continue
            out: dict[str, dict[str, str]] = {}
            for grp, cd, nm in rows:
                out.setdefault(grp, {})[str(cd)] = str(nm)
            return out
        return {}

    def _profile_column(self, conn: sqlite3.Connection, table: str, column: str, dtype: str) -> ColumnProfile:
        cp = ColumnProfile(table=table, column=column)
        q = f'"{column}"'
        t = f'"{table}"'
        try:
            total, nulls = conn.execute(
                f"SELECT COUNT(*), SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) FROM {t}"
            ).fetchone()
            total = total or 0
            cp.null_ratio = round((nulls or 0) / total, 4) if total else 0.0
            cp.distinct_count = conn.execute(f"SELECT COUNT(DISTINCT {q}) FROM {t}").fetchone()[0]

            row = conn.execute(f"SELECT MIN({q}), MAX({q}) FROM {t}").fetchone()
            cp.min_value = None if row[0] is None else str(row[0])
            cp.max_value = None if row[1] is None else str(row[1])

            if dtype.upper() not in _NUMERIC:
                cp.avg_length = float(
                    conn.execute(f"SELECT COALESCE(AVG(LENGTH({q})),0) FROM {t}").fetchone()[0] or 0.0
                )

            # Frequency-ranked values keep the profile informative on skewed columns.
            rows = conn.execute(
                f"SELECT {q}, COUNT(*) c FROM {t} WHERE {q} IS NOT NULL "
                f"GROUP BY {q} ORDER BY c DESC LIMIT {self.max_values}"
            ).fetchall()
            cp.values = [str(r[0]) for r in rows]
            cp.is_categorical = 0 < cp.distinct_count <= 40 and dtype.upper() not in _NUMERIC
            cp.is_yyyymmdd = bool(
                cp.values
                and all(len(v) == 8 and v.isdigit() and "1900" <= v[:4] <= "2100" for v in cp.values[:8])
            )
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            log.warning("profile failed", table=table, column=column, error=str(exc))
        return cp
