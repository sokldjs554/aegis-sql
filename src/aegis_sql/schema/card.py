"""Schema serialisation for prompts ("schema cards").

How you *write* a schema into a prompt is worth several accuracy points, and
the right format differs by tier: a frontier LLM benefits from rich M-Schema
style cards with Korean labels, code dictionaries and value samples, while the
in-house sLLM was trained on a terse fixed-order format and degrades when the
representation drifts.  Both live here so the two never disagree.

References: ``docs/PAPERS.md`` — M-Schema (XiYan-SQL), DAIL-SQL schema
representation ablation, CodeS value-augmented schema prompts.
"""

from __future__ import annotations

from typing import Literal

from aegis_sql.schema.graph import JoinGraph
from aegis_sql.schema.profile import SchemaProfile
from aegis_sql.types import LinkedSchema, SchemaGraph

Style = Literal["mschema", "ddl", "compact", "slm"]


class SchemaCardBuilder:
    """Renders a (possibly pruned) schema into a prompt-ready string."""

    def __init__(
        self,
        schema: SchemaGraph,
        profile: SchemaProfile | None = None,
        join_graph: JoinGraph | None = None,
    ) -> None:
        self.schema = schema
        self.profile = profile
        self.join_graph = join_graph or JoinGraph(schema)

    # -- entry point ------------------------------------------------------ #

    def render(
        self,
        linked: LinkedSchema | None = None,
        style: Style = "mschema",
        max_values: int = 6,
        include_code_dict: bool = True,
    ) -> str:
        tables = linked.tables if linked and linked.tables else list(self.schema.tables)
        keep_cols = set(linked.columns) if linked and linked.columns else None

        if style == "ddl":
            body = self._render_ddl(tables, keep_cols)
        elif style == "compact":
            body = self._render_compact(tables, keep_cols)
        elif style == "slm":
            body = self._render_slm(tables, keep_cols)
        else:
            body = self._render_mschema(tables, keep_cols, max_values)

        sections = [body]
        if style != "slm":
            fk_block = self._render_foreign_keys(tables)
            if fk_block:
                sections.append(fk_block)
        if include_code_dict and style in {"mschema", "compact"}:
            code_block = self._render_code_dictionary(tables, keep_cols)
            if code_block:
                sections.append(code_block)
        return "\n\n".join(s for s in sections if s)

    # -- styles ------------------------------------------------------------ #

    def _render_mschema(self, tables: list[str], keep: set[str] | None, max_values: int) -> str:
        """Rich format: physical name, Korean label, type, key markers, samples."""
        out: list[str] = ["【DATABASE SCHEMA】"]
        for tname in tables:
            table = self.schema.table(tname)
            if not table:
                continue
            head = f"# {table.name}"
            if table.comment:
                head += f"  ({table.comment})"
            if table.row_count >= 0:
                head += f"  [rows={table.row_count:,}]"
            out.append(head)
            for col in table.columns:
                if keep is not None and col.qualified not in keep and not col.is_primary_key:
                    continue
                marks: list[str] = []
                if col.is_primary_key:
                    marks.append("PK")
                if col.foreign_key:
                    marks.append(f"FK→{col.foreign_key.to_table}.{col.foreign_key.to_column}")
                if col.code_group:
                    marks.append(f"CODE:{col.code_group}")
                mark = f" [{', '.join(marks)}]" if marks else ""
                label = f" — {col.comment}" if col.comment else ""
                line = f"  - {col.name}: {col.dtype}{mark}{label}"
                sample = self._samples(col.table, col.name, max_values)
                if sample:
                    line += f"  e.g. {sample}"
                out.append(line)
        return "\n".join(out)

    def _render_ddl(self, tables: list[str], keep: set[str] | None) -> str:
        out: list[str] = []
        for tname in tables:
            table = self.schema.table(tname)
            if not table:
                continue
            cols: list[str] = []
            for col in table.columns:
                if keep is not None and col.qualified not in keep and not col.is_primary_key:
                    continue
                piece = f"  {col.name} {col.dtype}"
                if col.is_primary_key:
                    piece += " PRIMARY KEY"
                if not col.nullable:
                    piece += " NOT NULL"
                if col.comment:
                    piece += f" -- {col.comment}"
                cols.append(piece)
            for fk in table.foreign_keys:
                if fk.to_table in tables:
                    cols.append(f"  FOREIGN KEY ({fk.from_column}) REFERENCES {fk.to_table}({fk.to_column})")
            comment = f" -- {table.comment}" if table.comment else ""
            out.append(f"CREATE TABLE {table.name} ({comment}\n" + ",\n".join(cols) + "\n);")
        return "\n\n".join(out)

    def _render_compact(self, tables: list[str], keep: set[str] | None) -> str:
        """One line per table — cheapest useful representation."""
        out: list[str] = []
        for tname in tables:
            table = self.schema.table(tname)
            if not table:
                continue
            parts = []
            for col in table.columns:
                if keep is not None and col.qualified not in keep and not col.is_primary_key:
                    continue
                token = col.name
                if col.comment:
                    token += f"({col.comment})"
                parts.append(token)
            label = f"[{table.comment}]" if table.comment else ""
            out.append(f"{table.name}{label}: " + ", ".join(parts))
        return "\n".join(out)

    def _render_slm(self, tables: list[str], keep: set[str] | None) -> str:
        """Terse, deterministic, fixed-order format — must match training data."""
        out: list[str] = []
        for tname in sorted(tables):
            table = self.schema.table(tname)
            if not table:
                continue
            cols = [
                c.name
                for c in table.columns
                if keep is None or c.qualified in keep or c.is_primary_key
            ]
            out.append(f"{table.name}({','.join(cols)})")
        fks = sorted(
            f"{fk.from_table}.{fk.from_column}={fk.to_table}.{fk.to_column}"
            for fk in self.schema.foreign_keys
            if fk.from_table in tables and fk.to_table in tables
        )
        line = " | ".join(out)
        return line + (" || " + " , ".join(fks) if fks else "")

    # -- fragments --------------------------------------------------------- #

    def _render_foreign_keys(self, tables: list[str]) -> str:
        rows = [
            f"  {fk.from_table}.{fk.from_column} = {fk.to_table}.{fk.to_column}"
            for fk in self.schema.foreign_keys
            if fk.from_table in tables and fk.to_table in tables
        ]
        if not rows:
            return ""
        return "【FOREIGN KEYS】\n" + "\n".join(rows)

    def _render_code_dictionary(self, tables: list[str], keep: set[str] | None) -> str:
        """Emit the code→label dictionary only for the code columns actually linked.

        This is the single highest-leverage block in the whole prompt on this
        schema: without it the model has no way to know ``'02' = 실효``.
        """
        if not self.profile:
            return ""
        seen: set[str] = set()
        rows: list[str] = []
        for tname in tables:
            table = self.schema.table(tname)
            if not table:
                continue
            for col in table.columns:
                if not col.code_group or col.code_group in seen:
                    continue
                if keep is not None and col.qualified not in keep:
                    continue
                cp = self.profile.get(col.table, col.name)
                if not cp or not cp.code_labels:
                    continue
                seen.add(col.code_group)
                pairs = ", ".join(f"'{k}'={v}" for k, v in sorted(cp.code_labels.items()))
                rows.append(f"  {col.name} (TB_COMM_CD.CD_GRP='{col.code_group}'): {pairs}")
        if not rows:
            return ""
        return (
            "【CODE DICTIONARY】 코드값은 TB_COMM_CD 조인 또는 아래 값으로 직접 비교하세요.\n"
            + "\n".join(rows)
        )

    def _samples(self, table: str, column: str, limit: int) -> str:
        if not self.profile:
            return ""
        cp = self.profile.get(table, column)
        if not cp or not cp.values:
            return ""
        if cp.code_labels:
            return ", ".join(f"{v}({cp.code_labels.get(v, '?')})" for v in cp.values[:limit])
        if cp.is_yyyymmdd:
            return f"{cp.min_value}~{cp.max_value} (YYYYMMDD 문자열)"
        if cp.distinct_count > 200 and not cp.is_categorical:
            return ""
        vals = [v if len(v) <= 18 else v[:16] + "…" for v in cp.values[:limit]]
        return ", ".join(vals)


def token_estimate(text: str) -> int:
    """Cheap token estimate (Korean ≈ 1.4 chars/token, ASCII ≈ 4 chars/token)."""
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    other = len(text) - hangul
    return int(hangul / 1.4 + other / 4) + 1
