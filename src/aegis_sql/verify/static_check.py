"""Failures this schema produces, caught before the database is touched.

Executing a wrong statement is cheap in wall-clock terms but expensive in
diagnosis: SQLite answers a mis-typed date predicate with ``0`` and answers a
Korean label compared against a code column with ``0`` as well, and neither is
an error.  A silent empty result is the worst possible feedback for a repair
loop, because there is nothing to feed back.

So the checker looks for the failure modes this particular schema family
actually produces, in decreasing order of observed frequency:

1. ``CHAR(8)`` ``'YYYYMMDD'`` date columns compared against ``'YYYY-MM-DD'``
   literals, or wrapped in ``DATE()``/``YEAR()``/``strftime()``.  Every model
   does this, because every model has seen a million ISO dates.
2. code columns compared against their Korean *label* (``'정상'``) instead of
   the code value (``'01'``) — detected against the profiled code dictionary,
   which also yields the exact literal to substitute.
3. tables reachable only by a cartesian product: the checker unions the
   equalities from both ``ON`` and the ``WHERE`` clause (old-style joins are
   valid SQL) and reports only genuinely disconnected components.
4. unknown tables/columns, with the nearest real identifier by edit distance.
5. non-aggregated projections missing from ``GROUP BY``.

Every violation carries a concrete fix hint, because its second consumer is
:mod:`aegis_sql.verify.repair`, which turns hints into rewrites.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.optimizer.scope import Scope

from aegis_sql.observability.logging import get_logger
from aegis_sql.schema.graph import JoinGraph
from aegis_sql.schema.profile import SchemaProfile
from aegis_sql.types import SchemaGraph, Violation
from aegis_sql.verify.ast_guard import (
    DIALECT,
    ColumnResolver,
    aggregate_ancestor,
    function_name,
    in_group_by,
)

log = get_logger("verify.static_check")

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$")
_HANGUL_RE = re.compile(r"[가-힣]")
#: Functions that assume an ISO/epoch date and therefore break on 'YYYYMMDD'.
DATE_FUNCTIONS = {"date", "datetime", "year", "month", "day", "strftime", "julianday", "date_trunc"}
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.ILike)


class StaticChecker:
    """Execution-free validation of a candidate statement against the schema."""

    def __init__(
        self,
        schema: SchemaGraph,
        join_graph: JoinGraph,
        profile: SchemaProfile | None = None,
    ) -> None:
        self.schema = schema
        self.join_graph = join_graph
        self.profile = profile
        self._columns_by_table = {t.name.upper(): t.column_names for t in schema.tables.values()}

    # -- entry point ------------------------------------------------------ #

    def check(self, sql: str) -> list[Violation]:
        try:
            root = sqlglot.parse_one(sql, dialect=DIALECT)
        except (ParseError, TokenError, RecursionError) as exc:
            msg = str(exc).splitlines()[0][:160]
            return [Violation("SQL_PARSE", f"SQL을 파싱할 수 없습니다: {msg}", "error")]
        if root is None:
            return [Violation("SQL_PARSE", "빈 SQL 구문입니다.", "error")]

        resolver = ColumnResolver(self.schema, root)
        out: list[Violation] = []
        out += self._check_tables(root)
        out += self._check_columns(resolver)
        out += self._check_connectivity(resolver)
        out += self._check_group_by(resolver)
        out += self._check_predicates(resolver)
        return _dedupe(out)

    # -- unknown identifiers ---------------------------------------------- #

    def _check_tables(self, root: exp.Expr) -> list[Violation]:
        cte_names = {c.alias_or_name.upper() for c in root.find_all(exp.CTE)}
        known = set(self._columns_by_table)
        out: list[Violation] = []
        for table in root.find_all(exp.Table):
            name = table.name
            if not name or name.upper() in cte_names or name.upper() in known:
                continue
            near = nearest(name, list(self._columns_by_table))
            hint = f" '{near}'을(를) 의도한 것 같습니다." if near else ""
            out.append(
                Violation(
                    "UNKNOWN_TABLE",
                    f"스키마에 없는 테이블입니다: {name}.{hint}",
                    "error",
                    name,
                )
            )
        return out

    def _check_columns(self, resolver: ColumnResolver) -> list[Violation]:
        out: list[Violation] = []
        for scope in resolver.scopes:
            physical = resolver.physical_tables(scope)
            derived = [a for a, s in resolver.sources(scope).items() if a not in physical]
            if any(t.upper() not in self._columns_by_table for t in physical.values()):
                continue  # the unknown *table* is the finding; its columns are noise
            for column in scope.columns:
                binding = resolver.resolve(column, scope)
                if binding.ambiguous:
                    out.append(
                        Violation(
                            "AMBIGUOUS_COLUMN",
                            f"'{binding.column}'이(가) 여러 테이블에 존재합니다: "
                            f"{', '.join(binding.candidates)}. 별칭으로 한정하세요.",
                            "error",
                            binding.column,
                        )
                    )
                    continue
                if binding.table is not None or derived:
                    # A derived source's column list is only known after the
                    # inner query runs; do not guess against it.
                    continue
                scoped = self._scoped_columns(physical, column.table)
                near = nearest(column.name, scoped)
                hint = f" '{near}'을(를) 의도한 것 같습니다." if near else ""
                owner = f"{column.table}." if column.table else ""
                out.append(
                    Violation(
                        "UNKNOWN_COLUMN",
                        f"스키마에 없는 컬럼입니다: {owner}{column.name}.{hint}",
                        "error",
                        f"{owner}{column.name}",
                    )
                )
        return out

    def _scoped_columns(self, physical: dict[str, str], qualifier: str | None) -> list[str]:
        owned = _ci_lookup(physical, qualifier) if qualifier else None
        tables = [owned] if owned else list(physical.values())
        return [c for t in tables for c in self._columns_by_table.get(t.upper(), [])]

    # -- cartesian products ------------------------------------------------ #

    def _check_connectivity(self, resolver: ColumnResolver) -> list[Violation]:
        out: list[Violation] = []
        for scope in resolver.scopes:
            select = scope.expression
            if not isinstance(select, exp.Select):
                continue
            aliases = list(resolver.sources(scope))
            if len(aliases) < 2:
                continue
            union = _UnionFind(aliases)
            for eq in select.find_all(exp.EQ):
                if _crosses_scope(eq, select):
                    continue
                left, right = eq.this, eq.expression
                if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                    a = self._alias_of(resolver, scope, left)
                    b = self._alias_of(resolver, scope, right)
                    if a and b:
                        union.union(a, b)
            groups = union.groups()
            if len(groups) > 1:
                names = [
                    "+".join(sorted(resolver.physical_tables(scope).get(a, a) for a in group))
                    for group in groups
                ]
                out.append(
                    Violation(
                        "CARTESIAN_JOIN",
                        "조인 조건 없이 결합된 테이블 그룹이 있습니다: "
                        f"{' × '.join(names)}. FK 경로로 ON 조건을 추가하세요.",
                        "error",
                        " × ".join(names),
                    )
                )
        return out

    def _alias_of(self, resolver: ColumnResolver, scope: Scope, column: exp.Column) -> str | None:
        if column.table:
            return resolver.alias_of(scope, column.table)
        binding = resolver.resolve(column, scope)
        if binding.table is None:
            return None
        for alias, table in resolver.physical_tables(scope).items():
            if table.upper() == binding.table.upper():
                return alias
        return None

    # -- GROUP BY ---------------------------------------------------------- #

    def _check_group_by(self, resolver: ColumnResolver) -> list[Violation]:
        out: list[Violation] = []
        for scope in resolver.scopes:
            select = scope.expression
            if not isinstance(select, exp.Select) or not select.args.get("group"):
                continue
            for projection in select.expressions:
                for column in projection.find_all(exp.Column):
                    if aggregate_ancestor(column) is not None or in_group_by(select, column):
                        continue
                    ref = column.sql(dialect=DIALECT)
                    out.append(
                        Violation(
                            "GROUPBY_MISMATCH",
                            f"집계되지 않은 {ref}가 GROUP BY에 없습니다. "
                            f"GROUP BY에 {ref}를 추가하거나 집계 함수로 감싸세요.",
                            "error",
                            ref,
                        )
                    )
        return out

    # -- literal shape ------------------------------------------------------ #

    def _check_predicates(self, resolver: ColumnResolver) -> list[Violation]:
        out: list[Violation] = []
        for scope in resolver.scopes:
            for column in scope.columns:
                binding = resolver.resolve(column, scope)
                if binding.table is None:
                    continue
                if self.is_date_column(binding.table, binding.column):
                    out += self._check_date_usage(column, binding.qualified)
                out += self._check_code_literal(column, binding.table, binding.column)
        return out

    def _check_date_usage(self, column: exp.Column, qualified: str) -> list[Violation]:
        out: list[Violation] = []
        for literal in _compared_literals(column):
            if _ISO_DATE_RE.match(literal.this or ""):
                fixed = str(literal.this)[:10].replace("-", "")
                out.append(
                    Violation(
                        "DATE_FORMAT_MISMATCH",
                        f"{qualified}는 'YYYYMMDD' 8자리 문자열입니다. "
                        f"'{literal.this}' 대신 '{fixed}'로 비교하세요.",
                        "error",
                        qualified,
                    )
                )
        wrapper = _date_function_ancestor(column)
        if wrapper is not None:
            out.append(
                Violation(
                    "DATE_FORMAT_MISMATCH",
                    f"{qualified}는 'YYYYMMDD' 문자열이므로 {wrapper}() 함수가 "
                    f"동작하지 않습니다. 연도는 substr({qualified},1,4), "
                    "기간 비교는 문자열 BETWEEN을 사용하세요.",
                    "error",
                    qualified,
                )
            )
        return out

    def _check_code_literal(self, column: exp.Column, table: str, name: str) -> list[Violation]:
        labels = self.code_labels(table, name)
        if not labels:
            return []
        out: list[Violation] = []
        reverse = {v: k for k, v in labels.items()}
        for literal in _compared_literals(column):
            value = str(literal.this)
            if value in labels:
                continue
            code = reverse.get(value)
            if code is not None:
                out.append(
                    Violation(
                        "CODE_LITERAL_MISMATCH",
                        f"{table}.{name}은(는) 코드값 컬럼입니다. "
                        f"'{value}'(코드명) 대신 '{code}'로 비교하세요.",
                        "error",
                        f"{table}.{name}",
                    )
                )
            elif _HANGUL_RE.search(value):
                sample = ", ".join(f"{k}={v}" for k, v in list(labels.items())[:5])
                out.append(
                    Violation(
                        "CODE_LITERAL_MISMATCH",
                        f"{table}.{name}에 한글 리터럴 '{value}'을(를) 비교했습니다. "
                        f"공통코드 값을 사용하세요 ({sample}).",
                        "warn",
                        f"{table}.{name}",
                    )
                )
        return out

    # -- schema facts ------------------------------------------------------- #

    def is_date_column(self, table: str, column: str) -> bool:
        """``CHAR(8)`` 'YYYYMMDD' column, per the profile with a naming fallback."""
        if self.profile is not None:
            profiled = self.profile.get(table, column)
            if profiled is not None and profiled.values:
                return profiled.is_yyyymmdd
        info = self.schema.column(table, column)
        return bool(info and info.dtype.upper().startswith("TEXT") and _looks_like_date(column))

    def code_labels(self, table: str, column: str) -> dict[str, str]:
        if self.profile is None:
            return {}
        profiled = self.profile.get(table, column)
        return dict(profiled.code_labels) if profiled else {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ci_lookup(mapping: dict[str, str], key: str) -> str | None:
    lowered = key.lower()
    return next((v for k, v in mapping.items() if k.lower() == lowered), None)


def _looks_like_date(name: str) -> bool:
    upper = name.upper()
    return upper.endswith(("_DT", "_DATE", "_YMD")) or upper in {"DT", "BRDT"}


def _compared_literals(column: exp.Column) -> list[exp.Literal]:
    """String literals this column is directly compared against."""
    parent = column.parent
    out: list[exp.Literal] = []
    if isinstance(parent, _COMPARISONS):
        other = parent.expression if parent.this is column else parent.this
        if isinstance(other, exp.Literal) and other.is_string:
            out.append(other)
    elif isinstance(parent, exp.Between) and parent.this is column:
        out.extend(
            node for node in (parent.args.get("low"), parent.args.get("high"))
            if isinstance(node, exp.Literal) and node.is_string
        )
    elif isinstance(parent, exp.In) and parent.this is column:
        out.extend(
            node for node in parent.expressions
            if isinstance(node, exp.Literal) and node.is_string
        )
    return out


def _date_function_ancestor(column: exp.Column) -> str | None:
    """Name of the date function wrapping this column, if any."""
    node: exp.Expr | None = column.parent
    while node is not None and not isinstance(node, exp.Select):
        if isinstance(node, exp.Func) and function_name(node) in DATE_FUNCTIONS:
            return function_name(node)
        node = node.parent
    return None


def _crosses_scope(node: exp.Expr, select: exp.Select) -> bool:
    """True when ``node`` lives inside a nested SELECT rather than in ``select``."""
    cursor: exp.Expr | None = node
    while cursor is not None:
        if cursor is select:
            return False
        if isinstance(cursor, exp.Select):
            return True
        cursor = cursor.parent
    return True


def _dedupe(violations: list[Violation]) -> list[Violation]:
    seen: set[tuple[str, str | None, str]] = set()
    out: list[Violation] = []
    for v in violations:
        key = (v.code, v.subject, v.message)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def levenshtein(a: str, b: str, cap: int = 8) -> int:
    """Iterative edit distance over a single row buffer, with early exit at ``cap``."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def nearest(name: str, candidates: list[str]) -> str | None:
    """Closest identifier by case-insensitive edit distance, or ``None`` if none is close."""
    if not candidates:
        return None
    target = name.upper()
    threshold = max(2, len(target) // 3)
    best: tuple[int, str] | None = None
    for candidate in candidates:
        distance = levenshtein(target, candidate.upper(), cap=threshold)
        if distance <= threshold and (best is None or distance < best[0]):
            best = (distance, candidate)
    return best[1] if best else None


class _UnionFind:
    """Minimal disjoint-set over table aliases, used for join connectivity."""

    __slots__ = ("_parent",)

    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        if a in self._parent and b in self._parent:
            self._parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list[str]]:
        buckets: dict[str, list[str]] = {}
        for item in self._parent:
            buckets.setdefault(self.find(item), []).append(item)
        return list(buckets.values())
