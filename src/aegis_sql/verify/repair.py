"""Deterministic repair first, model repair only as a last resort.

The reflexive design for "the SQL failed" is to hand the error back to the
model.  That costs a round trip, a few thousand tokens and reproducibility, and
for this schema it is usually unnecessary: the observed failure distribution is
dominated by a handful of mechanical defects — a column name off by two
characters, an ISO date literal against a ``CHAR(8)`` column, a Korean label
where a code value belongs, a missing join condition, a projection missing from
``GROUP BY``.  Each of those has a *closed-form* fix given the schema, the
column profile and the FK graph, all of which this process already holds.

So the repair loop is ordered by cost: rule-based rewrites on the AST run
first, each one re-executed to confirm it actually helped, and the model is
called only when the rules are exhausted — with one attempt always held in
reserve for it, so a run of failed rules can never starve the fallback.

Every attempt is recorded as a :class:`~aegis_sql.types.RepairStep` with a
named strategy, which is what makes the flywheel able to answer "which repair
strategy is carrying the eval score" instead of "the model retried".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.optimizer.scope import Scope

from aegis_sql.observability.logging import get_logger
from aegis_sql.schema.graph import JoinEdge, JoinGraph
from aegis_sql.schema.profile import SchemaProfile
from aegis_sql.types import LinkedSchema, RepairStep, SchemaGraph, Violation
from aegis_sql.verify.ast_guard import DIALECT, ColumnResolver, is_scalar_aggregate
from aegis_sql.verify.executor import SQLExecutor
from aegis_sql.verify.static_check import StaticChecker, nearest

log = get_logger("verify.repair")

_NO_TABLE = "no such table:"
_NO_COLUMN = "no such column:"
_AMBIGUOUS = "ambiguous column name:"
#: strftime format → equivalent substring slice over a 'YYYYMMDD' string.
_STRFTIME_SLICES = {"%Y": (1, 4), "%m": (5, 2), "%d": (7, 2), "%Y%m": (1, 6), "%Y%m%d": (1, 8)}


@dataclass(slots=True)
class RepairContext:
    """Everything a repair strategy — rule or model — is allowed to look at."""

    question: str = ""
    schema_card: str = ""
    linked: LinkedSchema | None = None
    sql: str = ""
    error: str = ""
    attempt: int = 0


class SelfRepairer:
    """Rule-based SQL repair with an optional model-backed final attempt."""

    def __init__(
        self,
        schema: SchemaGraph,
        profile: SchemaProfile | None,
        join_graph: JoinGraph,
        executor: SQLExecutor,
        static_checker: StaticChecker | None = None,
        llm_repair: Callable[[RepairContext], str | None] | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.schema = schema
        self.profile = profile
        self.join_graph = join_graph
        self.executor = executor
        self.static_checker = static_checker
        self.llm_repair = llm_repair
        self.max_attempts = max(1, int(max_attempts))
        self._strategies: tuple[tuple[str, Callable[[exp.Expr, str, list[Violation]], bool]], ...] = (
            ("unknown-table", self._fix_unknown_table),
            ("unknown-column", self._fix_unknown_column),
            ("ambiguous-column", self._fix_ambiguous_column),
            ("date-format", self._fix_date_format),
            ("code-literal", self._fix_code_literal),
            ("missing-join", self._fix_missing_join),
            ("agg-groupby", self._fix_agg_groupby),
            ("limit", self._fix_limit),
        )

    # -- entry point ------------------------------------------------------- #

    def repair(
        self, sql: str, error: str, ctx: RepairContext | None = None
    ) -> tuple[str | None, list[RepairStep]]:
        """Return the first statement that executes, plus every attempt made."""
        context = ctx or RepairContext(sql=sql, error=error)
        steps: list[RepairStep] = []
        current, current_error = sql, error

        reserve = 1 if self.llm_repair is not None else 0
        rule_budget = max(1, self.max_attempts - reserve)

        issues = self._issues(current)
        for name, strategy in self._strategies:
            if len(steps) >= rule_budget:
                break
            candidate = self._apply(strategy, current, current_error, issues)
            if candidate is None:
                continue
            result = self.executor.execute(candidate)
            steps.append(
                RepairStep(
                    attempt=len(steps) + 1,
                    before_sql=current,
                    after_sql=candidate,
                    error=current_error,
                    strategy=name,
                    fixed=result.ok,
                )
            )
            if result.ok:
                log.info("repaired", strategy=name, attempts=len(steps))
                return candidate, steps
            new_error = result.error or ""
            if new_error == current_error:
                continue  # the rewrite changed nothing observable — drop it
            current, current_error = candidate, new_error
            issues = self._issues(current)

        while self.llm_repair is not None and len(steps) < self.max_attempts:
            candidate = self._call_llm(context, current, current_error, len(steps) + 1)
            if candidate is None:
                break
            result = self.executor.execute(candidate)
            steps.append(
                RepairStep(
                    attempt=len(steps) + 1,
                    before_sql=current,
                    after_sql=candidate,
                    error=current_error,
                    strategy="llm",
                    fixed=result.ok,
                )
            )
            if result.ok:
                log.info("repaired", strategy="llm", attempts=len(steps))
                return candidate, steps
            current, current_error = candidate, result.error or ""

        log.warning("repair exhausted", attempts=len(steps), error=current_error[:160])
        return None, steps

    # -- strategy plumbing --------------------------------------------------- #

    def _issues(self, sql: str) -> list[Violation]:
        return self.static_checker.check(sql) if self.static_checker else []

    def _apply(
        self,
        strategy: Callable[[exp.Expr, str, list[Violation]], bool],
        sql: str,
        error: str,
        issues: list[Violation],
    ) -> str | None:
        """Run one strategy on a fresh parse; return new SQL only if it changed."""
        root = _parse(sql)
        if root is None:
            return None
        try:
            changed = strategy(root, error, issues)
        except Exception as exc:  # noqa: BLE001 - a broken rule must not kill the loop
            log.warning("repair strategy raised", error=str(exc)[:160])
            return None
        if not changed:
            return None
        candidate = root.sql(dialect=DIALECT, pretty=False)
        return None if _normalize(candidate) == _normalize(sql) else candidate

    def _call_llm(
        self, context: RepairContext, sql: str, error: str, attempt: int
    ) -> str | None:
        assert self.llm_repair is not None
        payload = RepairContext(
            question=context.question,
            schema_card=context.schema_card,
            linked=context.linked,
            sql=sql,
            error=error,
            attempt=attempt,
        )
        try:
            raw = self.llm_repair(payload)
        except Exception as exc:  # noqa: BLE001 - provider failures are not fatal
            log.warning("llm repair failed", error=str(exc)[:160])
            return None
        cleaned = _strip_fences(raw or "")
        if not cleaned or _normalize(cleaned) == _normalize(sql) or _parse(cleaned) is None:
            return None
        return cleaned

    # -- 1. unknown table ---------------------------------------------------- #

    def _fix_unknown_table(self, root: exp.Expr, error: str, issues: list[Violation]) -> bool:
        wanted = _after(error, _NO_TABLE) or _subjects(issues, "UNKNOWN_TABLE")
        known = list(self.schema.tables)
        changed = False
        for name in _as_list(wanted):
            replacement = nearest(name.split(".")[-1], known)
            if replacement is None:
                continue
            for table in root.find_all(exp.Table):
                if table.name.upper() == name.upper():
                    table.set("this", exp.to_identifier(replacement))
                    changed = True
        return changed

    # -- 2. unknown column ---------------------------------------------------- #

    def _fix_unknown_column(self, root: exp.Expr, error: str, issues: list[Violation]) -> bool:
        wanted = _after(error, _NO_COLUMN) or _subjects(issues, "UNKNOWN_COLUMN")
        if not wanted:
            return False
        resolver = ColumnResolver(self.schema, root)
        changed = False
        for reference in _as_list(wanted):
            qualifier, _, name = reference.rpartition(".")
            for scope in resolver.scopes:
                pool = self._candidate_columns(resolver, scope, qualifier)
                replacement = nearest(name, [c for _t, c in pool])
                if replacement is None:
                    continue
                owner = next((t for t, c in pool if c.upper() == replacement.upper()), None)
                for column in list(scope.columns):
                    if column.name.upper() != name.upper():
                        continue
                    column.set("this", exp.to_identifier(replacement))
                    if column.table and owner:
                        alias = _alias_for(resolver, scope, owner)
                        if alias:
                            column.set("table", exp.to_identifier(alias))
                    changed = True
        return changed

    def _candidate_columns(
        self, resolver: ColumnResolver, scope: Scope, qualifier: str
    ) -> list[tuple[str, str]]:
        """``[(table, column)]`` reachable from this scope, narrowed by the qualifier."""
        physical = resolver.physical_tables(scope)
        owned = next(
            (t for a, t in physical.items() if qualifier and a.lower() == qualifier.lower()), None
        )
        tables = [owned] if owned else list(physical.values())
        out: list[tuple[str, str]] = []
        for table_name in tables:
            table = self.schema.table(table_name)
            if table:
                out.extend((table.name, c) for c in table.column_names)
        return out

    # -- 3. ambiguous column --------------------------------------------------- #

    def _fix_ambiguous_column(self, root: exp.Expr, error: str, issues: list[Violation]) -> bool:
        wanted = _after(error, _AMBIGUOUS) or _subjects(issues, "AMBIGUOUS_COLUMN")
        if not wanted:
            return False
        resolver = ColumnResolver(self.schema, root)
        changed = False
        for reference in _as_list(wanted):
            name = reference.rpartition(".")[2]
            for scope in resolver.scopes:
                owners = [
                    (alias, table)
                    for alias, table in resolver.physical_tables(scope).items()
                    if (t := self.schema.table(table)) is not None and t.column(name) is not None
                ]
                if len(owners) < 2:
                    continue
                alias = self._preferred_owner(owners, name)
                for column in list(scope.columns):
                    if column.name.upper() == name.upper() and not column.table:
                        column.set("table", exp.to_identifier(alias))
                        changed = True
        return changed

    def _preferred_owner(self, owners: list[tuple[str, str]], column: str) -> str:
        """Prefer the table where the column is the primary key — the canonical side."""
        for alias, table_name in owners:
            table = self.schema.table(table_name)
            if table and column.upper() in {p.upper() for p in table.primary_key}:
                return alias
        return owners[0][0]

    # -- 4. date format --------------------------------------------------------- #

    def _fix_date_format(self, root: exp.Expr, _error: str, _issues: list[Violation]) -> bool:
        resolver = ColumnResolver(self.schema, root)
        changed = False
        replacements: list[tuple[exp.Expr, exp.Expr]] = []
        for scope in resolver.scopes:
            for column in list(scope.columns):
                binding = resolver.resolve(column, scope)
                if binding.table is None or not self._is_date_column(binding.table, binding.column):
                    continue
                for literal in _literals_against(column):
                    text = str(literal.this)
                    if _looks_iso(text):
                        literal.set("this", text[:10].replace("-", ""))
                        changed = True
                wrapper = _date_wrapper(column)
                if wrapper is not None:
                    unwrapped = _unwrap_date(wrapper, column)
                    if unwrapped is not None:
                        replacements.append((wrapper, unwrapped))
        for node, replacement in replacements:
            node.replace(replacement)
            changed = True
        return changed

    def _is_date_column(self, table: str, column: str) -> bool:
        if self.static_checker is not None:
            return self.static_checker.is_date_column(table, column)
        profiled = self.profile.get(table, column) if self.profile else None
        return bool(profiled and profiled.is_yyyymmdd)

    # -- 5. code literal --------------------------------------------------------- #

    def _fix_code_literal(self, root: exp.Expr, _error: str, _issues: list[Violation]) -> bool:
        if self.profile is None:
            return False
        resolver = ColumnResolver(self.schema, root)
        changed = False
        for scope in resolver.scopes:
            for column in list(scope.columns):
                binding = resolver.resolve(column, scope)
                if binding.table is None:
                    continue
                profiled = self.profile.get(binding.table, binding.column)
                if not profiled or not profiled.code_labels:
                    continue
                reverse = {label: code for code, label in profiled.code_labels.items()}
                for literal in _literals_against(column):
                    text = str(literal.this)
                    if text in profiled.code_labels:
                        continue
                    code = reverse.get(text)
                    if code is not None:
                        literal.set("this", code)
                        changed = True
        return changed

    # -- 6. missing join ---------------------------------------------------------- #

    def _fix_missing_join(self, root: exp.Expr, _error: str, _issues: list[Violation]) -> bool:
        resolver = ColumnResolver(self.schema, root)
        changed = False
        for scope in resolver.scopes:
            select = scope.expression
            if not isinstance(select, exp.Select):
                continue
            physical = resolver.physical_tables(scope)
            if len(physical) < 2:
                continue
            connected = _connected_tables(select, resolver, scope)
            tables = list(dict.fromkeys(physical.values()))
            anchor = tables[0]
            for table in tables[1:]:
                if _same_component(connected, anchor, table):
                    continue
                path = self.join_graph.shortest_path(anchor, table)
                if not path:
                    continue
                for edge in path:
                    if self._attach_edge(select, resolver, scope, edge):
                        changed = True
        return changed

    def _attach_edge(
        self, select: exp.Select, resolver: ColumnResolver, scope: Scope, edge: JoinEdge
    ) -> bool:
        physical = resolver.physical_tables(scope)
        left_alias = _alias_for(resolver, scope, edge.left_table)
        right_alias = _alias_for(resolver, scope, edge.right_table)
        if left_alias is None:
            return False
        if right_alias is None:
            # A bridge table on the FK path is missing from the query entirely.
            right_alias = edge.right_table
            condition = sqlglot.parse_one(edge.to_sql(left_alias, right_alias), dialect=DIALECT)
            select.args.setdefault("joins", []).append(
                exp.Join(this=exp.to_table(edge.right_table), on=condition, kind="INNER")
            )
            return True
        if edge.right_table in physical.values() and edge.left_table in physical.values():
            condition = sqlglot.parse_one(edge.to_sql(left_alias, right_alias), dialect=DIALECT)
            if _has_condition(select, condition):
                return False
            select.where(condition, copy=False)
            return True
        return False

    # -- 7. GROUP BY ---------------------------------------------------------------- #

    def _fix_agg_groupby(self, root: exp.Expr, _error: str, issues: list[Violation]) -> bool:
        missing = {s for s in _subjects(issues, "GROUPBY_MISMATCH") if s}
        if not missing:
            return False
        changed = False
        for select in root.find_all(exp.Select):
            group = select.args.get("group")
            if not group:
                continue
            existing = {g.sql(dialect=DIALECT).lower() for g in group.expressions}
            additions = [
                sqlglot.parse_one(subject, dialect=DIALECT)
                for subject in sorted(missing)
                if subject.lower() not in existing and _mentions(select, subject)
            ]
            if additions:
                group.set("expressions", [*group.expressions, *additions])
                changed = True
        return changed

    # -- 8. illegal LIMIT -------------------------------------------------------------- #

    def _fix_limit(self, root: exp.Expr, _error: str, _issues: list[Violation]) -> bool:
        changed = False
        for select in root.find_all(exp.Select):
            if select.args.get("limit") and is_scalar_aggregate(select):
                select.set("limit", None)
                changed = True
        return changed


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _parse(sql: str) -> exp.Expr | None:
    try:
        return sqlglot.parse_one(sql, dialect=DIALECT)
    except (ParseError, TokenError, RecursionError):
        return None


def _normalize(sql: str) -> str:
    return " ".join(sql.lower().split())


def _strip_fences(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if "```" in body[3:] else body[3:]
        if body.lower().startswith("sql"):
            body = body[3:]
    return body.strip().rstrip(";").strip()


def _after(error: str, marker: str) -> str | None:
    lowered = (error or "").lower()
    index = lowered.find(marker)
    if index < 0:
        return None
    return error[index + len(marker) :].strip().split()[0].strip("\"'`") or None


def _subjects(issues: Iterable[Violation], code: str) -> list[str]:
    return [v.subject for v in issues if v.code == code and v.subject]


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _alias_for(resolver: ColumnResolver, scope: Scope, table: str) -> str | None:
    for alias, name in resolver.physical_tables(scope).items():
        if name.upper() == table.upper():
            return alias
    return None


def _literals_against(column: exp.Column) -> list[exp.Literal]:
    parent = column.parent
    out: list[exp.Literal] = []
    if isinstance(parent, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        other = parent.expression if parent.this is column else parent.this
        if isinstance(other, exp.Literal) and other.is_string:
            out.append(other)
    elif isinstance(parent, exp.Between) and parent.this is column:
        out.extend(
            n for n in (parent.args.get("low"), parent.args.get("high"))
            if isinstance(n, exp.Literal) and n.is_string
        )
    elif isinstance(parent, exp.In) and parent.this is column:
        out.extend(n for n in parent.expressions if isinstance(n, exp.Literal) and n.is_string)
    return out


def _looks_iso(text: str) -> bool:
    return len(text) >= 10 and text[4] == "-" and text[7] == "-" and text[:4].isdigit()


def _date_wrapper(column: exp.Column) -> exp.Func | None:
    """The nearest DATE()/YEAR()/strftime() call wrapping this column."""
    node: exp.Expr | None = column.parent
    while node is not None and not isinstance(node, exp.Select):
        name = str(node.this) if isinstance(node, exp.Anonymous) else (
            node.sql_name() if isinstance(node, exp.Func) else None
        )
        if name and name.lower() in {
            "date", "datetime", "year", "month", "day", "strftime", "julianday", "date_trunc"
        }:
            return node
        node = node.parent
    return None


def _unwrap_date(wrapper: exp.Func, column: exp.Column) -> exp.Expr | None:
    """Replace a date function over a 'YYYYMMDD' string with the equivalent substring."""
    name = (str(wrapper.this) if isinstance(wrapper, exp.Anonymous) else wrapper.sql_name()).lower()
    reference = column.copy()
    if name in {"date", "datetime", "julianday", "date_trunc"}:
        return reference
    if name in {"year", "month", "day"}:
        start, length = {"year": (1, 4), "month": (5, 2), "day": (7, 2)}[name]
        return _substr(reference, start, length)
    if name == "strftime":
        arguments = wrapper.expressions if isinstance(wrapper, exp.Anonymous) else []
        fmt = arguments[0] if arguments else None
        key = str(fmt.this) if isinstance(fmt, exp.Literal) else "%Y%m%d"
        start, length = _STRFTIME_SLICES.get(key, (1, 8))
        return _substr(reference, start, length)
    return None


def _substr(column: exp.Expr, start: int, length: int) -> exp.Expr:
    return exp.Substring(
        this=column,
        start=exp.Literal.number(start),
        length=exp.Literal.number(length),
    )


def _connected_tables(
    select: exp.Select, resolver: ColumnResolver, scope: Scope
) -> list[set[str]]:
    """Components of physical tables linked by an equality in ON or WHERE."""
    physical = resolver.physical_tables(scope)
    components = [{t} for t in dict.fromkeys(physical.values())]
    for eq in select.find_all(exp.EQ):
        left, right = eq.this, eq.expression
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            continue
        a = resolver.resolve(left, scope).table
        b = resolver.resolve(right, scope).table
        if not a or not b or a == b:
            continue
        merged = {a, b}
        rest: list[set[str]] = []
        for component in components:
            if component & merged:
                merged |= component
            else:
                rest.append(component)
        components = [merged, *rest]
    return components


def _same_component(components: list[set[str]], a: str, b: str) -> bool:
    return any(a in component and b in component for component in components)


def _has_condition(select: exp.Select, condition: exp.Expr) -> bool:
    target = condition.sql(dialect=DIALECT).lower()
    return target in select.sql(dialect=DIALECT).lower()


def _mentions(select: exp.Select, reference: str) -> bool:
    name = reference.rpartition(".")[2].lower()
    return any(c.name.lower() == name for p in select.expressions for c in p.find_all(exp.Column))
