"""Governance enforcement on the parsed statement, not on the prompt.

A statement produced by a language model has exactly the trust level of a
string pasted by an anonymous user, so the only place a *guarantee* about it
can be made is the parse tree.  This module turns ``configs/policy/*.yaml``
into an executable control over that tree:

* **Shape** — a single read-only ``SELECT``/``WITH`` root; anything that can
  write, re-configure or widen the session (``INSERT``, ``PRAGMA``, ``ATTACH``,
  stacked statements, ``load_extension``) is rejected before analysis starts.
* **Column grade** — every ``Column`` node is *resolved* to a physical
  ``(table, column)`` pair, walking through CTEs and derived tables, so that
  ``WITH x AS (SELECT TELNO FROM TB_CUST) SELECT TELNO FROM x`` is classified
  the same as the direct reference.  A ``SELECT *`` is expanded against the
  schema first, because an unexpanded star is the cheapest way to smuggle
  ``RRNO_ENC`` past a naive column matcher.
* **Position matters** — a masked column may leave the engine only through the
  outermost projection, where it is *rewritten* into the masking expression.
  The same column in ``WHERE``/``GROUP BY``/``ORDER BY``/``ON`` is blocked:
  a predicate over a masked value is an oracle that reconstructs it one
  comparison at a time.
* **Rewrites** — row-level policies (session context → mandatory filter),
  ``LIMIT`` injection and k-anonymity ``HAVING`` are applied as AST edits, so
  the statement handed to the executor is the statement that was audited.

Config may *tighten* the policy (``Settings.verify``) but never loosen it;
the YAML document is the upper bound of what the engine will run.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
import yaml
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.optimizer.scope import Scope, traverse_scope

from aegis_sql.config import Settings, get_settings
from aegis_sql.observability.logging import get_logger
from aegis_sql.types import GuardVerdict, SchemaGraph, Sensitivity, Violation

log = get_logger("verify.ast_guard")

DIALECT = "sqlite"

#: Statement roots that can never be read-only, mapped to their violation code.
_FORBIDDEN_ROOTS: tuple[tuple[type[exp.Expr], str], ...] = (
    (exp.Insert, "WRITE_FORBIDDEN"),
    (exp.Update, "WRITE_FORBIDDEN"),
    (exp.Delete, "WRITE_FORBIDDEN"),
    (exp.Create, "WRITE_FORBIDDEN"),
    (exp.Drop, "WRITE_FORBIDDEN"),
    (exp.Alter, "WRITE_FORBIDDEN"),
    (exp.TruncateTable, "WRITE_FORBIDDEN"),
    (exp.Pragma, "PRAGMA_FORBIDDEN"),
    (exp.Attach, "ATTACH_FORBIDDEN"),
    (exp.Detach, "ATTACH_FORBIDDEN"),
)

#: Clauses in which a value is compared rather than returned.
_PREDICATE_CLAUSES = frozenset({"where", "having", "join_on", "group", "order", "qualify"})

_TEMPLATE_RE = re.compile(r"\{\{\s*ctx\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_WHEN_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<op>==|!=)
      | (?P<bool>\b(?:and|or|not)\b)
      | ctx\.(?P<var>[A-Za-z_][A-Za-z0-9_]*)
      | '(?P<sq>[^']*)'
      | "(?P<dq>[^"]*)"
      | (?P<num>-?\d+(?:\.\d+)?)
      | (?P<lit>\btrue\b|\bfalse\b)
    )""",
    re.VERBOSE | re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Policy document
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class EnginePolicy:
    """The ``engine:`` block — hard limits that hold regardless of the caller."""

    read_only: bool = True
    allow_temp_tables: bool = False
    max_rows: int = 500
    force_limit: bool = True
    default_limit: int = 200
    max_join_tables: int = 8
    statement_timeout_s: float = 8.0
    forbid_functions: tuple[str, ...] = ()
    forbid_pragma: bool = True
    forbid_attach: bool = True
    single_statement: bool = True


@dataclass(slots=True)
class RowPolicy:
    """A session-context-conditional filter that is ANDed into the query."""

    id: str
    table: str
    when: str = ""
    filter: str = ""
    description: str = ""


@dataclass(slots=True)
class KAnonymity:
    enabled: bool = False
    k: int = 5
    applies_to_tables: tuple[str, ...] = ()

    def applies_to(self, table: str) -> bool:
        return table.upper() in {t.upper() for t in self.applies_to_tables}


@dataclass(slots=True)
class PolicyDocument:
    """Parsed ``configs/policy/*.yaml``.  Column keys are upper-cased on load."""

    version: int = 1
    name: str = "default"
    description: str = ""
    engine: EnginePolicy = field(default_factory=EnginePolicy)
    default_sensitivity: Sensitivity = Sensitivity.PUBLIC
    columns: dict[str, Sensitivity] = field(default_factory=dict)
    mask_strategy: str = "partial"
    mask_rules: dict[str, str] = field(default_factory=dict)
    row_policies: list[RowPolicy] = field(default_factory=list)
    k_anonymity: KAnonymity = field(default_factory=KAnonymity)
    audit: dict[str, Any] = field(default_factory=dict)

    # -- loading ---------------------------------------------------------- #

    @classmethod
    def load(cls, path: str | Path) -> PolicyDocument:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        engine_raw = dict(raw.get("engine") or {})
        engine_raw["forbid_functions"] = tuple(
            str(f).lower() for f in (engine_raw.get("forbid_functions") or ())
        )
        engine = EnginePolicy(
            **{k: v for k, v in engine_raw.items() if k in EnginePolicy.__dataclass_fields__}
        )
        masking = raw.get("masking") or {}
        kan_raw = raw.get("k_anonymity") or {}
        doc = cls(
            version=int(raw.get("version", 1)),
            name=str(raw.get("name", "default")),
            description=str(raw.get("description", "")),
            engine=engine,
            default_sensitivity=_as_sensitivity(raw.get("default_sensitivity"), Sensitivity.PUBLIC),
            columns={
                str(k).upper(): _as_sensitivity(v, Sensitivity.PUBLIC)
                for k, v in (raw.get("columns") or {}).items()
            },
            mask_strategy=str(masking.get("strategy", "partial")),
            mask_rules={str(k).upper(): str(v) for k, v in (masking.get("rules") or {}).items()},
            row_policies=[
                RowPolicy(
                    id=str(p.get("id", f"POLICY_{i}")),
                    table=str(p.get("table", "")),
                    when=str(p.get("when", "")),
                    filter=str(p.get("filter", "")),
                    description=str(p.get("description", "")),
                )
                for i, p in enumerate(raw.get("row_policies") or [])
            ],
            k_anonymity=KAnonymity(
                enabled=bool(kan_raw.get("enabled", False)),
                k=int(kan_raw.get("k", 5)),
                applies_to_tables=tuple(str(t) for t in (kan_raw.get("applies_to_tables") or ())),
            ),
            audit=dict(raw.get("audit") or {}),
        )
        log.info(
            "policy loaded",
            policy=doc.name,
            classified_columns=len(doc.columns),
            row_policies=len(doc.row_policies),
            k_anonymity=doc.k_anonymity.enabled,
        )
        return doc

    @classmethod
    def permissive(cls) -> PolicyDocument:
        """No column grades and no row policies — but still read-only.

        Read-only is an engine invariant rather than a governance choice: even
        with ``policy.enabled = false`` the engine must not be able to write.
        """
        return cls(name="permissive")

    # -- queries ---------------------------------------------------------- #

    def sensitivity(self, table: str, column: str) -> Sensitivity:
        return self.columns.get(f"{table}.{column}".upper(), self.default_sensitivity)

    def mask_expression(self, qualified: str) -> str | None:
        """Masking template for a MASKED column, with ``{col}`` as the placeholder."""
        key = qualified.upper()
        if self.columns.get(key) is not Sensitivity.MASKED:
            return None
        if self.mask_strategy == "null":
            return "NULL"
        if self.mask_strategy == "hash":
            # Stock SQLite ships no digest function; hex-truncation is a stable
            # pseudonym for the demo and is swapped for a UDF in deployment.
            return "'#' || substr(hex({col}), 1, 16)"
        return self.mask_rules.get(key, "'***'")


def _as_sensitivity(value: Any, default: Sensitivity) -> Sensitivity:
    try:
        return Sensitivity(str(value).lower())
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Binding:
    """Where a ``Column`` node ultimately comes from."""

    table: str | None
    column: str
    #: True when the reference passed through a computed projection, i.e. the
    #: value that leaves is a *function of* the base column, not the column.
    derived: bool = False
    ambiguous: bool = False
    candidates: tuple[str, ...] = ()

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.column}" if self.table else self.column


class ColumnResolver:
    """Alias/CTE aware ``Column`` → physical ``(table, column)`` resolution.

    Shared by the guard, the static checker and the repairer: all three need
    the same answer to "which physical column is this identifier?", and three
    different answers would be three different bugs.
    """

    def __init__(self, schema: SchemaGraph, root: exp.Expr) -> None:
        self.schema = schema
        try:
            self.scopes: list[Scope] = list(traverse_scope(root))
        except Exception:  # pragma: no cover - sqlglot cannot scope exotic trees
            self.scopes = []
        self._by_select: dict[int, Scope] = {id(s.expression): s for s in self.scopes}

    # -- scope helpers ---------------------------------------------------- #

    def scope_of(self, select: exp.Expr) -> Scope | None:
        return self._by_select.get(id(select))

    @staticmethod
    def sources(scope: Scope) -> dict[str, Any]:
        """Alias → ``exp.Table`` | ``Scope``, with the alias exactly as written.

        The case is preserved because these aliases end up in rewritten SQL that
        a human reads in the audit log.
        """
        return {str(k): v for k, v in scope.sources.items()}

    @classmethod
    def source_of(cls, scope: Scope, alias: str) -> Any | None:
        """Case-insensitive source lookup — SQL identifiers are not case sensitive."""
        lowered = alias.lower()
        for name, src in scope.sources.items():
            if str(name).lower() == lowered:
                return src
        return None

    @classmethod
    def alias_of(cls, scope: Scope, alias: str) -> str | None:
        """The canonical spelling of ``alias`` as it appears in the query."""
        lowered = alias.lower()
        return next((str(n) for n in scope.sources if str(n).lower() == lowered), None)

    def physical_tables(self, scope: Scope) -> dict[str, str]:
        """Alias → real table name, for the physical (non-derived) sources only."""
        out: dict[str, str] = {}
        for alias, src in self.sources(scope).items():
            if isinstance(src, exp.Table):
                tbl = self.schema.table(src.name)
                out[alias] = tbl.name if tbl else src.name
        return out

    # -- resolution ------------------------------------------------------- #

    def resolve(self, column: exp.Column, scope: Scope) -> Binding:
        name = column.name
        if column.table:
            src = self.source_of(scope, column.table)
            if src is None:
                return Binding(None, name)
            return self._from_source(src, name)

        matches: list[Binding] = []
        for src in self.sources(scope).values():
            found = self._from_source(src, name)
            if found.table is not None:
                matches.append(found)
        if not matches:
            return Binding(None, name)
        distinct = {m.qualified for m in matches}
        if len(distinct) > 1:
            return Binding(None, name, ambiguous=True, candidates=tuple(sorted(distinct)))
        return matches[0]

    def _from_source(self, src: Any, name: str) -> Binding:
        if isinstance(src, exp.Table):
            table = self.schema.table(src.name)
            if table is None or table.column(name) is None:
                return Binding(None, name)
            return Binding(table.name, table.column(name).name)  # type: ignore[union-attr]
        if isinstance(src, Scope):
            return self._through_scope(src, name)
        return Binding(None, name)

    def _through_scope(self, scope: Scope, name: str, depth: int = 0) -> Binding:
        """Follow ``name`` through a CTE / derived table to its physical origin."""
        if depth > 8:  # pragma: no cover - pathological nesting
            return Binding(None, name)
        select = scope.expression
        if not isinstance(select, exp.Select):
            inner = select.this if isinstance(select, exp.SetOperation) else None
            if not isinstance(inner, exp.Select):
                return Binding(None, name)
            select = inner

        for projection in select.expressions:
            if isinstance(projection, exp.Star) or (
                isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
            ):
                # ``SELECT *`` passes the name straight through to the inner scope.
                inner_binding = self.resolve(exp.column(name), scope)
                if inner_binding.table is not None:
                    return inner_binding
                continue
            if projection.output_name.lower() != name.lower():
                continue
            target = projection.this if isinstance(projection, exp.Alias) else projection
            if isinstance(target, exp.Column):
                return self.resolve(target, scope)
            return self._most_sensitive(target, scope)
        return Binding(None, name)

    def _most_sensitive(self, expression: exp.Expr, scope: Scope) -> Binding:
        """A computed projection inherits the origin of the columns it consumes.

        ``SELECT substr(TELNO, 1, 3) AS p`` must not launder ``TELNO`` into a
        public column, so the derived reference keeps the base binding.
        """
        best: Binding | None = None
        for col in expression.find_all(exp.Column):
            bound = self.resolve(col, scope)
            if bound.table is None:
                continue
            if best is None:
                best = bound
        if best is None:
            return Binding(None, expression.output_name or "?")
        return Binding(best.table, best.column, derived=True)


# --------------------------------------------------------------------------- #
# Guard
# --------------------------------------------------------------------------- #


class PolicyGuard:
    """Applies a :class:`PolicyDocument` to one statement, returning a verdict."""

    def __init__(
        self,
        schema: SchemaGraph,
        policy: PolicyDocument,
        settings: Settings | None = None,
    ) -> None:
        self.schema = schema
        self.policy = policy
        self.settings = settings or get_settings()
        eng, ver = policy.engine, self.settings.verify
        self.force_limit = eng.force_limit and ver.force_limit
        self.default_limit = min(eng.default_limit, ver.default_limit)
        self.max_rows = min(eng.max_rows, self.settings.database.max_rows)
        self.max_join_tables = min(eng.max_join_tables, ver.max_join_tables)
        self._stamp_schema()

    def _stamp_schema(self) -> None:
        """Publish the classification onto the SchemaGraph (see ``ColumnInfo.sensitivity``)."""
        for col in self.schema.all_columns:
            col.sensitivity = self.policy.sensitivity(col.table, col.name)

    # -- entry point ------------------------------------------------------ #

    def check(self, sql: str, ctx: dict | None = None) -> GuardVerdict:
        ctx = dict(ctx or {})
        violations: list[Violation] = []
        rewrites: list[str] = []

        try:
            statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
        except (ParseError, TokenError, RecursionError) as exc:
            msg = str(exc).splitlines()[0][:160]
            return GuardVerdict(
                allowed=False,
                violations=[Violation("SQL_PARSE", f"SQL을 파싱할 수 없습니다: {msg}", "error")],
                rewritten_sql=None,
            )
        if not statements:
            return GuardVerdict(
                allowed=False,
                violations=[Violation("SQL_PARSE", "빈 SQL 구문입니다.", "error")],
            )

        if len(statements) > 1 and self.policy.engine.single_statement:
            violations.append(
                Violation(
                    "MULTI_STATEMENT",
                    f"단일 SELECT 문만 허용됩니다 ({len(statements)}개 구문이 감지되었습니다).",
                    "block",
                )
            )
        root = statements[0]
        violations.extend(self._check_shape(statements))
        violations.extend(self._check_functions(root))
        if any(v.severity == "block" for v in violations):
            # A statement of the wrong shape must not be analysed further; its
            # scopes are meaningless and any rewrite would be misleading.
            return GuardVerdict(False, violations, rewritten_sql=None)

        outputs = output_selects(root)
        violations.extend(self._expand_stars(root, outputs))

        resolver = ColumnResolver(self.schema, root)
        mask_plan: list[tuple[exp.Select, int, exp.Column, str]] = []
        violations.extend(self._check_columns(resolver, outputs, mask_plan))
        violations.extend(self._check_join_width(resolver))

        for select, index, column, qualified in mask_plan:
            rewrites.append(self._apply_mask(select, index, column, qualified))
        rewrites.extend(self._apply_row_policies(resolver, ctx, violations))
        rewrites.extend(self._apply_k_anonymity(resolver, outputs))
        rewrites.extend(self._apply_limits(root, outputs))

        blocking = [v for v in violations if v.severity == "block"]
        rewritten = root.sql(dialect=DIALECT, pretty=False)
        if blocking:
            log.warning("statement blocked", codes=[v.code for v in blocking], policy=self.policy.name)
        return GuardVerdict(
            allowed=not blocking,
            violations=violations,
            rewritten_sql=rewritten,
            applied_rewrites=[r for r in rewrites if r],
        )

    # -- 2. statement shape ----------------------------------------------- #

    def _check_shape(self, statements: list[exp.Expr]) -> list[Violation]:
        out: list[Violation] = []
        for statement in statements:
            for node in statement.walk():
                for kind, code in _FORBIDDEN_ROOTS:
                    if isinstance(node, kind):
                        out.append(
                            Violation(
                                code,
                                f"읽기 전용 정책: {type(node).__name__.upper()} 구문은 실행할 수 없습니다.",
                                "block",
                                type(node).__name__.upper(),
                            )
                        )
                        break
                else:
                    if isinstance(node, exp.Command):
                        out.append(
                            Violation(
                                "WRITE_FORBIDDEN",
                                f"지원하지 않는 명령입니다: {str(node.this).upper()}",
                                "block",
                                str(node.this).upper(),
                            )
                        )
            if not isinstance(statement, (exp.Select, exp.SetOperation, exp.Subquery)) and not out:
                out.append(
                    Violation(
                        "WRITE_FORBIDDEN",
                        f"SELECT/WITH 구문만 허용됩니다 (감지: {type(statement).__name__}).",
                        "block",
                        type(statement).__name__,
                    )
                )
        return _dedupe(out)

    # -- 3. forbidden functions ------------------------------------------- #

    def _check_functions(self, root: exp.Expr) -> list[Violation]:
        forbidden = set(self.policy.engine.forbid_functions)
        if not forbidden:
            return []
        out: list[Violation] = []
        for node in root.find_all(exp.Func):
            name = function_name(node)
            if name in forbidden:
                out.append(
                    Violation(
                        "FUNCTION_FORBIDDEN",
                        f"금지된 함수입니다: {name}()",
                        "block",
                        name.lower(),
                    )
                )
        return _dedupe(out)

    # -- 4. star expansion ------------------------------------------------- #

    def _expand_stars(self, root: exp.Expr, outputs: list[exp.Select]) -> list[Violation]:
        """Rewrite ``SELECT *`` into an explicit column list before classifying.

        An unexpanded star is invisible to column-level governance while still
        returning every column of the table — including the forbidden ones.
        """
        resolver = ColumnResolver(self.schema, root)
        out: list[Violation] = []
        for select in outputs:
            scope = resolver.scope_of(select)
            if scope is None or not _has_star(select):
                continue
            expanded: list[exp.Expr] = []
            resolved = True
            for projection in select.expressions:
                names = self._star_columns(resolver, scope, projection)
                if names is None:
                    expanded.append(projection)
                    if _is_star(projection):
                        resolved = False
                    continue
                expanded.extend(exp.column(col, table=alias) for alias, col in names)
            if not resolved:
                out.append(
                    Violation(
                        "STAR_UNRESOLVED",
                        "스키마에 없는 소스의 '*'는 컬럼 등급을 판정할 수 없습니다.",
                        "warn",
                    )
                )
                continue
            select.set("expressions", expanded)
        return out

    def _star_columns(
        self, resolver: ColumnResolver, scope: Scope, projection: exp.Expr
    ) -> list[tuple[str, str]] | None:
        """``[(alias, column), ...]`` for a star projection, or ``None`` if not one."""
        if isinstance(projection, exp.Star):
            wanted = list(resolver.sources(scope).items())
        elif isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
            alias = resolver.alias_of(scope, projection.table)
            src = resolver.source_of(scope, projection.table)
            wanted = [(alias, src)] if alias and src is not None else []
        else:
            return None

        out: list[tuple[str, str]] = []
        for alias, src in wanted:
            if isinstance(src, exp.Table):
                table = self.schema.table(src.name)
                if table is None:
                    return None
                qualifier = src.alias or table.name
                out.extend((qualifier, c) for c in table.column_names)
            elif isinstance(src, Scope) and isinstance(src.expression, exp.Select):
                inner = src.expression.expressions
                if any(_is_star(p) for p in inner):
                    return None
                out.extend((alias, p.output_name) for p in inner if p.output_name)
            else:
                return None
        return out or None

    # -- 5. column governance ---------------------------------------------- #

    def _check_columns(
        self,
        resolver: ColumnResolver,
        outputs: list[exp.Select],
        mask_plan: list[tuple[exp.Select, int, exp.Column, str]],
    ) -> list[Violation]:
        out: list[Violation] = []
        output_ids = {id(s) for s in outputs}
        for scope in resolver.scopes:
            for column in scope.columns:
                binding = resolver.resolve(column, scope)
                if binding.ambiguous:
                    out.append(
                        Violation(
                            "AMBIGUOUS_COLUMN",
                            f"'{binding.column}' 컬럼이 여러 테이블에 존재합니다: "
                            f"{', '.join(binding.candidates)}. 테이블 별칭으로 한정하세요.",
                            "error",
                            binding.column,
                        )
                    )
                    continue
                if binding.table is None:
                    continue
                grade = self.policy.sensitivity(binding.table, binding.column)
                if grade is Sensitivity.PUBLIC:
                    continue
                select, clause = occurrence_of(column)
                is_output = select is not None and id(select) in output_ids
                out.extend(
                    self._classify(binding, column, select, clause, is_output, grade, mask_plan)
                )
        return _dedupe(out)

    def _classify(
        self,
        binding: Binding,
        column: exp.Column,
        select: exp.Select | None,
        clause: str,
        is_output: bool,
        grade: Sensitivity,
        mask_plan: list[tuple[exp.Select, int, exp.Column, str]],
    ) -> list[Violation]:
        qualified = binding.qualified
        if grade is Sensitivity.FORBIDDEN:
            return [
                Violation(
                    "PII_FORBIDDEN",
                    f"{qualified}는 고유식별정보로 어떤 형태로도 조회할 수 없습니다.",
                    "block",
                    qualified,
                )
            ]

        if grade is Sensitivity.MASKED:
            if clause in _PREDICATE_CLAUSES:
                return [
                    Violation(
                        "PII_PREDICATE",
                        f"{qualified}는 마스킹 대상이므로 조건절(WHERE/GROUP BY/ORDER BY/ON)에 "
                        "사용할 수 없습니다. 값 비교로 원본이 복원됩니다.",
                        "block",
                        qualified,
                    )
                ]
            if clause != "expressions" or not is_output or select is None:
                return []  # inner projection: the outer boundary decides
            index, projection = _projection_of(select, column)
            if index < 0:
                return []
            if isinstance(aggregate_ancestor(column), exp.Count):
                return []  # cardinality only, no value leaves
            target = projection.this if isinstance(projection, exp.Alias) else projection
            if not isinstance(target, exp.Column) or binding.derived:
                return [
                    Violation(
                        "PII_EXPRESSION",
                        f"{qualified}는 가공식 안에서 조회할 수 없습니다 "
                        "(마스킹 적용을 보증할 수 없습니다).",
                        "block",
                        qualified,
                    )
                ]
            mask_plan.append((select, index, column, qualified))
            return []

        if grade is Sensitivity.INTERNAL:
            if aggregate_ancestor(column) is not None or clause == "group":
                return []
            if clause != "expressions" or not is_output or select is None:
                return []
            if in_group_by(select, column):
                return []
            return [
                Violation(
                    "INTERNAL_ROWLEVEL",
                    f"{qualified}는 집계 결과로만 조회할 수 있습니다 "
                    "(COUNT/SUM/AVG/MIN/MAX 또는 GROUP BY 키로 사용하세요).",
                    "block",
                    qualified,
                )
            ]
        return []

    def _apply_mask(
        self, select: exp.Select, index: int, column: exp.Column, qualified: str
    ) -> str:
        template = self.policy.mask_expression(qualified) or "'***'"
        original = select.expressions[index]
        output_name = original.output_name or column.name
        rendered = template.replace("{col}", column.sql(dialect=DIALECT))
        masked = exp.alias_(sqlglot.parse_one(rendered, dialect=DIALECT), output_name)
        expressions = list(select.expressions)
        expressions[index] = masked
        select.set("expressions", expressions)
        return f"mask:{qualified}"

    # -- 6. row policies ---------------------------------------------------- #

    def _apply_row_policies(
        self, resolver: ColumnResolver, ctx: dict, violations: list[Violation]
    ) -> list[str]:
        applied: list[str] = []
        for policy in self.policy.row_policies:
            verdict = _eval_when(policy.when, ctx)
            if verdict is None:
                violations.append(
                    Violation(
                        "POLICY_EXPR_UNSUPPORTED",
                        f"행 정책 {policy.id}의 조건식을 해석할 수 없습니다: {policy.when}",
                        "warn",
                        policy.id,
                    )
                )
                continue
            if not verdict:
                continue
            rendered = _render_filter(policy.filter, ctx)
            if rendered is None:
                continue
            for scope in resolver.scopes:
                select = scope.expression
                if not isinstance(select, exp.Select):
                    continue
                alias = next(
                    (a for a, t in resolver.physical_tables(scope).items()
                     if t.upper() == policy.table.upper()),
                    None,
                )
                if alias is None:
                    continue
                condition = sqlglot.parse_one(rendered, dialect=DIALECT)
                for col in condition.find_all(exp.Column):
                    if col.table and col.table.upper() == policy.table.upper():
                        col.set("table", exp.to_identifier(alias))
                select.where(condition, copy=False)
                applied.append(f"row-policy:{policy.id}")
        return applied

    # -- 7-8. limits and k-anonymity ---------------------------------------- #

    def _apply_limits(self, root: exp.Expr, outputs: list[exp.Select]) -> list[str]:
        applied: list[str] = []
        limit = root.args.get("limit")
        if limit is not None:
            current = _int_literal(limit.expression)
            if current is not None and current > self.max_rows:
                limit.set("expression", exp.Literal.number(self.max_rows))
                applied.append(f"limit-capped:{self.max_rows}")
            return applied
        if not self.force_limit:
            return applied
        if len(outputs) == 1 and is_scalar_aggregate(outputs[0]):
            return applied  # a scalar aggregate returns exactly one row already
        root.set("limit", exp.Limit(expression=exp.Literal.number(self.default_limit)))
        applied.append(f"limit-injected:{self.default_limit}")
        return applied

    def _apply_k_anonymity(self, resolver: ColumnResolver, outputs: list[exp.Select]) -> list[str]:
        kan = self.policy.k_anonymity
        if not kan.enabled:
            return []
        applied: list[str] = []
        for select in outputs:
            if not select.args.get("group"):
                continue
            scope = resolver.scope_of(select)
            tables = set(resolver.physical_tables(scope).values()) if scope else set()
            if not any(kan.applies_to(t) for t in tables):
                continue
            select.having(
                exp.GTE(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(kan.k)),
                copy=False,
            )
            applied.append(f"k-anonymity:{kan.k}")
        return applied

    # -- 9. join width ------------------------------------------------------ #

    def _check_join_width(self, resolver: ColumnResolver) -> list[Violation]:
        tables = {
            table
            for scope in resolver.scopes
            for table in resolver.physical_tables(scope).values()
        }
        if len(tables) <= self.max_join_tables:
            return []
        return [
            Violation(
                "TOO_MANY_JOINS",
                f"조인 테이블 수가 정책 한도를 초과했습니다 "
                f"({len(tables)} > {self.max_join_tables}).",
                "block",
                ",".join(sorted(tables)),
            )
        ]


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #


def function_name(node: exp.Func) -> str:
    """Lower-cased call name, for both known nodes and unparsed ``Anonymous`` ones."""
    return (str(node.this) if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def output_selects(root: exp.Expr) -> list[exp.Select]:
    """The SELECTs whose projections form the final result set."""
    if isinstance(root, exp.Subquery):
        return output_selects(root.this)
    if isinstance(root, exp.Select):
        return [root]
    if isinstance(root, exp.SetOperation):
        return output_selects(root.this) + output_selects(root.expression)
    return []


def _is_star(node: exp.Expr) -> bool:
    return isinstance(node, exp.Star) or (
        isinstance(node, exp.Column) and isinstance(node.this, exp.Star)
    )


def _has_star(select: exp.Select) -> bool:
    return any(_is_star(p) for p in select.expressions)


def occurrence_of(column: exp.Column) -> tuple[exp.Select | None, str]:
    """Nearest enclosing SELECT and the clause the column sits in."""
    node: exp.Expr = column
    while node.parent is not None:
        parent = node.parent
        if isinstance(parent, exp.Join) and node.arg_key in {"on", "using"}:
            select = parent.parent
            return (select if isinstance(select, exp.Select) else None), "join_on"
        if isinstance(parent, exp.Select):
            key = node.arg_key or ""
            return parent, {"expressions": "expressions"}.get(key, key)
        node = parent
    return None, "unknown"


def _projection_of(select: exp.Select, column: exp.Column) -> tuple[int, exp.Expr]:
    for index, projection in enumerate(select.expressions):
        if projection is column or any(c is column for c in projection.find_all(exp.Column)):
            return index, projection
    return -1, column


def aggregate_ancestor(column: exp.Column) -> exp.AggFunc | None:
    node: exp.Expr | None = column.parent
    while node is not None and not isinstance(node, exp.Select):
        if isinstance(node, exp.AggFunc):
            return node
        node = node.parent
    return None


def in_group_by(select: exp.Select, column: exp.Column) -> bool:
    group = select.args.get("group")
    if not group:
        return False
    target = column.sql(dialect=DIALECT).lower()
    return any(g.sql(dialect=DIALECT).lower() == target for g in group.expressions) or any(
        g.name.lower() == column.name.lower() for g in group.find_all(exp.Column)
    )


def is_scalar_aggregate(select: exp.Select) -> bool:
    if select.args.get("group") or not select.expressions:
        return False
    return all(
        any(isinstance(n, exp.AggFunc) for n in p.find_all(exp.AggFunc))
        or isinstance(p, (exp.Literal, exp.Alias))
        and not list(p.find_all(exp.Column))
        for p in select.expressions
    )


def _int_literal(node: exp.Expr | None) -> int | None:
    if isinstance(node, exp.Literal) and node.is_int:
        return int(node.this)
    return None


def _dedupe(violations: Iterable[Violation]) -> list[Violation]:
    seen: set[tuple[str, str | None]] = set()
    out: list[Violation] = []
    for v in violations:
        key = (v.code, v.subject)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# The tiny session-context expression language
# --------------------------------------------------------------------------- #


def _sql_escape(value: str) -> str:
    """Session context is caller-supplied; it must not be able to close a literal."""
    return value.replace("'", "''")


def _render_filter(template: str, ctx: dict) -> str | None:
    """Substitute ``{{ctx.var}}``; returns ``None`` when a variable is missing."""
    missing = False

    def substitute(match: re.Match[str]) -> str:
        nonlocal missing
        value = ctx.get(match.group(1))
        if value is None:
            missing = True
            return ""
        return _sql_escape(str(value))

    rendered = _TEMPLATE_RE.sub(substitute, template)
    return None if missing else rendered


def _eval_when(expression: str, ctx: dict) -> bool | None:
    """Evaluate the policy ``when:`` mini-language.

    Supported forms are exactly the ones the YAML uses — ``ctx.var`` (truthy),
    ``ctx.var == 'literal'``, ``!=``, and ``and``/``or``/``not`` combinations
    evaluated left to right.  Anything else returns ``None`` so the caller can
    surface it instead of silently ignoring a policy.  ``eval()`` is never
    involved: a policy file is configuration, not code.
    """
    text = (expression or "").strip()
    if not text:
        return False

    tokens: list[str] = []
    position = 0
    while position < len(text):
        match = _WHEN_TOKEN_RE.match(text, position)
        if match is None or match.end() == position:
            return None
        position = match.end()
        for group in ("op", "bool", "var", "sq", "dq", "num", "lit"):
            value = match.group(group)
            if value is not None:
                tokens.append(f"{group}:{value}")
                break
    if position != len(text.rstrip()):  # pragma: no cover - trailing garbage
        return None

    result: bool | None = None
    operator = "and"
    negate = False
    index = 0
    while index < len(tokens):
        kind, _, value = tokens[index].partition(":")
        if kind == "bool":
            lowered = value.lower()
            if lowered == "not":
                negate = True
            else:
                operator = lowered
            index += 1
            continue
        if kind != "var":
            return None
        term = ctx.get(value)
        index += 1
        if index + 1 < len(tokens) and tokens[index].startswith("op:"):
            comparison = tokens[index].split(":", 1)[1]
            literal_kind, _, literal = tokens[index + 1].partition(":")
            if literal_kind not in {"sq", "dq", "num", "lit"}:
                return None
            outcome = str(term) == literal if comparison == "==" else str(term) != literal
            index += 2
        else:
            outcome = bool(term)
        if negate:
            outcome = not outcome
            negate = False
        result = outcome if result is None else (result and outcome if operator == "and" else result or outcome)
    return result
