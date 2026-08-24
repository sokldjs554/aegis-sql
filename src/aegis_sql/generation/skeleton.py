"""Structural views of a SQL statement — canonical form, skeleton, components.

Raw SQL strings are a terrible unit of comparison.  ``SELECT COUNT(*) FROM
TB_CTRT AS a WHERE a.MON_PRM >= 200000`` and ``select count(*) from TB_CTRT t1
where t1.mon_prm>=200000`` are the same query, yet no string metric says so, and
every downstream consumer in this repository needs them to be equal:

* **Evaluation.**  ``eval/metrics.py`` computes exact-set-match against a gold
  query written by a different author, with different aliases and different
  whitespace.  :func:`normalize_sql` is the equivalence class that makes that
  metric measure semantics instead of typing habits.
* **Diagnosis.**  Execution accuracy tells you *that* a query is wrong; it never
  tells you *why*.  Comparing the RESDSQL-style :func:`sql_skeleton` of the
  prediction against the gold separates "right plan, wrong constant" (skeleton
  matches, execution differs — a value-linking bug) from "wrong plan" (skeleton
  differs — a schema-linking or reasoning bug).  Those two failures are fixed in
  completely different parts of the pipeline, so the split is worth a metric.
* **Data curation.**  The flywheel de-duplicates synthesised training pairs and
  the retriever diversifies few-shot exemplars along the skeleton axis: two
  examples with the same skeleton teach the model the same lesson twice, and a
  prompt has room for six examples, not six copies of one.
* **Routing.**  :func:`difficulty_of` labels a corpus of gold SQL so the cascade
  router has supervision that does not require a human to grade 4000 queries.

Every function here is total: SQL that sqlglot cannot parse (a half-generated
LLM completion, a dialect quirk) degrades to a regex path rather than raising,
because these functions run inside metric loops where one bad row must not
abort the evaluation.  Nothing depends on the clock, the environment, or hash
ordering, so the same statement always yields the same key.
"""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError, TokenError

from aegis_sql.observability.logging import get_logger

log = get_logger("generation.skeleton")

__all__ = [
    "normalize_sql",
    "sql_skeleton",
    "mask_question",
    "sql_components",
    "difficulty_of",
]

#: Placeholders shared with ``retrieval/fewshot.py`` so masked questions coming
#: from either module land in the same vector space.
PLACEHOLDER_NUM = "<NUM>"
PLACEHOLDER_DATE = "<DATE>"
PLACEHOLDER_VAL = "<VAL>"

#: Skeleton vocabulary (RESDSQL §3.2): structure survives, content does not.
SKELETON_IDENT = "_"
SKELETON_LITERAL = "?"

_PARSE_ERRORS = (ParseError, TokenError, SqlglotError, ValueError, RecursionError)

_SQL_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'(?:[^']|'')*'")
_WS_RE = re.compile(r"\s+")
_QUOTE_CHARS = re.compile(r'[`"\[\]]')
_SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([),])")
_SPACE_AFTER_OPEN_RE = re.compile(r"\(\s+")
_SPACE_BEFORE_OPEN_RE = re.compile(r"(\w)\s+\(")
#: Only *comparison* operators are re-spaced; touching ``*`` would break ``count(*)``.
_OPERATOR_SPACING_RE = re.compile(r"\s*(<=|>=|<>|!=|\|\||[=<>])\s*")

_SQL_TOKEN_RE = re.compile(
    r"'(?:[^']|'')*'|[A-Za-z_가-힣][A-Za-z0-9_$가-힣]*"
    r"|\d+\.\d+|\d+|<=|>=|<>|!=|\|\||[(),;.*=<>+\-/%]"
)

#: Reserved words that must survive skeletonisation.  Anything else that looks
#: like a word is an identifier and collapses to ``_``.
_SQL_KEYWORDS = frozenset((
    "SELECT", "DISTINCT", "ALL", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT",
    "OFFSET", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL", "ON",
    "USING", "AND", "OR", "NOT", "IN", "EXISTS", "BETWEEN", "LIKE", "GLOB", "IS", "NULL", "AS",
    "UNION", "INTERSECT", "EXCEPT", "CASE", "WHEN", "THEN", "ELSE", "END", "CAST", "ASC", "DESC",
    "WITH", "RECURSIVE", "OVER", "PARTITION", "COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "ABS",
    "COALESCE", "NULLIF", "SUBSTR", "STRFTIME", "JULIANDAY", "LENGTH", "IFNULL", "REAL",
    "INTEGER", "TEXT", "NUMERIC",
))

#: Aggregate node types.  ``sqlglot`` has no single base class for these.
_AGG_NODES: tuple[type[exp.Expression], ...] = (
    exp.Count,
    exp.Sum,
    exp.Avg,
    exp.Min,
    exp.Max,
    exp.Stddev,
    exp.Variance,
    exp.GroupConcat,
)


# --------------------------------------------------------------------------- #
# canonical form
# --------------------------------------------------------------------------- #


def normalize_sql(sql: str, dialect: str = "sqlite") -> str:
    """Canonical string for exact-set-match comparison.

    Four kinds of noise are removed, and nothing else:

    1. **Alias naming.**  Table aliases are renumbered ``t1..tn`` in the order
       their sources appear, and every column qualifier is rewritten to follow.
       Two authors who wrote ``a``/``b`` and ``c``/``ctr`` now agree.
    2. **Output aliases.**  ``SUM(x) AS total`` and ``SUM(x)`` select the same
       values; the label is presentation, not semantics, so it is dropped.
    3. **Identifier case.**  Lower-cased — the target dialects are all
       case-insensitive on identifiers.  String *literals* keep their case,
       because ``'V1'`` and ``'v1'`` are genuinely different predicates.
    4. **Layout.**  Keyword case, comments and whitespace are folded away.

    Clause *order* is deliberately preserved: ``ORDER BY a, b`` is not the same
    query as ``ORDER BY b, a``, and a normaliser that sorted them would report
    false matches — the failure mode that makes exact-set-match untrustworthy.
    """
    text = (sql or "").strip()
    if not text:
        return ""
    try:
        tree = sqlglot.parse_one(text, read=dialect)
    except _PARSE_ERRORS as exc:
        log.debug("normalize_sql falling back to regex", error=str(exc))
        return _regex_normalize(text)
    if tree is None:  # pragma: no cover - empty statement
        return _regex_normalize(text)

    tree = _canonicalise_aliases(tree)
    tree = tree.transform(_drop_output_alias)
    tree = tree.transform(_lower_identifier)
    try:
        rendered = tree.sql(dialect=dialect, comments=False)
    except _PARSE_ERRORS as exc:  # pragma: no cover - generator failure
        log.debug("normalize_sql generation failed", error=str(exc))
        return _regex_normalize(text)
    return _tidy(_lower_outside_literals(rendered))


def _canonicalise_aliases(tree: exp.Expr) -> exp.Expr:
    """Rename table aliases to ``t1..tn`` and rewrite the column qualifiers.

    Aliases are numbered across the whole statement rather than per scope.  A
    correlated sub-query that shadows an outer alias would therefore be
    renumbered as one namespace; that is rare enough in analytics SQL to be
    worth the simplicity, and it stays *consistent* (both sides of a comparison
    are treated identically), which is what the metric actually needs.
    """
    mapping: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        alias = table.alias or table.name
        if alias and alias.lower() not in mapping:
            mapping[alias.lower()] = f"t{len(mapping) + 1}"

    for table in tree.find_all(exp.Table):
        alias = (table.alias or table.name).lower()
        replacement = mapping.get(alias)
        if replacement:
            table.set("alias", exp.TableAlias(this=exp.to_identifier(replacement)))

    for column in tree.find_all(exp.Column):
        qualifier = column.table
        if qualifier and qualifier.lower() in mapping:
            column.set("table", exp.to_identifier(mapping[qualifier.lower()]))
    return tree


def _drop_output_alias(node: exp.Expression) -> exp.Expression:
    """Strip ``AS label`` from projections while leaving table aliases alone."""
    if isinstance(node, exp.Alias) and not isinstance(node.parent, exp.Table):
        return node.this
    return node


def _lower_identifier(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Identifier):
        return exp.to_identifier(node.name.lower(), quoted=node.quoted)
    return node


def _regex_normalize(sql: str) -> str:
    """Dependency-free fallback for statements sqlglot rejects."""
    body = _SQL_COMMENT_RE.sub(" ", sql)
    body = _QUOTE_CHARS.sub("", body)
    body = _lower_outside_literals(body)
    return _tidy(body)


def _lower_outside_literals(sql: str) -> str:
    """Lower-case everything except the contents of single-quoted strings."""
    out: list[str] = []
    cursor = 0
    for match in _STRING_RE.finditer(sql):
        out.append(sql[cursor : match.start()].lower())
        out.append(match.group(0))
        cursor = match.end()
    out.append(sql[cursor:].lower())
    return "".join(out)


def _tidy(sql: str) -> str:
    """Collapse layout without touching literals."""
    pieces: list[str] = []
    cursor = 0
    for match in _STRING_RE.finditer(sql):
        pieces.append(_tidy_fragment(sql[cursor : match.start()]))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(_tidy_fragment(sql[cursor:]))
    joined = "".join(pieces)
    return _WS_RE.sub(" ", joined).strip().rstrip(";").strip()


def _tidy_fragment(fragment: str) -> str:
    fragment = _WS_RE.sub(" ", fragment)
    fragment = _SPACE_BEFORE_CLOSE_RE.sub(r"\1", fragment)
    fragment = _SPACE_AFTER_OPEN_RE.sub("(", fragment)
    fragment = _SPACE_BEFORE_OPEN_RE.sub(r"\1(", fragment)
    return _OPERATOR_SPACING_RE.sub(r" \1 ", fragment)


# --------------------------------------------------------------------------- #
# skeleton
# --------------------------------------------------------------------------- #


def sql_skeleton(sql: str, dialect: str = "sqlite") -> str:
    """Query shape with all content erased (RESDSQL-style skeleton).

    Identifiers become ``_`` and literals become ``?``, so
    ``SELECT COUNT(*) FROM TB_CTRT WHERE MON_PRM >= 200000`` collapses to
    ``select count(_) from _ where _ >= ?``.  What survives is exactly the part
    a model has to *reason* about — clause structure, join count, operators,
    nesting — and what disappears is the part it has to *retrieve*.
    """
    text = (sql or "").strip()
    if not text:
        return ""
    try:
        tree = sqlglot.parse_one(text, read=dialect)
    except _PARSE_ERRORS as exc:
        log.debug("sql_skeleton falling back to regex", error=str(exc))
        return _regex_skeleton(text)
    if tree is None:  # pragma: no cover - empty statement
        return _regex_skeleton(text)

    tree = tree.transform(_drop_output_alias)
    try:
        rendered = tree.transform(_skeletonise).sql(dialect=dialect, comments=False)
    except _PARSE_ERRORS as exc:  # pragma: no cover - generator failure
        log.debug("sql_skeleton generation failed", error=str(exc))
        return _regex_skeleton(text)
    return _tidy(_lower_outside_literals(rendered))


def _skeletonise(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Literal):
        return exp.Var(this=SKELETON_LITERAL)
    if isinstance(node, (exp.Placeholder, exp.Parameter)):
        return exp.Var(this=SKELETON_LITERAL)
    if isinstance(node, exp.Star):
        return exp.Var(this=SKELETON_IDENT)
    if isinstance(node, exp.Column):
        return exp.Column(this=exp.to_identifier(SKELETON_IDENT))
    if isinstance(node, exp.Table):
        return exp.Table(this=exp.to_identifier(SKELETON_IDENT))
    return node


def _regex_skeleton(sql: str) -> str:
    body = _SQL_COMMENT_RE.sub(" ", sql)
    out: list[str] = []
    for token in _SQL_TOKEN_RE.findall(body):
        head = token[0]
        if head == "'" or head.isdigit():
            out.append(SKELETON_LITERAL)
        elif head == "*":
            out.append(SKELETON_IDENT)
        elif head.isalpha() or head == "_":
            upper = token.upper()
            out.append(upper.lower() if upper in _SQL_KEYWORDS else SKELETON_IDENT)
        elif token == ";":
            continue
        else:
            out.append(token)
    # Collapse ``_ . _`` (a qualified column) into a single identifier slot.
    joined = _tidy(" ".join(out))
    return re.sub(r"_ \. _", SKELETON_IDENT, joined)


# --------------------------------------------------------------------------- #
# question masking
# --------------------------------------------------------------------------- #

_MASK_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"'[^']*'|\"[^\"]*\"|[“][^”]*[”]|[‘][^’]*[’]"), PLACEHOLDER_VAL),
    (re.compile(r"\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}"), PLACEHOLDER_DATE),
    (re.compile(r"\d{4}\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?"), PLACEHOLDER_DATE),
    (re.compile(r"(?<!\d)\d{8}(?!\d)"), PLACEHOLDER_DATE),
    (re.compile(r"\d{4}\s*[-/.]\s*\d{1,2}(?!\s*[-/.]?\s*\d)"), PLACEHOLDER_DATE),
    (re.compile(r"\d{4}\s*년|\d{1,2}\s*월(?!\s*\d)|\d{1,2}\s*분기|\d{1,2}\s*일(?![\d수])"), PLACEHOLDER_DATE),
    (re.compile(r"\d[\d,.]*\s*(?:조|억|만|천|백)?\s*(?:원|달러|퍼센트|%)"), PLACEHOLDER_NUM),
    (re.compile(r"\d[\d,.]*\s*(?:조|억|만|천)(?![가-힣])"), PLACEHOLDER_NUM),
    (re.compile(r"\d[\d,.]*\s*(?:건|명|개|위|회|점|세|년차|개월|주|가지)"), PLACEHOLDER_NUM),
    (re.compile(r"\d[\d,.]*"), PLACEHOLDER_NUM),
)


def mask_question(text: str) -> str:
    """Replace values in a question with ``<NUM>`` / ``<DATE>`` / ``<VAL>``.

    Few-shot retrieval (DAIL-SQL §4.1) should rank exemplars by *what is being
    asked*, not by which numbers happen to appear.  Without masking, "2025년
    보험금 지급액 합계" and "2024년 보험금 지급액 합계" look like different
    questions to a lexical scorer and identical exemplars crowd each other out.

    Rules are applied most-specific first — quoted strings, then dates, then
    amounts with a unit, then bare numbers — so ``20만원`` never degrades to
    ``<NUM>만원``.
    """
    masked = (text or "").strip()
    for pattern, placeholder in _MASK_RULES:
        masked = pattern.sub(placeholder, masked)
    return _WS_RE.sub(" ", masked).strip()


# --------------------------------------------------------------------------- #
# components & difficulty
# --------------------------------------------------------------------------- #


def sql_components(sql: str) -> dict[str, Any]:
    """Decompose a statement into the structural features the engine reasons about.

    Returns ``{"tables", "columns", "aggregates", "has_group_by", "has_order_by",
    "has_subquery", "n_joins", "limit"}``.  Column references are resolved back
    through table aliases to ``TABLE.COLUMN`` wherever the alias is bound in the
    statement, which is what makes the output comparable against a gold query
    that used different alias letters.
    """
    empty: dict[str, Any] = {
        "tables": [],
        "columns": [],
        "aggregates": [],
        "has_group_by": False,
        "has_order_by": False,
        "has_subquery": False,
        "n_joins": 0,
        "limit": None,
    }
    text = (sql or "").strip()
    if not text:
        return empty
    try:
        tree = sqlglot.parse_one(text, read="sqlite")
    except _PARSE_ERRORS as exc:
        log.debug("sql_components falling back to regex", error=str(exc))
        return _regex_components(text, empty)
    if tree is None:  # pragma: no cover - empty statement
        return empty

    # Identifiers are case-insensitive in every dialect this engine targets, so
    # de-duplication is case-folded while the first spelling seen is preserved.
    alias_to_table: dict[str, str] = {}
    tables: list[str] = []
    seen_tables: set[str] = set()
    for table in tree.find_all(exp.Table):
        name = table.name
        if not name:
            continue
        if name.lower() not in seen_tables:
            seen_tables.add(name.lower())
            tables.append(name)
        for key in (table.alias, name):
            if key:
                alias_to_table.setdefault(key.lower(), name)

    columns: list[str] = []
    seen_columns: set[str] = set()
    for column in tree.find_all(exp.Column):
        if not column.name:
            continue
        owner = alias_to_table.get(column.table.lower()) if column.table else None
        ref = f"{owner}.{column.name}" if owner else column.name
        if ref.lower() not in seen_columns:
            seen_columns.add(ref.lower())
            columns.append(ref)

    aggregates: list[str] = []
    for node in tree.find_all(*_AGG_NODES):
        name = type(node).__name__.upper()
        if name not in aggregates:
            aggregates.append(name)

    selects = list(tree.find_all(exp.Select))
    limit = tree.args.get("limit")
    limit_value: int | None = None
    if isinstance(limit, exp.Limit):
        try:
            limit_value = int(limit.expression.name)
        except (AttributeError, TypeError, ValueError):
            limit_value = None

    return {
        "tables": tables,
        "columns": columns,
        "aggregates": aggregates,
        "has_group_by": any(s.args.get("group") for s in selects),
        "has_order_by": bool(tree.args.get("order")) or any(s.args.get("order") for s in selects),
        "has_subquery": len(selects) > 1 or bool(list(tree.find_all(exp.CTE))),
        "n_joins": len(list(tree.find_all(exp.Join))),
        "limit": limit_value,
    }


def _regex_components(sql: str, empty: dict[str, Any]) -> dict[str, Any]:
    """Coarse feature extraction for statements sqlglot cannot parse."""
    lowered = _lower_outside_literals(_SQL_COMMENT_RE.sub(" ", sql))
    out = dict(empty)
    out["tables"] = re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_]*)", lowered)
    out["aggregates"] = [
        fn.upper() for fn in dict.fromkeys(re.findall(r"\b(count|sum|avg|min|max)\s*\(", lowered))
    ]
    out["has_group_by"] = "group by" in lowered
    out["has_order_by"] = "order by" in lowered
    out["has_subquery"] = lowered.count("select") > 1
    out["n_joins"] = len(re.findall(r"\bjoin\b", lowered))
    match = re.search(r"\blimit\s+(\d+)", lowered)
    out["limit"] = int(match.group(1)) if match else None
    return out


def difficulty_of(sql: str) -> str:
    """Grade a statement ``"easy" | "medium" | "hard"`` from its components.

    The thresholds mirror Spider's hardness criteria, adapted to the shapes this
    engine actually emits: a second join means a Steiner path through a bridge
    table was needed, and nesting means the question could not be answered by a
    single scan.  Both are the points where the template tier stops being enough
    and the router should be paying for a bigger model — which is precisely what
    this label supervises.
    """
    comp = sql_components(sql)
    n_agg = len(comp["aggregates"])
    n_joins = int(comp["n_joins"])

    if comp["has_subquery"] or n_joins >= 2 or n_agg >= 3:
        return "hard"
    if comp["has_group_by"] and n_agg >= 2:
        return "hard"
    if n_joins == 0 and not comp["has_group_by"] and n_agg <= 1:
        return "easy"
    return "medium"
