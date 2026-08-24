"""FK join-graph reasoning.

Schema linking gives us a *set* of tables; a generator needs the *joins* that
connect them.  This module answers that with a weighted shortest-path search
over the FK graph plus a small Steiner-tree approximation for the multi-table
case, and it knows about the common-code table so that ``TB_COMM_CD`` is
attached via an equality on both ``CD_GRP`` and ``CD`` rather than a bare FK
(there is no FK — that is exactly why naive schema linking mis-joins it).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from aegis_sql.schema.introspect import CODE_GROUP_COLUMN, CODE_VALUE_COLUMN
from aegis_sql.types import ForeignKey, SchemaGraph


@dataclass(slots=True)
class JoinEdge:
    left_table: str
    right_table: str
    on: list[tuple[str, str]]          # [(left_col, right_col), ...]
    literal_filters: list[tuple[str, str]] = None  # type: ignore[assignment]
    kind: str = "fk"                   # "fk" | "code"

    def __post_init__(self) -> None:
        if self.literal_filters is None:
            self.literal_filters = []

    def to_sql(self, left_alias: str, right_alias: str) -> str:
        conds = [f"{left_alias}.{lc} = {right_alias}.{rc}" for lc, rc in self.on]
        conds += [f"{right_alias}.{col} = '{val}'" for col, val in self.literal_filters]
        return " AND ".join(conds)


class JoinGraph:
    """Undirected view of the schema's FK edges with join-path search."""

    def __init__(self, schema: SchemaGraph) -> None:
        self.schema = schema
        self._adj: dict[str, list[tuple[str, JoinEdge]]] = defaultdict(list)
        self._build()

    def _build(self) -> None:
        for fk in self.schema.foreign_keys:
            fwd = JoinEdge(fk.from_table, fk.to_table, [(fk.from_column, fk.to_column)])
            rev = JoinEdge(fk.to_table, fk.from_table, [(fk.to_column, fk.from_column)])
            self._adj[fk.from_table].append((fk.to_table, fwd))
            self._adj[fk.to_table].append((fk.from_table, rev))

    # -- neighbourhood ---------------------------------------------------- #

    def neighbours(self, table: str) -> list[str]:
        return [t for t, _ in self._adj.get(table, [])]

    def expand(self, tables: set[str], hops: int = 1) -> set[str]:
        """Return ``tables`` plus everything reachable within ``hops`` FK hops."""
        frontier = set(tables)
        seen = set(tables)
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for t in frontier:
                for n in self.neighbours(t):
                    if n not in seen:
                        nxt.add(n)
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return seen

    # -- paths ------------------------------------------------------------ #

    def shortest_path(self, src: str, dst: str) -> list[JoinEdge] | None:
        """BFS over FK edges.  Returns the edge chain, or ``None`` if disjoint."""
        if src == dst:
            return []
        queue: deque[tuple[str, list[JoinEdge]]] = deque([(src, [])])
        visited = {src}
        while queue:
            node, path = queue.popleft()
            for nxt, edge in self._adj.get(node, []):
                if nxt in visited:
                    continue
                new_path = [*path, edge]
                if nxt == dst:
                    return new_path
                visited.add(nxt)
                queue.append((nxt, new_path))
        return None

    def connect(self, tables: list[str]) -> tuple[list[str], list[JoinEdge]]:
        """Approximate a Steiner tree joining every table in ``tables``.

        Greedy: start from the table with the highest degree (usually the fact
        table), then repeatedly attach the unconnected table whose shortest path
        to the current tree is cheapest, adding any intermediate bridge tables.
        """
        wanted = [t for t in dict.fromkeys(tables) if self.schema.table(t)]
        if len(wanted) <= 1:
            return wanted, []

        root = max(wanted, key=lambda t: len(self._adj.get(t, [])))
        tree: set[str] = {root}
        edges: list[JoinEdge] = []
        remaining = [t for t in wanted if t != root]

        while remaining:
            best: tuple[int, str, list[JoinEdge], str] | None = None
            for target in remaining:
                for anchor in tree:
                    path = self.shortest_path(anchor, target)
                    if path is None:
                        continue
                    if best is None or len(path) < best[0]:
                        best = (len(path), target, path, anchor)
            if best is None:
                # Disconnected component — keep the table, let the generator
                # decide (usually a cross-join-free sub-select).
                remaining.pop(0)
                continue
            _, target, path, _anchor = best
            for edge in path:
                if edge not in edges:
                    edges.append(edge)
                tree.add(edge.left_table)
                tree.add(edge.right_table)
            remaining = [t for t in remaining if t not in tree]

        ordered = [root] + [t for t in _edge_order(edges) if t != root]
        for t in wanted:
            if t not in ordered:
                ordered.append(t)
        return ordered, edges

    # -- code-table joins -------------------------------------------------- #

    def code_join(self, table: str, column: str, code_table: str = "TB_COMM_CD") -> JoinEdge | None:
        """Build the two-predicate join that resolves a code column to its label."""
        col = self.schema.column(table, column)
        if not col or not col.code_group or not self.schema.table(code_table):
            return None
        return JoinEdge(
            left_table=table,
            right_table=code_table,
            on=[(column, CODE_VALUE_COLUMN)],
            literal_filters=[(CODE_GROUP_COLUMN, col.code_group)],
            kind="code",
        )

    # -- diagnostics ------------------------------------------------------- #

    def degree(self, table: str) -> int:
        return len(self._adj.get(table, []))

    def max_depth(self, tables: list[str]) -> int:
        """Longest shortest-path between any pair — a difficulty feature."""
        depth = 0
        for i, a in enumerate(tables):
            for b in tables[i + 1 :]:
                path = self.shortest_path(a, b)
                if path is not None:
                    depth = max(depth, len(path))
        return depth

    def as_edges(self) -> list[ForeignKey]:
        return self.schema.foreign_keys


def _edge_order(edges: list[JoinEdge]) -> list[str]:
    order: list[str] = []
    for e in edges:
        for t in (e.left_table, e.right_table):
            if t not in order:
                order.append(t)
    return order
