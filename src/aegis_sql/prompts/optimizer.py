"""Automatic prompt optimisation.

Prompt engineering is usually done by editing a string and forming an
impression.  This module turns it into a search with a held-out objective:
mutate the prompt, score every variant on a dev split of the benchmark with the
real engine, keep what wins, and record the delta.  The result is a versioned
prompt whose changelog says *how many points* a change was worth — which is the
only defensible way to answer "is this prompt better?".

Two mutation sources:

* **Deterministic operators** (always available, no API key) — insert or drop a
  rule, reorder sections, tighten an output contract, swap the schema-card style.
  These encode the edits a practitioner actually makes.
* **LLM rewrites** (when a provider is configured) — an APE-style "propose a
  better instruction" step, seeded with the current prompt and its failure cases.

Both are evaluated identically.  Nothing is accepted on vibes.

Reference: automatic instruction search (APE, Zhou et al. ICLR 2023) adapted to a
task with a hard, executable objective — execution accuracy rather than a
likelihood proxy.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aegis_sql.config import PROJECT_ROOT, Settings
from aegis_sql.observability.logging import get_logger
from aegis_sql.prompts.registry import PromptRecord, PromptRegistry

log = get_logger("prompts.optimizer")


# --------------------------------------------------------------------------- #
# mutation operators
# --------------------------------------------------------------------------- #

#: Candidate rules the optimiser may add.  Each one is a hypothesis about a
#: failure mode observed on this schema; the search decides which actually pay.
CANDIDATE_RULES: list[tuple[str, str]] = [
    ("date-substr", "날짜 컬럼에서 연/월을 뽑을 때는 substr(컬럼,1,4) / substr(컬럼,1,6) 을 사용하세요."),
    ("no-code-join", "코드명이 결과에 필요 없으면 TB_COMM_CD 조인을 생략하고 코드값을 직접 비교하세요."),
    ("real-div", "비율 계산은 반드시 CAST(... AS REAL) / NULLIF(..., 0) 형태로 쓰세요."),
    ("distinct-count", "1:N 조인 후 개체 수를 셀 때는 COUNT(DISTINCT 키) 를 사용하세요."),
    ("alias-required", "모든 테이블에 짧은 별칭을 붙이고 모든 컬럼을 별칭으로 한정하세요."),
    ("null-guard", "NULL 이 섞일 수 있는 컬럼을 집계할 때는 의도한 NULL 처리(WHERE ... IS NOT NULL)를 명시하세요."),
    ("between-inclusive", "기간 조건은 양끝을 포함하는 BETWEEN 을 기본으로 하세요."),
    ("order-explicit", "상위/하위를 묻는 질문에는 반드시 ORDER BY 와 LIMIT 을 함께 쓰세요."),
]

#: Rules that can be dropped to test whether they are still earning their tokens.
DROPPABLE_MARKERS = ["SELECT *", "별칭", "코드값을 직접 비교", "정수 나눗셈"]


@dataclass(slots=True)
class Candidate:
    name: str
    template: str
    operator: str
    parent: str = ""
    score: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        from aegis_sql.schema.card import token_estimate

        return token_estimate(self.template)


@dataclass(slots=True)
class OptimizationResult:
    prompt_id: str
    baseline_score: float
    best_score: float
    best_template: str
    best_operator: str
    generations: int
    evaluated: int
    history: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def delta(self) -> float:
        return self.best_score - self.baseline_score

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["delta"] = round(self.delta, 4)
        return d


class PromptOptimizer:
    """Evolutionary search over one prompt, scored by real execution accuracy."""

    def __init__(
        self,
        registry: PromptRegistry,
        prompt_id: str,
        evaluate: Callable[[PromptRegistry], tuple[float, dict[str, Any]]],
        settings: Settings | None = None,
        seed: int = 20260824,
        llm: Any = None,
    ) -> None:
        self.registry = registry
        self.prompt_id = prompt_id
        self.evaluate = evaluate
        self.settings = settings
        self.rng = random.Random(seed)
        self.llm = llm
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    # -- public ----------------------------------------------------------- #

    def optimize(
        self, generations: int = 3, population: int = 5, keep: int = 2
    ) -> OptimizationResult:
        started = time.perf_counter()
        base: PromptRecord = self.registry.get(self.prompt_id)
        base_cand = Candidate("baseline", base.template, "none")
        base_cand.score, base_cand.detail = self._score(base_cand)
        log.info("baseline scored", prompt=self.prompt_id, score=round(base_cand.score, 4))

        pool = [base_cand]
        history: list[dict[str, Any]] = [self._row(base_cand, 0)]
        evaluated = 1

        for gen in range(1, generations + 1):
            children: list[Candidate] = []
            for parent in pool[:keep]:
                for child in self._mutate(parent, population):
                    if child.template in {c.template for c in pool + children}:
                        continue
                    children.append(child)
            children = children[:population]
            for child in children:
                child.score, child.detail = self._score(child)
                evaluated += 1
                history.append(self._row(child, gen))
                log.info("variant scored", gen=gen, name=child.name,
                         operator=child.operator, score=round(child.score, 4))
            pool = sorted(pool + children, key=lambda c: (-c.score, c.tokens))[: keep * 2]

        best = pool[0]
        return OptimizationResult(
            prompt_id=self.prompt_id,
            baseline_score=base_cand.score,
            best_score=best.score,
            best_template=best.template,
            best_operator=best.operator,
            generations=generations,
            evaluated=evaluated,
            history=history,
            wall_s=time.perf_counter() - started,
        )

    # -- mutation --------------------------------------------------------- #

    def _mutate(self, parent: Candidate, n: int) -> list[Candidate]:
        ops: list[Callable[[str], tuple[str, str] | None]] = [
            self._op_add_rule,
            self._op_drop_rule,
            self._op_reorder_sections,
            self._op_tighten_output,
            self._op_shorten,
        ]
        if self.llm is not None and getattr(self.llm, "available", lambda: False)():
            ops.append(self._op_llm_rewrite)

        out: list[Candidate] = []
        for i in range(n * 2):
            op = ops[i % len(ops)]
            result = op(parent.template)
            if not result:
                continue
            template, label = result
            if template.strip() == parent.template.strip():
                continue
            out.append(
                Candidate(
                    name=f"{parent.name}+{label}"[-48:],
                    template=template,
                    operator=label,
                    parent=parent.name,
                )
            )
            if len(out) >= n:
                break
        return out

    def _op_add_rule(self, template: str) -> tuple[str, str] | None:
        unused = [(k, r) for k, r in CANDIDATE_RULES if r not in template]
        if not unused:
            return None
        key, rule = self.rng.choice(unused)
        anchor = "【질문】"
        if anchor in template:
            head, _, tail = template.partition(anchor)
            return f"{head}【추가 규칙】\n- {rule}\n\n{anchor}{tail}", f"add:{key}"
        return template + f"\n\n【추가 규칙】\n- {rule}\n", f"add:{key}"

    def _op_drop_rule(self, template: str) -> tuple[str, str] | None:
        lines = template.splitlines()
        droppable = [
            i for i, ln in enumerate(lines)
            if any(m in ln for m in DROPPABLE_MARKERS) and ln.strip().startswith(("-", "*", "1", "2", "3", "4", "5", "6", "7", "8"))
        ]
        if not droppable:
            return None
        idx = self.rng.choice(droppable)
        marker = next((m for m in DROPPABLE_MARKERS if m in lines[idx]), "rule")
        return "\n".join(lines[:idx] + lines[idx + 1 :]), f"drop:{marker[:12]}"

    def _op_reorder_sections(self, template: str) -> tuple[str, str] | None:
        """Move the glossary block ahead of the few-shot block (or back)."""
        blocks = re.split(r"(?m)^(?=【)", template)
        if len(blocks) < 3:
            return None
        i, j = 1, 2
        blocks[i], blocks[j] = blocks[j], blocks[i]
        return "".join(blocks), "reorder:sections"

    def _op_tighten_output(self, template: str) -> tuple[str, str] | None:
        clause = "\n**출력은 ```sql 코드블록 하나뿐입니다. 그 밖의 텍스트를 출력하면 실패로 처리됩니다.**\n"
        if clause.strip() in template:
            return None
        return template.rstrip() + "\n" + clause, "tighten:output"

    def _op_shorten(self, template: str) -> tuple[str, str] | None:
        """Drop the step-by-step scaffold — is the reasoning preamble paying for itself?"""
        m = re.search(r"(?ms)^【작업 절차】.*?$", template)
        if not m:
            return None
        return template[: m.start()].rstrip() + "\n", "shorten:drop-scaffold"

    def _op_llm_rewrite(self, template: str) -> tuple[str, str] | None:
        try:
            from aegis_sql.llm.base import Message

            msg = [
                Message("system", "당신은 프롬프트 엔지니어입니다. 지시문을 더 명확하고 짧게 개선합니다."),
                Message(
                    "user",
                    "아래는 Text-to-SQL 생성 프롬프트입니다. Jinja2 변수({{ }}, {% %})는 그대로 두고, "
                    "규칙의 의미를 바꾸지 않으면서 더 명확하게 다듬은 버전을 출력하세요. "
                    "프롬프트 본문만 출력하세요.\n\n----\n" + template,
                ),
            ]
            resp = self.llm.complete(msg, temperature=0.7)
            text = resp.text.strip()
            if "{{" not in text or len(text) < 120:
                return None
            return text, "llm:rewrite"
        except Exception as exc:  # pragma: no cover - provider dependent
            log.debug("llm rewrite failed", error=str(exc))
            return None

    # -- scoring ---------------------------------------------------------- #

    def _score(self, cand: Candidate) -> tuple[float, dict[str, Any]]:
        key = cand.template
        if key in self._cache:
            return self._cache[key]
        registry = self.registry.with_override(self.prompt_id, cand.template)
        score, detail = self.evaluate(registry)
        self._cache[key] = (score, detail)
        return score, detail

    def _row(self, cand: Candidate, gen: int) -> dict[str, Any]:
        return {
            "generation": gen,
            "name": cand.name,
            "operator": cand.operator,
            "parent": cand.parent,
            "score": round(cand.score, 4),
            "tokens": cand.tokens,
            "detail": cand.detail,
        }


# --------------------------------------------------------------------------- #
# convenience: score a registry against a dev split of the benchmark
# --------------------------------------------------------------------------- #


def benchmark_evaluator(
    settings: Settings, dev_limit: int = 30, difficulties: tuple[str, ...] = ("medium", "hard")
) -> Callable[[PromptRegistry], tuple[float, dict[str, Any]]]:
    """Build an ``evaluate`` callable that runs the real engine on a dev subset.

    The subset is fixed (first N of the chosen difficulties) so variants are
    comparable, and it deliberately excludes the easy tier: prompts rarely move
    easy questions, so scoring them only adds variance.
    """
    from aegis_sql.eval.harness import EvalHarness, Variant

    harness = EvalHarness(settings)
    items = harness.select(limit=dev_limit, difficulties=difficulties, include_probes=False)

    def evaluate(registry: PromptRegistry) -> tuple[float, dict[str, Any]]:
        variant = Variant(
            name=f"prompt-{registry.name}",
            description="prompt variant",
            mutate=lambda engine: setattr(engine.c, "prompt_registry", registry)
            or _rebind_registry(engine, registry),
        )
        result = harness.run_variant(variant, items=items, progress=False)
        return result.overall.execution_accuracy, {
            "n": result.overall.n,
            "ex": round(result.overall.execution_accuracy, 4),
            "em": round(result.overall.exact_match, 4),
            "executable": round(result.overall.executable_rate, 4),
        }

    return evaluate


def _rebind_registry(engine: Any, registry: PromptRegistry) -> None:
    gen = getattr(engine.c, "llm_generator", None)
    if gen is not None:
        gen.registry = registry


def save_result(result: OptimizationResult, path: str | Path = "reports/prompt_opt.json") -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p
