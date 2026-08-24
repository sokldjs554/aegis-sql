"""Versioned prompt registry.

Prompts are the single most-edited artefact in an LLM system and the one least
likely to be under review.  Here every prompt is a YAML record with an id, a
semantic version, a content hash and metadata, so that an evaluation report can
say *which prompt* produced a number, an A/B test can pin two versions
side-by-side, and `aegis prompt diff` can show what changed.

See ``docs/PROMPT_ENGINEERING.md`` for the methodology and the measured deltas.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import ChainableUndefined, Template

from aegis_sql.config import CONFIG_DIR
from aegis_sql.observability.logging import get_logger

log = get_logger("prompts.registry")


@dataclass(slots=True)
class PromptRecord:
    id: str
    version: str
    template: str
    role: str = "user"          # "system" | "user"
    description: str = ""
    variables: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    #: Free-form notes: what changed, which eval run motivated it.
    changelog: str = ""

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()[:12]

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}#{self.hash}"

    def render(self, **variables: Any) -> str:
        missing = [v for v in self.variables if v not in variables]
        if missing:
            raise KeyError(f"prompt '{self.id}' missing variables: {missing}")
        # Declared variables are enforced above; optional blocks ({% if few_shots %})
        # are allowed to be absent, which is what keeps call sites terse.
        return Template(
            self.template, undefined=ChainableUndefined, trim_blocks=True, lstrip_blocks=True
        ).render(**variables)


class PromptRegistry:
    """Loads ``configs/prompts/<set>.yaml`` and serves prompts by id."""

    def __init__(self, records: dict[str, PromptRecord], name: str = "default") -> None:
        self._records = records
        self.name = name

    # -- loading ----------------------------------------------------------- #

    @classmethod
    def load(cls, prompt_set: str = "default", directory: str | Path | None = None) -> PromptRegistry:
        base = Path(directory) if directory else CONFIG_DIR / "prompts"
        path = base / f"{prompt_set}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"prompt set not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        records: dict[str, PromptRecord] = {}
        for pid, spec in (payload.get("prompts") or {}).items():
            records[pid] = PromptRecord(
                id=pid,
                version=str(spec.get("version", "1.0.0")),
                template=spec["template"],
                role=spec.get("role", "user"),
                description=spec.get("description", ""),
                variables=list(spec.get("variables", [])),
                tags=list(spec.get("tags", [])),
                changelog=spec.get("changelog", ""),
            )
        log.debug("prompt set loaded", set=prompt_set, prompts=len(records))
        return cls(records, name=prompt_set)

    @classmethod
    def from_records(cls, records: list[PromptRecord], name: str = "adhoc") -> PromptRegistry:
        return cls({r.id: r for r in records}, name=name)

    # -- access ------------------------------------------------------------ #

    def get(self, prompt_id: str) -> PromptRecord:
        if prompt_id not in self._records:
            raise KeyError(f"unknown prompt '{prompt_id}' in set '{self.name}' (have: {sorted(self._records)})")
        return self._records[prompt_id]

    def render(self, prompt_id: str, **variables: Any) -> str:
        return self.get(prompt_id).render(**variables)

    def with_override(self, prompt_id: str, template: str, version_suffix: str = "-exp") -> PromptRegistry:
        """Return a copy with one prompt swapped — the unit of A/B testing."""
        base = self.get(prompt_id)
        clone = dict(self._records)
        clone[prompt_id] = PromptRecord(
            id=base.id,
            version=f"{base.version}{version_suffix}",
            template=template,
            role=base.role,
            description=base.description,
            variables=base.variables,
            tags=[*base.tags, "experiment"],
            changelog="runtime override",
        )
        return PromptRegistry(clone, name=f"{self.name}{version_suffix}")

    def __contains__(self, prompt_id: str) -> bool:
        return prompt_id in self._records

    def __iter__(self):
        return iter(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def manifest(self) -> dict[str, str]:
        """``{prompt_id: ref}`` — embedded in every evaluation report."""
        return {r.id: r.ref for r in self._records.values()}
