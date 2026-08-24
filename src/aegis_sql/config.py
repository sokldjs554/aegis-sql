"""Typed configuration.

Layering (last wins): built-in defaults → ``configs/default.yaml`` →
``AEGIS_*`` environment variables → explicit constructor overrides.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"


class DatabaseConfig(BaseModel):
    dialect: str = "sqlite"
    path: str = str(DATA_DIR / "demo" / "aegis_demo.sqlite")
    #: Statement timeout for the sandboxed executor.
    timeout_s: float = 8.0
    #: Hard cap on rows returned to the caller.
    max_rows: int = 500
    #: Rows sampled per column when profiling for value linking.
    profile_sample: int = 200


class RetrievalConfig(BaseModel):
    #: "auto" picks sentence-transformers when installed, else the hashing embedder.
    embedder: str = "auto"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 256  # used by the dependency-free hashing embedder
    #: "auto" → chroma, else faiss, else the built-in numpy store.
    vector_store: str = "auto"
    persist_dir: str = str(DATA_DIR / "generated" / "vectorstore")
    top_k_tables: int = 6
    top_k_columns: int = 32
    #: Weight of dense similarity in the hybrid score (lexical gets 1 - alpha).
    dense_weight: float = 0.5
    glossary_weight: float = 0.35
    value_match_weight: float = 0.25
    #: Always pull in FK-reachable tables within this hop count.  0 keeps only
    #: the join bridges the path search actually needs, which the ablation showed
    #: costs nothing in accuracy and saves prompt tokens.
    fk_expand_hops: int = 0
    few_shot_k: int = 6
    #: Minimum recall guard: never prune below this fraction of tables.
    min_tables: int = 2


class GenerationConfig(BaseModel):
    provider: str = "auto"  # auto | anthropic | openai | slm | template | mock
    model: str = "claude-sonnet-5"
    temperature: float = 0.0
    max_tokens: int = 1024
    #: >1 enables self-consistency sampling on the LLM/ensemble tier.
    n_samples: int = 1
    ensemble_samples: int = 5
    ensemble_temperature: float = 0.7
    slm_checkpoint: str = str(DATA_DIR / "generated" / "slm" / "aegis-lm-tiny")
    prompt_set: str = "default"
    #: How the linked schema is written into the prompt (see schema/card.py).
    schema_card_style: str = "mschema"
    request_timeout_s: float = 60.0


class RouterConfig(BaseModel):
    enabled: bool = True
    model_dir: str = str(DATA_DIR / "generated" / "router")
    #: P(hard) above which we escalate from SLM to LLM.
    escalate_threshold: float = 0.55
    #: Calibrated confidence below which we sample an ensemble.
    ensemble_threshold: float = 0.35
    #: Cost ceiling per query in USD; the router downgrades tiers to respect it.
    budget_usd: float = 0.05
    #: Fallback difficulty threshold used when no trained router is present.
    heuristic_threshold: float = 0.5


class VerifyConfig(BaseModel):
    max_repair_attempts: int = 3
    self_consistency: bool = True
    #: Inject a LIMIT when the statement has none.
    force_limit: bool = True
    default_limit: int = 200
    #: Reject queries whose estimated cartesian blow-up exceeds this.
    max_join_tables: int = 8
    dry_run_first: bool = True


class PolicyConfig(BaseModel):
    enabled: bool = True
    path: str = str(CONFIG_DIR / "policy" / "insurance.yaml")
    #: Deny by default when a column has no explicit classification.
    default_sensitivity: str = "public"
    mask_strategy: str = "partial"  # partial | hash | null


class FlywheelConfig(BaseModel):
    output_dir: str = str(DATA_DIR / "generated" / "flywheel")
    n_programs: int = 4000
    augment_per_example: int = 3
    difficulty_mix: dict[str, float] = Field(
        default_factory=lambda: {"easy": 0.3, "medium": 0.45, "hard": 0.25}
    )
    seed: int = 20260824
    #: Drop generated pairs whose SQL returns an empty result set.
    require_nonempty_result: bool = True
    dedupe_threshold: float = 0.92


class TrainingConfig(BaseModel):
    output_dir: str = str(DATA_DIR / "generated" / "slm")
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    vocab_size: int = 8000
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 3e-4
    epochs: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj", "o_proj"])
    dpo_beta: float = 0.1
    seed: int = 20260824
    device: str = "auto"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    enable_docs: bool = True
    request_log: bool = True


class Settings(BaseModel):
    """Root settings object; obtain it via :func:`get_settings`."""

    env: str = "dev"
    log_level: str = "INFO"
    log_json: bool = False
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    flywheel: FlywheelConfig = Field(default_factory=FlywheelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # -- loading ---------------------------------------------------------- #

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> Settings:
        payload: dict[str, Any] = {}
        cfg_path = Path(path) if path else CONFIG_DIR / "default.yaml"
        if cfg_path.exists():
            payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        payload = _deep_merge(payload, _env_overrides())
        payload = _deep_merge(payload, overrides)
        return cls.model_validate(payload)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(raw: str) -> Any:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _env_overrides() -> dict[str, Any]:
    """``AEGIS_ROUTER__ESCALATE_THRESHOLD=0.7`` → ``{"router": {"escalate_threshold": 0.7}}``."""
    out: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith("AEGIS_"):
            continue
        path = key[len("AEGIS_") :].lower().split("__")
        cursor = out
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(value)
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
