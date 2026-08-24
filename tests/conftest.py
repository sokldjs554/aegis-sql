"""Shared fixtures.

The demo database is built once per session (at a small scale) and every fixture
below is derived from it, so the whole suite runs against the *real* schema —
cryptic physical names, YYYYMMDD dates, common-code table and all — rather than a
toy stand-in that would not exercise the parts that matter.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DB_PATH = ROOT / "data" / "demo" / "aegis_demo.sqlite"
REFERENCE_DATE = date(2026, 8, 24)


@pytest.fixture(scope="session", autouse=True)
def demo_db() -> Path:
    if not DB_PATH.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_demo_db.py"), "--scale", "0.25"],
            check=True, cwd=ROOT,
        )
    return DB_PATH


@pytest.fixture(scope="session")
def settings(demo_db):
    from aegis_sql.config import Settings

    return Settings.load(
        database={"path": str(demo_db)},
        generation={"provider": "template"},
        log_level="ERROR",
    )


@pytest.fixture(scope="session")
def schema(demo_db):
    from aegis_sql.schema.introspect import introspect

    return introspect(demo_db)


@pytest.fixture(scope="session")
def profile(demo_db, schema):
    from aegis_sql.schema.profile import Profiler

    return Profiler(demo_db).profile(schema)


@pytest.fixture(scope="session")
def join_graph(schema):
    from aegis_sql.schema.graph import JoinGraph

    return JoinGraph(schema)


@pytest.fixture(scope="session")
def glossary():
    from aegis_sql.retrieval.glossary import Glossary

    return Glossary.load(ROOT / "data" / "demo" / "glossary.yaml")


@pytest.fixture(scope="session")
def normalizer():
    from aegis_sql.nlu.korean import KoreanNormalizer

    return KoreanNormalizer(today=REFERENCE_DATE)


@pytest.fixture(scope="session")
def executor(demo_db):
    from aegis_sql.verify.executor import SQLExecutor

    ex = SQLExecutor(demo_db, timeout_s=10.0, max_rows=500)
    yield ex
    ex.close()


@pytest.fixture(scope="session")
def linker(schema, profile, join_graph, glossary, settings):
    from aegis_sql.retrieval.embedder import get_embedder
    from aegis_sql.retrieval.schema_linker import SchemaLinker

    lk = SchemaLinker(
        schema=schema, profile=profile, join_graph=join_graph,
        glossary=glossary, embedder=get_embedder(settings), settings=settings,
    )
    lk.build_index()
    return lk


@pytest.fixture(scope="session")
def guard(schema, settings):
    from aegis_sql.verify.ast_guard import PolicyDocument, PolicyGuard

    return PolicyGuard(schema, PolicyDocument.load(ROOT / "configs" / "policy" / "insurance.yaml"), settings)


@pytest.fixture(scope="session")
def engine(settings):
    from aegis_sql.pipeline import AegisEngine

    eng = AegisEngine.build(settings)
    yield eng
    eng.close()


@pytest.fixture(scope="session")
def benchmark():
    from aegis_sql.eval.harness import load_benchmark

    path = ROOT / "data" / "benchmark" / "korfin_bench.jsonl"
    if not path.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_benchmark.py")], check=True, cwd=ROOT)
    return load_benchmark(path)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: heavy tests (model training, full benchmark)")
