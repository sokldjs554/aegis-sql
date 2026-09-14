"""Service-level guardrails for expensive query execution."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient


def _with_runtime_limits(settings, *, timeout_s: float, max_concurrent: int):
    server = settings.server.model_copy(
        update={
            "query_timeout_s": timeout_s,
            "max_concurrent_queries": max_concurrent,
        }
    )
    return settings.model_copy(update={"server": server})


def test_server_config_declares_query_runtime_limits():
    from aegis_sql.config import ServerConfig

    cfg = ServerConfig()
    assert cfg.query_timeout_s > 0
    assert cfg.max_concurrent_queries >= 1


def test_query_timeout_returns_504(settings, monkeypatch):
    from aegis_sql.api import app as app_mod
    from aegis_sql.api.app import create_app

    limited = _with_runtime_limits(settings, timeout_s=0.02, max_concurrent=1)
    with TestClient(create_app(limited), raise_server_exceptions=False) as client:
        engine_type = type(app_mod._ENGINE)
        real_ask = engine_type.ask

        def slow_ask(self, *args, **kwargs):
            time.sleep(0.08)
            return real_ask(self, *args, **kwargs)

        monkeypatch.setattr(engine_type, "ask", slow_ask)
        response = client.post("/v1/query", json={"question": "전체 계약은 몇 건인가요?"})

        assert response.status_code == 504
        assert response.json()["detail"]["code"] == "QUERY_TIMEOUT"
        metrics = client.get("/metrics").text
        assert 'aegis_query_runtime_rejections_total{reason="timeout"}' in metrics
        # The worker thread is intentionally allowed to finish before app teardown.
        time.sleep(0.1)


def test_query_capacity_returns_429_while_first_request_is_still_running(settings, monkeypatch):
    from aegis_sql.api import app as app_mod
    from aegis_sql.api.app import create_app

    limited = _with_runtime_limits(settings, timeout_s=2.0, max_concurrent=1)
    started = threading.Event()
    release = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    with TestClient(create_app(limited), raise_server_exceptions=False) as client:
        engine_type = type(app_mod._ENGINE)
        real_ask = engine_type.ask

        def block_first_ask(self, *args, **kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                number = call_count
            if number == 1:
                started.set()
                release.wait(timeout=1.5)
            return real_ask(self, *args, **kwargs)

        monkeypatch.setattr(engine_type, "ask", block_first_ask)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                client.post,
                "/v1/query",
                json={"question": "전체 계약은 몇 건인가요?"},
            )
            assert started.wait(timeout=1.0)

            second = client.post(
                "/v1/query",
                json={"question": "실효된 계약은 몇 건인가요?"},
            )
            assert second.status_code == 429
            assert second.json()["detail"]["code"] == "QUERY_CAPACITY_EXCEEDED"
            assert second.headers["retry-after"] == "1"
            metrics = client.get("/metrics").text
            assert 'aegis_query_runtime_rejections_total{reason="capacity"}' in metrics

            release.set()
            assert first.result(timeout=2.0).status_code == 200


def test_stream_timeout_is_reported_as_sse_error(settings, monkeypatch):
    from aegis_sql.api import app as app_mod
    from aegis_sql.api.app import create_app

    limited = _with_runtime_limits(settings, timeout_s=0.02, max_concurrent=1)
    with TestClient(create_app(limited), raise_server_exceptions=False) as client:
        engine_type = type(app_mod._ENGINE)
        real_ask = engine_type.ask

        def slow_ask(self, *args, **kwargs):
            time.sleep(0.08)
            return real_ask(self, *args, **kwargs)

        monkeypatch.setattr(engine_type, "ask", slow_ask)
        with client.stream(
            "POST",
            "/v1/query/stream",
            json={"question": "전체 계약은 몇 건인가요?"},
        ) as response:
            payload = "".join(response.iter_text())

        assert response.status_code == 200
        assert "event: error" in payload
        assert "QUERY_TIMEOUT" in payload
        time.sleep(0.1)
