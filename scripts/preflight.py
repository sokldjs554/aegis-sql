#!/usr/bin/env python3
"""AEGIS-SQL 데모 사전 점검 — 이 기계에서 면접 데모가 도는지 확인한다.

저장소 루트에서:  make preflight   (또는 .venv/bin/python scripts/preflight.py)

실패하면 무엇이 왜 안 되는지와 다음에 칠 명령을 알려준다.  통과하면
그 자리에서 서버를 띄워 첫 질의까지 태워 본 것이므로, 면접장에서
처음 겪는 일이 없다.
"""
from __future__ import annotations

import os
import platform
import shutil
import socket
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OK, WARN, BAD = "\033[32m✓\033[0m", "\033[33m!\033[0m", "\033[31m✗\033[0m"
failures: list[str] = []
warnings_: list[str] = []


def say(mark: str, label: str, detail: str = "") -> None:
    print(f"  {mark} {label}" + (f"  — {detail}" if detail else ""))


def fail(label: str, detail: str, fix: str) -> None:
    say(BAD, label, detail)
    failures.append(f"{label}: {fix}")


def warn(label: str, detail: str, note: str) -> None:
    say(WARN, label, detail)
    warnings_.append(f"{label}: {note}")


print("\n\033[1mAEGIS-SQL 데모 사전 점검\033[0m")

# ---------------------------------------------------------------- 1. 런타임
print("\n[1/6] 런타임")
v = sys.version_info
say(OK if v >= (3, 10) else BAD, "Python", f"{v.major}.{v.minor}.{v.micro} ({platform.machine()})")
if v < (3, 10):
    fail("Python", f"{v.major}.{v.minor}", "3.10 이상이 필요합니다 — brew install python@3.12")

say(OK, "OS", f"{platform.system()} {platform.release()}")

# SQLite: 윈도우 함수(3.25+)·JSON1 을 쓰므로 하한을 확인한다.
sq = sqlite3.sqlite_version_info
say(OK if sq >= (3, 25, 0) else BAD, "SQLite", sqlite3.sqlite_version)
if sq < (3, 25, 0):
    fail("SQLite", sqlite3.sqlite_version, "윈도우 함수에 3.25+ 가 필요합니다")
try:
    sqlite3.connect(":memory:").execute("SELECT ROW_NUMBER() OVER (ORDER BY 1)").fetchone()
    say(OK, "윈도우 함수", "ROW_NUMBER 동작")
except sqlite3.OperationalError as exc:
    fail("윈도우 함수", str(exc), "이 Python 의 sqlite3 가 너무 낡았습니다")

# ---------------------------------------------------------------- 2. 패키지
print("\n[2/6] 패키지")
for mod, label in [
    ("aegis_sql", "aegis_sql"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("sqlglot", "sqlglot"),
    ("numpy", "numpy"),
]:
    try:
        m = __import__(mod)
        say(OK, label, getattr(m, "__version__", "설치됨"))
    except ImportError as exc:
        fail(label, str(exc), 'make install  (또는 pip install -e ".[llm,dev]")')

try:
    import sqlglot
    from sqlglot import expressions as exp

    if not hasattr(exp, "Attach"):
        fail("sqlglot", sqlglot.__version__, "27.4 이상이 필요합니다 — pip install -U 'sqlglot>=27.4'")
    else:
        say(OK, "sqlglot ATTACH 가드", "exp.Attach 존재")
except ImportError:
    pass

# ---------------------------------------------------------------- 3. 산출물
print("\n[3/6] 데이터 산출물")
artifacts = [
    (ROOT / "data/demo/aegis_demo.sqlite", "데모 DB", "make demo-db"),
    (ROOT / "data/benchmark/korfin_bench.jsonl", "벤치마크 106문항", "make benchmark"),
    (ROOT / "data/generated/profile.json", "컬럼 프로파일", "make profile"),
    (ROOT / "models/router/router_weights.npz", "라우터 가중치", "저장소에 커밋되어 있어야 합니다"),
    (ROOT / "models/router/calibrator.json", "온도 보정", "저장소에 커밋되어 있어야 합니다"),
]
for path, label, fix in artifacts:
    if path.exists() and path.stat().st_size > 0:
        say(OK, label, f"{path.stat().st_size / 1e6:.1f} MB" if path.stat().st_size > 1e6 else "있음")
    else:
        fail(label, "없음", fix)

# ---------------------------------------------------------------- 4. 포트
print("\n[4/6] 포트")
PORT = int(os.environ.get("PORT", "8000"))
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", PORT))
        say(OK, f"{PORT} 포트", "사용 가능")
    except OSError:
        warn(f"{PORT} 포트", "이미 사용 중", "make serve PORT=8010 으로 다른 포트를 쓰세요")

# ---------------------------------------------------------------- 5. 엔진
print("\n[5/6] 엔진 실기동")
if failures:
    say(WARN, "건너뜀", "위 실패를 먼저 해결하세요")
else:
    os.environ.setdefault("AEGIS_LOG_LEVEL", "ERROR")
    import time

    from fastapi.testclient import TestClient

    from aegis_sql.api.app import create_app

    t0 = time.perf_counter()
    with TestClient(create_app()) as c:
        boot = time.perf_counter() - t0
        h = c.get("/v1/health").json()
        say(OK, "기동", f"{boot:.1f}초")

        fp = h.get("schema_fingerprint")
        expected = "26cee9e1989d6426"
        if fp == expected:
            say(OK, "스키마 지문", f"{fp} — README 와 일치")
        else:
            fail("스키마 지문", f"{fp} (기대 {expected})", "make demo-db-force && make profile")

        if h.get("router_loaded"):
            say(OK, "라우터", "적재됨")
        else:
            fail("라우터", "미적재", "models/router/ 가 클론에 있는지 확인하세요")

        # 3막: 정상 조회 / PII 차단 / 되묻기
        acts = [
            ("전체 계약은 몇 건인가요?", "ok", "정상 조회"),
            ("고객 이름이랑 주민등록번호 좀 뽑아줘", "blocked", "PII 차단"),
            ("설계사 실적 좀 보여줘", "clarify", "되묻기"),
        ]
        for q, want, label in acts:
            r = c.post("/v1/query", json={"question": q, "max_rows": 5})
            got = r.json().get("status") if r.status_code == 200 else f"HTTP {r.status_code}"
            if got == want:
                say(OK, label, f"status={got}")
            else:
                fail(label, f"status={got} (기대 {want})", "엔진 상태를 확인하세요")

# ---------------------------------------------------------------- 6. 브라우저
print("\n[6/6] 콘솔")
console = ROOT / "src/aegis_sql/api/console.html"
if console.exists():
    html = console.read_text(encoding="utf-8")
    say(OK, "콘솔 파일", f"{len(html) / 1024:.0f} KB")
    # 외부 리소스가 있으면 오프라인 면접장에서 깨진다.
    import re

    ext = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)', html)
    if ext:
        warn("외부 리소스", f"{len(ext)}개", "오프라인에서 깨질 수 있습니다: " + ", ".join(ext[:3]))
    else:
        say(OK, "외부 의존", "없음 — 오프라인에서도 완전히 렌더됩니다")
else:
    fail("콘솔 파일", "없음", "클론이 온전한지 확인하세요")

if shutil.which("open"):
    say(OK, "브라우저 열기", f"open http://127.0.0.1:{PORT}/")

# ---------------------------------------------------------------- 결과
print()
if failures:
    print(f"\033[31m{len(failures)}건 실패 — 데모 전에 해결하세요\033[0m")
    for f in failures:
        print(f"  · {f}")
    sys.exit(1)

if warnings_:
    print(f"\033[33m{len(warnings_)}건 주의\033[0m")
    for w in warnings_:
        print(f"  · {w}")

print("\033[32m준비 완료.\033[0m  데모 시작:")
print(f"  make serve HOST=127.0.0.1 PORT={PORT}   그리고  open http://127.0.0.1:{PORT}/")
print()
