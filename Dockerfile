# ---------------------------------------------------------------------
# AEGIS-SQL 런타임 이미지
# 코어 + LangChain 만 설치한다. PyTorch/TensorFlow는 학습 전용이고,
# 서빙 경로는 numpy 추론만 사용하므로 이미지에 넣지 않는다 (~4GB 절감).
# ---------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[llm]"

COPY configs ./configs
COPY scripts ./scripts
COPY data/demo/schema.sql data/demo/glossary.yaml ./data/demo/
COPY data/benchmark ./data/benchmark

# 데모 DB는 결정론적으로 생성되므로 이미지에 굽는다 (리뷰어가 바로 실행 가능).
RUN python scripts/build_demo_db.py --scale 0.5 \
 && python -c "import sys; sys.path.insert(0,'src'); \
from aegis_sql.schema.introspect import introspect; \
from aegis_sql.schema.profile import Profiler; \
g=introspect('data/demo/aegis_demo.sqlite'); \
Profiler('data/demo/aegis_demo.sqlite').profile(g, cache_path='data/generated/profile.json')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/v1/health || exit 1

CMD ["python", "-m", "aegis_sql.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
