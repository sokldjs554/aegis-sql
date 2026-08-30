# ---------------------------------------------------------------------
# AEGIS-SQL 런타임 이미지
# 코어 + LangChain 만 설치한다. PyTorch/TensorFlow는 학습 전용이고,
# 서빙 경로는 numpy 추론만 사용하므로 이미지에 넣지 않는다 (~4GB 절감).
# ---------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    # build_demo_db.py 가 RRNO_ENC 에 파이썬 내장 hash() 를 쓴다. 문자열 해시는
    # 프로세스마다 랜덤화되므로, 고정하지 않으면 '결정론적' 이라는 주장이 거짓이 된다.
    PYTHONHASHSEED=0 \
    # 공개 배포에서 질의 하나가 붙잡을 수 있는 최대 시간을 8초에서 3초로 줄인다.
    AEGIS_DATABASE__TIMEOUT_S=3.0

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[llm]"

COPY configs ./configs
COPY scripts ./scripts
# 학습된 라우터 가중치(36KB, numpy). 빠지면 router_loaded=False 로 떠서 콘솔이
# '라우터 미적재 — 규칙 기반으로 동작' 을 노출한다. 캐스케이드 라우팅이 이
# 프로젝트의 핵심 주장인데 그게 꺼진 채로 보이게 된다.
COPY models ./models
COPY data/demo/schema.sql data/demo/glossary.yaml ./data/demo/
COPY data/benchmark ./data/benchmark

# 데모 DB는 결정론적으로 생성되므로 이미지에 굽는다 (리뷰어가 바로 실행 가능).
RUN python scripts/build_demo_db.py --scale 0.5 \
 && python -c "import sys; sys.path.insert(0,'src'); \
from aegis_sql.schema.introspect import introspect; \
from aegis_sql.schema.profile import Profiler; \
g=introspect('data/demo/aegis_demo.sqlite'); \
Profiler('data/demo/aegis_demo.sqlite').profile(g, cache_path='data/generated/profile.json')"

# 빌드 단계에서 프로파일 캐시가 실제로 만들어졌는지 확인한다 — 없으면 런타임에
# 재계산을 시도하고, 쓰기 불가한 환경에서는 기동 자체가 실패한다.
RUN test -s data/generated/profile.json

# 많은 호스트(Hugging Face Spaces 등)가 컨테이너를 uid 1000 으로 돌린다.
# /app 전체를 재귀 chown 하면 레이어가 통째로 복제되므로, 런타임에 쓰기가
# 필요한 단 하나의 디렉터리만 넘긴다.
RUN useradd -m -u 1000 user \
 && chown -R user:user /app/data/generated
USER user
ENV HOME=/home/user

# 포트는 환경변수로 받는다 — Cloud Run 같은 호스트는 PORT 를 주입하고(기본 8080)
# 컨테이너가 그 포트를 듣지 않으면 기동 실패로 처리한다. 로컬·CI 는 8000 그대로.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/v1/health" || exit 1

CMD ["sh", "-c", "exec python -m aegis_sql.cli serve --host 0.0.0.0 --port ${PORT}"]
