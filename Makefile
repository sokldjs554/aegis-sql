# =====================================================================
# AEGIS-SQL — 개발/실행 진입점
#   make setup   : 가상환경 + 의존성 + 데모 DB + 벤치마크까지 한 번에
#   make demo    : 엔진이 실제로 도는 것을 30초 안에 확인
#   make eval    : 재현 가능한 평가 리포트 생성
# =====================================================================
SHELL      := /bin/bash
PY         ?= python3
VENV       ?= .venv
BIN        := $(VENV)/bin
PYTHON     := $(BIN)/python
PIP        := $(BIN)/pip
export PYTHONPATH := src

DB         ?= data/demo/aegis_demo.sqlite
BENCH      ?= data/benchmark/korfin_bench.jsonl
SCALE      ?= 1.0
HOST       ?= 127.0.0.1
PORT       ?= 8000

# 데모 DB 생성기가 RRNO_ENC 에 내장 hash() 를 쓴다.  str 해시는 프로세스마다
# 무작위라 고정하지 않으면 그 한 컬럼만 실행마다 달라진다 — 눈에 보이는 수치는
# 아니지만 '같은 시드면 같은 산출물'이 이 프로젝트의 약속이라 여기서 못박는다.
# (Dockerfile 은 이미 같은 값을 박고 있다.)
export PYTHONHASHSEED := 0
Q          ?= 작년 하반기에 체결된 계약 중 월납보험료가 20만원 이상인 건수를 지점별로 알려줘

.DEFAULT_GOAL := help
.PHONY: help setup venv install install-all demo-db benchmark profile demo ask serve \
        preflight flywheel train-slm train-slm-quick routing-data train-router eval eval-quick \
        test test-fast lint fmt typecheck \
        check docker-build docker-up docker-down clean distclean tree

help: ## 사용 가능한 타깃 목록
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = "## "}; {sub(/:.*/, "", $$1); printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------- 환경
# 쓸 수 있는 인터프리터를 스스로 찾는다.  PY= 로 직접 주면 그것만 쓴다.
#
# macOS 의 /usr/bin/python3 는 3.9 라 그대로는 못 쓴다.  그런데 사용자가
# Homebrew 로 새 파이썬을 깔아도 `brew link` 는 기존 파일과 충돌하면 실패한다 —
# 그때도 바이너리 자체는 Cellar 에 멀쩡히 있고 PATH 에만 없다.  PATH 를 보고
# 없다고 포기하면, 이미 해결된 문제를 안 풀린 것처럼 보이게 만든다.
# 그래서 PATH 다음으로 Homebrew·python.org 의 실제 설치 경로까지 뒤진다.
#
# 우선순위는 CI 가 도는 버전(3.13~3.10)이 먼저다.  3.14 는 아직 아무도
# 이 프로젝트를 돌려 본 적이 없어 마지막에 둔다.
venv:
	@set -e; \
	ok() { "$$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' >/dev/null 2>&1; }; \
	PYBIN="$(PY)"; \
	if ! ok "$$PYBIN"; then \
	  found=""; \
	  for p in python3.13 python3.12 python3.11 python3.10 python3.14; do \
	    c=$$(command -v "$$p" 2>/dev/null) || continue; \
	    if ok "$$c"; then found="$$c"; break; fi; \
	  done; \
	  if [ -z "$$found" ]; then \
	    for g in /opt/homebrew/opt/python@3.1[0-9]/bin/python3.1[0-9] \
	             /usr/local/opt/python@3.1[0-9]/bin/python3.1[0-9] \
	             /Library/Frameworks/Python.framework/Versions/3.1[0-9]/bin/python3.1[0-9]; do \
	      if [ -x "$$g" ] && ok "$$g"; then found="$$g"; break; fi; \
	    done; \
	  fi; \
	  if [ -n "$$found" ]; then \
	    echo "· $(PY) 는 $$("$(PY)" -V 2>&1) 이라 $$found 를 씁니다"; \
	    PYBIN="$$found"; \
	  else \
	    echo "✗ Python 3.10+ 를 찾지 못했습니다 ($(PY) 는 $$("$(PY)" -V 2>&1))."; \
	    echo "  macOS 기본 /usr/bin/python3 는 3.9 입니다.  설치:"; \
	    echo "      brew install python@3.12"; \
	    echo "  'brew link' 단계가 실패해도 괜찮습니다 — 바이너리는 깔려 있고 PATH 에만 없습니다."; \
	    echo "  다시 이 명령을 돌리면 알아서 찾고, 안 되면 경로를 직접 주세요:"; \
	    echo "      make $(or $(MAKECMDGOALS),setup) PY=\"\$$(brew --prefix python@3.12)/bin/python3.12\""; \
	    exit 1; \
	  fi; \
	fi; \
	if [ -x $(BIN)/python ] && ! ok $(BIN)/python; then \
	  echo "· 기존 $(VENV) 가 Python 3.10 미만이라 다시 만듭니다"; rm -rf $(VENV); \
	fi; \
	test -d $(VENV) || "$$PYBIN" -m venv $(VENV); \
	$(PIP) install -q --upgrade pip setuptools wheel

install: venv ## 코어 의존성만 설치 (LLM/학습 없이도 전부 동작)
	@$(PIP) install -q -e ".[llm,dev]"

install-all: venv ## PyTorch / TensorFlow / VectorDB 포함 전체 설치
	@$(PIP) install -q -e ".[all]"

setup: install demo-db benchmark profile ## 처음 한 번: 설치 + 데모DB + 벤치마크 + 프로파일
	@echo "✅ setup 완료 — 'make demo' 로 확인해 보세요."

# --------------------------------------------------------------------- 데이터
demo-db: ## 한국 보험 레거시 스키마 데모 DB 생성 (결정론적, 37만 행)
	@$(PYTHON) scripts/build_demo_db.py --scale $(SCALE)

demo-db-force: ## 데모 DB 재생성
	@$(PYTHON) scripts/build_demo_db.py --scale $(SCALE) --force

benchmark: ## KorFin-Bench 생성 + gold SQL 실행 검증 (106문항)
	@$(PYTHON) scripts/build_benchmark.py

profile: ## 컬럼 프로파일 캐시 생성 (값 링킹용)
	@$(PYTHON) -m aegis_sql.cli profile

# --------------------------------------------------------------------- 실행
preflight: ## 이 기계에서 데모가 도는지 사전 점검 (면접 직전에 한 번)
	@$(PYTHON) scripts/preflight.py

demo: ## 대표 질의 5개를 엔진에 태워 SQL + 결과 + 트레이스 출력
	@$(PYTHON) -m aegis_sql.cli demo

ask: ## 임의 질문 실행:  make ask Q="실효된 계약의 채널별 비중은?"
	@$(PYTHON) -m aegis_sql.cli ask "$(Q)" --explain

serve: ## FastAPI 서버 기동 (웹 콘솔 포함).  외부 노출은 HOST=0.0.0.0
	@$(PYTHON) -m aegis_sql.cli serve --host $(HOST) --port $(PORT)

# --------------------------------------------------------------------- 학습 파이프라인
flywheel: ## 스키마 → SQL 샘플링 → 역번역 → 증강 → 실행검증 → 학습셋
	@$(PYTHON) -m aegis_sql.cli flywheel --n-programs 4000

train-slm: ## 자체 구현 PyTorch Transformer 학습 (BPE → SFT → DPO; --lora 로 어댑터 학습)
	@$(PYTHON) scripts/train_slm.py --data-dir data/generated/flywheel --out data/generated/slm \
	  --epochs 3 --limit 9000 --d-model 256 --n-layers 4 --n-heads 8 --d-ff 1024 \
	  --max-seq-len 288 --vocab-size 8000 --batch-size 32 --dpo

train-slm-quick: ## 빠른 학습 스모크 (CPU 2분)
	@$(PYTHON) scripts/train_slm.py --data-dir data/generated/flywheel --out data/generated/slm \
	  --epochs 1 --limit 300 --d-model 96 --n-layers 2 --n-heads 4 --d-ff 256 \
	  --max-seq-len 192 --vocab-size 2000 --batch-size 16

routing-data: ## 평가 로그에서 라우터 학습용 (features, label) 수집
	@$(PYTHON) -m aegis_sql.cli eval --bench data/generated/flywheel/test.jsonl \
	  --routing-log data/generated/router/routing_train.jsonl --report reports/flywheel_eval.md

train-router: routing-data ## TensorFlow 라우터 학습 → numpy 가중치 export (관측 라벨 사용)
	@$(PYTHON) scripts/train_router.py --out data/generated/router \
	  --data data/generated/router/routing_train.jsonl

# --------------------------------------------------------------------- 평가
eval: ## 전체 벤치마크 평가 + 어블레이션 리포트 (reports/)
	@$(PYTHON) -m aegis_sql.cli eval --bench $(BENCH) --ablation --report reports/eval.md

eval-quick: ## 빠른 평가 (easy+medium 20문항)
	@$(PYTHON) -m aegis_sql.cli eval --bench $(BENCH) --limit 20

# --------------------------------------------------------------------- 품질
# TensorFlow(라우터 학습)와 PyTorch(sLLM 학습)를 한 프로세스에 함께 적재하면
# 네이티브 런타임이 종료 단계에서 충돌한다.  두 학습 테스트를 각각 별도
# 프로세스로 돌려 격리한다 — CI 도 같은 이유로 잡을 나눠 실행한다.
# torch/TensorFlow 가 없는 환경에서는 해당 학습 테스트가 전부 skip 되고,
# pytest 는 '수집된 테스트 없음'을 종료코드 5 로 알린다.  실패가 아니므로
# 5 만 성공으로 접고 나머지 코드는 그대로 전파한다.
slow = $(BIN)/pytest -q -m slow $(1); c=$$?; test $$c -eq 0 -o $$c -eq 5

test: ## 전체 테스트 (학습 테스트는 프로세스 분리)
	@$(BIN)/pytest -q -m "not slow"
	@$(call slow,tests/test_cli.py tests/test_flywheel.py)
	@$(call slow,tests/test_router.py)
	@$(call slow,tests/test_training.py)

test-fast: ## 무거운 학습 테스트 제외
	@$(BIN)/pytest -q -m "not slow"

lint: ## ruff 검사
	@$(BIN)/ruff check src tests scripts

fmt: ## ruff 포맷 적용
	@$(BIN)/ruff format src tests scripts && $(BIN)/ruff check --fix src tests scripts

typecheck: ## mypy
	@$(BIN)/mypy src/aegis_sql --ignore-missing-imports

check: lint test ## CI가 도는 것과 동일한 검사

# --------------------------------------------------------------------- 배포
docker-build: ## 컨테이너 이미지 빌드
	@docker build -t aegis-sql:latest .

docker-up: ## docker compose 기동
	@docker compose up --build

docker-down:
	@docker compose down -v

# --------------------------------------------------------------------- 정리
clean: ## 캐시 정리
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov

distclean: clean ## 생성물 전부 삭제 (DB/학습 산출물 포함)
	@rm -rf data/generated $(DB) reports/*.md reports/*.json

tree: ## 소스 트리 요약
	@find src scripts tests -name "*.py" | sort | xargs wc -l | tail -1
	@find src -name "*.py" | sort
