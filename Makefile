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
PORT       ?= 8000
Q          ?= 작년 하반기에 체결된 계약 중 월납보험료가 20만원 이상인 건수를 지점별로 알려줘

.DEFAULT_GOAL := help
.PHONY: help setup venv install install-all demo-db benchmark profile demo ask serve \
        flywheel train-slm train-slm-quick routing-data train-router eval eval-quick \
        test test-fast lint fmt typecheck \
        check docker-build docker-up docker-down clean distclean tree

help: ## 사용 가능한 타깃 목록
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------- 환경
venv:
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	@$(PIP) install -q --upgrade pip setuptools wheel

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
demo: ## 대표 질의 5개를 엔진에 태워 SQL + 결과 + 트레이스 출력
	@$(PYTHON) -m aegis_sql.cli demo

ask: ## 임의 질문 실행:  make ask Q="실효된 계약의 채널별 비중은?"
	@$(PYTHON) -m aegis_sql.cli ask "$(Q)" --explain

serve: ## FastAPI 서버 기동 (웹 콘솔 포함)
	@$(PYTHON) -m aegis_sql.cli serve --port $(PORT)

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
test: ## 전체 테스트 (학습 테스트는 프로세스 분리)
	@$(BIN)/pytest -q -m "not slow"
	@$(BIN)/pytest -q -m slow tests/test_cli.py tests/test_flywheel.py
	@$(BIN)/pytest -q -m slow tests/test_router.py
	@$(BIN)/pytest -q -m slow tests/test_training.py

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
