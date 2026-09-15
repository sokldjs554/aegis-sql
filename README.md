# AEGIS-SQL

**한국 금융·보험 레거시 스키마를 위한 거버넌스 내장형 Text-to-SQL 연구 시스템입니다.**

[![CI](https://github.com/sokldjs554/aegis-sql/actions/workflows/ci.yml/badge.svg)](https://github.com/sokldjs554/aegis-sql/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/live%20demo-aegis--sql.onrender.com-2563eb)](https://aegis-sql.onrender.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sokldjs554/aegis-sql/blob/main/notebooks/aegis_sql_colab.ipynb)

> 데이터는 모두 **합성 보험 데이터**입니다. 실제 고객 데이터는 사용하지 않았습니다.

## 서비스 데모

**Live**: <https://aegis-sql.onrender.com>  
무료 인스턴스라 첫 접속은 1~2분 걸릴 수 있습니다.

![AEGIS-SQL 서비스 데모](docs/images/console-demo.gif)

공개 데모의 **질의 탭은 API 키 없는 `template` 경로를 실시간 실행**합니다.  
`LLM 실험` 탭은 보관된 Claude·Qwen·R³ 실측을 API 호출 없이 보여줍니다.

## 문제와 접근

보험사 DB에는 `CTRT_STAT_CD`, `YYYYMMDD` 문자열 날짜, 공통코드 테이블, 민감정보 컬럼처럼
LLM이 그대로 이해하기 어려운 요소가 함께 존재합니다.

AEGIS-SQL은 다음 흐름으로 처리합니다.

`한국어 질문 → 의도 검사 → 스키마 링킹 → 난이도 라우팅 → SQL 생성 → AST 거버넌스 → 실행 → 제한적 repair`

- **스키마 링킹** — dense + BM25 + 보험 도메인 용어사전 + 값 프로파일 + FK 그래프
- **4-tier generation ladder** — `template → SLM → LLM → ensemble`
- **거버넌스** — SQL 생성 전 요청 의도 검사 + 생성 후 `sqlglot` AST 정책 검사
- **데이터 플라이휠** — SQL 샘플링 → 한국어 생성·증강 → 실행 검증 → 누수 없는 분할
- **평가 우선** — 같은 벤치마크에서 기준을 통과한 구성만 서비스 경로에 승격

직접 학습한 5.3M SLM은 최종 EX가 0%여서 기본 경로에서 비활성화했습니다.

## 검증된 범위

### KorFin-Bench — 합성 보험 도메인

| 항목 | 결과 |
|---|---:|
| SQL 생성 90문항 | template **44.4%** · cascade **52.2%** · hosted LLM **57.8%** |
| hard 20문항 | template **0/20** · cascade **4/20** · hosted LLM **6/20** |
| 변경 요청 | **10/10 사전 차단** |
| 정상 조회 오차단 | **0/17** |
| 모호성 | **6/6 되묻기** |
| 용어사전 제거 | EX **-10.0%p** |

### Spider-KO — 외부 일반화 1,034문항

`Qwen2.5-Coder-1.5B-Instruct` · NF4 4-bit · Tesla T4 기준입니다.

| 구성 | EX | 실행 실패 | schema-reference 실패 |
|---|---:|---:|---:|
| `mschema` | **38.01%** (393/1,034) | 320 | 304 |
| `mschema + 1회 bounded repair` | **41.97%** (434/1,034) | **224** | **209** |

- EX **+3.97%p / +41문항**
- 기존 정답 회귀 **0건**
- 실행 복구 96건 중 실제 정답 41건
- total generation p95 **4.16s → 7.58s**

이 수치는 **공식 Spider leaderboard 점수가 아니라 AEGIS `execution_match` 기준 외부 일반화 실험**입니다.
상세 조건과 provenance는 [`docs/SPIDER_KO.md`](docs/SPIDER_KO.md)에 있습니다.

## 데이터 플라이휠

```text
스키마 11테이블
→ SQL 프로그램 4,000개
→ 한국어 생성·증강
→ 실행 검증·퇴화 제거·중복 제거·난이도 균형
→ 12,416쌍
```

- train 9,914 / dev 1,192 / test 1,310
- train↔test SQL 스켈레톤 누수 **0건**
- 난이도 교차검증 일치도 **0.833**
- full-scale 생성 **113초 (CPU)**

## 기술 구성

`Python` · `PyTorch` · `TensorFlow/Keras` · `HuggingFace Transformers` · `Qwen2.5-Coder`  
`FastAPI` · `SSE` · `SQLite` · `sqlglot` · `Prometheus`  
`Docker` · `GitHub Actions` · `pytest` · `ruff` · `mypy`

## 실행

```bash
git clone https://github.com/sokldjs554/aegis-sql
cd aegis-sql
make setup
make demo
make serve
```

- `make eval` — KorFin-Bench 평가
- `make flywheel` — 데이터 플라이휠 재생성
- `make train-slm` — 자체 SLM 학습

API 키 없이도 template 경로와 거버넌스, 평가 파이프라인을 확인할 수 있습니다.

## 문서

- [Architecture](docs/ARCHITECTURE.md)
- [Evaluation](docs/EVALUATION.md)
- [Governance](docs/GOVERNANCE.md)
- [Spider-KO external evaluation](docs/SPIDER_KO.md)
- [SLM](docs/SLM.md)
- [Data flywheel](docs/FLYWHEEL.md)
- [Papers & research mapping](docs/PAPERS.md)
- [Prompt system](docs/PROMPTS.md)

## 범위와 한계

- 보험 데이터와 질문 분포는 합성 환경입니다.
- KorFin-Bench는 106문항으로 작으며 1~2%p 차이를 강한 결론으로 해석하지 않습니다.
- Spider-KO 41.97%는 공식 leaderboard 수치가 아닙니다.
- bounded repair는 실행 오류를 줄였지만 의미·구성 추론 오류까지 해결하지는 못했습니다.
- 10B+ 모델 학습·서빙과 실제 금융권 운영 트래픽은 검증 범위에 포함되지 않습니다.
- 공개 데모의 실시간 질의는 template 경로이며, hosted LLM 수치는 저장된 실험 증거입니다.

Copyright © 2026 sokldjs554. All rights reserved.  
이 저장소는 포트폴리오 목적으로 공개되었습니다.
