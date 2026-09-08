# SEED — Automatic Evidence Generation for Practical Text-to-SQL

- 읽은 날짜: 2026-09-08
- 논문: Janghyeon Yun, Sang Goo Lee. *ICDE Workshops 2025*
- 서울대학교 공식 연구 페이지: https://snu.elsevierpure.com/en/publications/seed-enhancing-text-to-sql-performance-and-practical-usability-th/
- DOI: https://doi.org/10.1109/ICDEW67478.2025.00005
- 상태: **[비교 · AEGIS의 evidence/retrieval 설계를 재검토하는 근거]**

## 1. 논문이 푸는 문제

BIRD는 질문과 함께 사람이 만든 evidence를 제공한다. 그러나 실제 사용자는 DB schema와 도메인 지식을 이미 알고 evidence까지 작성해 줄 수 없고, BIRD의 human-written evidence 자체에도 누락·오류가 있을 수 있다.

SEED는 이 전제를 문제 삼고, **질문에 필요한 domain evidence를 시스템이 자동으로 생성해야 한다**고 본다.

## 2. 핵심 방법

SEED(System for Evidence Extraction and Domain knowledge generation)는 다음 정보를 체계적으로 분석해 evidence를 만든다.

- database schema
- schema/column description files
- 실제 database values

저자들은 BIRD와 Spider에서 no-evidence 조건을 평가했고, 자동 생성 evidence가 SQL generation accuracy를 유의미하게 개선하며 일부 조건에서는 BIRD의 수동 evidence를 제공한 설정보다도 높은 결과를 보였다고 보고한다.

## 3. AEGIS-SQL과 겹치는 지점

AEGIS의 schema linking은 이미 다음 정보를 결합한다.

- dense retrieval
- 직접 구현한 BM25
- 보험 도메인 glossary
- column sample/value profile
- FK graph / join path

즉 AEGIS도 사용자가 evidence를 직접 제공하지 않는다는 현실적 전제를 공유한다. `retrieval/schema_linker.py`가 만든 `LinkedSchema.evidence`는 generation 앞단에서 필요한 schema/domain 정보를 복원한다.

## 4. 차이

SEED의 중심 산출물은 **자동 생성된 evidence**다. AEGIS는 evidence 생성 자체보다 그것을 schema pruning, routing, prompt context, SQL generation, governance로 연결하는 전체 파이프라인에 초점을 둔다.

또 AEGIS의 glossary는 합성 보험 환경을 위해 직접 만든 도메인 지식이므로, 실제 고객사에서 수동 glossary를 얼마나 구축해야 하는지 아직 검증되지 않았다. SEED는 이 부분을 더 자동화할 수 있는 연구 방향을 제시한다.

## 5. AEGIS에서 확인된 관련 실험

현재 어블레이션에서 glossary 제거는 template EX를 **44.4% → 34.4%(-10.0%p)**로 떨어뜨렸다. dense-only는 41.1%였다. 이 결과는 "모델 크기보다 glossary가 더 중요하다"는 일반 명제를 증명하지는 않는다. 다만 **이 합성 보험 schema에서 domain evidence 주입이 큰 영향을 미친다**는 것은 지지한다.

따라서 문서 표현은 다음처럼 제한한다.

> 모델 성능만으로 해결되지 않았고, schema·현업 용어·값에서 domain knowledge를 복원해 생성 단계에 제공하는 것이 이 환경에서 큰 영향을 보였다.

## 6. 후속 실험 가설

실제 보험사 schema가 주어졌다고 가정했을 때:

1. 수동 glossary 없이 schema comment + code table + sampled values만으로 evidence 생성
2. 현재 glossary-on 기준과 비교
3. EX뿐 아니라 schema recall, linked column 수, prompt token 수 측정

핵심 질문은 **"41개 glossary를 사람이 만들어야 하는가, 아니면 schema/value에서 상당 부분 복원할 수 있는가"**다.

## 7. 면접에서 30초 설명

> SEED는 BIRD처럼 사용자가 evidence를 제공해 준다는 가정이 현실적이지 않다고 보고 schema, description, DB values에서 evidence를 자동 생성합니다. AEGIS도 사용자가 domain hint를 주지 않는 전제에서 glossary·값 프로파일·FK graph를 retrieval에 넣었는데, 다음 단계는 실제 고객 schema에서 수동 glossary 의존도를 얼마나 줄일 수 있는지 SEED식 자동 evidence 생성과 비교하는 것입니다.
