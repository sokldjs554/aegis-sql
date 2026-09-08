# SafeQL — Search-based Refinement for Safe and Efficient LLM-based Text-to-SQL

- Authors: Geonho Lee, Min-Soo Kim
- Affiliation: KAIST
- Venue: VLDB 2026
- arXiv: https://arxiv.org/abs/2608.09260
- KAIST research note: https://kaist.ac.kr/newsen/html/news/?mng_no=66630&mode=V
- DOI: 10.14778/3819518.3819545

## 1. 논문이 푸는 문제

LLM 기반 Text-to-SQL이 schema에 없는 table/column/function/value를 사용해 실행 오류를 만들었을 때, 흔한 대응은 오류 메시지를 다시 LLM에 주고 **SQL 전체를 재생성**하는 것이다.

SafeQL은 이 방식이 이미 맞게 생성된 SQL 구조까지 다시 흔들고, 추가 LLM token과 latency를 소모한다는 문제를 본다.

## 2. 핵심 방법

SafeQL은 DBMS를 단순 실행기가 아니라 **refinement guide**로 사용한다.

1. DBMS parser/binder/type analyzer가 오류 component를 특정한다.
2. 전체 SQL을 다시 생성하지 않고 해당 component만 교체할 후보를 만든다.
3. 실행 가능한 후보들로 `safe query space`를 구성한다.
4. 원래 SQL과 가까운 수정부터 search/pruning한다.
5. 정해진 threshold 안에서 해결되지 않을 때만 LLM 재호출을 허용한다.

KAIST 공개 설명에 따르면 BIRD/Spider에서 초기 execution error의 최대 87.4%를 해결했고, unrefined baseline 대비 execution accuracy를 최대 5.8%p 높였으며, 전체 재생성 대비 token 사용을 최대 15.1배, refinement latency를 최대 29.6배 줄였다고 보고한다.

## 3. AEGIS와 같은 점

AEGIS도 `generate → static/AST check → execute → repair` 구조를 사용하고, 일부 오류는 LLM을 다시 부르기 전에 deterministic하게 고친다.

특히 다음 철학은 같다.

- 실행 오류를 단순 실패로 끝내지 않는다.
- 맞는 SQL 전체를 버리기보다 가능한 한 작은 수정으로 복구한다.
- 비용이 큰 LLM 재호출은 마지막 수단으로 둔다.

## 4. AEGIS와 다른 점

SafeQL은 PostgreSQL extension으로 DBMS parser/binder/type analyzer 내부와 통합돼 **오류 위치를 더 구조적으로 찾고 search space를 구성**한다.

현재 AEGIS의 `SelfRepairer`는 날짜 형식, code literal, join/FK 등 프로젝트에서 관찰한 failure pattern과 AST/static rule에 더 강하게 의존한다. 따라서 AEGIS가 SafeQL을 구현했다고 말하면 안 된다.

정확한 분류는 **문제의식 일치 + 부분 수정 철학 유사 / 구현은 별개**다.

## 5. AEGIS 후속 가설

### H1. 오류 localization을 clause/component 단위로 기록

현재 repair log의 `strategy/error/before_sql/after_sql` 외에 오류가 발생한 `SELECT/FROM/WHERE/JOIN/value/function` component를 구조화하면, repair 성공률을 유형별로 측정할 수 있다.

### H2. deterministic repair 후보를 search space로 비교

하나의 rule이 바로 SQL을 결정하는 대신 안전하게 실행 가능한 복수 후보가 있을 때 다음 순서로 rank한다.

1. 원 SQL과 AST edit distance가 작은 후보
2. policy guard를 통과하는 후보
3. execution error가 사라지는 후보
4. result shape가 질문 의도와 맞는 후보

### H3. LLM fallback gate

local repair 후보가 없거나 제한된 탐색 budget 안에서 해결되지 않을 때만 LLM repair를 부른다.

## 6. 면접용 30초 답변

> SafeQL은 DBMS 오류를 이용해 틀린 SQL 전체를 다시 생성하지 않고 오류 component만 search-based로 고칩니다. AEGIS도 비용 때문에 deterministic repair를 먼저 두지만, SafeQL처럼 PostgreSQL parser/binder 내부에서 safe query space를 탐색하는 수준은 아닙니다. 그래서 현재 repair log를 component 단위로 구조화하고, 여러 안전 후보를 비교한 뒤 LLM을 마지막 fallback으로 쓰는 방향을 후속 실험으로 보고 있습니다.

## 7. 과장 금지

- SafeQL을 AEGIS에 구현했다고 말하지 않는다.
- KAIST 공개 수치는 SafeQL의 BIRD/Spider 결과이지 AEGIS 결과가 아니다.
- 현재 AEGIS에서 주장 가능한 것은 기존 deterministic repair와의 설계 비교 및 후속 가설까지다.
