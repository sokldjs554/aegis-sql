# 실제 논문 독해 기록

이 파일은 원문에서 방법·실험·한계까지 확인한 논문만 기록하는 완료 목록이다. 읽을 예정인 자료는 `READING-LIST-KO-EN.md`에만 두며, 각 완료 기록은 아래 네 줄을 넘기지 않는다.

## K1 · [윤장현, 「Text-to-SQL Evidence 자동생성을 통한 DB 접근성 향상」](https://s-space.snu.ac.kr/bitstream/10371/234012/1/000000193888.pdf) (원문 검토: 2026-09-12)

- 해결하는 문제: 실제 비전문가는 BIRD처럼 수동 evidence를 제공하기 어렵고, BIRD dev의 evidence도 9.65%가 누락되고 6.84%가 오류를 포함했으며, evidence 제거 시 비교 모델의 EX가 8.35~20.86%p 하락하는 연구-현장 간극을 다룬다.
- 핵심 방법: 질문·DB 스키마·설명 문서·값 샘플에서 키워드를 찾고 샘플 SQL을 생성·실행한 뒤, 유사 질문 few-shot과 실행 결과를 조합해 evidence를 생성하며 컨텍스트가 짧은 모델에는 스키마 요약을 선행한다.
- AEGIS와 같은 점/다른 점: 둘 다 사용자의 수동 힌트 없이 스키마·설명·값에서 도메인 근거를 복원하지만, SEED는 LLM이 자유형 evidence를 생성하고 일부 경로에서 전체 스키마를 보는 반면 AEGIS는 glossary·BM25/dense·값/코드 프로파일·FK 경로를 결합해 제한된 스키마와 출처가 남는 typed evidence를 결정론적으로 선택한다.
- 적용·변형·기각 판단: **변형 적용** — 전체 스키마와 자유형 evidence의 온라인 생성을 그대로 도입하지 않고, 스키마 설명·코드 테이블·범주형 값에서 glossary 후보를 오프라인 자동 추출한 뒤 고정 KorFin 평가와 어블레이션을 통과한 항목만 채택한다; 논문에서도 evidence 형식에 따라 모델별 효과가 달라졌기 때문이다.

## K2 · [신건우, 「PromptBase 기반 Text-to-SQL의 동적 교정 기법」](https://s-space.snu.ac.kr/bitstream/10371/220733/1/000000188472.pdf) (원문 검토: 2026-09-12)

- 해결하는 문제: LLM이 만든 SQL에는 DBMS가 드러내는 구문·실행 오류뿐 아니라 정상 실행돼도 사용자 의도와 어긋나는 암묵적 오류가 남는데, 모든 쿼리에 같은 체크리스트를 주는 정적 교정은 쿼리·DB별 오류 차이를 반영하지 못한다.
- 핵심 방법: 최초 SQL의 실행 결과를 0행·1행 이상·DBMS 오류로 나눠 각각 self-feedback과 유사 성공 이력, SQL 키워드별 교정 지침, 오류 메시지별 팁을 검색해 반복 교정하고, 테이블·컬럼 관계·FK 위반·NULL 여부를 조회하는 DB-as-a-Tool을 보탠다; BIRD dev에서 refinement는 EX 64.54%→67.76%, SR-DB 업데이트는 교정 성공률 20.3%→30.0%를 기록했다.
- AEGIS와 같은 점/다른 점: 둘 다 실행 결과와 스키마·값·FK 정보를 이용해 SQL을 다시 검증하지만, PB-SQL은 세 실행 분기에 LLM 교정과 PromptBase 이력 검색·갱신을 결합하는 반면 AEGIS는 DBMS 실패 또는 세 가지 알려진 silent defect만 교정 대상으로 삼고 8종 deterministic AST rewrite를 먼저 실행한 뒤 LLM을 마지막 fallback으로 둔다.
- 적용·변형·기각 판단: **변형 적용** — repair log에 `empty/nonempty/error` 분기와 오류 유형을 남기고 검증된 성공 이력만 오프라인 검색 후보로 쓰되, 정상 실행 SQL을 전부 LLM으로 재작성하거나 0행 탈출만으로 성공 처리하는 방식은 기각한다; 고정 KorFin에서 `no-repair` 대비 EX·교정 trigger/성공/회귀율·비용·p50/p95를 함께 측정한 뒤 채택한다.

## E1 · [Lee and Kim, *SafeQL: Search-based Refinement for Safe and Efficient LLM-based Text-to-SQL*](https://arxiv.org/pdf/2608.09260) (원문 검토: 2026-09-12)

- 해결하는 문제: 실행 실패 때마다 SQL 전체를 LLM으로 재생성하면 이미 맞는 구조까지 버리고 같은 오류를 다시 만들 수 있으며 token·latency가 누적되는데, 기존 방식은 DBMS를 오류 문자열만 주는 수동적 검사기로 제한한다.
- 핵심 방법: PostgreSQL의 parser·binder·type analyzer가 실패한 AST component를 찾고 relation·join·attribute·value·function 단위 후보를 만든 뒤, AST tree-edit distance와 embedding 거리를 섞은 best-first search에 type/top-K pruning·cache·HNSW index를 적용하고 100회 탐색 실패 때만 LLM 재생성으로 전환한다; GPT-OSS-120B의 BIRD full dev에서 prompt baseline 57.5%를 search-only 62.5%(추가 token 0), hybrid 63.3%로 높였고 agent baseline 64.2%는 hybrid 69.4%가 됐다.
- AEGIS와 같은 점/다른 점: 둘 다 실행 피드백으로 맞는 조각을 보존한 작은 AST 수정을 먼저 하고 LLM을 마지막 fallback으로 두지만, SafeQL은 PostgreSQL 내부에서 오류 위치별 복수 후보를 만들고 거리순으로 탐색하는 반면 AEGIS는 SQLite 바깥의 sqlglot AST에서 8종 규칙을 고정 순서로 한 번씩 적용해 첫 실행 성공안을 택하며 component label·후보 순위·safe-query-space 탐색은 없다.
- 적용·변형·기각 판단: **변형 적용** — AEGIS repair log에 오류 component와 탐색 budget을 추가하고 복수 deterministic 후보를 AST 변화량·도메인 evidence·guard 통과 여부로 순위화하되 PostgreSQL extension 이식과 모든 0행의 자동 교정은 기각한다; 실행 가능성은 의미 정답 보장이 아니므로 고정 KorFin fault set에서 현행 first-match 대비 repair 성공률·회귀율·EX·LLM 호출·p50/p95를 함께 통과한 경우에만 채택한다.
