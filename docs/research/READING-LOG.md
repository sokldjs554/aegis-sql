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

## E2 · [Lee et al., *EXPO-SQL: Execution-based Clause-level Policy Optimization for Text-to-SQL*](https://aclanthology.org/2026.findings-acl.1107/) (원문 검토: 2026-09-12)

- 해결하는 문제: 기존 execution-based RL은 SQL 전체의 성공·실패를 모든 token과 clause에 같은 query-level reward로 주므로, WHERE 하나만 틀린 부분 정답에서도 이미 맞은 SELECT·JOIN까지 함께 벌점받는 coarse credit-assignment 문제를 다룬다.
- 핵심 방법: 실행을 정답·실행 가능 오답·실행 오류로 나누고, 오답에는 10종 결과 차이와 `FROM/JOIN→WHERE→GROUP BY→HAVING→SELECT→ORDER BY→LIMIT` 순 incremental execution을, 오류에는 메시지 parsing과 역방향 clause tracing을 적용해 오류 절을 찾은 뒤 clause별 token reward로 REINFORCE++를 학습한다; Qwen2.5-Coder-7B·SynSQL-Complex-5K 조건의 BIRD dev에서 query-level reward 68.5% 대비 71.3%(+2.8%p)를 기록했다.
- AEGIS와 같은 점/다른 점: 둘 다 실행 피드백으로 모델 실패를 학습 신호화하고 맞는 부분을 보존하려 하지만, 현행 AEGIS DPO는 repair log의 gold 전체를 `chosen`, failed SQL 전체를 `rejected`로 두고 target-span sequence log-probability를 비교하는 반면 EXPO-SQL은 결과·오류 원인을 clause에 귀속해 같은 SQL 안의 token마다 다른 보상을 주며 학습 중 DB를 반복 실행한다.
- 적용·변형·기각 판단: **변형 적용** — 8×H100 REINFORCE++를 재현하지 않고 gold/failed SQL의 AST 차이와 기존 실행 로그로 clause label·단계적 preference를 오프라인 생성해 현행 query-level DPO와 같은 데이터·seed·budget에서 EX·difficulty·invalid SQL·clause 오류·학습 시간/메모리를 비교한다; SQLite만 검증됐고 실행 오류 보상이 본문 식(−1.5)과 부록 사례(−1.0)에서 불일치하므로 수치를 그대로 복제하지 않는다.

## E3 · [Piao et al., *LitE-SQL: A Lightweight and Efficient Text-to-SQL Framework with Vector-based Schema Linking and Execution-Guided Self-Correction*](https://aclanthology.org/2026.findings-eacl.186/) (원문 검토: 2026-09-12)

- 해결하는 문제: 전체 스키마와 대형 proprietary LLM·다중 후보 생성에 의존하면 context·연산·지연·privacy 비용이 커지고, 작은 모델은 불필요한 column이 섞일 때 성능이 더 크게 떨어지는 배포 간극을 다룬다.
- 핵심 방법: Qwen3-Embedding-0.6B로 column 문서를 미리 색인하고 positive와 유사한 hard negative만 남기는 HN-SupCon으로 top-k schema를 검색한 뒤, Qwen2.5-Coder 1.5B/3B/7B를 LoRA SFT와 DPO+NLL RFT로 학습해 실행 실패 SQL·오류 메시지를 반복 교정한다; 다만 대표 7B BIRD 72.10%는 정답 schema 조건이고 retriever·SFT·RFT를 함께 쓴 top-25 결과는 60.56%(SFT 58.21%→RFT 60.56%)다.
- AEGIS와 같은 점/다른 점: 둘 다 column 단위 vector schema linking, pretrained Qwen LoRA 계열 적응, 실행 검증·교정을 결합하지만, LitE-SQL은 learned dense retriever와 반복 LLM 교정·DPO+NLL을 쓰는 반면 AEGIS는 dense·BM25·glossary·값·FK 경로의 출처가 남는 hybrid retrieval과 deterministic AST repair 우선 정책을 쓰고 이번 Qwen 비교군에는 RFT를 적용하지 않았다.
- 적용·변형·기각 판단: **변형 적용·실측 완료** — 이 논문을 근거로 같은 AEGIS snapshot·retrieval·KorFin 90문항에서 Qwen2.5-Coder-1.5B base와 QLoRA를 Tesla T4로 비교해 EX 11.1%→12.2%(+1.11%p), easy 33.3%→30.0%, medium 0%→5%, hard 0% 유지와 p50 3.78s→5.39s를 확인했다; 서로 다른 benchmark의 절대 점수 비교와 ‘QLoRA가 크게 개선했다’는 해석은 기각하고 HN-SupCon/RFT는 동일 조건 어블레이션 전까지 후속 후보로 둔다.
