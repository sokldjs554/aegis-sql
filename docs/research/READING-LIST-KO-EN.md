# AEGIS-SQL 논문 독서 목록 — 국문 / 영문

이 목록은 **노리스페이스 LLM Research Engineer 지원에서 실제로 설명할 수 있을 정도로 읽을 논문**만 추린다. 숫자를 늘리는 것이 목적이 아니라, 논문에서 제기한 문제를 AEGIS-SQL의 설계·실험과 연결하는 것이 목적이다.

## 0. 읽는 방법

한 편당 1차 독해는 20~30분으로 제한한다.

1. Abstract
2. Introduction
3. Architecture / Method figure
4. 핵심 실험표와 ablation
5. Limitations / Conclusion

읽고 나서는 반드시 아래 네 줄을 직접 적는다.

- 이 논문이 푸는 문제:
- 핵심 방법:
- AEGIS와 같은 점 / 다른 점:
- 내가 적용·변형·기각할 부분:

---

# A. 국문 — 먼저 읽기

## K1. 윤장현, 「Text-to-SQL Evidence 자동생성을 통한 DB 접근성 향상」 — 서울대학교, 2025

- 구분: 서울대학교 공학전문대학원 공학전문석사 학위 연구보고서
- 원문 PDF: https://s-space.snu.ac.kr/bitstream/10371/234012/1/000000193888.pdf
- 영어 후속/연결 연구: SEED — https://arxiv.org/abs/2506.07423
- AEGIS 연결: glossary, schema description, DB value profile, domain evidence

**왜 읽는가**

BIRD처럼 사용자가 evidence를 미리 제공한다는 가정이 실제 환경에서 왜 약한지, schema·description·value에서 evidence를 자동으로 만드는 문제를 한국어로 먼저 이해할 수 있다. AEGIS의 `glossary + value profile + schema linking`을 설명할 때 가장 직접적인 국내 자료다.

**읽고 답할 질문**

1. evidence가 없는 실제 사용 환경에서 Text-to-SQL이 왜 어려워지는가?
2. schema 이름만 제공하는 것과 DB value/description까지 제공하는 것은 무엇이 다른가?
3. AEGIS의 glossary는 SEED의 evidence generation과 어디까지 같고 어디부터 다른가?

---

## K2. 신건우, 「PromptBase 기반 Text-to-SQL의 동적 교정 기법」 — 서울대학교, 2025

- 구분: 서울대학교 데이터사이언스대학원 석사학위논문
- 영어 제목: *Dynamic Refinement Techniques in PromptBase Based Text-to-SQL*
- 원문 PDF: https://s-space.snu.ac.kr/bitstream/10371/220733/1/000000188472.pdf
- AEGIS 연결: execution feedback, silent failure, deterministic repair, iterative refinement

**왜 읽는가**

SQL이 실행됐다는 사실과 SQL이 의미적으로 맞다는 사실을 구분하고, 실행 결과와 오류 유형에 따라 교정하는 문제를 다룬다. AEGIS의 `static check → execute → repair`와 직접 비교할 수 있다.

**읽고 답할 질문**

1. 실행 성공만으로 정답을 판정할 수 없는 사례는 무엇인가?
2. 동적 교정이 SQL 전체 재생성과 다른 점은 무엇인가?
3. AEGIS가 일부 오류를 LLM이 아니라 deterministic repair로 처리하는 이유는 무엇인가?

---

## K3. 조예진·김무철·이남연, 「효율적인 프롬프트 엔지니어링과 사용자 피드백 기반 인터랙티브 Text-to-SQL 시스템 설계」 — 2026

- 소속: 연세대학교 컴퓨터과학과 / 중앙대학교 / 한신대학교
- 학술지: 한국경영공학회지, 2026, 31(1), 37-47
- KCI: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003319623
- 핵심: Llama 3.2 3B, structured CoT prompting, 사용자 feedback 기반 instruction tuning
- AEGIS 연결: lightweight model, prompt engineering, feedback flywheel

**왜 읽는가**

2026년 국내 국문 Text-to-SQL 연구라서 최신 용어와 실험 흐름을 한국어로 익히기 좋다. 특히 AEGIS의 Qwen lightweight-model 실험과 사용자 피드백 기반 데이터 개선을 비교하기 쉽다.

---

## K4. 남명국·김남규, 「운영 SQL을 활용한 Text-to-SQL 신뢰성 향상 방안」 — 국민대학교, 2026

- 학술지: 한국컴퓨터정보학회논문지, 2026, 31(4), 147-156
- DOI: 10.9708/jksci.2026.31.04.147
- KCI: https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART003329500
- 핵심: 실제 운영 환경에 축적된 SQL을 외부 지식 자산으로 활용하는 OpSQL-Leverage(OSL)
- AEGIS 연결: few-shot retrieval, SQL asset reuse, data flywheel, RAG for Text-to-SQL

**왜 읽는가**

합성 데이터만 보지 않고 실제 운영 SQL을 어떻게 재사용할지 생각하게 해 준다. AEGIS를 실제 고객사 환경으로 옮길 때 `KorFin-Bench / flywheel → 고객 운영 SQL 및 실제 질문`으로 확장하는 근거가 된다.

---

# B. 영문 — 반드시 읽을 최신 핵심

## E1. SafeQL — KAIST, VLDB 2026

- 제목: *SafeQL: Search-based Refinement for Safe and Efficient LLM-based Text-to-SQL*
- 저자: Geonho Lee, Min-Soo Kim
- arXiv: https://arxiv.org/abs/2608.09260
- KAIST 소개: https://kaist.ac.kr/newsen/html/news/?mng_no=66630&mode=V
- AEGIS 연결: execution error localization, partial repair, cost/latency-aware refinement

**핵심 포인트**

SQL 전체를 다시 생성하지 않고 DBMS feedback으로 오류 component를 찾아 안전한 query space 안에서 부분 수정한다. AEGIS의 repair가 현재 어디까지 SafeQL과 같고, 어디서 단순 규칙 기반인지 비교한다.

---

## E2. EXPO-SQL — 성균관대학교 연구진, ACL Findings 2026

- 제목: *EXPO-SQL: Execution-based Clause-level Policy Optimization for Text-to-SQL*
- 원문: https://aclanthology.org/2026.findings-acl.1107/
- AEGIS 연결: DPO / execution feedback / clause-level error signal

**핵심 포인트**

기존 query-level reward가 SELECT/FROM/WHERE의 맞고 틀림을 구분하지 못한다는 문제를 제기하고 clause-level reward를 사용한다. AEGIS의 현재 `gold SQL = chosen / 실패 SQL = rejected` 방식과 직접 비교한다.

---

## E3. LitE-SQL — 연세대학교 연구진, EACL Findings 2026

- 제목: *LitE-SQL: A Lightweight and Efficient Text-to-SQL Framework with Vector-based Schema Linking and Execution-Guided Self-Correction*
- 원문: https://aclanthology.org/2026.findings-eacl.186/
- AEGIS 연결: VectorDB schema linking, lightweight pretrained model, SFT, execution-guided learning

**핵심 포인트**

현재 AEGIS의 `5.3M from-scratch EX 0%`를 해석하는 데 가장 중요한 비교 논문이다. 이 논문 때문에 Qwen2.5-Coder 1.5B pretrained adaptation 비교군을 별도로 만들었다.

---

## E4. R³-SQL — ETRI + 서울대학교 + Snowflake AI Research, ACL Findings 2026

- 제목: *R³-SQL: Ranking Reward and Resampling for Text-to-SQL*
- 원문: https://aclanthology.org/2026.findings-acl.2146/
- AEGIS 연결: execution-result grouping, candidate ranking, selective resampling

**핵심 포인트**

후보 SQL들이 기능적으로 같은 결과를 내는지 execution result 기준으로 묶고, 정답 후보가 pool에 없을 가능성이 있을 때만 resampling한다. AEGIS의 cascade 손실과 ensemble candidate disagreement를 분석하는 후속 실험과 연결된다.

---

## E5. SEED — 서울대학교, ICDE Workshop 2025

- 제목: *SEED: Enhancing Text-to-SQL Performance and Practical Usability Through Automatic Evidence Generation*
- 원문: https://arxiv.org/abs/2506.07423
- 코드: https://github.com/felix01189/SEED
- AEGIS 연결: domain evidence, schema/value extraction, practical deployment

국문 K1을 먼저 읽은 뒤 영어 논문을 읽으면 된다. 같은 연구축을 한국어 → 국제 논문 순서로 확인하는 용도다.

---

## E6. EHRSQL — KAIST 중심 연구, NeurIPS benchmark

- 제목: *EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records*
- arXiv: https://arxiv.org/abs/2301.07695
- AEGIS 연결: real-user questions, answerable/unanswerable, reliability

AEGIS의 30개 clear-but-unanswerable probe를 만든 직접적인 문제의식이다. 최신성보다는 **실제 업무 질문과 abstention을 benchmark에 포함했다는 점** 때문에 반드시 이해한다.

---

# C. 최종 읽기 순서

시간이 부족하면 이 순서대로 읽는다.

1. **K1 서울대 Evidence 학위 연구보고서** — 한국어로 schema/evidence 개념 잡기
2. **K2 서울대 동적 교정 논문** — 한국어로 execution/refinement 개념 잡기
3. **K3 2026 국문 인터랙티브 Text-to-SQL** — 최신 lightweight/prompt/feedback 흐름
4. **E1 SafeQL** — 최신 KAIST repair 연구
5. **E2 EXPO-SQL** — 최신 성균관대 execution/RL 연구
6. **E3 LitE-SQL** — 연세대 lightweight model 연구
7. **E4 R³-SQL** — 서울대 candidate ranking/resampling 연구
8. **K4 운영 SQL 활용 논문** — 실제 고객 데이터 확장 관점
9. **E5 SEED** — K1의 국제 논문 버전
10. **E6 EHRSQL** — answerability benchmark 원전

면접 직전 최소 완독 목표는 **K1, K2, E1, E2, E3, E4 총 6편**이다.

## 증거 규칙

포트폴리오에 `논문 조사`라고 쓰려면 최소한 다음 중 하나를 GitHub에 남긴다.

- 논문 리뷰 노트 + AEGIS 설계 비교
- 논문에서 나온 가설을 검증하는 코드/테스트
- 적용하지 않은 경우 기각 이유
- 실제 실험 결과와 실패 해석

`읽을 예정`인 논문과 `실제로 리뷰를 끝낸` 논문은 숫자를 섞지 않는다.
