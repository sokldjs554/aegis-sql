# Research Notes — 논문 조사 → 설계 판단 → 실험 연결

이 디렉터리는 "논문을 읽었다"는 목록이 아니라 **논문의 주장과 AEGIS-SQL의 설계·코드·실험을 대조한 읽기 기록**이다.

각 노트는 다음을 구분한다.

1. 논문이 실제로 주장한 것
2. AEGIS-SQL과 겹치는 문제
3. 그대로 적용하지 않은 이유 또는 변형 지점
4. 현재 저장소에서 이미 측정한 근거
5. 아직 측정하지 않은 후속 가설

측정하지 않은 것은 결과처럼 쓰지 않는다.

## 먼저 볼 파일

- [국문/영문 실제 독서 목록](READING-LIST-KO-EN.md) — 면접 전 읽을 논문만 최신성·AEGIS 연관성 순으로 정리
- [기존 면접형 읽기 가이드](READING-GUIDE.md) — 논문별 질문과 30초 답변
- [Paper-driven experiments](EXPERIMENTS.md) — 어떤 가설을 코드/측정으로 연결했는지와 현재 상태

## 핵심 리뷰

| 논문 | 분야 | AEGIS와의 관계 | 상태 |
|---|---|---|---|
| [SafeQL](SAFEQL.md) — VLDB 2026, KAIST | DBMS-guided partial repair | deterministic repair와 최신 search-based refinement 대조 | gap 분석 완료 |
| [EXPO-SQL](EXPO-SQL.md) — Findings ACL 2026 | clause-level execution reward | query-level DPO의 다음 학습 신호 후보 | 후속실험 후보 |
| [R³-SQL](R3-SQL.md) — Findings ACL 2026 | 후보 ranking · resampling | execution-result grouping과 selective resampling 대조 | policy 구현 완료 / 측정 대기 |
| [LitE-SQL](LitE-SQL.md) — Findings EACL 2026 | lightweight model · vector schema linking | AEGIS 5.3M sLLM 실패와 가장 직접적인 비교군 | Qwen 경로 구현 완료 / GPU 측정 대기 |
| [SEED](SEED.md) — ICDEW 2025, 서울대 | automatic evidence generation | glossary/value/FK evidence와 대조 | 비교 · 자동화 후보 |
| [EHRSQL](EHRSQL.md) — KAIST 중심 | 실무 benchmark · unanswerable | KorFin-Bench에 없던 answerability 축 발견 | 30-probe 측정 완료 |
| [RAT-SQL](RAT-SQL.md) — ACL 2020 | schema encoding/linking | neural relation encoder 대신 명시적 FK/evidence layer | 변형 |
| [DIN-SQL](DIN-SQL.md) — NeurIPS 2023 | decomposition · self-correction | LLM subtask를 deterministic component로 분해 | 변형 |

## 현재 확인된 연구 질문

### Q1. 작은 모델 실패의 원인은 "크기"인가 "사전학습 부재"인가?

AEGIS의 5.3M from-scratch 모델은 DPO reward margin이 개선됐지만 KorFin-Bench EX는 0.0%였다. LitE-SQL을 대조해 **Qwen2.5-Coder 1.5B pretrained adaptation 비교 경로**를 만들었다. 같은 flywheel split, schema representation, retrieval, guard, execution-match를 고정했고 실제 GPU full run만 남아 있다.

### Q2. 보험 domain evidence를 사람이 얼마나 만들어야 하는가?

현재 glossary 제거 시 template EX가 44.4% → 34.4%로 떨어진다. SEED는 schema/description/value에서 evidence를 자동 생성한다. 따라서 **수동 glossary 의존도를 automatic evidence generation으로 얼마나 줄일 수 있는지**가 다음 질문이다.

### Q3. 질문이 명확해도 DB가 답을 가지고 있지 않다면?

EHRSQL을 검토한 뒤 15 unanswerable + 15 answerable hard-negative probe를 고정했고, 실제 schema/glossary capability detector로 **15/15 recall, false abstention 0/15**를 측정했다. `CapabilityAwareEngine`으로 executable abstention path도 만들었다. 단, curated 30문항만으로 default core behavior를 바꾸지는 않았다.

### Q4. 후보가 없을 때 ranking을 잘하는 것으로 충분한가?

R³-SQL을 대조해 route confidence + execution-result group dispersion을 사용하는 **side-effect-free selective-resampling policy**와 offline evaluator를 구현했다. 현재 과거 eval report에는 candidate-level group/agreement 및 실제 resampled outcome이 없어 개선 수치는 주장하지 않는다.

### Q5. SQL을 틀렸을 때 전체를 다시 만들 필요가 있는가?

SafeQL은 DBMS parser/binder/type analyzer feedback으로 오류 component만 search-based로 고친다. AEGIS도 deterministic repair를 먼저 사용하지만 PostgreSQL 내부 safe-query-space search를 구현한 것은 아니다. 이 차이를 명시적으로 기록했다.

### Q6. DPO가 실패 SQL 전체에만 벌점을 주는 것이 충분한가?

EXPO-SQL은 query-level reward가 맞는 clause와 틀린 clause를 구분하지 못한다는 문제를 다룬다. AEGIS의 query-level preference를 clause-aware preference로 바꾸는 것은 별도 후속 가설로 남긴다.

## 증거 수준

- **읽기 증거**: source/venue/핵심 주장/한계 기록
- **판단 증거**: 적용·변형·기각 또는 후속실험 후보를 이유와 함께 기록
- **구현 증거**: 관련 모듈 경로를 노트에서 직접 연결
- **실험 증거**: 이미 측정한 값만 명시하고, 미실행 가설은 별도로 표시
- **Git 증거**: 리뷰 노트와 후속 구현을 별도 커밋/PR로 남겨 변경 순서를 추적

구체적인 실험 설계와 남은 측정은 [EXPERIMENTS.md](EXPERIMENTS.md)에 기록한다. 실제 측정값이 생기기 전에는 README/포트폴리오의 성능 주장에 반영하지 않는다.
