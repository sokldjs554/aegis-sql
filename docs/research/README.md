# Research Notes — 논문 조사 → 설계 판단 → 실험 연결

이 디렉터리는 "논문을 읽었다"는 목록이 아니라 **논문의 주장과 AEGIS-SQL의 설계·코드·실험을 대조한 읽기 기록**이다.

각 노트는 다음을 구분한다.

1. 논문이 실제로 주장한 것
2. AEGIS-SQL과 겹치는 문제
3. 그대로 적용하지 않은 이유 또는 변형 지점
4. 현재 저장소에서 이미 측정한 근거
5. 아직 측정하지 않은 후속 가설

측정하지 않은 것은 결과처럼 쓰지 않는다.

## 핵심 리뷰 6편

| 논문 | 분야 | AEGIS와의 관계 | 상태 |
|---|---|---|---|
| [R³-SQL](R3-SQL.md) — Findings ACL 2026 | 후보 ranking · resampling | 실행 결과 기반 self-consistency와 직접 대조 | 후속실험 후보 |
| [LitE-SQL](LitE-SQL.md) — Findings EACL 2026 | lightweight model · vector schema linking | AEGIS sLLM 실패와 가장 직접적인 비교군 | 후속실험 핵심 |
| [SEED](SEED.md) — ICDEW 2025, 서울대 | automatic evidence generation | glossary/value/FK evidence와 대조 | 비교 · 자동화 후보 |
| [EHRSQL](EHRSQL.md) — NeurIPS 2022, KAIST 중심 | 실무 benchmark · unanswerable | KorFin-Bench에 없는 abstention 축 발견 | 평가 확장 후보 |
| [RAT-SQL](RAT-SQL.md) — ACL 2020 | schema encoding/linking | neural relation encoder 대신 명시적 FK/evidence layer | 변형 |
| [DIN-SQL](DIN-SQL.md) — NeurIPS 2023 | decomposition · self-correction | LLM subtask를 deterministic component로 분해 | 변형 |

## 현재 확인된 연구 질문

### Q1. 작은 모델 실패의 원인은 "크기"인가 "사전학습 부재"인가?

AEGIS의 5.3M from-scratch 모델은 DPO reward margin이 개선됐지만 KorFin-Bench EX는 0.0%였다. LitE-SQL을 대조하면 다음 실험은 단순 scale-up이 아니라 **pretrained lightweight model adaptation과 동일 조건 비교**여야 한다.

### Q2. 보험 domain evidence를 사람이 얼마나 만들어야 하는가?

현재 glossary 제거 시 template EX가 44.4% → 34.4%로 떨어진다. SEED는 schema/description/value에서 evidence를 자동 생성한다. 따라서 **수동 glossary 의존도를 자동 evidence generation으로 얼마나 줄일 수 있는지**가 다음 질문이다.

### Q3. 질문이 명확해도 DB가 답을 가지고 있지 않다면?

현재 KorFin-Bench는 answerable 90 + governance 10 + ambiguity 6이다. EHRSQL을 대조하면 **schema capability 밖의 unanswerable 질문**이 독립 평가축으로 빠져 있다.

### Q4. 후보가 없을 때 ranking을 잘하는 것으로 충분한가?

AEGIS는 execution-result signature로 후보를 묶어 self-consistency하지만, 후보 풀이 부족한지 판정해 선택적으로 다시 생성하지 않는다. R³-SQL을 대조하면 **candidate disagreement + router confidence 기반 resampling**이 실험 후보가 된다.

## 증거 수준

- **읽기 증거**: 각 논문별 source/venue/핵심 주장/한계 기록
- **판단 증거**: 적용·변형·기각 또는 후속실험 후보를 이유와 함께 기록
- **구현 증거**: 관련 모듈 경로를 노트에서 직접 연결
- **실험 증거**: 이미 측정한 값만 명시하고, 미실행 가설은 별도로 표시
- **Git 증거**: 리뷰 노트와 후속 구현을 별도 커밋으로 남겨 변경 순서를 추적

## 다음 실험

구체적인 실험 설계는 [EXPERIMENTS.md](EXPERIMENTS.md)에 기록한다. 실제 측정값이 생기기 전에는 README/포트폴리오의 성능 주장에 반영하지 않는다.
