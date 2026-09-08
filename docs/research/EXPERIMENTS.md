# Paper-driven Experiments

이 문서는 논문 리뷰에서 나온 **실제로 측정 가능한 후속 실험**만 기록한다. 결과가 나오기 전에는 성공/개선처럼 표현하지 않는다.

## EXP-01 — EHRSQL-inspired Unanswerable Probe

**근거 논문:** [EHRSQL](EHRSQL.md)

### 문제

현재 KorFin-Bench는 answerable 90문항, governance 10문항, ambiguity 6문항으로 구성되어 있다. 그러나 질문이 명확하고 안전해도 **현재 schema에 필요한 정보가 없어서 답할 수 없는 경우**를 독립적으로 측정하지 않는다.

### 가설

schema capability 밖의 질문을 별도 probe로 만들면, 현재 엔진이 그럴듯한 SQL을 억지로 만드는지 또는 명시적으로 abstain하는지 확인할 수 있다.

### 첫 probe 후보

- "신용점수가 700점 이하인 고객의 계약 유지율을 알려줘" — 신용점수 정보 없음
- "직업군별 실효율을 보여줘" — 고객 직업 이력 없음
- "태풍 발생일 전후 보험금 청구 증가율은?" — 외부 기상 데이터 없음
- "고객의 최근 건강검진 결과별 보험료를 비교해줘" — 건강검진 데이터 없음
- "경쟁사 상품 대비 보험료가 비싼 상품을 알려줘" — 경쟁사 데이터 없음

### 필요한 구현

1. `AnswerStatus.UNANSWERABLE` 또는 동등한 명시적 abstention 상태
2. answerability 판단의 근거 기록
3. `unanswerable_recall`과 `false_abstention_rate` 지표
4. 정상 answerable 질문을 거부하지 않는 회귀 테스트

### 완료 조건

- 최소 15개 unanswerable probe
- 최소 15개 유사하지만 answerable한 hard-negative probe
- 두 지표를 리포트에 별도 표시

**현재 상태:** 설계 완료 / 코드·측정 미실행

---

## EXP-02 — LitE-SQL-inspired Pretrained Lightweight Model Comparison

**근거 논문:** [LitE-SQL](LitE-SQL.md)

### 문제

현재 AegisLM 5.3M from-scratch 모델은 DPO reward margin은 개선되지만 KorFin-Bench EX 0.0%다. 이 결과만으로는 실패 원인이 model size인지, pretraining 부재인지, 데이터 규모인지 분리할 수 없다.

### 가설

동일 AEGIS flywheel 데이터에서 pretrained 1.5B/3B 모델을 LoRA/SFT하면 from-scratch 5.3M보다 downstream EX가 크게 높아질 가능성이 있다.

### 비교 조건

| 축 | 조건 |
|---|---|
| 데이터 | 동일 train/dev/test split |
| 평가 | 동일 KorFin-Bench |
| retrieval | 동일 schema linking 결과 |
| 모델 A | AegisLM 5.3M from-scratch |
| 모델 B | pretrained 1.5B + LoRA/SFT |
| 모델 C | pretrained 3B + LoRA/SFT (자원 허용 시) |

### 측정

- EX / execution success
- easy/medium/hard
- model memory
- p50/p95 latency
- 학습 시간
- glossary on/off 교차 실험

**현재 상태:** 설계 완료 / HuggingFace·Qwen 스택 미추가 / 측정 미실행

---

## EXP-03 — R³-SQL-inspired Selective Resampling

**근거 논문:** [R³-SQL](R3-SQL.md)

### 문제

현재 AEGIS 캐스케이드는 LLM 단독보다 5.6%p 낮다. 문항 단위 재분석에서 ensemble 구간은 오히려 +1문항이었고, 손실은 template에 남긴 구간에서 순 -6문항이었다. 따라서 무조건 sample을 늘리는 것이 아니라 **언제 다시 생성할지**가 핵심이다.

### 가설

router confidence가 낮고 후보의 execution-result group이 분산된 경우에만 상위 tier/resampling을 발동하면 비용 증가를 제한하면서 EX를 개선할 수 있다.

### 측정

- unique execution-result group 수
- top-group share
- router confidence
- resampling trigger rate
- EX delta
- 추가 LLM 호출 수
- cost/query
- p95 latency

### 성공 기준

정확도만 올리는 것을 성공으로 보지 않는다. 다음을 같이 보고 판단한다.

- EX 증가
- 질의당 비용 증가율
- p95 지연 증가율
- hard 문항 개선 여부

**현재 상태:** 설계 완료 / 측정 미실행

---

## 실험 기록 규칙

실행한 실험은 다음을 반드시 남긴다.

- commit SHA
- schema fingerprint
- benchmark hash 또는 item count
- provider/model
- seed
- raw JSON 결과 경로
- 실패한 가설도 삭제하지 않고 결과와 해석을 함께 기록
