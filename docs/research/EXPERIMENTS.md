# Paper-driven Experiments

이 문서는 논문 리뷰에서 나온 **실제로 측정 가능한 후속 실험**만 기록한다. 결과가 나오기 전에는 성공/개선처럼 표현하지 않는다.

## EXP-01 — EHRSQL-inspired Unanswerable Probe

**근거 논문:** [EHRSQL](EHRSQL.md)

### 문제

KorFin-Bench는 answerable 90문항, governance 10문항, ambiguity 6문항으로 구성되어 있다. 그러나 질문이 명확하고 안전해도 **현재 schema에 필요한 정보가 없어서 답할 수 없는 경우**를 독립적으로 측정하지 않았다.

### 가설

schema capability 밖의 질문을 별도 probe로 만들면, 현재 엔진이 그럴듯한 SQL을 억지로 만드는지 또는 명시적으로 abstain해야 하는지 확인할 수 있다.

### 구현

- `data/research/ehrsql_unanswerable_probes.jsonl`
  - unanswerable 15문항
  - 유사하지만 실제 schema로 답할 수 있는 hard-negative 15문항
- `src/aegis_sql/nlu/answerability.py`
  - 실제 demo schema + glossary evidence에 반응하는 deterministic capability detector
- `scripts/eval_answerability.py`
  - unanswerable recall / false abstention / accuracy 측정
- `tests/test_answerability_research.py`
  - 실제 demo schema integration test
  - schema에 matching evidence를 추가하면 기존 abstention이 해제되는지 확인
- `src/aegis_sql/research/capability_engine.py`
  - default engine을 바꾸지 않고 명시적으로 opt-in하는 executable abstention adapter
- `scripts/run_answerability_engine.py`
  - 실제 질의를 `unanswerable` 또는 기존 AEGIS 결과로 실행

### 측정 결과

고정한 30개 research probe에서:

- unanswerable recall: **15/15 = 100.0%**
- false abstention: **0/15 = 0.0%**
- accuracy: **30/30 = 100.0%**

이 수치는 일반 OOD 성능이 아니다. 이번 가설을 위해 직접 고정한 30개 probe 범위의 결과다.

### production 적용 판단

`CapabilityAwareEngine`으로 end-to-end abstention path까지 실행 가능하게 만들었지만, **default `AegisEngine.ask()`의 동작은 아직 바꾸지 않았다.** 30개 curated probe만으로 모든 실제 고객 질문에 default abstention을 켜는 것은 근거가 부족하기 때문이다.

즉 `코드 미완성`이 아니라 **연구 adapter 구현 완료 / default promotion gate 보류** 상태다. 실제 고객 질문이나 더 넓은 OOD set에서 false-abstention을 검증한 뒤 core path로 승격한다.

**현재 상태:** probe + detector + 지표 + 측정 + executable adapter **완료** / default core promotion은 추가 외부 검증 대기

---

## EXP-02 — LitE-SQL-inspired Pretrained Lightweight Model Comparison

**근거 논문:** [LitE-SQL](LitE-SQL.md)

### 문제

현재 AegisLM 5.3M from-scratch 모델은 DPO reward margin은 개선되지만 KorFin-Bench EX 0.0%다. 이 결과만으로는 실패 원인이 model size인지, pretraining 부재인지, 데이터 규모인지 분리할 수 없다.

### 가설

고정한 AEGIS flywheel snapshot에서 Qwen2.5-Coder 1.5B base와 QLoRA 이후 모델을 비교하면 pretrained adaptation의 downstream EX 변화를 측정할 수 있을 것이다. 과거 from-scratch 5.3M 결과는 원본 JSONL이 보존되지 않아 이번 paired comparison에 포함하지 않는다.

### 구현 완료

- HuggingFace `transformers` + PEFT + bitsandbytes optional stack
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` LoRA/QLoRA training entry point
- 기존 flywheel train/dev split 그대로 재사용
- 기존 `SchemaCardBuilder(style="slm")` 재사용
- train/dev SHA-256 + schema fingerprint + seed + LoRA 설정 manifest 기록
- Git에 고정한 합성 snapshot train 9,000 / dev 1,153과 materialize hash 검증
- target SQL이 prompt에 leakage되지 않는 CI test
- 기존 AEGIS normalizer + schema linker + policy guard + executor + execution-match를 그대로 쓰는 evaluator
- Colab-ready notebook
- `scripts/run_qwen_colab.sh` full-run runner

자세한 실행 절차: [LitE-SQL-QWEN-EXPERIMENT.md](LitE-SQL-QWEN-EXPERIMENT.md)

### 실제 측정 결과

Git revision `27aeccc681f6a7341ae2375c75a5e101ad2f2a30`을 Colab Tesla T4에서 full run했다.

- base EX: **10/90 = 11.1%**
- QLoRA 이후 EX: **11/90 = 12.2%**
- delta: **+1문항 / +1.11%p**
- difficulty: easy **33.3% → 30.0%**, medium **0.0% → 5.0%**, hard **0.0% → 0.0%**
- p50 latency: **3,776.95 ms → 5,388.40 ms**
- p95 latency: **7,692.91 ms → 10,199.67 ms**
- inference peak CUDA: **1.134 GiB → 1.150 GiB**
- QLoRA training: **9,788.2초**, peak CUDA **3.085 GiB**

base 실패를 복구한 문항은 5개, base 성공에서 회귀한 문항은 4개다. 순개선이 1개뿐이므로 큰 성능 향상으로 표현하지 않는다. 상세 결과와 증거 제한은 [LitE-SQL-QWEN-EXPERIMENT.md](LitE-SQL-QWEN-EXPERIMENT.md)에 기록했다.

과거 AegisLM 5.3M과 `동일 데이터` 비교를 주장하려면 이 snapshot으로 AegisLM을 다시 학습·평가해야 한다. 그 전까지 0.0% EX는 역사적 참고값으로만 표시한다.

Colab runtime reset 때문에 생성된 원본 row-level JSON/ZIP은 보존하지 못했다. 다운로드한 notebook에는 180개 개별 OK/MISS, 학습 로그, strict summary 출력이 남아 있어 `data/research/qwen_t4_full_console_evidence.json`으로 복구했다. aggregate 측정은 완료됐지만 완전한 row-level audit bundle은 향후 재실행 과제다.

**현재 상태:** 코드·Colab·평가 경로·실제 1.5B GPU full run **완료** / console evidence 복구 완료 / 원본 ZIP 보존 미완료

---

## EXP-03 — R³-SQL-inspired Selective Resampling

**근거 논문:** [R³-SQL](R3-SQL.md)

### 문제

현재 AEGIS 캐스케이드는 LLM 단독보다 5.6%p 낮다. 문항 단위 재분석에서 ensemble 구간은 오히려 +1문항이었고, 손실은 template에 남긴 구간에서 순 -6문항이었다. 따라서 무조건 sample을 늘리는 것이 아니라 **언제 다시 생성할지**가 핵심이다.

### 가설

router confidence가 낮고 후보의 execution-result group이 분산된 경우에만 resampling을 발동하면 비용 증가를 제한하면서 실패 후보 pool을 복구할 가능성이 있다.

### 구현 완료

- `src/aegis_sql/research/selective_resampling.py`
  - route confidence
  - execution-result group 수
  - winner agreement
  - candidate count
  를 이용한 side-effect-free trigger policy
- `scripts/eval_selective_resampling.py`
  - 실제로 기록된 candidate-pool JSONL을 받아 trigger rate 계산
  - `baseline_correct / resampled_correct`가 실제 관측돼 있을 때만 counterfactual accuracy 계산
  - cost / latency 증가도 같이 계산
- `tests/test_paper_driven_runtime.py`
  - low-confidence + disagreement에서만 trigger되는지 회귀 테스트

### 중요한 구분

이 policy는 R³-SQL의 learned/agentic judge를 재현한 것이 아니다. 현재 AEGIS가 이미 보유한 observable signal로 만든 **보수적인 heuristic approximation**이다.

또한 현재 저장된 `reports/eval_llm.json`은 최종 EX/tier mix는 있지만 candidate별 execution group/agreement와 실제 resampled outcome을 저장하지 않는다. 따라서 지금 숫자를 만들어 `EX가 개선됐다`고 주장하지 않는다.

### 다음 실제 측정 조건

LLM/ensemble을 다시 실행할 때 candidate pool마다 다음을 기록한다.

- route confidence
- candidate count
- unique execution-result groups
- winner agreement
- baseline correctness
- trigger 시 추가 sample 결과의 correctness
- extra cost / latency

그 로그를 `scripts/eval_selective_resampling.py`에 넣어 실제 delta를 계산한다.

**현재 상태:** trigger policy + offline evaluator + tests **완료** / 실제 resampled outcome 로그 측정은 LLM 재실행 대기

---

## EXP-04 — SafeQL-inspired Component Repair Audit

**근거 논문:** [SAFEQL.md](SAFEQL.md)

SafeQL은 DBMS parser/binder/type analyzer feedback으로 오류 component를 찾고, SQL 전체 재생성 대신 safe query space에서 부분 수정한다. AEGIS의 deterministic repair와 문제의식은 유사하지만 구현 수준은 다르다.

후속 검토 항목:

- repair log에 SELECT/FROM/WHERE/JOIN/value/function component label 추가
- 복수 deterministic repair 후보가 가능한 경우 AST edit distance 기반 ranking
- local repair budget을 소진했을 때만 LLM repair를 부르는 gate 정량화

**현재 상태:** 논문 리뷰 및 AEGIS gap 분석 완료 / 신규 구현은 EXP-01~03의 측정 이후 우선순위 재평가

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

특히 `준비된 실험`, `smoke run`, `full benchmark`를 같은 숫자로 취급하지 않는다.
