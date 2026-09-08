# EHRSQL-inspired Answerability Experiment

이 실험은 EHRSQL을 읽고 KorFin-Bench에 빠져 있던 축을 확인한 뒤 만든 **clear-but-unanswerable** 평가다.

질문이 모호하거나 위험해서 답하지 않는 경우와, 질문은 명확하지만 **현재 DB에 필요한 정보가 없어서 답할 수 없는 경우**를 분리한다.

## 연구 질문

> 현재 보험 레거시 스키마에 없는 정보를 요구할 때, 그럴듯한 SQL을 억지로 만들지 않고 abstain할 수 있는가? 동시에 비슷한 정상 질문을 잘못 거부하지 않는가?

## 데이터

`data/research/ehrsql_unanswerable_probes.jsonl`

- unanswerable 15개
- answerable hard negative 15개
- 총 30개

이 30개는 KorFin-Bench 본 점수와 합치지 않는다. 연구 가설을 확인하기 위해 별도로 고정한 probe set이다.

## 구현

`src/aegis_sql/nlu/answerability.py`

`SchemaAnswerabilityDetector`는 질문 문자열만 보고 무조건 거부하지 않는다.

1. 질문에서 capability family를 찾는다.
2. 실제 demo schema의 table/column 이름과 comment, business glossary의 term/alias/definition을 evidence vocabulary로 만든다.
3. 질문이 capability를 요구해도 현재 schema/glossary에 해당 evidence가 실제로 존재하면 거부를 해제한다.
4. evidence가 없을 때만 missing concept와 이유를 반환한다.

예를 들어 현재 schema에는 `위험점수`가 있기 때문에 위험점수 질의는 정상 평가 대상이다. 반면 `신용점수`는 존재하지 않아 별도 capability gap으로 처리한다.

## 지표

- **unanswerable recall** = 실제 unanswerable 중 abstain한 비율
- **false abstention rate** = 실제 answerable을 잘못 abstain한 비율
- **accuracy** = 두 클래스를 합친 단순 정확도

실행:

```bash
python scripts/eval_answerability.py
```

CI에서는 `tests/test_answerability_research.py`가 실제 demo schema + glossary를 사용해 같은 가설을 검증한다. 또한 schema에 `신용점수` column을 임시로 추가했을 때 기존 거부가 사라지는지도 확인해, 단순 benchmark ID 암기가 아니라 schema evidence에 반응하는지 검사한다.

## 결과 해석 규칙

이 실험에서 높은 점수가 나와도 **일반적인 OOD detector 성능**으로 주장하지 않는다. 30개 probe는 이번 연구 질문을 위해 직접 구성한 작은 데이터셋이다.

포트폴리오에서 사용할 수 있는 표현은 다음 수준까지다.

> EHRSQL을 검토한 뒤 기존 평가셋에 `명확하지만 DB로 답할 수 없는 질문`이 빠져 있음을 확인했다. 15개 unanswerable과 15개 answerable hard-negative probe를 만들고, 실제 schema/glossary evidence에 기반한 capability gate를 구현해 별도 평가했다.

수치는 CI/실험 실행이 통과한 뒤에만 이 문서에 기록한다.

## 현재 상태

- probe set: 구현 완료
- schema-capability detector: 구현 완료
- 독립 평가 runner: 구현 완료
- 실제 demo schema integration test: 구현 완료
- 측정 결과: **CI 확인 중**
- production `AegisEngine.ask()` abstention 경로: 아직 미통합
