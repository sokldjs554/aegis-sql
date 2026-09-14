# EHRSQL — A Practical Text-to-SQL Benchmark for Electronic Health Records

- 초기 검토: 2026-09-08 (완독 기록 아님)
- 논문: Gyubok Lee et al. *NeurIPS 2022 Datasets and Benchmarks Track*
- 공식 원문: https://proceedings.neurips.cc/paper_files/paper/2022/hash/643e347250cf9289e5a2a6c1ed5ee42e-Abstract-Datasets_and_Benchmarks.html
- 공식 코드/데이터: https://github.com/glee4810/EHRSQL
- 상태: **[평가 설계 참고 · 30-probe 구현·실측 완료 · default 승격 보류]**

## 1. 논문이 푸는 문제

기존 Text-to-SQL benchmark가 실제 현업 질문 분포와 멀 수 있다는 문제를 다룬다. EHRSQL은 전자건강기록(EHR) 환경에서 **실제 병원 종사자가 필요로 하는 질문을 기반으로 benchmark를 구성**했다.

질문은 의사, 간호사, 보험심사, 의무기록 등 **222명의 병원 직원**에게서 수집되었다. 저자들은 설문 응답을 seed question으로 templatize한 뒤 MIMIC-III와 eICU schema에 수동 연결했다.

## 2. 핵심 평가 특징

EHRSQL은 단순 SQL 생성만 평가하지 않는다.

- 단순 조회부터 survival rate 같은 복잡 계산까지 폭넓은 현업 요구
- 다양한 시간 표현
- **answerable / unanswerable 구분**

특히 held-out unanswerable question을 포함해, 질문이 들어오면 무조건 SQL을 생성하는 시스템이 아니라 **현재 DB로 답할 수 없는 질문을 confidence에 따라 거부할 수 있는가**를 평가한다.

## 3. AEGIS-SQL과 겹치는 지점

KorFin-Bench도 단순 EX만 보지 않고 governance 10문항과 ambiguity 6문항을 포함한다. 그러나 현재의 106문항에는 **"질문은 명확하지만 schema에 필요한 정보 자체가 없어 답할 수 없음"**이라는 축이 독립적으로 없다.

예:

- DB에 신용점수 컬럼/연결 정보가 없는데 "신용점수 700점 이하 고객의 계약 유지율"을 요청
- 외부 날씨 데이터가 없는데 "태풍 발생일의 보험금 청구 증가율"을 요청
- 고객 직업 이력이 없는데 "직업군별 실효율"을 요청

이 질문들은 모호하지도 않고 governance 위반도 아니다. **schema capability 밖**에 있다는 이유로 abstain해야 한다.

## 4. EHRSQL 2024 후속 연구에서 확인한 방향

LG AI Research & KAIST의 EHRSQL 2024 시스템은 pseudo-labeled unanswerable question으로 self-training하고, token entropy와 query execution 기반 filtering으로 uncertain prediction을 걸러냈다.

공식 원문: https://aclanthology.org/2024.clinicalnlp-1.61/

이 연구는 unanswerable detection을 단순 규칙 하나가 아니라 **학습 데이터 + confidence + execution signal** 문제로 볼 근거를 준다.

## 5. 후속 실험과 실제 측정

KorFin-Bench 본 점수는 당장 변경하지 않고 별도의 research probe로 먼저 추가한다.

다음 고정 probe와 지표를 구현했다.

- 15개 unanswerable 질문
- 예상 동작: SQL 생성 대신 명시적 abstain
- false abstention: 실제로 답할 수 있는 정상 질문을 거부하는 비율
- unanswerable recall: 답할 수 없는 질문을 제대로 거부한 비율

`SchemaAnswerabilityDetector`와 opt-in `CapabilityAwareEngine`을 추가해 실제 demo
schema/glossary evidence가 없을 때 SQL 생성 전에 abstain하도록 했다. 고정한
unanswerable 15개와 answerable hard-negative 15개에서 recall **15/15**,
false abstention **0/15**를 측정했다. 직접 구성한 30문항 결과이므로 일반 OOD
성능으로 확대하지 않고, default `AegisEngine.ask()` 승격은 외부 검증 전까지 보류한다.
세부 조건은 [answerability 실험 문서](EHRSQL-ANSWERABILITY.md)에 있다.

## 6. 면접에서 30초 설명

> EHRSQL을 보고 기존 KorFin 평가에 ‘질문은 명확하지만 현재 DB로 답할 수 없음’ 축이 빠졌다고 판단했습니다. 15개 unanswerable과 15개 answerable hard-negative를 고정하고 schema/glossary evidence 기반 gate를 구현해 recall 15/15, false abstention 0/15를 확인했습니다. 다만 직접 만든 30문항 결과라 일반 OOD 성능으로 부풀리지 않고 default 경로에는 아직 승격하지 않았습니다.
