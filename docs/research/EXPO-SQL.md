# EXPO-SQL — Execution-based Clause-level Policy Optimization for Text-to-SQL

- Authors: Jaehoon Lee, CheolWon Na, Suyoung Bae, Jin-Seop Lee, Jihyung Lee, YunSeok Choi, Jee-Hyong Lee
- Venue: Findings of ACL 2026
- Paper: https://aclanthology.org/2026.findings-acl.1107/
- DOI: 10.18653/v1/2026.findings-acl.1107

## 1. 논문이 푸는 문제

execution feedback을 이용한 RL이 Text-to-SQL에서 유용하지만, 기존 방식은 SQL 전체에 하나의 query-level reward를 주는 경우가 많다.

예를 들어 SELECT와 GROUP BY는 맞고 WHERE만 틀린 SQL도 전체 query가 실패했다는 이유로 모든 clause가 같은 낮은 reward를 받는다. 그러면 모델 입장에서는 **어느 부분을 유지하고 어느 부분을 바꿔야 하는지 학습 신호가 거칠다.**

## 2. 핵심 방법

EXPO-SQL은 execution 결과, DB 오류 메시지, clause-wise incremental execution을 이용해 오류가 있는 SQL clause를 찾고 **clause-level reward**를 제공한다.

핵심 차이는 다음과 같다.

- 기존: SQL 전체 하나 → 하나의 reward
- EXPO-SQL: SELECT / FROM / WHERE / GROUP BY 등 clause 단위 → 더 세밀한 reward

즉 실행 성공/실패를 단순 binary label로 쓰지 않고, 오류 localization을 학습 신호로 바꾼다.

## 3. AEGIS와 연결

현재 AEGIS의 DPO 데이터는 대체로 다음 구조다.

- chosen: gold / 성공 SQL
- rejected: 실패 SQL

따라서 preference는 **query 전체 단위**다. DPO objective에서 reward margin이 개선돼도 downstream EX가 개선되지 않았던 결과를 해석할 때, `어떤 clause가 틀렸는지 구분하지 않는 coarse preference`도 검토할 수 있는 후속 변수다.

다만 AEGIS의 5.3M from-scratch EX 0%를 EXPO-SQL 하나로 설명하면 안 된다. pretraining, model scale, data scale 등 다른 변수가 함께 있으므로 clause-level signal은 별도 실험 가설이다.

## 4. AEGIS에서 가능한 후속 실험

### H1. failed SQL clause localization

실패 SQL과 gold SQL을 AST로 비교해 다음 단위로 mismatch label을 만든다.

- SELECT
- FROM / JOIN
- WHERE
- GROUP BY / HAVING
- ORDER BY / LIMIT
- nested subquery / CTE

### H2. clause-aware preference pair

전체 실패 SQL 하나를 rejected로만 쓰지 않고, 맞는 clause는 유지하면서 오류 clause만 수정한 intermediate SQL을 만든다.

예:

- `rejected_full`: WHERE + JOIN 모두 틀림
- `partial`: JOIN은 수정, WHERE는 아직 틀림
- `chosen`: gold SQL

이렇게 하면 preference가 `full < partial < chosen`의 단계적 구조를 가질 수 있다.

### H3. downstream metric 분리

objective margin만 보지 않고 다음을 함께 본다.

- KorFin execution accuracy
- clause별 정확도
- execution success
- hard-query EX
- repair 횟수

## 5. AEGIS 분류

**현재 상태: 연구 검토 / 후속 실험 후보.**

EXPO-SQL을 현재 AEGIS DPO에 직접 구현했다고 말하지 않는다. 정확한 표현은:

> query-level DPO의 한계를 검토하면서, EXPO-SQL의 clause-level execution feedback을 후속 학습 신호 후보로 분석했다.

## 6. 면접용 30초 답변

> 현재 AEGIS의 DPO는 성공 SQL과 실패 SQL을 query 단위로 비교합니다. EXPO-SQL은 이 방식이 SELECT나 JOIN처럼 맞는 clause까지 동일하게 벌점을 줄 수 있다는 문제를 지적하고 execution feedback으로 clause별 reward를 만듭니다. 제 DPO margin 개선이 EX로 이어지지 않았기 때문에, 다음 단계에서는 실패 SQL을 clause 단위로 localization한 preference가 실제 EX를 올리는지 별도 검증할 가치가 있다고 봤습니다.
