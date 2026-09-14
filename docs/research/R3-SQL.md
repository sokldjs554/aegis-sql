# R³-SQL — Ranking Reward and Resampling for Text-to-SQL

- 초기 검토: 2026-09-08 (완독 기록 아님)
- 논문: Hojae Han, Yeonseok Jeong, Seung-won Hwang, Zhewei Yao, Yuxiong He. *Findings of ACL 2026*
- 공식 원문: https://aclanthology.org/2026.findings-acl.2146/
- DOI: https://doi.org/10.18653/v1/2026.findings-acl.2146
- 상태: **[검토 · heuristic 변형 구현 및 실제 full run 완료 · 기본 승격 기각]**

## 1. 논문이 푸는 문제

다중 후보를 생성하는 Text-to-SQL 시스템에는 두 가지 실패가 남는다.

1. **실행 결과가 같은 SQL을 서로 다르게 평가**할 수 있다. 표면 형태는 달라도 기능적으로 같은 SQL인데, 문자열·개별 후보 단위 ranking은 이를 안정적으로 다루지 못한다.
2. **정답 SQL이 후보 집합에 아예 없으면 ranking으로는 복구할 수 없다.** 후보를 아무리 잘 정렬해도 candidate recall이 0이면 최종 정답도 없다.

## 2. 핵심 방법

R³-SQL은 후보를 **실행 결과 기준으로 그룹화**한 뒤 그룹을 ranking한다. 그룹 평가는 pairwise preference와 pointwise utility를 함께 사용한다. 또 현재 후보 풀에 정답이 없을 가능성이 높다고 판단하면 **agentic resampling**으로 후보를 다시 생성한다.

논문은 BIRD-dev에서 **75.03 execution accuracy**를 보고하고, 공개된 모델 크기를 사용하는 방법 가운데 당시 강한 성능을 제시한다. 저자들은 다섯 benchmark에서 일관된 개선을 보고한다.

## 3. AEGIS-SQL과 겹치는 지점

AEGIS-SQL의 `verify/selfconsistency.py`도 문자열 다수결이 아니라 `ExecutionResult.result_signature()`를 이용해 **실행 결과 단위로 후보를 묶어 투표**한다. 따라서 R³-SQL의 첫 번째 문제의식과 이미 직접 겹친다.

AEGIS에는 이 분석을 바탕으로 router confidence와 execution-result group 분산이 모두 낮을 때만 새 ensemble을 호출하는 selective-resampling runner를 추가했다. 다만 이는 R³-SQL의 learned ranking/agentic judge가 아니라 관측 가능한 신호만 쓴 heuristic approximation이다.

## 4. 그대로 적용하지 않는 이유

R³-SQL을 그대로 복제하면 AEGIS가 이미 가진 cost-aware router와 역할이 겹칠 수 있다. AEGIS의 목표는 최고 정확도만이 아니라 비용·지연·거버넌스를 함께 관리하는 것이므로, resampling은 항상 실행할 기능이 아니라 **불확실성이 높은 경우에만 발동하는 선택적 단계**여야 한다.

## 5. 실제 실험과 후속 판단

**가설:** 현재 ensemble 결과에서 실행 결과 그룹의 분포가 분산되어 있고 router confidence도 낮은 경우에만 resampling하면, 전체 LLM 호출 수를 크게 늘리지 않고 hard/medium EX를 개선할 수 있다.

`claude-sonnet-5` 실제 호출로 KorFin answerable 90문항을 paired 측정한 결과,
trigger는 4/90(4.4%)였고 EX는 47/90(52.2%)에서 46/90(51.1%)로
**1문항·1.11%p 하락**했다. gain 0, regression 1이었으며 추가 비용은
$0.331545(+9.29%), p95는 37.58초에서 43.17초로 증가했다.

따라서 이 heuristic은 기본 파이프라인에 승격하지 않는다. 실패 원인은 불확실한
문항을 찾는 trigger와 새 후보가 더 낫다고 판정하는 acceptance/ranking을 같은 것으로
취급한 데 있다. 저장된 기존·신규 후보로 수용 gate를 먼저 오프라인 설계하고, 추가
유료 full run은 보류한다. 원본·조건·행별 전이는
[R3-SQL-EXPERIMENT.md](R3-SQL-EXPERIMENT.md)에 기록했다.

## 6. 면접에서 30초 설명

> R³-SQL을 참고해 낮은 router confidence와 실행 결과 분산이 동시에 나타날 때만 새 후보를 생성하는 정책을 실제 90문항에서 검증했습니다. 4문항에 발동했지만 EX가 52.2%에서 51.1%로 1문항 하락하고 비용은 9.29% 늘었습니다. 불확실성 탐지와 새 답 수용 판단은 다른 문제라는 결론을 얻어 현재 정책은 승격하지 않았고, 다음에는 저장된 후보로 acceptance gate부터 오프라인 검증합니다.
