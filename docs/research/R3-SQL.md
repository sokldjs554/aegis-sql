# R³-SQL — Ranking Reward and Resampling for Text-to-SQL

- 읽은 날짜: 2026-09-08
- 논문: Hojae Han, Yeonseok Jeong, Seung-won Hwang, Zhewei Yao, Yuxiong He. *Findings of ACL 2026*
- 공식 원문: https://aclanthology.org/2026.findings-acl.2146/
- DOI: https://doi.org/10.18653/v1/2026.findings-acl.2146
- 상태: **[검토 · 후속실험 후보]**

## 1. 논문이 푸는 문제

다중 후보를 생성하는 Text-to-SQL 시스템에는 두 가지 실패가 남는다.

1. **실행 결과가 같은 SQL을 서로 다르게 평가**할 수 있다. 표면 형태는 달라도 기능적으로 같은 SQL인데, 문자열·개별 후보 단위 ranking은 이를 안정적으로 다루지 못한다.
2. **정답 SQL이 후보 집합에 아예 없으면 ranking으로는 복구할 수 없다.** 후보를 아무리 잘 정렬해도 candidate recall이 0이면 최종 정답도 없다.

## 2. 핵심 방법

R³-SQL은 후보를 **실행 결과 기준으로 그룹화**한 뒤 그룹을 ranking한다. 그룹 평가는 pairwise preference와 pointwise utility를 함께 사용한다. 또 현재 후보 풀에 정답이 없을 가능성이 높다고 판단하면 **agentic resampling**으로 후보를 다시 생성한다.

논문은 BIRD-dev에서 **75.03 execution accuracy**를 보고하고, 공개된 모델 크기를 사용하는 방법 가운데 당시 강한 성능을 제시한다. 저자들은 다섯 benchmark에서 일관된 개선을 보고한다.

## 3. AEGIS-SQL과 겹치는 지점

AEGIS-SQL의 `verify/selfconsistency.py`도 문자열 다수결이 아니라 `ExecutionResult.result_signature()`를 이용해 **실행 결과 단위로 후보를 묶어 투표**한다. 따라서 R³-SQL의 첫 번째 문제의식과 이미 직접 겹친다.

다만 현재 AEGIS의 캐스케이드는 후보 풀이 부족한지 판단해 **선택적으로 재생성하는 단계는 없다.** 현재 측정에서 캐스케이드 52.2%가 LLM 단독 57.8%보다 낮았고, 문항 단위 재분석 결과 ensemble 자체보다 `escalate_threshold`가 template에 너무 많은 문항을 남긴 것이 주원인이었다. 즉 지금 문제는 무조건 sample 수를 늘리는 것보다 **언제 상위 생성기를 다시 호출해야 하는지**가 더 중요하다.

## 4. 그대로 적용하지 않는 이유

R³-SQL을 그대로 복제하면 AEGIS가 이미 가진 cost-aware router와 역할이 겹칠 수 있다. AEGIS의 목표는 최고 정확도만이 아니라 비용·지연·거버넌스를 함께 관리하는 것이므로, resampling은 항상 실행할 기능이 아니라 **불확실성이 높은 경우에만 발동하는 선택적 단계**여야 한다.

## 5. 후속 실험 가설

**가설:** 현재 ensemble 결과에서 실행 결과 그룹의 분포가 분산되어 있고 router confidence도 낮은 경우에만 resampling하면, 전체 LLM 호출 수를 크게 늘리지 않고 hard/medium EX를 개선할 수 있다.

측정해야 할 것:

- candidate pool별 unique execution-result group 수
- top group 비중
- router confidence와 candidate disagreement의 상관
- resampling 전/후 EX
- 추가 호출 수, 질의당 비용, p95 지연

실험은 아직 실행하지 않았다. 결과가 나오기 전까지는 `PAPERS.md`에서 **후속실험 후보**로만 표기한다.

## 6. 면접에서 30초 설명

> R³-SQL은 후보 SQL을 개별 문자열로 보지 않고 실행 결과가 같은 후보끼리 그룹화해 ranking하고, 정답 후보가 없다고 판단되면 선택적으로 다시 생성합니다. AEGIS도 이미 실행 결과 해시로 self-consistency를 하고 있어서 첫 아이디어는 유사하지만, resampling은 아직 없습니다. 현재 AEGIS의 병목이 router threshold로 확인된 만큼, 후보 불일치와 router confidence를 함께 써서 선택적 resampling을 거는 실험이 다음 단계라고 봤습니다.
