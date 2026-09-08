# DIN-SQL — Decomposed In-Context Learning of Text-to-SQL with Self-Correction

- 읽은 날짜: 2026-09-08
- 논문: Mohammadreza Pourreza, Davood Rafiei. *NeurIPS 2023*
- 공식 원문: https://papers.neurips.cc/paper_files/paper/2023/hash/72223cc66f63ca1aa59edaec1b3670e6-Abstract-Conference.html
- DOI: https://doi.org/10.52202/075280-1577
- 상태: **[변형 · 파이프라인 설계 참고]**

## 1. 논문이 푸는 문제

LLM에게 자연어 질문에서 SQL 전체를 한 번에 생성하게 하면 복잡한 schema reasoning과 SQL composition에서 오류가 커진다. DIN-SQL은 Text-to-SQL을 더 작은 하위 문제로 나누고 각 단계의 결과를 다음 단계에 전달하는 **decomposition**이 LLM reasoning을 개선하는지 검증한다.

저자들은 세 LLM에서 simple few-shot 대비 대략 **10% 수준의 일관된 개선**을 보고했고, 당시 Spider holdout에서 **85.3 execution accuracy**, BIRD에서 **55.9 execution accuracy**를 보고했다.

## 2. 핵심 방법

DIN-SQL의 중요한 아이디어는 "좋은 prompt 하나"보다 **문제를 단계로 쪼개는 것**이다.

논문의 흐름은 schema linking, 난이도/문제 유형 판단, SQL generation, self-correction 같은 하위 단계로 구성된다. 복잡 질의를 별도 전략으로 처리해 한 번의 monolithic generation에 모든 추론 부담을 주지 않는다.

## 3. AEGIS-SQL과 겹치는 지점

AEGIS의 production pipeline도 하나의 LLM call에 전부 맡기지 않는다.

- 한국어 normalize
- intent guard
- schema linking
- ambiguity detection
- difficulty routing
- SQL generation
- static/AST governance
- execution
- deterministic repair

즉 DIN-SQL의 **decomposition 철학**과는 직접 연결된다.

## 4. 중요한 차이

AEGIS에서 decomposition의 상당 부분은 LLM prompt chain이 아니라 **결정론적 component**다.

- schema linking: retrieval component
- difficulty: Keras router → numpy serving
- governance: sqlglot AST
- 반복되는 repair: deterministic rules 우선

이유는 비용과 지연뿐 아니라 **동일 입력에서 같은 판단을 재현하고 테스트로 고정하기 위해서**다.

따라서 `PAPERS.md`에서 DIN-SQL을 "파이프라인 골격 그대로 구현"이라고 표현하면 지나치게 강할 수 있다. 더 정확한 표현은 **"decomposition 철학을 변형해, LLM sub-agent 대신 결정론적 component와 학습 router로 분리"**다.

## 5. AEGIS에서 확인할 수 있는 검증 포인트

DIN-SQL의 아이디어를 AEGIS가 그대로 재현했다는 별도 ablation은 아직 없다. 현재 `no-repair`, `no-schema-linking` 등 개별 component ablation은 있지만, monolithic LLM baseline과 동일 조건에서 decomposition 전체의 효과를 분리해 측정한 실험은 없다.

따라서 이 논문과의 관계는 **설계 근거**이지 "DIN-SQL 방식으로 X% 개선"이라는 실험 주장으로 쓰면 안 된다.

## 6. 후속 실험

가능한 비교:

- 동일 LLM / 동일 schema card / 동일 few-shot
- A: 한 번에 question → SQL
- B: AEGIS retrieval + structured scaffold → SQL
- C: B + deterministic repair

EX, token cost, latency, repair rate를 비교하면 decomposition과 repair의 기여를 더 직접적으로 볼 수 있다.

## 7. 면접에서 30초 설명

> DIN-SQL은 Text-to-SQL을 schema linking, 문제 분류, 생성, self-correction으로 나눠 LLM의 추론 부담을 줄입니다. AEGIS도 decomposition 철학은 가져왔지만 각 단계를 LLM agent로 두지 않았습니다. retrieval, Keras router, AST guard, deterministic repair처럼 테스트 가능한 component로 분리했습니다. 다만 아직 monolithic LLM baseline과 decomposition 전체를 직접 ablation한 것은 아니어서, 논문과의 관계는 구현 재현보다 설계 변형이라고 설명하는 게 정확합니다.
