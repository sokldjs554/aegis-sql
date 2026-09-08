# RAT-SQL — Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers

- 읽은 날짜: 2026-09-08
- 논문: Bailin Wang, Richard Shin, Xiaodong Liu, Oleksandr Polozov, Matthew Richardson. *ACL 2020*
- 공식 원문: https://aclanthology.org/2020.acl-main.677/
- 공식 코드: https://github.com/microsoft/rat-sql
- DOI: https://doi.org/10.18653/v1/2020.acl-main.677
- 상태: **[변형]**

## 1. 논문이 푸는 문제

Text-to-SQL이 unseen database schema에 일반화하려면 두 가지가 중요하다.

1. table/column/foreign-key 같은 **schema relation을 encoder가 이해할 수 있어야 한다.**
2. 자연어 질문의 표현과 실제 schema element를 **정확히 alignment**해야 한다.

RAT-SQL은 relation-aware self-attention으로 schema encoding, schema linking, feature representation을 하나의 encoder에서 다룬다.

논문은 Spider에서 **57.2% exact match**, BERT 결합 시 **65.6%**를 보고한다.

## 2. AEGIS-SQL에서의 변형

AEGIS는 RAT-SQL의 neural relation encoder를 재현하지 않았다. 대신 schema relation을 학습시키기보다 **명시적으로 제공**한다.

- `schema/graph.py`: foreign-key join graph
- `retrieval/schema_linker.py`: question ↔ table/column linking
- `retrieval/glossary.py`: 보험 업무용어 ↔ schema mapping
- value profile: 실제 code/value에 대한 evidence

즉 RAT-SQL이 relation을 attention 내부에서 학습한다면, AEGIS는 작은 합성 도메인에서 **FK·용어·값을 외부 구조로 유지하고 generator에 주입**한다.

## 3. 왜 그대로 구현하지 않았나

AEGIS의 현재 목표는 RAT-SQL 재현이나 Spider SOTA가 아니라 한국 보험 레거시 schema에서의 실제 실패 유형을 관측하는 것이다. 물리명이 `CTRT_STAT_CD`처럼 암호 같고, 업무용어 `유지율`이 특정 code predicate로 연결되는 문제는 relation-aware encoder만으로 해결된다고 가정하기 어렵다.

또 AEGIS는 template, 자체 sLLM, hosted LLM을 같은 retrieval 결과 위에서 사용하므로, schema relation을 특정 neural encoder 안에 묶기보다 **generator와 분리된 공통 evidence layer**로 두었다.

## 4. AEGIS에서 확인된 관련 결과

현재 어블레이션에서:

- full template EX: 44.4%
- no-glossary: 34.4% (**-10.0%p**)
- dense-only: 41.1% (**-3.3%p**)

이 결과는 RAT-SQL보다 AEGIS 방식이 우월하다는 뜻이 아니다. 비교 benchmark와 model이 다르므로 직접 우열 비교는 불가능하다. 다만 **이 프로젝트 환경에서 schema/domain linking 구성요소가 실제 결과를 바꾼다**는 근거다.

## 5. 남은 질문

- 명시적 FK graph가 unseen schema에서도 충분한가?
- schema가 수백~수천 table로 커지면 relation-aware learned representation이 더 유리한가?
- DCG-SQL/LitE-SQL 같은 최근 schema retrieval과 동일 benchmark에서 비교하면 어떤가?

현재 AEGIS는 단일 합성 insurance schema이므로 unseen-schema 일반화는 검증하지 않았다.

## 6. 면접에서 30초 설명

> RAT-SQL은 unseen DB에서 schema relation과 질문-schema alignment를 relation-aware attention으로 함께 학습합니다. AEGIS는 그 모델을 복제한 게 아니라 같은 문제를 FK graph, glossary, value profile이라는 명시적 evidence layer로 변형했습니다. 여러 generator가 같은 retrieval을 공유해야 했고, 보험 레거시 schema의 code/업무용어 문제를 직접 넣기 위해서였습니다. 다만 unseen schema 일반화는 아직 검증하지 않았다는 한계도 분명합니다.
