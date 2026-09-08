# AEGIS-SQL Paper Reading Guide

목적은 논문 제목을 나열하는 것이 아니라, **AEGIS-SQL에서 실제로 어떤 가설과 구현으로 이어졌는지 설명할 수 있을 정도로 읽는 것**이다. 처음부터 모든 수식과 표를 완독할 필요는 없다. 1차 독해에서는 Abstract → Introduction → Method figure → 핵심 실험표 → Limitations/Conclusion 순서로 읽고, 아래 질문에 답을 남긴다.

## 1순위 — 반드시 읽기

### 1. EHRSQL — A Practical Text-to-SQL Benchmark for Electronic Health Records

- Paper: https://arxiv.org/abs/2301.07695
- Code/data: https://github.com/glee4810/EHRSQL
- 왜 읽나: 실제 업무 사용자가 만든 질문, 시간 표현, 그리고 **answerable / unanswerable 구분**을 text-to-SQL 신뢰성 문제로 다룬다.
- AEGIS 연결: `docs/research/EHRSQL-ANSWERABILITY.md`의 clear-but-unanswerable probe와 직접 연결된다.

읽으면서 답할 것:
1. 저자들은 왜 단순 SQL 정확도만으로 실제 배포 신뢰성을 설명할 수 없다고 보는가?
2. unanswerable 질문을 benchmark에 넣은 이유는 무엇인가?
3. EHRSQL의 unanswerable 설정과 AEGIS의 schema-capability gate는 어디까지 같고 어디부터 다른가?
4. AEGIS의 30개 probe 결과를 왜 일반 OOD 성능이라고 부르면 안 되는가?

면접용 한 문장:
> EHRSQL에서 명확하지만 DB로 답할 수 없는 질문을 별도 신뢰성 축으로 다루는 점을 보고, KorFin-Bench에 빠진 answerability 축을 별도 probe로 구현했습니다.

---

### 2. LitE-SQL — A Lightweight and Efficient Text-to-SQL Framework

- Paper: https://arxiv.org/abs/2510.09014
- 왜 읽나: 대형 proprietary LLM만 쓰지 않고 **lightweight pretrained model + schema linking + execution feedback**으로 실용적인 text-to-SQL을 구성한다.
- AEGIS 연결: `docs/research/LitE-SQL-QWEN-EXPERIMENT.md`와 Qwen2.5-Coder 1.5B LoRA/QLoRA 비교군의 직접 근거다.

읽으면서 답할 것:
1. LitE-SQL이 작은 모델을 쓸 수 있게 만든 핵심은 모델 크기 자체인가, retrieval/training/feedback의 조합인가?
2. schema retriever가 generator 앞에서 하는 역할은 무엇인가?
3. execution-guided correction은 단순 문자열 정확도와 어떻게 다른가?
4. AEGIS의 5.3M from-scratch 실패와 pretrained 1.5B 비교에서 무엇을 통제해야 공정한가?

면접용 한 문장:
> 5.3M scratch 모델의 실패를 단순히 모델 크기 문제로 결론내리지 않고, LitE-SQL을 참고해 동일 데이터·retrieval·평가를 고정한 pretrained Qwen 비교군을 만들었습니다.

---

### 3. DIN-SQL — Decomposed In-Context Learning of Text-to-SQL with Self-Correction

- Paper: https://arxiv.org/abs/2304.11015
- 왜 읽나: schema linking → 난이도 분류/분해 → SQL generation → self-correction으로 복잡한 질의를 단계화한다.
- AEGIS 연결: `src/aegis_sql/nlu/decompose.py`의 EASY / JOIN / NESTED 분해와 repair 경로를 이해할 때 가장 직접적이다.

읽으면서 답할 것:
1. DIN-SQL은 왜 모든 질문을 같은 방식으로 생성하지 않는가?
2. EASY, NON-NESTED COMPLEX, NESTED COMPLEX를 나누면 어떤 종류의 오류를 줄일 수 있는가?
3. AEGIS가 분류 자체를 LLM 호출이 아니라 deterministic rule로 둔 이유는 무엇인가?
4. decomposition이 router feature와 결합될 때 장단점은 무엇인가?

면접용 한 문장:
> DIN-SQL의 분해 아이디어를 그대로 복제하기보다, 운영 재현성과 비용 때문에 난이도 판정은 deterministic하게 만들고 generation 단계에만 활용했습니다.

---

### 4. R³-SQL — Ranking Reward and Resampling for Text-to-SQL

- Paper: https://arxiv.org/abs/2604.25325
- 왜 읽나: 후보 SQL을 execution-result 기준으로 묶어 평가하고, **정답 후보 자체가 없을 때만 선택적으로 resampling**하는 문제를 다룬다.
- AEGIS 연결: `docs/research/EXPERIMENTS.md`의 selective resampling 실험 설계와 직접 연결된다.

읽으면서 답할 것:
1. 후보 ranking만 잘해도 해결되지 않는 상황은 무엇인가?
2. execution-equivalent SQL을 개별 문자열로 순위화하면 왜 불안정한가?
3. selective resampling이 무조건 여러 번 생성하는 것보다 어떤 비용 이점이 있는가?
4. AEGIS에서 resampling trigger를 router confidence, execution-result dispersion과 연결한 이유는 무엇인가?

면접용 한 문장:
> 기존 캐스케이드의 손실이 후보 선택보다 저비용 tier에 남겨둔 질문에서 컸기 때문에, R³-SQL을 참고해 낮은 confidence와 결과 분산이 있을 때만 다시 생성하는 실험을 설계했습니다.

## 2순위 — 구조 이해를 위해 읽기

### 5. DAIL-SQL — Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation

- Paper: https://arxiv.org/abs/2308.15363
- Code: https://github.com/BeachWang/DAIL-SQL
- 핵심 관점: prompt representation, demonstration selection, example organization을 체계적으로 비교하고 **token efficiency**까지 본다.
- AEGIS 연결: few-shot selector, SQL skeleton 유사도, prompt 비용 관리.

읽으면서 볼 것:
- 단순히 예제를 많이 넣는 것보다 어떤 예제를 어떤 형식으로 넣는지가 왜 중요한가?
- token efficiency를 정확도와 같이 봐야 하는 이유는 무엇인가?

---

### 6. CodeS — Towards Building Open-source Language Models for Text-to-SQL

- Paper: https://arxiv.org/abs/2402.16347
- Code: https://github.com/RUCKBReasoning/codes
- 핵심 관점: 1B~15B open-source model, SQL-centric pretraining, schema linking, domain adaptation, data augmentation.
- AEGIS 연결: Qwen pretrained baseline을 단순 모델 교체가 아니라 schema prompt와 domain adaptation 문제로 보는 근거.

읽으면서 볼 것:
- 사전학습된 code model과 text-to-SQL 특화 추가 학습의 역할을 분리해 볼 것.
- real-world financial dataset 평가를 왜 포함했는지 볼 것.

---

### 7. XiYan-SQL — A Multi-Generator Ensemble Framework for Text-to-SQL

- Paper: https://arxiv.org/abs/2411.08599
- 핵심 관점: M-Schema representation, multiple generators, refiner, selector.
- AEGIS 연결: `SchemaCardBuilder(style="mschema")`, ensemble/selector 설계.

읽으면서 볼 것:
- M-Schema가 단순 DDL보다 어떤 정보를 추가하는지.
- 여러 generator를 쓰는 것이 정확도만의 문제가 아니라 candidate diversity 문제인 이유.

## 읽는 순서

처음에는 아래 네 편만 읽어도 충분하다.

1. **EHRSQL** — 신뢰성/answerability
2. **LitE-SQL** — lightweight pretrained model
3. **DIN-SQL** — decomposition
4. **R³-SQL** — selective resampling

각 논문마다 20~30분 1차 독해 후 이 파일의 질문에 2~4문장으로 답을 적는다. 그 다음 실제 AEGIS 코드에서 대응 파일을 열어 보면 논문을 '읽었다'가 아니라 **설계 차이를 설명할 수 있는 상태**가 된다.

## 읽은 뒤 반드시 구분할 표현

- 논문에서 제안한 것
- AEGIS가 그대로 가져온 것
- AEGIS가 다르게 구현한 것
- 아직 실험만 설계했고 측정하지 않은 것
- 실제 CI/benchmark로 측정한 것

이 다섯 가지를 섞지 않는 것이 가장 중요하다. 면접에서도 논문 이름보다 이 구분을 정확히 말하는 편이 훨씬 강하다.
