# 논문 → 구현 매핑 (Paper-to-Module Map)

이 문서는 AEGIS-SQL의 각 모듈이 **어떤 논문의 어떤 아이디어**에서 왔는지,
그리고 **무엇을 그대로 쓰고 · 무엇을 바꾸고 · 무엇을 의도적으로 버렸는지** 기록한다.
"논문을 읽었다"가 아니라 "논문의 주장을 이 도메인에서 검증했다"를 남기는 것이 목적이다.

> 표기: **[적용]** 구현에 반영 / **[변형]** 아이디어는 채택하되 도메인에 맞게 수정 /
> **[기각]** 검토 후 채택하지 않음 (사유 명시)

---

## 1. 스키마 링킹 · 검색 (Schema Linking / Retrieval)

| 논문 | 핵심 아이디어 | 이 저장소에서 |
|---|---|---|
| **RAT-SQL** — Wang et al., *ACL 2020* | 스키마 요소 간 관계(FK, 이름 일치)를 relation-aware self-attention으로 인코딩 | **[변형]** 인코더를 학습하는 대신, 같은 관계 정보를 **FK 조인 그래프**(`schema/graph.py`)로 명시적으로 유지하고 프롬프트/생성 단계에 주입. 소규모 데이터에서 관계를 *학습*시키는 것보다 *부여*하는 것이 안정적이었다. |
| **CHESS** — Talaei et al., *arXiv 2024* | LSH 기반 값 검색 + 스키마 프루닝으로 대형 DB에서 컨텍스트 축소 | **[적용]** `schema/profile.py`가 컬럼별 대표값을 프로파일링하고 `retrieval/schema_linker.py`가 **값 매칭**을 하이브리드 점수에 합산. 다만 LSH 대신 (테이블 수가 수백 규모라) 정확 매칭 + 역인덱스를 사용. |
| **DAIL-SQL** — Gao et al., *VLDB 2024* | 질문의 **값을 마스킹**한 뒤 유사도를 계산해 few-shot 예시 선택 | **[적용]** `retrieval/fewshot.py`. 리터럴이 유사도를 지배하는 문제("20만원"과 "30만원"이 같은 질문 취급) 해결. |
| **MMR** — Carbonell & Goldstein, *SIGIR 1998* | 관련성과 다양성을 동시에 최적화하는 재랭킹 | **[적용]** few-shot k개가 **구조적으로 다양**하도록 SQL 스켈레톤 기준 MMR 적용. 같은 패턴 6개를 넣는 것보다 유의미하게 낫다. |
| **BM25** — Robertson & Zaragoza, *FnTIR 2009* | 어휘 기반 랭킹 | **[적용]** 외부 의존성 없이 직접 구현(`schema_linker.py`). **암호 같은 물리명(`CTRT_STAT_CD`)에서는 임베딩보다 어휘 매칭이 강하다** — 하이브리드가 필수인 이유. |
| *(도메인 고유)* | — | **[신규]** **사내 용어사전 주입**(`retrieval/glossary.py`). 어떤 논문도 "유지율 → `CTRT_STAT_CD IN ('01','05')`"를 알려주지 않는다. 금융권 Text-to-SQL 정확도를 가르는 실제 요인. |

## 2. 프롬프트 · 추론 (Prompting / Reasoning)

| 논문 | 핵심 아이디어 | 이 저장소에서 |
|---|---|---|
| **Chain-of-Thought** — Wei et al., *NeurIPS 2022* | 중간 추론 단계를 유도 | **[변형]** 자유 CoT는 출력 파싱을 불안정하게 만든다. `configs/prompts/default.yaml`의 `nl2sql.user`는 **4단계 고정 스캐폴드**(조인 경로 → 필터 → 집계 → SQL) 후 코드블록 1개만 허용. |
| **Least-to-Most** — Zhou et al., *ICLR 2023* | 큰 문제를 하위 문제로 분해 | **[적용]** `nlu/decompose.py` |
| **DIN-SQL** — Pourreza & Rafiei, *NeurIPS 2023* | 난이도 분류 → 분해 → 생성 → 자가교정의 4단계 파이프라인 | **[적용]** 파이프라인 골격 그대로. 단, 난이도 분류를 LLM 호출이 아니라 **규칙 + 학습된 라우터**로 대체(비용·지연 이유). |
| **MAC-SQL** — Wang et al., *arXiv 2023* | Selector / Decomposer / Refiner 멀티에이전트 | **[변형]** 에이전트 3개를 각각 LLM으로 돌리면 비용이 3배가 된다. Selector=검색기, Decomposer=규칙, Refiner=실행 기반 교정으로 **LLM 호출을 1회로 유지**. |
| **C3** — Dong et al., *arXiv 2023* | Clear Prompting / Calibration with Hints / Consistent Output | **[적용]** Hint 주입(용어사전 SQL 조각)과 Consistent Output(실행 결과 투표)을 채택. |
| **XiYan-SQL / M-Schema** — *arXiv 2024* | 스키마 직렬화 형식이 정확도에 유의미한 영향 | **[적용]** `schema/card.py`가 4가지 스타일(`mschema`/`ddl`/`compact`/`slm`)을 제공하고 어블레이션으로 비교. |
| **SQL-PaLM** — Sun et al., *arXiv 2023* | 실행 기반 self-consistency + 프롬프트 설계 | **[적용]** `verify/selfconsistency.py` |

## 3. 생성 · 디코딩 (Generation / Decoding)

| 논문 | 핵심 아이디어 | 이 저장소에서 |
|---|---|---|
| **RESDSQL** — Li et al., *AAAI 2023* | 스키마 랭킹 + **스켈레톤 인지 디코딩**으로 구조와 내용을 분리 | **[적용]** `generation/skeleton.py`의 `sql_skeleton()`이 few-shot 다양성·중복 제거·난이도 판정의 공통 축으로 재사용된다. |
| **PICARD** — Scholak et al., *EMNLP 2021* | 증분 파싱으로 문법적으로 불가능한 토큰을 디코딩 중 차단 | **[기각]** 토큰 단위 제약 디코딩은 자체 sLLM에는 적용 가능하지만 **호스팅 LLM API에는 불가능**하고, 우리 실패의 대부분은 문법 오류가 아니라 **의미 오류**(잘못된 코드값, 날짜 형식)였다. 대신 **생성 후 AST 검증 + 실행 기반 교정**에 투자. |
| **Execution-Guided Decoding** — Wang et al., *arXiv 2018* | 부분 실행 결과로 후보를 걸러냄 | **[적용]** 전체 실행 기반이지만 동일 철학. `verify/executor.py` + `selfconsistency.py`. |
| **Self-Consistency** — Wang et al., *ICLR 2023* | 여러 샘플의 다수결 | **[변형]** 문자열이 아니라 **실행 결과 해시**(`ExecutionResult.result_signature()`)로 투표. 서로 다른 SQL이 같은 답을 낼 수 있으므로 문자열 투표는 과소 집계된다. |
| **Self-Debugging** — Chen et al., *ICLR 2024* | 오류 메시지를 되먹여 모델이 스스로 고치게 함 | **[변형]** LLM 호출 전에 **규칙 기반 수리 8종**을 먼저 시도(`verify/repair.py`). 날짜 형식·코드 리터럴·미지정 조인 같은 반복 실패는 결정론적으로 고치는 편이 싸고 빠르고 정확하다. |

## 4. 소형 모델 · 학습 (Small LM / Training)

| 논문 | 핵심 아이디어 | 이 저장소에서 |
|---|---|---|
| **LLaMA** — Touvron et al., *arXiv 2023* | Pre-norm + RMSNorm + RoPE + SwiGLU 디코더 구조 | **[적용]** `training/model.py`에 **PyTorch로 직접 구현**(HF 미사용, 다운로드 없음). |
| **RMSNorm** — Zhang & Sennrich, *NeurIPS 2019* | LayerNorm의 재중심화 제거 | **[적용]** |
| **RoFormer(RoPE)** — Su et al., *arXiv 2021* | 회전 위치 임베딩 | **[적용]** cos/sin 캐시 + KV 캐시 생성 경로 포함 |
| **GLU Variants(SwiGLU)** — Shazeer, *arXiv 2020* | FFN 게이팅 | **[적용]** |
| **Byte-level BPE** — Sennrich et al., *ACL 2016* / Radford et al., 2019 | 서브워드 분절 | **[적용]** `training/tokenizer.py`에서 **바이트 단위 BPE를 처음부터 학습**. 한글+SQL 혼합 코퍼스에서 무손실 왕복을 보장해야 하므로 바이트 레벨이 필수. |
| **LoRA** — Hu et al., *ICLR 2022* | 저랭크 어댑터로 파라미터 효율 미세조정 | **[적용]** `training/lora.py`에 `peft` 없이 직접 구현. `B=0` 초기화로 적용 직후 출력이 동일함(`allclose`, atol 1e-6)을 테스트로 보장. |
| **DPO** — Rafailov et al., *NeurIPS 2023* | 보상모델 없이 선호쌍으로 직접 정렬 | **[적용]** `training/dpo.py`. **선호쌍을 엔진이 스스로 만든다** — `chosen`=gold SQL, `rejected`=자가교정 로그에 남은 실패 SQL. 사람 라벨링 0건. |
| **CodeS** — Li et al., *SIGMOD 2024* | 양방향 데이터 증강 + 점진적 사전학습으로 오픈 LLM의 Text-to-SQL 성능 확보 | **[적용]** 플라이휠의 직접적 근거. 다만 사전학습 대신 **스키마 기반 프로그램 샘플링 → 역번역**으로 도메인 데이터를 0에서 생성. |

## 5. 데이터 구축 · 증강 (Data Flywheel)

| 논문 | 핵심 아이디어 | 이 저장소에서 |
|---|---|---|
| **Back-Translation** — Sennrich et al., *ACL 2016* | 타깃→소스 역생성으로 학습쌍 확보 | **[적용]** `flywheel/back_translate.py`가 **SQL → 한국어 질문**을 역생성. 정답(SQL)이 먼저 있으므로 라벨 노이즈가 원천적으로 없다. |
| **EDA** — Wei & Zou, *EMNLP-IJCNLP 2019* | 동의어 치환·삽입·교환·삭제 | **[변형]** 한국어는 영어식 EDA가 잘 듣지 않는다. **조사 변형 · 존댓말/반말 · 띄어쓰기 오류 · 자모 단위 오타 · 용어사전 별칭 치환**으로 재설계(`flywheel/augment.py`). 숫자·날짜·코드값은 증강 전후 불변임을 검증. |
| **Self-Instruct** — Wang et al., *ACL 2023* | 모델이 스스로 instruction 데이터를 생성 | **[변형]** 모델이 아니라 **스키마 문법**이 시드를 만든다. 환각이 구조적으로 불가능하고 API 키 없이 돈다. |
| **AlpaGasus류 품질 필터링** | 저품질 합성 데이터 제거가 양보다 중요 | **[적용]** `flywheel/quality_filter.py`: 실행 실패 · 빈 결과 · 퇴화 결과 · 중복 제거 후, **스켈레톤 클러스터 단위로 train/dev/test 분할**하여 누수 차단. |

## 6. 평가 (Evaluation)

| 논문 | 핵심 아이디어 | 이 저장소에서 |
|---|---|---|
| **Spider** — Yu et al., *EMNLP 2018* | 크로스도메인 Text-to-SQL 벤치마크, Exact Set Match | **[적용]** EM을 보조 지표로 사용. 단, **EM은 정답을 오답으로 만드는 경우가 많아** 주지표로 쓰지 않는다. |
| **Test-Suite Accuracy** — Zhong et al., *EMNLP 2020* | 여러 DB 인스턴스에서 실행해 우연히 맞는 SQL을 걸러냄 | **[변형]** 다중 DB 대신 **결과 집합 해시 비교 + 컬럼 수/행 수 일치**로 근사(`eval/metrics.py`). 한계는 문서에 명시. |
| **BIRD** — Li et al., *NeurIPS 2023* | 대규모 실DB 기반 벤치마크, **VES(Valid Efficiency Score)** 도입 | **[적용]** 실행 시간까지 점수화하는 VES 개념을 차용해 리포트에 지연/비용 축을 함께 싣는다. |
| *(도메인 고유)* | — | **[신규]** **KorFin-Bench 106문항**: 90개 정답 SQL + **10개 거버넌스 프로브**(요청 거부/문장 차단·마스킹) + **6개 모호성 프로브**(반드시 되물음). 정확도만 높고 주민번호를 뱉는 시스템은 배포 불가라는 관점을 점수에 넣었다. |

## 7. 라우팅 · 비용 · 신뢰도 (Cascade / Calibration)

| 논문 | 핵심 아이디어 | 이 저장소에서 |
|---|---|---|
| **FrugalGPT** — Chen et al., *arXiv 2023* | LLM 캐스케이드로 비용을 크게 줄이면서 정확도 유지 | **[적용]** `router/cascade.py`. 다만 FrugalGPT의 순차 캐스케이드 대신 **사전 난이도 예측(1회 판정)** 방식 — 실패 후 재시도는 지연을 두 배로 만들기 때문. 실패 시 사후 에스컬레이션은 보조 경로로만 유지. |
| **On Calibration of Modern Neural Networks** — Guo et al., *ICML 2017* | 현대 신경망은 과신하며, temperature scaling으로 교정 가능 | **[적용]** `router/calibrator.py` + ECE/신뢰도 표. **보정되지 않은 sigmoid 출력을 에스컬레이션 임계값으로 쓰면 안 되는 이유**를 문서화. |

## 8. 거버넌스 (Governance)

| 논문/표준 | 핵심 아이디어 | 이 저장소에서 |
|---|---|---|
| **k-Anonymity** — Sweeney, *IJUFKS 2002* | 소집단 재식별 방지 | **[적용, 옵션]** `configs/policy/insurance.yaml`의 `k_anonymity`. 활성화 시 GROUP BY 질의에 `HAVING COUNT(*) >= k` 자동 주입. |
| 개인정보보호법 / 신용정보법 (국내) | 고유식별정보·민감정보 처리 제한, 목적 외 이용 금지 | **[적용]** 컬럼 4등급(`public`/`internal`/`masked`/`forbidden`) + 행 수준 정책(마케팅 목적은 수신동의 고객으로 한정). |

---

## 의도적으로 채택하지 않은 것들

| 기법 | 기각 사유 |
|---|---|
| **제약 디코딩(PICARD)** | 호스팅 API에 적용 불가. 우리 오류의 다수는 문법이 아니라 의미(코드값·날짜형식)였고, 이는 AST 검증 + 실행 교정으로 더 싸게 잡힌다. |
| **멀티에이전트 3회 LLM 호출(MAC-SQL 원형)** | 비용 3배 대비 이득이 크지 않았다. 검색·분해를 비-LLM으로 대체해 호출 1회를 유지. |
| **10B+ 모델 파인튜닝** | 이 프로젝트의 sLLM 파트는 *구현 이해도*를 보이는 것이 목적이다. 대신 아키텍처·LoRA·SFT·DPO를 **직접 구현**해 스케일업 시 그대로 확장되는 코드를 남겼다. |
| **컬럼 단위 임베딩만 사용하는 스키마 링킹** | 물리명이 암호 같은 국내 레거시에서는 단독으로 실패. 하이브리드 + 용어사전이 필요하다. |
| **문자열 기반 self-consistency 투표** | 서로 다른 SQL이 동일한 정답을 낼 수 있어 과소 집계된다. 실행 결과 해시로 대체. |

---

## 참고 문헌

1. Yu et al. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.* EMNLP 2018.
2. Wang et al. *RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers.* ACL 2020.
3. Scholak et al. *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models.* EMNLP 2021.
4. Li et al. *RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL.* AAAI 2023.
5. Pourreza & Rafiei. *DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction.* NeurIPS 2023.
6. Gao et al. *Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation (DAIL-SQL).* VLDB 2024.
7. Dong et al. *C3: Zero-shot Text-to-SQL with ChatGPT.* arXiv 2023.
8. Wang et al. *MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL.* arXiv 2023.
9. Talaei et al. *CHESS: Contextual Harnessing for Efficient SQL Synthesis.* arXiv 2024.
10. Li et al. *Can LLM Already Serve as A Database Interface? A Big Bench for Large-Scale Database Grounded Text-to-SQLs (BIRD).* NeurIPS 2023.
11. Zhong et al. *Semantic Evaluation for Text-to-SQL with Distilled Test Suites.* EMNLP 2020.
12. Chen et al. *Teaching Large Language Models to Self-Debug.* ICLR 2024.
13. Wang et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.
14. Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
15. Zhou et al. *Least-to-Most Prompting Enables Complex Reasoning in Large Language Models.* ICLR 2023.
16. Li et al. *CodeS: Towards Building Open-source Language Models for Text-to-SQL.* SIGMOD 2024.
17. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
18. Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
19. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv 2021.
20. Zhang & Sennrich. *Root Mean Square Layer Normalization.* NeurIPS 2019.
21. Shazeer. *GLU Variants Improve Transformer.* arXiv 2020.
22. Sennrich et al. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016.
23. Sennrich et al. *Improving Neural Machine Translation Models with Monolingual Data.* ACL 2016.
24. Wei & Zou. *EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks.* EMNLP-IJCNLP 2019.
25. Wang et al. *Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL 2023.
26. Chen et al. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* arXiv 2023.
27. Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017.
28. Carbonell & Goldstein. *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR 1998.
29. Robertson & Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR, 2009.
30. Wang et al. *Robust Text-to-SQL Generation with Execution-Guided Decoding.* arXiv 2018.
31. Sun et al. *SQL-PaLM: Improved Large Language Model Adaptation for Text-to-SQL.* arXiv 2023.
32. Sweeney. *k-Anonymity: A Model for Protecting Privacy.* IJUFKS 2002.
33. Touvron et al. *LLaMA: Open and Efficient Foundation Language Models.* arXiv 2023.
34. XiYan-SQL. *A Multi-Generator Ensemble Framework for Text-to-SQL.* arXiv 2024.
