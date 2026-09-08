# LitE-SQL — Lightweight Text-to-SQL with Vector Schema Linking and Execution-Guided Self-Correction

- 읽은 날짜: 2026-09-08
- 논문: Shengmin Piao, Jieun Lee, Sanghyun Park. *Findings of EACL 2026*
- 공식 원문: https://aclanthology.org/2026.findings-eacl.186/
- 공식 코드: https://github.com/shengminp/LitE-SQL
- DOI: https://doi.org/10.18653/v1/2026.findings-eacl.186
- 상태: **[검토 · sLLM 후속실험의 핵심 비교군]**

## 1. 논문이 푸는 문제

최근 Text-to-SQL은 대형 proprietary LLM에 많이 의존하지만, 실제 기업 환경에서는 비용·지연·데이터 프라이버시 때문에 그대로 배포하기 어렵다. LitE-SQL은 **작은 모델로도 schema linking과 execution feedback을 제대로 설계하면 경쟁력 있는 Text-to-SQL을 만들 수 있는가**를 다룬다.

## 2. 핵심 방법

구성은 두 축이다.

1. **Schema Retriever**
   - schema embedding을 미리 계산해 vector database에 저장
   - hard-negative supervised contrastive objective로, 의미는 비슷하지만 실제 SQL에는 불필요한 column을 구분
2. **SQL Generator**
   - supervised fine-tuning(SFT)
   - execution-guided reinforcement를 이어서 적용
   - multi-candidate sampling에 의존하지 않고 execution feedback으로 self-correction

논문은 **BIRD 72.10% EX, Spider 1.0 88.45%**를 보고하고, 비교 대상의 대형 LLM보다 **2~30배 적은 parameter**로 경쟁력 있는 결과를 제시한다.

## 3. AEGIS-SQL과 겹치는 지점

AEGIS에도 이미 다음 구성요소가 있다.

- `retrieval/vectorstore.py`: Chroma / FAISS / numpy backend
- `retrieval/schema_linker.py`: dense + BM25 + glossary + value profile + FK graph
- `training/`: from-scratch BPE / Transformer / SFT / DPO
- `verify/`: 실행 검증과 repair

따라서 연구 질문은 거의 정면으로 겹친다. 차이는 **모델 출발점**이다. AEGIS의 AegisLM은 5.3M parameter from-scratch 모델이고 KorFin-Bench EX가 0.0%였다. LitE-SQL은 pretrained lightweight model을 fine-tune하는 접근이라, AEGIS의 실패는 "작은 모델이 불가능하다"가 아니라 **사전학습 없이 지나치게 작은 모델을 짧게 학습한 조건의 한계**로 읽어야 한다.

## 4. 이 논문 때문에 바뀐 판단

기존 AEGIS 문서에서 `100M급 이상이면 될 것`처럼 특정 규모를 필요조건처럼 쓰는 것은 실험으로 증명되지 않았다. 더 정확한 다음 질문은 다음과 같다.

> 같은 AEGIS 데이터와 같은 KorFin-Bench에서 from-scratch 5.3M과 pretrained 1.5B/3B adaptation을 비교하면 어느 정도 차이가 나는가?

즉 다음 sLLM 실험은 단순 scale-up이 아니라 **pretraining의 효과를 분리해 보는 비교실험**이어야 한다.

## 5. 후속 실험 가설

후속 후보:

- Qwen 계열 1.5B 또는 3B + HuggingFace + LoRA/SFT
- 동일한 AEGIS flywheel train/dev/test split 사용
- 동일 KorFin-Bench에서 EX / execution success / latency / model memory 비교
- glossary/schema-linking on/off를 같이 측정해 모델 크기와 domain retrieval 효과를 분리

이 실험은 아직 측정하지 않았다. 실제 결과가 없으므로 현재는 기술스택에 Qwen/HuggingFace를 추가하지 않는다.

## 6. 면접에서 30초 설명

> LitE-SQL은 vector schema retrieval과 lightweight pretrained generator를 결합하고 SFT 뒤 execution-guided training을 해서 BIRD 72.10%를 냈습니다. 제 5.3M from-scratch 모델은 학습 신호는 개선됐지만 downstream EX가 0%였기 때문에, 이 논문을 보고 다음 실험은 단순히 scratch 모델을 키우는 게 아니라 pretrained 1.5B/3B adaptation과 동일 데이터·동일 benchmark에서 비교해야 한다고 판단했습니다.
