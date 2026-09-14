# LitE-SQL — Lightweight Text-to-SQL with Vector Schema Linking and Execution-Guided Self-Correction

- 원문 검토: 2026-09-12
- 논문: Shengmin Piao, Jieun Lee, Sanghyun Park. *Findings of EACL 2026*
- 공식 원문: https://aclanthology.org/2026.findings-eacl.186/
- 공식 코드: https://github.com/shengminp/LitE-SQL
- DOI: https://doi.org/10.18653/v1/2026.findings-eacl.186
- 상태: **[원문 독해 완료 · Qwen 1.5B 변형 실측 완료]**

## 1. 논문이 푸는 문제

최근 Text-to-SQL은 대형 proprietary LLM에 많이 의존하지만, 실제 기업 환경에서는 비용·지연·데이터 프라이버시 때문에 그대로 배포하기 어렵다. LitE-SQL은 **작은 모델로도 schema linking과 execution feedback을 제대로 설계하면 경쟁력 있는 Text-to-SQL을 만들 수 있는가**를 다룬다.

## 2. 핵심 방법

구성은 두 축이다.

1. **Schema Retriever**
   - schema embedding을 미리 계산해 vector database에 저장
   - hard-negative supervised contrastive objective로, 의미는 비슷하지만 실제 SQL에는 불필요한 column을 구분
2. **SQL Generator**
   - supervised fine-tuning(SFT)
   - 여러 생성 후보와 gold로 preference를 만든 뒤 DPO+NLL reinforcement fine-tuning(RFT)을 적용
   - 추론에서는 multi-candidate voting 대신 실패 SQL과 오류 메시지를 넣어 반복 self-correction

논문의 대표 **BIRD 72.10% EX, Spider 1.0 88.45%**는 Table 1 캡션상
`correct schema`를 제공한 7B 조건이다. 자체 retriever의 top-25 schema를 실제로
사용한 end-to-end 7B 결과는 Table 2/12에서 **BIRD 60.56%, Spider 84.35%**이며,
RFT의 순증가는 SFT 58.21%→60.56%, 83.56%→84.35%다. 따라서 72.10%를
retriever 포함 전체 파이프라인 수치로 쓰지 않는다.

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

## 5. AEGIS 변형 실험

실험 조건:

- `Qwen/Qwen2.5-Coder-1.5B-Instruct` base와 같은 모델의 QLoRA 이후를 비교
- 고정 AEGIS snapshot train 9,000 / dev 1,153 사용
- 같은 schema card·retrieval·guard·KorFin answerable 90문항 사용
- EX / easy·medium·hard / latency / GPU memory / 학습 시간을 함께 측정

Tesla T4 full run에서 base EX **11.1%(10/90)**, QLoRA 이후
**12.2%(11/90)**로 +1문항/+1.11%p였다. easy는 33.3%→30.0%, medium은
0%→5.0%, hard는 0%로 유지됐다. p50은 3,776.95ms→5,388.40ms,
p95는 7,692.91ms→10,199.67ms였고 학습은 9,788.2초, peak 3.085GiB였다.
base 실패 복구 5건과 회귀 4건이 함께 있어 “QLoRA가 크게 개선했다”는 주장은
기각한다. 자세한 증거 한계는 [Qwen 실험 문서](LitE-SQL-QWEN-EXPERIMENT.md)에 있다.

## 6. 면접에서 30초 설명

> LitE-SQL을 읽고 5.3M scratch 모델의 0%를 ‘작은 모델은 안 된다’로 결론내리지 않고 pretrained Qwen 1.5B 비교군을 만들었습니다. 같은 AEGIS snapshot과 KorFin 90문항에서 T4로 측정하니 base 11.1%에서 QLoRA 12.2%로 순증가는 1문항뿐이었고 지연도 늘었습니다. 그래서 pretrained adaptation의 가능성은 확인했지만 큰 개선으로 포장하지 않았습니다. 논문의 72.10%도 정답 schema 조건이라 제 점수와 직접 비교하지 않습니다.
