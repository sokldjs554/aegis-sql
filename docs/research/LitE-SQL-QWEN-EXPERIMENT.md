# LitE-SQL-inspired Qwen Baseline

이 실험의 목적은 "작은 모델을 더 키우면 된다"를 증명하는 것이 아니다. 현재 AegisLM 5.3M from-scratch가 DPO objective에서는 개선됐지만 KorFin-Bench EX 0.0%였기 때문에, 실패 원인을 **모델 크기 / 사전학습 부재 / 데이터 규모** 중 하나로 단정할 수 없다.

LitE-SQL의 lightweight pretrained model + schema linking + execution feedback 방향을 보고, AEGIS에서도 **같은 데이터·같은 retrieval·같은 평가를 고정한 pretrained adaptation 비교군**을 추가한다.

## 비교군

| arm | 모델 | 학습 |
|---|---|---|
| A | AegisLM 5.3M from-scratch | 기존 SFT → DPO |
| B | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | LoRA 또는 QLoRA |
| C | `Qwen/Qwen2.5-Coder-3B-Instruct` | 자원 허용 시 같은 설정으로 후속 비교 |

공식 Hugging Face 모델:

- https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct

Qwen2.5-Coder 공식 모델 카드가 요구하는 Hugging Face `transformers` 경로를 사용하며, 이 실험 전용 의존성은 `.[hf]` extra로 분리했다. 정상 AEGIS 설치와 CI가 multi-GB checkpoint에 의존하지 않게 하기 위해 `hf` extra는 `all`에 넣지 않았다.

## 통제하는 변수

모델만 바꾸고 아래는 고정한다.

1. **학습 데이터** — AEGIS flywheel의 기존 `train.jsonl` / `dev.jsonl`
2. **split** — 기존 skeleton-cluster split 그대로 사용, 다시 섞지 않음
3. **schema representation** — `SchemaCardBuilder(style="slm")` 재사용
4. **table context** — flywheel record의 `tables`만 제공하되 gold SQL column은 prompt에 쓰지 않음
5. **평가 retrieval** — KorFin 질문을 AEGIS normalizer + schema linker에 통과시킨 결과 사용
6. **평가 지표** — 기존 `execution_match` 기반 KorFin EX
7. **안전 실행** — 생성 SQL을 기존 AEGIS policy guard를 거쳐 실행

따라서 Qwen 쪽 점수가 달라졌을 때 "다른 평가 코드라서" 생긴 차이를 최소화한다.

## 구현

### 데이터/manifest

`src/aegis_sql/training/hf_experiment.py`

- flywheel JSONL을 그대로 읽음
- AEGIS schema card 생성기를 재사용
- prompt에는 target SQL을 넣지 않음
- train/dev file SHA-256, schema fingerprint, seed, LoRA 설정을 manifest에 기록

### 학습

`scripts/train_hf_text2sql.py`

- 기본 모델: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- LoRA targets: `q_proj / k_proj / v_proj / o_proj`
- prompt 구간 label은 `-100`, SQL target에만 loss
- `--qlora`: CUDA에서 NF4 4-bit loading
- `--prepare-only`: 모델을 다운로드하지 않고 data/schema wiring과 manifest만 검증

### 평가

`scripts/eval_hf_text2sql.py`

- base model 또는 PEFT adapter 모두 평가 가능
- AEGIS normalizer → schema linker → schema card 재사용
- policy guard → 실제 DB 실행 → 기존 execution-match로 채점
- easy / medium / hard EX, p50/p95 generation latency, GPU peak memory 기록

## 재현 절차

```bash
# 1) 기존 데이터가 없으면 동일 flywheel을 만든다.
make flywheel

# 2) 모델 다운로드 없이 실험 입력과 해시부터 검증한다.
python scripts/train_hf_text2sql.py --prepare-only --limit-train 32 --limit-dev 16

# 3) GPU 환경(예: Colab)에서 HuggingFace/PEFT stack 설치
pip install -e ".[train,hf]"

# 4) 1.5B QLoRA
python scripts/train_hf_text2sql.py \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --qlora \
  --out data/generated/hf/qwen2.5-coder-1.5b

# 5) 같은 AEGIS retrieval + KorFin EX로 평가
python scripts/eval_hf_text2sql.py \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --adapter data/generated/hf/qwen2.5-coder-1.5b/adapter \
  --load-4bit \
  --out reports/qwen2.5-coder-1.5b.json
```

3B는 **1.5B 실험을 먼저 끝낸 뒤** 데이터·LoRA 설정을 바꾸지 않고 모델 ID만 교체해 비교한다. 자원 부족으로 설정을 바꾸면 같은 실험으로 취급하지 않고 manifest에 별도 run으로 남긴다.

## 성공/실패 해석

### Qwen adaptation이 크게 개선된다면

`5.3M scratch EX 0%`의 원인을 단순히 "sLLM은 안 된다"로 읽지 않는다. pretrained representation / scale / adaptation이 중요한 변수였다는 근거가 생긴다.

### Qwen도 낮다면

flywheel의 질문 분포, schema card, 데이터 규모나 objective를 다시 봐야 한다. 모델 규모만 늘리는 것으로 해결되지 않는다는 후속 가설이 된다.

### 어떤 결과든

- base Qwen과 fine-tuned Qwen을 분리해서 기록
- EX와 execution-success를 혼동하지 않음
- KorFin 90 answerable을 다 돌린 결과인지 limit run인지 명시
- 비용·메모리·지연을 정확도와 함께 기록

## 현재 상태

- HuggingFace/PEFT optional dependency: **구현 완료**
- 동일 flywheel/schema-card preparation: **구현 완료 + CI test 대상**
- LoRA/QLoRA training entry point: **구현 완료**
- AEGIS retrieval + KorFin EX evaluator: **구현 완료**
- 1.5B 실제 GPU 학습/평가: **미실행 — 성능 수치 주장 없음**
- 3B 실제 비교: **1.5B 결과 이후 진행**
