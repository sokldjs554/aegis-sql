# LitE-SQL-inspired Qwen Baseline

이 실험의 목적은 "작은 모델을 더 키우면 된다"를 증명하는 것이 아니다. 현재 AegisLM 5.3M from-scratch가 DPO objective에서는 개선됐지만 KorFin-Bench EX 0.0%였기 때문에, 실패 원인을 **모델 크기 / 사전학습 부재 / 데이터 규모** 중 하나로 단정할 수 없다.

LitE-SQL의 lightweight pretrained model + schema linking + execution feedback 방향을 보고, AEGIS에서도 **같은 데이터·같은 retrieval·같은 평가를 고정한 pretrained adaptation 비교군**을 추가한다. 이번 GPU 실험에서 직접 paired comparison으로 인정하는 범위는 **Qwen base와 같은 Qwen의 QLoRA 이후 결과**다.

과거 5.3M AegisLM이 사용한 12,540쌍의 원본 JSONL은 Git에 보존되지 않았다. 당시 profile cache는 스키마 fingerprint만 확인하고 DB 내용 hash를 확인하지 않아 현재 코드와 시드만으로 정확한 행을 복원할 수도 없다. 따라서 과거 AegisLM 0% EX는 역사적 참고값이며, 새 snapshot으로 AegisLM을 다시 학습하기 전에는 Qwen과 `동일 데이터 paired comparison`이라고 부르지 않는다.

## 비교군

| arm | 모델 | 학습 |
|---|---|---|
| A | AegisLM 5.3M from-scratch | 기존 SFT → DPO, 현재는 역사적 비대응 참고값 |
| B0 | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | frozen base, NF4 4-bit 평가 |
| B1 | B0와 동일한 Qwen | 고정 snapshot으로 QLoRA |
| C | `Qwen/Qwen2.5-Coder-3B-Instruct` | 자원 허용 시 같은 설정으로 후속 비교 |

공식 Hugging Face 모델:

- https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct

Qwen2.5-Coder 공식 모델 카드가 요구하는 Hugging Face `transformers` 경로를 사용하며, 이 실험 전용 의존성은 `.[hf]` extra로 분리했다. 정상 AEGIS 설치와 CI가 multi-GB checkpoint에 의존하지 않게 하기 위해 `hf` extra는 `all`에 넣지 않았다.

## 통제하는 변수

모델만 바꾸고 아래는 고정한다.

1. **학습 데이터** — `data/research/qwen_flywheel_v1`에 고정한 합성 snapshot
2. **split** — train 9,000 / dev 1,153의 행 수와 SHA-256을 Git에 고정
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

`scripts/qwen_dataset_snapshot.py`

- 현재 공개 데모 조건에서 새로 만든 합성 flywheel 중 train 9,000 / dev 1,153을 gzip snapshot으로 보존
- Colab에서는 재생성하지 않고 snapshot을 materialize한 뒤 count와 SHA-256을 다시 검증
- historical AegisLM은 이 snapshot으로 재학습하기 전까지 unpaired라고 manifest에 명시

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
- easy / medium / hard EX, CUDA 동기화 p50/p95 generation latency, GPU peak memory 기록
- 첫 generation은 warmup으로 제외하고 측정 범위를 결과 JSON에 명시
- GPU 이름·VRAM·CUDA·PyTorch·Transformers·PEFT·bitsandbytes 버전을 함께 기록

### 완료 판정

`scripts/summarize_qwen_experiment.py`는 다음 항목이 모두 실제 파일에 있을 때만 결과 요약을 만든다.

- base/adapted가 동일한 모델과 benchmark hash를 사용
- full run은 answerable KorFin **90문항**을 모두 평가
- 전체 EX와 easy/medium/hard count가 서로 일치
- base/adapted p50·p95와 peak CUDA memory가 존재
- QLoRA wall-clock 학습 시간과 학습 peak CUDA memory가 존재

10문항 smoke 결과는 별도 경로에 저장되며 `portfolio_evidence_ready=false`다. full run이 중단돼도 smoke 숫자가 full 결과 파일에 남지 않는다.

## 재현 절차

```bash
# 1) Colab GPU에서 wiring만 확인한다. 포트폴리오 수치로 사용하지 않는다.
SMOKE=1 bash scripts/run_qwen_colab.sh

# 2) smoke 성공 후 동일 런타임에서 전체 학습·평가를 실행한다.
bash scripts/run_qwen_colab.sh
```

full run이 성공하면 아래 네 원본과 이를 묶은 ZIP이 생성된다.

- `reports/qwen2.5-coder-1.5b-full-base.json`
- `reports/qwen2.5-coder-1.5b-full-adapted.json`
- `data/generated/hf/qwen2.5-coder-1.5b-full/experiment_manifest.json`
- `reports/qwen2.5-coder-1.5b-full-summary.json`
- `reports/qwen2.5-coder-1.5b-full-results.zip`

adapter 가중치나 결과 숫자는 Git에 자동 반영하지 않는다. 먼저 ZIP 원본을 보존하고, `portfolio_evidence_ready=true`와 90문항 row-level 결과를 검토한 뒤 README·포트폴리오를 별도 업데이트한다.

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
- 고정 train/dev snapshot + hash 검증: **완료**
- LoRA/QLoRA training entry point: **구현 완료**
- AEGIS retrieval + KorFin EX evaluator: **구현 완료**
- 1.5B 실제 GPU 학습/평가: **미실행 — 성능 수치 주장 없음**
- 과거 5.3M과 동일 데이터 비교: **미완료 — snapshot 기반 AegisLM 재학습 전에는 unpaired**
- 3B 실제 비교: **1.5B 결과 이후 진행**
