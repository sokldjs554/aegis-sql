# Spider-KO 외부 일반화 실험

> 이 문서의 수치는 **공식 Spider leaderboard 정확도**가 아닙니다. `huggingface-KREW/spider-ko` validation 1,034문항을 한국어 질문으로 입력하고, 각 문항의 공식 SQLite DB에서 예측 SQL과 gold SQL을 실행해 AEGIS `execution_match`로 비교한 외부 일반화 실험입니다.

## 실험 조건

- 모델: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- adapter: 없음
- 양자화: NF4 4-bit
- GPU: Tesla T4
- 질문 언어: 한국어 (`question_ko`)
- 문항: validation 전체 1,034개
- 생성: deterministic (`do_sample=False`)
- 데이터셋 SHA-256: `db802bab717deb2e16f2fa6cbb493712ac294515e68a69f1a243c5238cf99b52`

## 1. 스키마 표현 어블레이션

모델과 데이터는 고정하고, 모델에게 보여 주는 스키마 직렬화만 바꿨습니다.

| 스키마 표현 | EX | 실행 실패 | schema-reference 실패 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| `slm` | 369/1,034 = **35.69%** | 337 | 319 | 2,556.90 ms | 4,529.23 ms |
| `ddl` | 388/1,034 = **37.52%** | 318 | **294** | 2,424.85 ms | 4,286.85 ms |
| `compact` | 365/1,034 = **35.30%** | 368 | 344 | 2,538.06 ms | 4,259.01 ms |
| `mschema` | **393/1,034 = 38.01%** | 320 | 304 | **2,410.73 ms** | 4,260.92 ms |

`mschema`가 EX 기준으로 `slm`보다 **+2.32%p** 높아 다음 실험의 기준선으로 선택했습니다. `ddl`은 schema-reference 오류 수만 보면 가장 적었지만 최종 EX는 `mschema`가 더 높았습니다.

## 2. mschema + 1회 bounded execution-guided repair

다음 실험은 `mschema`를 그대로 고정하고 **초기 SQL이 실제 SQLite 실행에 실패했을 때만 최대 1회** 추가 생성을 허용했습니다.

repair 대상 오류도 아래 네 종류로 제한합니다.

- `no such column`
- `no such table`
- `ambiguous column name`
- `syntax error`

repair prompt에는 **원래 한국어 질문 + 동일한 mschema + 실패한 SQL + 실제 SQLite 오류**만 들어갑니다. gold SQL은 prompt에 없으며, gold query 실행은 repair 여부와 생성 경로가 최종 결정된 뒤에만 수행합니다.

### 결과

| 지표 | mschema 초기 | + 1회 bounded repair | 변화 |
|---|---:|---:|---:|
| **EX** | 393/1,034 = **38.01%** | 434/1,034 = **41.97%** | **+3.97%p / +41문항** |
| 실행 실패 | 320 | **224** | **−96 (−30.0%)** |
| schema-reference 실패 | 304 | **209** | **−95 (−31.25%)** |
| generation p50 | 2,327.92 ms | **2,533.56 ms** | +8.83% |
| generation p95 | 4,155.52 ms | **7,579.65 ms** | +82.4% |

기존에 맞힌 393문항의 **회귀는 0건**이고, 이전 오답에서 41문항을 새로 맞혔습니다. `slm` 최초 기준선 35.69%와 비교하면 최종 41.97%는 **+6.28%p**입니다.

### repair가 실제로 한 일

전체 1,034문항 중 307문항(29.69%)에서 repair가 발동했습니다.

| 원인 | 시도 | 실행 복구 | 최종 정답 |
|---|---:|---:|---:|
| `no such column` | 289 | 95 | **41** |
| `no such table` | 15 | 0 | 0 |
| `syntax error` | 3 | 1 | 0 |
| **합계** | **307** | **96** | **41** |

실행 복구율은 96/307 = **31.27%**, repair 발동 문항 중 최종 정답률은 41/307 = **13.36%**입니다. 실행까지 복구된 96문항 중 정답이 된 것은 41문항(**42.71%**)입니다.

이 차이가 중요한 실패 분석입니다. 단순히 컬럼·테이블 이름을 고쳐 SQL을 실행 가능하게 만드는 것만으로는 충분하지 않았습니다. 실행이 복구되고도 틀린 55문항은 잘못된 조인, 중첩 구조, 집합연산, 집계 의미 등 **semantic/compositional reasoning**이 다음 병목이라는 근거입니다.

## 판단

이 실험은 **채택**합니다.

- 같은 1,034문항에서 초기 `mschema` 393문항을 정확히 재현했습니다.
- 단일 변수인 one-shot repair를 추가해 EX가 38.01% → **41.97%**로 올랐습니다.
- 기존 정답 회귀가 0건입니다.
- 실행 실패와 schema-reference 실패가 각각 30% 이상 줄었습니다.
- 대신 repair가 걸리는 질의의 추가 생성 때문에 p95 지연은 크게 증가했습니다.

따라서 지금 단계에서 곧바로 더 큰 3B 모델로 옮기기보다, **오류 근거를 이용한 제한적 시스템 개선에도 아직 유의미한 성능 여지가 있다**고 판단합니다. 동시에 실행 복구 후에도 오답이 많이 남아 있으므로 향후 모델 크기/추론 능력 실험의 필요성도 분명합니다.

## 재현·증거

- 실행 코드: `scripts/eval_spider_ko_hf.py`
- one-click runner: `scripts/run_spider_ko_bounded_repair_colab.sh`
- Colab: `notebooks/spider_ko_bounded_repair.ipynb`
- compact evidence: `data/research/spider_ko_bounded_repair_evidence.json`
- 실행 Git SHA: `f2b8ccacb164caf5ccb7061ec4d768e5bab92791`
- Python 3.13.15 / torch 2.11.0+cu128 / CUDA 12.8
- transformers 5.16.1 / peft 0.20.0 / accelerate 1.14.0 / bitsandbytes 0.50.2
- 원본 1,034-row report SHA-256: `9cc36de92c5fbf18d97c704b60cd5ae1fe0509950670c1aee63c053d5d06a8c2`

원본 report는 1.3MB의 행 단위 SQL·오류·지연 증거를 포함합니다. 저장소에는 리뷰 가능한 compact evidence만 보존하고, 동일 형식의 원본은 Colab runner가 Google Drive persistent archive에 저장합니다.
