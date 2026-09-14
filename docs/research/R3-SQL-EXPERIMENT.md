# R³-SQL-inspired Selective Resampling — Live Experiment

이 문서는 `EXP-03`의 **실제 후보 재생성 실험**과 재현 절차를 기록한다.
2026-09-11 Anthropic full run과 2026-09-13 독립 재검증에서
`portfolio_evidence_ready=true`를 확인했다. 결과는 개선이 아니라 성능 하락이며,
그 실패까지 원본 행 단위 로그와 함께 보존한다.

## 고정한 비교 프로토콜

| 항목 | 설정 |
|---|---|
| 평가셋 | KorFin-Bench answerable 90문항 |
| baseline | 각 문항을 정상 AEGIS cascade로 1회 실행 |
| trigger | route confidence `< 0.60` **AND** winner agreement `< 0.60` **AND** execution-result groups `>= 2` **AND** candidates `>= 2` |
| resampling | trigger 문항만 새로운 hosted ensemble 5-sample을 강제 실행 |
| 정책 결과 | trigger면 새 ensemble 결과, 아니면 같은-run baseline 결과 |
| 정확도 | 기존 `execution_match` EX |
| 비용 | provider usage token과 저장소 내 고정 price table로 계산한 USD 추정치 |
| 지연시간 | end-to-end wall latency; 자연어 답변 합성은 양쪽 모두 제외 |
| retrieval | hashing embedder + numpy store, few-shot 0개 |
| seed | DB/로컬 구성은 `PYTHONHASHSEED=0`; hosted sampling seed는 지원되지 않아 `null`로 기록 |

여기서 resampling은 기존 후보와 새 후보를 합쳐 learned ranker로 재평가하는
R³-SQL 원 구현의 복제가 아니다. AEGIS의 observable signal로 trigger한 뒤 새
ensemble 결과로 교체하는 heuristic approximation이다. 따라서 결과도
`R³-SQL 재현`이 아니라 `R³-SQL-inspired selective resampling`으로 표기한다.

과거 `reports/eval_llm.json`의 52.2%는 candidate-level log가 없어 이 실험의
baseline으로 재사용하지 않는다. 이번 full run 안에서 새로 얻은 paired
baseline만 비교 기준이다.

## 실제 full run — 2026-09-11

Git revision `73a1a3c4a8e2875660471f01b18bb664ecdb975c`에서
`claude-sonnet-5`를 실제 호출해 KorFin-Bench answerable 90문항을 같은 run 안에서
paired 비교했다. 업로드된 bundle을 `--strict`로 다시 계산했으며 evidence error는
0건이었다.

| 지표 | 결과 |
|---|---:|
| trigger | 4/90 = **4.4%** (eligible 4/50 = 8.0%) |
| baseline EX | 47/90 = **52.2%** |
| selective-resampling EX | 46/90 = **51.1%** |
| ΔEX | **−1문항 / −1.11%p** |
| gain / regression / unchanged | **0 / 1 / 89** |
| easy / medium / hard | 100%→100% / 32.5%→32.5% / 20.0%→15.0% |
| 비용 | $3.567036→$3.898581, 추가 **$0.331545 (+9.29%)** |
| 추가 비용 | trigger당 $0.082886 / 전체 질의당 $0.003684 |
| p50 latency | 9,978.20 ms→9,978.20 ms |
| p95 latency | 37,578.92 ms→43,171.68 ms |
| trigger 추가 latency p50 / p95 | 32,224.72 / 44,735.07 ms |

4개 trigger 가운데 2개 오답은 다시 생성해도 오답이었고, 1개 정답은 유지됐으며,
hard 1개(`kfb-h17`)는 맞았던 baseline을 새 오답으로 교체했다. 즉 낮은 router
confidence와 execution-group 분산은 **불확실성**은 찾았지만 **재생성하면 복구될
문항**을 식별하지 못했다. 기존 답을 새 답으로 무조건 교체하는 현재 approximation은
R³-SQL의 learned ranking/judge가 맡는 수용 판단을 대체하지 못한다.

따라서 현재 heuristic은 기본 파이프라인에 승격하지 않는다. 후속안은 추가 유료 호출
전에 저장된 후보만으로 기존/신규 결과를 함께 비교하는 acceptance gate를 설계하는
것이며, 새 hosted full run은 비용을 쓸 수 있을 때까지 보류한다.

원본 증거: [manifest](../../data/research/r3_selective_resampling_full/r3-selective-resampling-manifest.json) ·
[raw JSONL](../../data/research/r3_selective_resampling_full/r3-selective-resampling-raw.jsonl) ·
[summary](../../data/research/r3_selective_resampling_full/r3-selective-resampling-summary.json)

## 기록되는 숫자

- 전체 trigger rate와 multi-candidate eligible pool 기준 trigger rate
- 같은-run baseline EX, selective-resampling EX, ΔEX(%p)
- easy / medium / hard별 paired EX와 ΔEX
- gain / regression / unchanged 문항 수
- baseline total cost, extra total cost, trigger당·전체 질의당 추가 비용
- baseline / policy p50·p95 latency
- trigger 문항의 extra p50·p95 latency
- 요청 sample 수와 실제로 반환된 provider response 수

각 문항에는 후보 SQL, raw model output, execution-result group 수, agreement,
provider token/cost/latency, 선택 SQL, 결과 signature, repair 기록이 JSONL로
남는다. API key는 어느 artifact에도 저장하지 않는다.

## GPU가 필요한가

필요 없다. 모델 추론은 hosted API가 수행하고 로컬에서는 schema linking,
SQLite 실행, EX 계산만 한다. Colab CPU runtime으로 충분하다. 다만 provider
호출은 순차적으로 여러 번 일어나므로 브라우저/runtime 연결을 유지하는 것이
좋다.

## Colab full run

### 1. 저장소와 Drive 준비

```python
from google.colab import drive

drive.mount("/content/drive")
```

fresh runtime이면 저장소를 clone하고, 이미 있으면 full-run 대상 commit으로
맞춘다. 결과 manifest가 dirty worktree를 감지하므로 실험 중 코드를 수정하지
않는다.

```bash
%cd /content
!git clone https://github.com/sokldjs554/aegis-sql.git
%cd /content/aegis-sql
!git pull --ff-only
```

### 2. API key를 Colab Secrets에서 환경변수로 로드

왼쪽 열쇠 아이콘의 Secrets에 `ANTHROPIC_API_KEY`를 저장한 다음 아래처럼
로드한다. key 값을 출력하거나 notebook cell에 문자열로 직접 쓰지 않는다.

```python
import os
from google.colab import userdata

os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
assert os.environ["ANTHROPIC_API_KEY"]
```

### 3. full run 실행

`MAX_COST_USD`는 이미 관측된 누적 비용이 그 값에 도달하면 **다음 문항 전에**
멈추는 안전장치다. 현재 문항의 paired 결과를 버리지 않기 때문에 최대 한
문항 비용만큼 초과할 수 있다. 처음 실행에서는 직접 승인한 한도를 넣는다.

```bash
%cd /content/aegis-sql
!PREFIX=/content/drive/MyDrive/aegis-sql-evidence/r3-selective-resampling \
  MAX_COST_USD=5 \
  bash scripts/run_r3_colab.sh
```

정상 완료 시 Drive에 다음 네 파일이 생기고 ZIP browser download도 즉시
시작된다.

- `r3-selective-resampling-manifest.json`
- `r3-selective-resampling-raw.jsonl`
- `r3-selective-resampling-summary.json`
- `r3-selective-resampling-results.zip`

### 4. 중단 후 재개

같은 commit, provider/model, threshold, package/runtime, DB hash여야 한다.
Drive의 기존 prefix를 그대로 지정하고 `RESUME=1`만 추가한다. 비용 한도는
실험 결과를 바꾸는 설정이 아니라 중단 안전장치이므로 재개할 때 올릴 수 있고,
사용한 한도 이력은 manifest에 모두 남는다.

```bash
%cd /content/aegis-sql
!PREFIX=/content/drive/MyDrive/aegis-sql-evidence/r3-selective-resampling \
  MAX_COST_USD=5 \
  RESUME=1 \
  bash scripts/run_r3_colab.sh
```

JSONL은 문항 하나가 끝날 때마다 flush + `fsync`된다. 이미 끝난 ordered prefix는
다시 과금하지 않는다. 설정이 달라졌으면 재개를 거부하고 새 prefix를 요구한다.

## 무과금 smoke

아래 실행은 파일 생성·summary·ZIP·resume 경로만 검사한다. mock 후보들은
기능적으로 같아서 기본 threshold에서는 보통 trigger가 0이며, 성능 증거로
사용할 수 없다.

```bash
PROVIDER=mock LIMIT=3 PREFIX=/tmp/r3-smoke bash scripts/run_r3_colab.sh
```

## evidence gate

다음 조건이 모두 참일 때만 summary의 `portfolio_evidence_ready`가 `true`다.

- 90개 answerable 문항과 frozen ordered-id hash 일치
- explicit Anthropic/OpenAI provider와 실제 선택 provider 일치
- mock/fallback 없음, hosted generation마다 요청한 response가 모두 관측됨
- trigger 문항마다 forced ensemble audit와 observed outcome 존재
- multi-candidate baseline마다 실제 execution-vote stats 존재
- benchmark·DB·router·prompt hash 기록
- clean git commit, `PYTHONHASHSEED=0`
- 최소 1개 문항에서 실제 resampling 실행

부분 실행이나 trigger 0도 디버깅용 summary는 만들지만 포트폴리오 증거로
승격하지 않는다.

## 결과 확인만 다시 하기

GPU나 API key 없이 raw log와 manifest만 있으면 된다.

```bash
PYTHONPATH=src python scripts/eval_selective_resampling.py \
  data/research/r3_selective_resampling_full/r3-selective-resampling-raw.jsonl \
  --manifest data/research/r3_selective_resampling_full/r3-selective-resampling-manifest.json \
  --strict \
  --out /tmp/r3-selective-resampling-recheck.json
```
