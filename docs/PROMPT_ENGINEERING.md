# 프롬프트 엔지니어링 방법론

> **입장:** "이 프롬프트가 더 좋은 것 같다"는 측정이 아니다.
> 이 저장소에서 프롬프트는 **버전·해시가 찍힌 산출물**이고, 변경은 **held-out 세트의
> 실행 정확도**로 검증되며, 어떤 수치가 어떤 프롬프트에서 나왔는지는 리포트에 남는다.

## 1. 프롬프트를 코드처럼 다룬다

`configs/prompts/default.yaml`의 모든 프롬프트는 다음을 가진다.

| 필드 | 역할 |
|---|---|
| `id` | `nl2sql.system`, `repair.user` … 호출 지점과 1:1 |
| `version` | semver. 의미가 바뀌면 minor, 표현만 바뀌면 patch |
| `hash` | 템플릿 본문의 SHA-256 앞 12자 — 자동 계산 |
| `variables` | 필수 변수. 누락 시 렌더 단계에서 `KeyError` |
| `changelog` | **관측된 실패 모드 → 그것을 겨냥한 변경** |

```bash
aegis prompt list           # id / version / hash / 설명
aegis prompt show nl2sql.user
curl localhost:8000/v1/prompts
```

평가 리포트의 "재현 정보" 블록에는 `prompt_manifest`가 통째로 박힌다.

```json
"prompt_manifest": {
  "nl2sql.system": "nl2sql.system@1.4.0#495f203eb4d1",
  "nl2sql.user":   "nl2sql.user@1.5.0#bdda2aab46db"
}
```

3개월 뒤 "이 88%는 어떤 프롬프트에서 나온 건가?"에 답할 수 있다는 뜻이다.

## 2. 이 도메인에서 실제로 효과가 있었던 것

프롬프트 규칙은 취향이 아니라 **관측된 실패 모드**에서 역산해야 한다.
이 스키마에서 반복적으로 나타난 실패와 대응은 다음과 같다.

| 관측된 실패 | 프롬프트 대응 | 왜 프롬프트인가 |
|---|---|---|
| `DATE(CTRT_DT) > '2025-01-01'` — 날짜 컬럼이 TEXT인데 날짜 함수를 씀 | `nl2sql.system` 규칙 5에 **예시와 함께** 명시 | 스키마 카드만으로는 타입이 TEXT라는 사실이 "문자열 비교하라"로 이어지지 않는다 |
| `WHERE CTRT_STAT_CD = '실효'` — 코드값 대신 한글 라벨 비교 | 스키마 카드의 **CODE DICTIONARY 블록** | 정답을 프롬프트에 주는 것이 가장 싸다 |
| 필요 없는 `TB_COMM_CD` 조인으로 행이 부풀려짐 | 규칙 6: 코드명이 결과에 필요 없으면 조인 금지 | 조인 자체를 막는 것이 사후 교정보다 정확 |
| `SUM(x)/COUNT(*)` 정수 나눗셈으로 비율이 0 | 규칙 8: `CAST(... AS REAL) / NULLIF(...,0)` | SQLite 방언 특성이라 모델이 자주 놓친다 |
| 설명문과 SQL을 섞어 출력해 파싱 실패 | 4단계 고정 스캐폴드 + 코드블록 1개 계약 | 자유 CoT는 정확도를 조금 올리고 파싱을 많이 깬다 |

**스키마 카드가 프롬프트의 대부분이다.** `schema/card.py`는 4가지 직렬화 스타일을 제공하고,
어느 쪽이 나은지는 어블레이션 표에서 결정한다 (`card-compact` 변형).

## 3. 컨텍스트 절약도 프롬프트 엔지니어링이다

전체 스키마를 프롬프트에 넣는 것은 가장 흔한 낭비다.
링킹된 서브스키마만 넣었을 때의 실제 절감량은 API로 직접 확인할 수 있다.

```bash
curl -s localhost:8000/v1/link -H 'content-type: application/json' \
  -d '{"question":"실효된 계약의 채널별 비중은?"}' | jq '{card_tokens, full_card_tokens}'
```

토큰 절감은 비용만의 문제가 아니다. 무관한 테이블 90개가 프롬프트에 있으면
모델이 그중 하나를 고르는 실패가 실제로 늘어난다.

## 4. 변경을 측정하는 절차

```bash
# 1) 기준선
make eval                                   # reports/eval.md

# 2) 후보 탐색 (변이 → dev 세트 실행정확도로 채점 → 상위 보존)
python scripts/optimize_prompts.py --prompt nl2sql.user --generations 3

# 3) 결과 확인 후 configs/prompts/default.yaml 에 직접 반영 (버전 올림)
cat reports/prompt_opt.json | jq '.history'

# 4) 전체 벤치마크로 재검증
make eval
```

`scripts/optimize_prompts.py`는 **레지스트리를 자동으로 덮어쓰지 않는다.**
승격은 사람이 하는 리뷰 가능한 커밋이어야 한다.

### 변이 연산자

| 연산자 | 하는 일 |
|---|---|
| `add:<rule>` | 후보 규칙 하나를 【추가 규칙】 블록으로 삽입 |
| `drop:<marker>` | 기존 규칙 하나를 제거 — **그 규칙이 아직 값을 하는지** 확인 |
| `reorder:sections` | 용어사전/few-shot 블록 순서 교체 |
| `tighten:output` | 출력 계약을 강화 |
| `shorten:drop-scaffold` | 4단계 사고 스캐폴드 제거 — 추론 preamble이 비용값을 하는지 |
| `llm:rewrite` | APE 방식 재작성 (LLM 필요) |

`drop`과 `shorten`이 있는 이유: 프롬프트는 규칙을 **더하기만** 하면 길어지고 서로 충돌한다.
"빼도 점수가 안 떨어지는 규칙"을 찾는 것이 더하는 것만큼 중요하다.

### 채점 방식

- dev 분할은 **medium/hard만** 사용한다. easy는 프롬프트로 거의 움직이지 않아 분산만 키운다.
- 지표는 실행 정확도(EX). 문자열 유사도가 아니다.
- 동점이면 **토큰 수가 적은 쪽**이 이긴다 (`sorted(key=(-score, tokens))`).

> **주의:** 기본 설정(`provider=template`)에서는 프롬프트가 사용되지 않으므로 델타가 0이다.
> `ANTHROPIC_API_KEY` 또는 `OPENAI_API_KEY`를 설정한 뒤 실행해야 의미 있는 수치가 나온다.
> 스크립트도 이 경우 경고를 출력한다.

## 5. 하지 않기로 한 것

| | 사유 |
|---|---|
| **프롬프트에 보안 규칙을 넣고 끝내기** | "주민등록번호를 조회하지 마세요"는 확률적 보증이다. 실제 차단은 `verify/ast_guard.py`가 AST에서 한다. 프롬프트의 보안 문구는 *보조*일 뿐 통제가 아니다. |
| **few-shot 개수를 무작정 늘리기** | 예시가 늘면 토큰이 선형으로 늘고, 유사한 예시가 반복되면 오히려 편향된다. MMR로 6개 내외의 **구조적으로 다양한** 예시를 고른다. |
| **모든 단계에 LLM 호출** | 정규화·링킹·난이도 판정·예시 선택은 결정론으로 충분하다. LLM 호출은 질의당 1회를 유지한다. |
| **자유 형식 CoT** | 정확도 이득보다 파싱 실패 비용이 컸다. 4단계 고정 스캐폴드로 대체. |
