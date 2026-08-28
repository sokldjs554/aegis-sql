# AEGIS-SQL 평가 리포트

생성 시각: 2026-08-28 14:40 KST  ·  벤치마크: `korfin_bench.jsonl` (106문항)

## 요약

| 지표 | 값 | 설명 |
|---|---:|---:|
| **실행 정확도 (EX)** | **52.2%** | 결과 집합이 정답과 일치 |
| Exact Set Match | 4.4% | SQL 문자열 구조 일치 (보조 지표) |
| Skeleton Match | 30.0% | 질의 구조는 맞고 상수/컬럼만 다름 |
| 실행 성공률 | 100.0% | 오류 없이 실행된 비율 |
| VES | 0.409 | 정확도 × 상대 실행 효율 (BIRD) |
| 자가교정 발동률 | 0.0% | 실행 실패 후 수리 시도 |
| 교정 성공률 | 0.0% | 수리된 질의 중 정답 |
| 에스컬레이션률 | 0.0% | 상위 티어로 재시도 |
| p50 / p95 지연 | 4456 / 39227 ms | 종단 지연 |
| 질의당 비용 | $0.027945 | LLM 토큰 비용 |

티어 분포: `{'template': 54, 'ensemble': 36}`

## 거버넌스 · 모호성 프로브

정확도만으로는 배포 가능성을 말할 수 없다. 아래 두 지표는 벤치마크의 일부다.

| 프로브 | n | 통과율 | 실패 항목 |
|---|---:|---:|---:|
| 거버넌스 (요청 거부 / 문장 차단·마스킹) | 10 | 100.0% | — |
| 모호성 (반드시 되물음) | 6 | 100.0% | — |

거버넌스 프로브는 **각자가 겨냥하는 계층에서** 채점한다. `intent` 프로브(파괴적 요청)는 종단 거부 여부로, `sql` 프로브는 가드가 해당 문장을 차단·마스킹하는지로 채점한다. 참고로 종단 거부율은 70.0% 이며, 이 값은 활성 티어가 위험한 컬럼을 실제로 생성했는지에 좌우되므로 안전성 수치로 읽으면 안 된다.

## 난이도별

| 난이도 | n | EX | EM | 실행성공 | p50(ms) |
|---|---:|---:|---:|---:|---:|
| easy | 30 | 96.7% | 13.3% | 100.0% | 1970 |
| medium | 40 | 35.0% | 0.0% | 100.0% | 4747 |
| hard | 20 | 20.0% | 0.0% | 100.0% | 29331 |

## 질의 유형별 (태그, n≥3)

| 태그 | n | EX |
|---|---:|---:|
| `code-join` | 23 | 13.0% |
| `count` | 22 | 100.0% |
| `glossary` | 16 | 31.2% |
| `rank` | 11 | 54.5% |
| `ratio` | 10 | 40.0% |
| `code` | 9 | 100.0% |
| `multi-join` | 9 | 44.4% |
| `cte` | 9 | 44.4% |
| `relative` | 7 | 57.1% |
| `avg` | 7 | 57.1% |
| `date` | 6 | 66.7% |
| `sum` | 6 | 66.7% |
| `group` | 6 | 33.3% |
| `having` | 6 | 16.7% |
| `null` | 4 | 100.0% |
| `date-bucket` | 4 | 75.0% |
| `join` | 4 | 0.0% |
| `compare` | 3 | 100.0% |

## 실패 사례 (43건)

| id | 난이도 | 상태 | 원인 |
|---|---:|---:|---:|
| kfb-e28 | easy | ok | result mismatch (pred 1행 × 1열 vs gold 1행 × 1열) |
| kfb-m02 | medium | ok | result mismatch (pred 5행 × 3열 vs gold 5행 × 3열) |
| kfb-m04 | medium | ok | result mismatch (pred 7행 × 2열 vs gold 7행 × 2열) |
| kfb-m08 | medium | ok | result mismatch (pred 8행 × 2열 vs gold 8행 × 2열) |
| kfb-m09 | medium | ok | result mismatch (pred 3행 × 2열 vs gold 6행 × 3열) |
| kfb-m10 | medium | ok | result mismatch (pred 4행 × 2열 vs gold 4행 × 2열) |
| kfb-m11 | medium | ok | result mismatch (pred 11행 × 2열 vs gold 11행 × 2열) |
| kfb-m12 | medium | ok | result mismatch (pred 10행 × 6열 vs gold 10행 × 2열) |
| kfb-m13 | medium | ok | result mismatch (pred 5행 × 4열 vs gold 5행 × 2열) |
| kfb-m17 | medium | ok | result mismatch (pred 24행 × 2열 vs gold 24행 × 2열) |
| kfb-m18 | medium | ok | result mismatch (pred 7행 × 2열 vs gold 7행 × 2열) |
| kfb-m19 | medium | ok | result mismatch (pred 5행 × 2열 vs gold 5행 × 2열) |
| kfb-m20 | medium | ok | result mismatch (pred 7행 × 4열 vs gold 7행 × 2열) |
| kfb-m21 | medium | ok | result mismatch (pred 1행 × 1열 vs gold 10행 × 2열) |
| kfb-m22 | medium | ok | result mismatch (pred 4행 × 2열 vs gold 4행 × 2열) |
| kfb-m23 | medium | ok | result mismatch (pred 6행 × 2열 vs gold 6행 × 2열) |
| kfb-m25 | medium | ok | result mismatch (pred 1행 × 1열 vs gold 8행 × 2열) |
| kfb-m27 | medium | ok | result mismatch (pred 4행 × 2열 vs gold 4행 × 2열) |
| kfb-m28 | medium | ok | result mismatch (pred 5행 × 2열 vs gold 5행 × 3열) |
| kfb-m29 | medium | ok | result mismatch (pred 7행 × 2열 vs gold 7행 × 2열) |
| kfb-m31 | medium | ok | result mismatch (pred 10행 × 6열 vs gold 10행 × 2열) |
| kfb-m32 | medium | ok | result mismatch (pred 1행 × 1열 vs gold 1행 × 1열) |
| kfb-m33 | medium | ok | result mismatch (pred 1행 × 1열 vs gold 11행 × 2열) |
| kfb-m34 | medium | ok | result mismatch (pred 1행 × 1열 vs gold 1행 × 1열) |
| kfb-m37 | medium | ok | result mismatch (pred 7행 × 2열 vs gold 7행 × 2열) |

…외 18건

## 재현 정보

```json
{
  "schema_fingerprint": "26cee9e1989d6426",
  "prompt_manifest": {
    "nl2sql.system": "nl2sql.system@1.4.0#495f203eb4d1",
    "nl2sql.user": "nl2sql.user@1.5.0#bdda2aab46db",
    "repair.user": "repair.user@1.2.0#9bf9b9d32dd9",
    "decompose.user": "decompose.user@1.1.0#832fb6306c4a",
    "clarify.user": "clarify.user@1.0.0#24a51d547243",
    "answer.user": "answer.user@1.2.0#e01b635c0f83",
    "backtranslate.user": "backtranslate.user@1.1.0#7aa76a6a6f83",
    "paraphrase.user": "paraphrase.user@1.0.0#9ac3af69ed37",
    "selfcheck.user": "selfcheck.user@1.0.0#d015e809041c"
  },
  "provider": "auto",
  "model": "claude-sonnet-5",
  "available_tiers": [
    "ensemble",
    "llm",
    "template"
  ],
  "embedder": "HashingEmbedder",
  "python": "3.11.9",
  "wall_s": 1188.4
}
```

```bash
make setup && make eval          # 이 표를 그대로 재생성
```
