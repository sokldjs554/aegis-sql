# Render 배포 — 카드 없이 공개 URL 만들기

Cloud Run 은 결제 계정(카드)이 필요하다. 이 문서는 **카드를 등록하지 않고**
공개 주소를 만드는 경로다. 대가는 속도이고, 아래에 정확히 적었다.

## 왜 Render 인가

카드 없이 되는 선택지를 원문 대조로 좁힌 결과다.

| | 결론 | 근거 |
|---|---|---|
| Hugging Face Spaces | ✗ | Docker Space **생성 자체가 유료 플랜 필요** |
| Fly.io | ✗ | 문서 원문: *"There is no 'free account/free tier' on Fly.io"* |
| Railway | △ | 무료 플랜은 있으나 **리전 선택이 Pro 전용**으로 보인다 — 한국 접속자가 미국 기본 리전을 쓰게 된다. 게다가 절전(Serverless)이 기본 꺼짐이라 켜지 않으면 $1 크레딧이 8일 만에 소진된다 |
| **Render** | ✓ | 무료 인스턴스 타입이 따로 있고, 한도 초과 시 **과금이 아니라 중단**된다. 싱가포르 리전이 있다 |

Render 를 고른 결정적 이유는 **구조적으로 청구가 불가능**하다는 점이다.
`plan: free` 는 크레딧 잔액이 아니라 별도의 인스턴스 타입이라, 초과분이
쌓여서 요금이 되는 경로 자체가 없다. 한도(워크스페이스당 월 750 인스턴스
시간)에 닿으면 다음 달까지 서비스가 정지된다.

## 대가 — 먼저 읽을 것

- **15분간 접속이 없으면 인스턴스가 내려간다.**
- **다시 깨는 데 Render 문서 기준 약 1분**이 걸리고, 그 위에 이 앱의 기동
  시간이 얹힌다. 무료 인스턴스는 **0.1 vCPU** 라 실제로는 더 걸린다.
- 즉 **채용 담당자가 링크를 처음 누르면 1~2분 로딩 화면을 볼 수 있다.**

그래서 이 URL 의 용도는 "이력서에 적어 둘 살아 있는 링크"다.
**면접 자리에서 직접 보여줄 때는 노트북 로컬(`make serve`)이 비교가 안 되게
낫다** — 0.8초에 뜬다.

메모리는 문제가 아니다. 무료 한도가 512MB 인데 이 앱의 실측 RSS 는
**78MB** 다(엔진 구성 + 질의 3회 후). 여유가 크다. sLLM 체크포인트가
이미지에 들어가지 않아 PyTorch 가 적재되지 않기 때문이다.

## 절차

**0. 카드부터 확인한다.** GitHub 로 가입하고, Free 웹 서비스를 만들기 전에
카드를 요구하는 화면이 나오면 **거기서 멈춘다.** 이 문서를 쓰면서
`render.com` 이 검증 환경의 이그레스 정책에 막혀 있어, "카드 불필요"만은
Render 가 쓴 문서로 대조하지 못했다. 30초짜리 확인이니 직접 보고 진행할 것.

1. Render 대시보드 → **New** → **Blueprint**
2. 이 저장소(`sokldjs554/aegis-sql`)를 연결
3. 루트의 `render.yaml` 을 자동으로 읽는다 — 추가 설정 없음
4. Apply

`render.yaml` 은 Render 공식 JSON 스키마(`render.yaml.json`)로 기계 검증했다
(0 errors). 키 이름을 손대지 말 것 — 특히 `runtime: docker` 는 옳고
`env: docker` 는 폐기된 키라 거부된다.

CLI 로 미리 확인하고 싶으면:

```bash
render blueprints validate      # CLI v2.7.0+
```

## 배포 후 확인

```bash
URL=https://aegis-sql.onrender.com     # 실제 URL 로 바꿀 것
curl -s "$URL/v1/health" | head -c 200
curl -s -X POST "$URL/v1/query" -H 'content-type: application/json' \
  -d '{"question":"전체 계약은 몇 건인가요?","max_rows":5}' | head -c 300
```

`schema_fingerprint` 가 `26cee9e1989d6426` 로 나오면 README 와 같은 빌드다.
첫 호출은 스핀업 때문에 1~2분 걸릴 수 있다.

## LLM API 키를 넣지 말 것

`render.yaml` 에 `ANTHROPIC_API_KEY` 를 넣으면 답변 문장을 LLM 이 써 주는
보조 호출이 켜진다. 그러면:

- 화면의 비용·지연이 `$0 · 6ms` 에서 **`$0.0013 · 2,600ms`** 로 바뀐다
- README 가 게시한 `template 44.4% ($0·6ms)` 와 화면이 어긋난다
- **인증 없는 공개 URL이므로 아무나 그 비용을 발생시킬 수 있다**

키가 없으면 LLM 티어는 파이프라인에 등록조차 되지 않아 과금이 **구조적으로 0**
이다. 이건 이 데모의 기본 구성이지 축소판이 아니다.

## 확인하지 못한 것

정직하게 남긴다. 검증 환경에서 `render.com` 계열 호스트가 전부 차단돼
(`render.com`, `docs.render.com`, `api.render.com`, `dashboard.render.com`)
Render 가 직접 쓴 페이지를 한 장도 열지 못했다.

- **스키마는 확인했다** — 공식 `render.yaml.json` 사본을 GitHub 에서 구해
  `jsonschema` 로 돌렸고, `runtime`/`plan`/`region`/`dockerfilePath`/
  `healthCheckPath`/`envVars` 전부 유효했다. `env: docker` 로 바꾸면 2건의
  오류가 나는 것도 확인했다.
- **750 인스턴스 시간**은 Render 의 공개 저장소(`render-oss/skills`)에서
  원문을 확인했다.
- **카드 불필요**와 **한도 도달 시 중단**은 검색 요약에서만 확인했다.
  구조적으로는 맞는 이야기지만(무료는 크레딧이 아니라 인스턴스 타입이다),
  Render 문서 원문으로 대조하지는 못했다. 위 0번 단계를 반드시 지킬 것.
