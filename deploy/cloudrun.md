# Google Cloud Run 배포

채용 담당자가 링크 하나로 웹 콘솔을 써볼 수 있게 하는 것이 목적입니다.

> **먼저 읽을 것 — "무료"는 조건부입니다.**
> 아래 사실은 구글 공식 요금 페이지 원문으로 확인했습니다.
>
> | 계정 종류 | 결론 |
> |---|---|
> | **무료 체험(Free Trial)** | 청구가 **구조적으로 0**. 구글 명시: *"you won't be charged unless you manually upgrade to a paid account."* $300 소진 또는 90일 경과 시 **자동 종료**되며 자동 유료 전환은 없습니다 |
> | **유료(pay-as-you-go)** | **0원이 아닙니다.** 아래 두 항목이 확정적으로 과금됩니다 |
>
> 유료 계정에서 새는 곳:
> 1. **한국행 아웃바운드 트래픽에 무료 구간이 없습니다.** 요금표의
>    `TO Australia, Indonesia, Korea, ...` 행은 **0 GiB부터** $0.19/GiB입니다.
>    다른 아시아 지역에 있는 "첫 1 GiB 무료"가 한국행에만 없습니다.
>    리전을 옮겨도 **목적지 기준**이라 회피할 수 없습니다.
> 2. **Artifact Registry 저장** — 무료 0.5 GiB는 프로젝트가 아니라
>    **결제 계정 단위 합산**입니다. 이 이미지는 200~400MB라 재배포 2~3회면
>    넘깁니다. 그리고 **트래픽이 0이어도, 잊어버린 뒤에도 매달** 나갑니다.
>
> **하드 지출 상한은 없습니다.** 예산(budget)은 알림일 뿐 차단하지 않습니다.
> Spend Cap이 있으나 공개 프리뷰이고 범위가 "프로젝트 1개 + 서비스 1개"라
> **Artifact Registry 저장과 네트워크 이그레스는 막지 못합니다.**

## 리전 — 서울이 아니라 도쿄입니다

이전 판에서는 서울을 권했는데, 요금 페이지를 다시 읽고 바꿨습니다.

Cloud Run 요금 페이지의 `Regional price tiers` 절은 `asia-northeast1
(Seoul, South Korea)` 를 **Tier 2** 로 분류합니다. 그리고 무료 한도는
*"The free tier is applied as a spending based discount using Tier 1
pricing"* 입니다. 즉 **사용은 Tier 2 단가로 계산되고 할인은 Tier 1 단가로
적립**되므로, 서울에 배포하면 컴퓨트가 첫 요청부터 $0 이 아닙니다
(CPU 기준 $0.0000336 vs 할인 $0.000024 — 약 71%만 덮입니다).

`asia-northeast1 (Tokyo)` 은 같은 페이지에서 **Tier 1** 로 분류됩니다.
도쿄에 배포하면 이 워크로드의 컴퓨트는 **실제로 $0** 이고, 한국 접속자가
치르는 대가는 왕복 30ms 남짓입니다.

이그레스는 어느 쪽이든 같습니다 — Premium Tier 이그레스는 **출발지가 아니라
목적지 기준**이라 서울에 두어도 한국행 $0.19/GiB 를 피하지 못합니다
(요금표의 한국 행은 45개 출발 리전 변형에서 전부 동일).

## 사전 준비 (최초 1회)

```bash
gcloud auth login
gcloud projects create aegis-sql-demo --name="AEGIS-SQL Demo"
gcloud config set project aegis-sql-demo
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Cloud Console에서 이 프로젝트에 **결제 계정을 연결**해야 합니다.

## 배포

```bash
cd ~/Downloads/aegis-sql

gcloud run deploy aegis-sql \
  --source . \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 20 \
  --timeout 60 \
  --min-instances 0 \
  --max-instances 1 \
  --set-env-vars AEGIS_DEMO_PUBLIC=1
```

| 플래그 | 이유 |
|---|---|
| `--region asia-northeast1` | 도쿄. **Tier 1 이라 무료 한도가 컴퓨트를 전액 덮습니다.** 서울(Tier 2)은 첫 요청부터 컴퓨트가 과금됩니다 |
| `--memory 1Gi` | Cloud Run 파일시스템은 **인메모리**라 쓰기가 RAM을 먹습니다 |
| `--concurrency 20` | 앱의 `limit_concurrency=20`과 맞춥니다 |
| `--max-instances 1` | **지출 상한이 아니라 발생 속도 상한**입니다. 봇이 붙었을 때 최악 금액을 낮추는, 실제로 쥘 수 있는 몇 안 되는 레버입니다 |
| `--min-instances 0` | 유휴 과금 0. 구글 원문: *"Idle instances that are not minimum instances are not charged"* |
| `--timeout 60` | 질의는 수백 ms에 끝납니다 |
| `AEGIS_DEMO_PUBLIC=1` | 인증 없는 공개 URL이므로 `/v1/feedback`이 디스크에 쓰지 않습니다 |

> **LLM API 키를 절대 넣지 마세요.** `--set-env-vars` 에 `ANTHROPIC_API_KEY`
> 를 넣으면 답변 문장을 LLM 이 써 주는 보조 호출이 켜지고, 화면의 비용·지연이
> `$0 · 6ms` 에서 `$0.0013 · 2,600ms` 로 바뀝니다. README 가 게시한
> `template 44.4% ($0·6ms)` 와 화면이 어긋나고, 공개 URL 이므로 아무나
> 그 비용을 발생시킬 수 있습니다. 키가 없으면 LLM 티어는 파이프라인에
> 등록조차 되지 않아 과금이 **구조적으로 0** 입니다.

`--no-cpu-throttling`과 `--cpu-boost`는 **절대 켜지 마세요.** 인스턴스 기반
과금으로 바뀌어 요청이 없어도 24시간 과금됩니다.

## 배포 직후 — 이 단계를 건너뛰지 마세요

```bash
gcloud artifacts docker images list \
  asia-northeast1-docker.pkg.dev/aegis-sql-demo/cloud-run-source-deploy
```

옛 이미지가 쌓여 있으면 지우거나 정리 정책을 거세요:

```bash
cat > /tmp/policy.json <<'JSON'
[{"name":"keep-latest","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":1}}]
JSON
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
  --location=asia-northeast1 --policy=/tmp/policy.json
```

`--source .` 가 만드는 저장소에 자동 정리 정책이 붙는지는 **확인하지 못했습니다.**
붙는다고 가정하지 마세요. 이 항목은 트래픽과 무관하게 매달 나가고 배포를
멈춰도 멈추지 않습니다 — "1년 뒤 청구서"의 주범 후보 1순위입니다.

## LLM 티어를 켤지

**켜지 마세요.** API 키를 넣지 않으면 `Tier.LLM`/`ENSEMBLE`이 파이프라인에
등록조차 되지 않아 과금이 구조적으로 0이고, template 티어만으로 콘솔은
정상 동작합니다.

굳이 켜려면 요청 본문의 `tier` 필드로 질의당 예산이 우회되므로
**프로바이더 대시보드의 하드 스펜드 리밋**이 유일하게 믿을 수 있는 방어선입니다.

## 배포 후 확인

```bash
URL=$(gcloud run services describe aegis-sql --region asia-northeast1 --format='value(status.url)')

curl -fsS "$URL/v1/health" | python3 -m json.tool
# router_loaded 가 true 인지 반드시 확인 — false 면 콘솔이 '라우터 미적재'를 노출합니다
```

브라우저에서 `$URL`을 열고:
1. 예시 칩 `작년 하반기에…` → 결과와 SQL 탭
2. `고객 이름이랑 주민등록번호 좀 뽑아줘` → 차단
3. 상단 `목적: 마케팅` → `마케팅 수신 대상 고객 수를 성별로 알려줘` → SQL에 `MKT_AGR_YN` 주입
4. SSE 스트리밍(단계별 진행)이 실시간으로 뜨는지 — 프록시가 버퍼링하면 한 번에 뜹니다(기능은 정상)

## 안전장치

예산 알림을 거세요 — **차단이 아니라 통보입니다.**

```
Cloud Console → 결제 → 예산 및 알림 → 예산 만들기
  금액 $1, 알림 임계값 50% / 90% / 100%
```

그리고 배포 3일 뒤와 다음 달 1일에 **결제 → 비용 분석**을 SKU 단위로 직접
열어보세요. 어떤 추정보다 실제 청구서 한 줄이 정확합니다.

## 갱신

```bash
git pull && gcloud run deploy aegis-sql --source . --region asia-northeast1
```

갱신할 때마다 이미지가 쌓이므로 위의 정리 정책을 꼭 걸어두세요.

## 내리기 — 채용 끝나면 즉시

```bash
gcloud run services delete aegis-sql --region asia-northeast1
gcloud artifacts repositories delete cloud-run-source-deploy --location asia-northeast1
```

컴퓨트는 요청이 없으면 0이지만 **저장 요금은 트래픽 0에서도 계속 나갑니다.**
포트폴리오를 상시 가동할 필요는 없습니다.

---

## 확인하지 못한 것

정직하게 남깁니다. 이 환경에서 `docs.cloud.google.com` 이 차단되어 아래는
1차 출처로 확인하지 못했습니다:

- Spend Cap의 정확한 동작·발동 지연·무료 체험 계정 지원 여부
- `cloud-run-source-deploy` 저장소의 기본 정리 정책 유무
- `--source` 스테이징 버킷의 정확한 수명주기 기본값
- 리전별 초당 단가 (가격 페이지가 JS로 로드해 정적 HTML에 없음)

따라서 이 문서의 **금액 추정치는 자릿수 감각 이상으로 믿지 마세요.**
확실한 것은 "한국행 egress에 무료 구간이 없다"와 "AR 저장은 매달 나간다"는
두 가지이며, 이 둘은 원문으로 확인했습니다.
