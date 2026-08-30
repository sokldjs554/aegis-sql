# Google Cloud Run 배포

채용 담당자가 링크 하나로 웹 콘솔을 써볼 수 있게 하는 것이 목적입니다.
요청이 없을 때 인스턴스가 0으로 줄어들어 무료 한도 안에서 운영됩니다.

## 왜 Cloud Run인가

| | |
|---|---|
| 무료 한도 | 월 200만 요청 · 180,000 vCPU-초 · 360,000 GiB-초 ([공식 요금](https://cloud.google.com/run/pricing)) |
| 유휴 시 | 인스턴스 0개 → 과금 0 |
| 콜드 스타트 | 이미지가 가볍고(torch·TF 미포함) DB·프로파일이 구워져 있어 수 초 |
| 카드 | 결제 계정 등록은 필요하지만 무료 한도 안에서는 청구되지 않음 |

## 사전 준비 (최초 1회)

```bash
# gcloud CLI 설치 후
gcloud auth login
gcloud projects create aegis-sql-demo --name="AEGIS-SQL Demo"   # 이미 있으면 생략
gcloud config set project aegis-sql-demo
```

Cloud Console에서 이 프로젝트에 **결제 계정을 연결**해야 합니다
(무료 한도 안에서는 청구되지 않지만 연결 자체는 필수입니다).

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 배포

저장소 루트에서 한 줄이면 됩니다. 소스를 업로드하면 Cloud Build가
`Dockerfile`로 이미지를 만들어 배포합니다.

```bash
cd ~/Downloads/aegis-sql

gcloud run deploy aegis-sql \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 20 \
  --timeout 60 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars AEGIS_DEMO_PUBLIC=1
```

끝나면 `https://aegis-sql-xxxxx-du.a.run.app` 형태의 URL이 출력됩니다.

### 각 플래그의 이유

| 플래그 | 이유 |
|---|---|
| `--region asia-northeast3` | 서울. 국내 채용 담당자 기준 지연이 가장 낮습니다 |
| `--memory 1Gi` | Cloud Run의 파일시스템은 **인메모리**라 쓰기가 RAM을 먹습니다. 512Mi는 여유가 없습니다 |
| `--concurrency 20` | 앱의 `limit_concurrency=20`과 맞춥니다 |
| `--timeout 60` | 질의는 수백 ms에 끝납니다. 길게 잡을 이유가 없습니다 |
| `--max-instances 3` | 링크가 퍼져도 지출 상한이 걸립니다 |
| `--min-instances 0` | 유휴 시 과금 0. 대신 첫 요청에 콜드 스타트가 붙습니다 |
| `AEGIS_DEMO_PUBLIC=1` | 인증 없는 공개 URL이므로 `/v1/feedback`이 디스크에 쓰지 않습니다 |

## LLM 티어를 켤지

**켜지 마세요.** API 키를 넣지 않으면 `Tier.LLM`/`ENSEMBLE`이 파이프라인에
등록조차 되지 않아 과금이 구조적으로 0이고, template 티어만으로 콘솔은
정상 동작합니다.

굳이 켜려면 요청 본문의 `tier` 필드로 질의당 예산이 우회되므로
**프로바이더 대시보드에서 해당 키에 하드 스펜드 리밋을 거는 것**이
유일하게 믿을 수 있는 방어선입니다.

## 배포 후 확인

```bash
URL=$(gcloud run services describe aegis-sql --region asia-northeast3 --format='value(status.url)')

curl -fsS "$URL/v1/health" | python3 -m json.tool
# router_loaded 가 true 인지 반드시 확인 — false 면 콘솔이 '라우터 미적재'를 노출합니다
```

브라우저에서 `$URL`을 열고:
1. 예시 칩 `작년 하반기에…` → 결과와 SQL 탭
2. `고객 이름이랑 주민등록번호 좀 뽑아줘` → 차단
3. 상단 `목적: 마케팅` → `마케팅 수신 대상 고객 수를 성별로 알려줘` → SQL에 `MKT_AGR_YN` 주입
4. SSE 스트리밍(단계별 진행)이 실시간으로 뜨는지 — 프록시가 버퍼링하면 한 번에 뜹니다(기능은 정상)

## 갱신

```bash
git pull && gcloud run deploy aegis-sql --source . --region asia-northeast3
```

## 내리기

```bash
gcloud run services delete aegis-sql --region asia-northeast3
```
