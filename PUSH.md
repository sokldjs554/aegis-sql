# 푸시 안내 (저장소 생성 후)

이 세션은 `sokldjs554/Titanic-project` 에만 바인딩되어 있어 새 저장소를 **생성**할 수 없습니다.
아래 한 줄만 해주시면 나머지는 자동입니다.

## 1) GitHub에서 빈 저장소 만들기

https://github.com/new
- Repository name: **`aegis-sql`**
- Public
- **Add a README / .gitignore / license 전부 체크 해제** (완전히 빈 저장소여야 합니다)

## 2) 알려주시면 이 세션에서 푸시합니다

또는 직접 푸시하실 경우:

```bash
cd aegis-sql
git remote add origin https://github.com/sokldjs554/aegis-sql.git
git push -u origin main
```

## 현재 상태

- 커밋 14개, 추적 파일 119개 (1.5 MB)
- 브랜치: `main`
- 생성물(`data/generated/`, `.venv/`, `reports/*.md`)은 `.gitignore` 처리됨
- 예외: `models/router/` — 서빙용 numpy 라우터 가중치 17 KB (클론 직후 `make demo` 에서 실제 라우팅이 보이도록)
