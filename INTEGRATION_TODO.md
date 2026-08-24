# 통합 체크리스트

- [x] template_generator.py `_top_k` 미정의 → 에이전트가 정의 추가, TOPN 경로 정상
- [x] skeleton.py mypy 우회 제거 (에이전트 최종본으로 대체)
- [x] docs/ARCHITECTURE.md 토큰 절감 수치를 실측 범위로 교체
- [x] README "실제 출력" 블록을 실제 캡처로 교체
- [x] fk_expand_hops 0 으로 확정 (어블레이션 Δ 0.0%p, 토큰 5.7x)
- [x] ruff / mypy 클린, 232 테스트 통과
- [ ] README `<!-- RESULTS -->` 구간 → sLLM 학습 완료 후 최종 어블레이션 표로 교체
- [ ] sLLM 티어 평가 수치 확보 후 README 티어 비교표 작성
