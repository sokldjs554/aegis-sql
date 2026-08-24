# 통합 시 확인 (워크플로 종료 후)
- [ ] template_generator.py:922 `_top_k(...)` 미정의 → NameError. TOPN 경로(측정값만 있고 그룹 차원 없는 상위 N 질의, 예: "총가입금액 상위 5개 계약")에서 터진다.
- [ ] skeleton.py:134 mypy 우회(`if False else`) 를 깔끔한 형태로 정리
- [ ] configs/default.yaml 의 retrieval 튜닝값 재확인
- [ ] docs/ARCHITECTURE.md 의 토큰 절감 수치를 실측값(3.0~6.3x)으로 교체
