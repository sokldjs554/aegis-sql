# 사전 학습된 캐스케이드 라우터 (서빙용)

여기 있는 파일은 **numpy 추론 전용 가중치**다. TensorFlow 없이 로드된다.

| 파일 | 내용 |
|---|---|
| `router_weights.npz` | Normalization mean/var + Dense 커널·바이어스 (15 KB) |
| `router_meta.json` | 특징 순서, 임계값, 학습 지표 |
| `calibrator.json` | temperature scaling 파라미터 |

`data/generated/router/` 에 새로 학습한 라우터가 있으면 그쪽이 우선한다.
재학습:

```bash
make eval        # 평가 로그에서 (features, label) 수집
python scripts/train_router.py --data data/generated/router/routing_train.jsonl
```

라벨은 **관측값**이다: 값싼 티어가 그 질문을 틀렸으면 1, 맞혔으면 0.
