# 자체 구현 sLLM: AegisLM-tiny

> **왜 직접 구현했는가.** `peft.get_peft_model(AutoModelForCausalLM.from_pretrained(...))`
> 두 줄은 LoRA를 이해했다는 증거가 되지 못한다. 이 디렉터리는 토크나이저·트랜스포머·LoRA·SFT·DPO를
> **PyTorch만으로 처음부터** 구현한다. 모델 다운로드가 없고, `transformers` 의존이 없고,
> 4코어 CPU에서 몇 분 안에 학습이 끝난다. 스케일업 시 그대로 확장되는 코드다.

```
src/aegis_sql/training/
├── tokenizer.py   바이트 레벨 BPE — 코퍼스에서 직접 학습
├── model.py       디코더 온리 트랜스포머 (RMSNorm · RoPE · SwiGLU · KV 캐시)
├── lora.py        LoRA 어댑터 (peft 미사용)
├── sft.py         프롬프트 마스킹 SFT 트레이너
├── dpo.py         DPO 트레이너 (참조 모델 고정)
└── infer.py       Generator 프로토콜 구현 — 파이프라인의 SLM 티어
```

---

## 1. 토크나이저 — 왜 바이트 레벨이어야 하는가

한글과 SQL이 한 문장에 섞인다.

```
계약상태코드가 '02'인 계약 수  →  SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_STAT_CD = '02'
```

문자 단위 어휘를 쓰면 한글 음절만 11,172자라 어휘가 폭발하고, 단어 단위는 미등록어에서 깨진다.
**바이트 레벨 BPE**는 256개 바이트에서 시작하므로 어떤 유니코드도 미등록어가 없고,
`decode(encode(s)) == s` 가 **항상** 성립한다. 테스트가 이 불변식을 한글·SQL·혼합 문자열로 강제한다.

학습 절차는 표준 BPE다: 인접 바이트쌍 빈도를 세고 → 최빈 쌍을 병합 → 어휘 크기까지 반복.
성능을 위해 원시 스트림이 아니라 **`tuple(bytes) → count` 딕셔너리** 위에서 병합한다.

특수 토큰으로 `<|sql|>` 구분자를 둔다 — 프롬프트와 정답 SQL의 경계이고, 손실 마스킹의 기준선이다.

---

## 2. 모델 — 2023년 이후 표준 디코더 구성

| 구성요소 | 선택 | 이유 |
|---|---|---|
| 정규화 | **RMSNorm** (pre-norm) | LayerNorm의 재중심화를 제거해 더 싸고, 학습이 안정적이다 (Zhang & Sennrich, NeurIPS 2019) |
| 위치 | **RoPE** | 절대 위치 임베딩과 달리 상대 위치가 attention 내적에 자연스럽게 들어간다. cos/sin 캐시를 미리 계산 |
| FFN | **SwiGLU** | 게이팅이 같은 파라미터 예산에서 더 낫다 (Shazeer, 2020) |
| Attention | causal mask + **KV 캐시** | 생성 시 O(n²)를 O(n)으로. `scaled_dot_product_attention` 사용 가능 시 활용 |
| 임베딩 | **tied** (입출력 공유) | 작은 모델에서 파라미터의 상당 비중이 임베딩이다 |
| 초기화 | `N(0, 0.02)`, 잔차 투영은 `1/√(2·n_layers)` 스케일 | 깊이에 따른 잔차 분산 누적 억제 |

투영 레이어 이름을 `q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj`로 둔 것은
**이름 기반 LoRA 타게팅**이 그대로 동작하게 하기 위해서다.

> **KV 캐시 정합성 테스트**: `use_cache=True`와 `False`의 greedy 출력이 **완전히 동일**해야 한다.
> 캐시 구현의 오프바이원은 이 테스트가 아니면 조용히 품질만 갉아먹는다.

---

## 3. LoRA — `B=0` 이 왜 중요한가

$$W' = W + \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times d_{in}},\; B \in \mathbb{R}^{d_{out} \times r}$$

`A`는 정규분포, **`B`는 0으로 초기화**한다. 따라서 어댑터를 붙인 직후 $BA = 0$ 이고
모델 출력은 **비트 단위로 동일**하다. 이것이 "튜닝을 시작하는 지점이 사전학습 모델과 같다"는 보증이다.

테스트가 이를 강제한다.

```python
before = model(ids)["logits"].clone()
apply_lora(model, ["q_proj", "v_proj", "o_proj"], r=16, alpha=32)
assert torch.allclose(before, model(ids)["logits"], atol=1e-6)   # B=0 보증
trainable, total = mark_only_lora_trainable(model)
assert trainable / total < 0.05                                   # 학습 파라미터 5% 미만
```

`merge_lora_()`는 $W \leftarrow W + \frac{\alpha}{r}BA$ 로 병합해 추론 시 오버헤드를 0으로 만든다.

---

## 4. SFT — 손실을 SQL에만 준다

가장 흔한 실수는 프롬프트 토큰에도 손실을 주는 것이다. 그러면 모델이 **스키마 카드를 외운다.**

```
[프롬프트: 질문 + 스키마 카드]  <|sql|>  [정답 SQL]  <eos>
 labels = -100 ...................        실제 토큰 ....
```

- 길이 초과 시 **프롬프트의 앞쪽을 자른다**. 질문과 스키마 카드의 뒷부분(가장 관련 높은 테이블)을 살리기 위해서다.
- AdamW(β=0.9/0.95), linear warmup + cosine decay, grad clipping 1.0, grad accumulation.
- 매 에폭 dev loss와 **토큰 정확도**를 기록하고 best 체크포인트만 저장한다.
- AMP는 CUDA에서만 켠다 — CPU에서 bf16 autocast는 이 크기에서 이득이 없다.

스모크 테스트는 **32개 예제를 의도적으로 과적합**시켜 손실이 떨어지는지 확인한다.
"학습 코드가 있다"가 아니라 "손실이 실제로 내려간다 = 마스킹·옵티마이저·데이터로더가 전부 맞다"를 증명한다.

---

## 5. DPO — 선호쌍을 엔진이 스스로 만든다

$$\mathcal{L} = -\log\sigma\Big(\beta\big[(\log\pi_\theta(y_w|x) - \log\pi_{ref}(y_w|x)) - (\log\pi_\theta(y_l|x) - \log\pi_{ref}(y_l|x))\big]\Big)$$

핵심은 **어디서 $(y_w, y_l)$ 를 얻느냐**다. 보통은 사람이 라벨링한다. 여기서는 그럴 필요가 없다.

```
y_w (chosen)   = 검증을 통과해 실행된 gold SQL
y_l (rejected) = 자가교정 로그에 남은, 실행에 실패했던 SQL
```

즉 **엔진이 운영 중에 만든 실패 기록이 그대로 선호 데이터가 된다** (`pairs_from_repair_log`).
사람 라벨링 0건으로 "예전에 틀렸던 패턴을 덜 생성하도록" 정렬된다. 플라이휠이 도는 지점이다.

구현 시 주의한 것:
- 로그확률은 **정답 구간 토큰에 대해서만** 합산한다 (프롬프트 구간 제외).
- 참조 모델은 `eval()` + `torch.no_grad()`. 실수로 학습되면 DPO가 무의미해진다.
- **implicit reward margin**과 chosen-preferred accuracy를 매 스텝 기록한다.
  손실만 보면 학습이 되는지 알기 어렵다. 마진이 올라가야 한다.

---

## 6. 실측 학습 곡선 — 그리고 그것이 말해 주는 것

플라이휠 코퍼스 9,000쌍, 5.5M 파라미터, CPU 4코어:

| epoch | train loss | dev loss | dev token acc |
|---:|---:|---:|---:|
| 1 | 2.295 | **2.487** | 0.648 |
| 2 | 0.269 | 2.878 | 0.660 |
| 3 | 0.177 | 2.688 | 0.680 |

**dev loss는 1에폭에서 바닥을 치는데 dev token accuracy는 계속 오릅니다.**
전형적인 신호입니다 — 손실은 소수의 어려운 토큰(리터럴, 희귀 컬럼명)이 지배하고,
그 토큰들에서 모델이 **과신**하기 시작하면서 손실이 커지는 동안에도
구조 토큰(SELECT/FROM/GROUP BY/별칭)의 예측은 계속 좋아집니다.

체크포인트 선택 기준은 **dev loss**입니다. 보수적인 선택이고, 그래서 저장된 가중치는
1에폭 것입니다. 생성 품질만 보면 token accuracy 기준이 더 나을 수 있지만,
"손실이 오르는데 저장한다"는 기준은 조용히 과적합을 통과시키기 쉬워
기본값으로 두지 않았습니다. `training_report.json` 에 두 곡선이 모두 남으므로
판단은 사후에 다시 할 수 있습니다.

이 규모에서 얻은 결론도 정직하게 적습니다: **이 모델은 SQL의 문법과 이 스키마의
테이블 이름을 배우지만, 질문의 의미를 SQL 조건으로 옮기는 데는 아직 이르다.**
9,000쌍 · 5.5M 파라미터 · CPU 25분이 만들 수 있는 것의 한계이고,
그래서 캐스케이드가 존재합니다.

## 7. 이 모델의 위치 — 정직하게

AegisLM-tiny는 프론티어 모델을 대체하지 않는다. **캐스케이드의 아래층**이다.

| | template | **sLLM** | LLM |
|---|---|---|---|
| 파라미터 | — | 약 10~20M (설정에 따라) | 수천억 |
| 한계 비용 | 0 | 0 (자체 호스팅) | 토큰당 과금 |
| 지연 | ~3ms | ~수십 ms (CPU) | 수백 ms ~ 수 초 |
| 강점 | 정형 패턴 | 학습 분포 내 질의 | 처음 보는 복잡 질의 |

라우터가 "이 질문은 아래층이 맞힐 것"이라고 판단할 때만 sLLM으로 보낸다.
즉 sLLM의 가치는 **절대 정확도가 아니라 "라우터가 신뢰할 수 있을 만큼 예측 가능한가"** 다.

10B급 모델로 확장할 때 바뀌는 것은 `AegisLMConfig`의 숫자와 체크포인트 로딩 경로뿐이며,
LoRA 타게팅·손실 마스킹·DPO 선호쌍 생성·서빙 어댑터는 그대로 재사용된다.

---

## 재현

```bash
make flywheel                 # 스키마 → 학습 데이터
make train-slm                # 토크나이저 학습 → SFT (+LoRA) → (옵션) DPO
cat data/generated/slm/training_report.json
pytest -m slow tests/test_training.py   # 실제로 학습되는지 검증
```
