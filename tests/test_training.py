"""The in-house sLLM: byte-BPE, the PyTorch Transformer, LoRA, SFT and DPO.

These tests are marked slow because they actually train.  That is the point —
the claim is not "there is a training script" but "this trains, on CPU, from
nothing, and the loss goes down".
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


KOREAN_SQL_CORPUS = [
    "계약상태코드가 '02'인 계약 수를 알려줘",
    "SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_STAT_CD = '02'",
    "지점별 신계약 건수 상위 5개",
    "SELECT b.BRCH_NM, COUNT(*) FROM TB_CTRT t JOIN TB_AGNT a ON a.AGNT_ID = t.AGNT_ID GROUP BY b.BRCH_NM",
    "작년 하반기 계약 중 월납보험료가 20만원 이상인 건수",
] * 400


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    from aegis_sql.training.tokenizer import ByteBPETokenizer

    return ByteBPETokenizer.train(KOREAN_SQL_CORPUS, vocab_size=1200, special_tokens=[])


def test_tokenizer_roundtrips_korean_and_sql(tokenizer):
    for text in [
        "계약상태코드가 '02'인 계약 수",
        "SELECT substr(CTRT_DT,1,6) AS ym, COUNT(*)\nFROM TB_CTRT\nGROUP BY ym",
        "혼합 text 123 ~!@#$ 한글",
    ]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_persists(tokenizer, tmp_path):
    from aegis_sql.training.tokenizer import ByteBPETokenizer

    tokenizer.save(tmp_path / "tok.json")
    loaded = ByteBPETokenizer.load(tmp_path / "tok.json")
    text = "실효된 계약의 채널별 비중"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.vocab_size == tokenizer.vocab_size


def _tiny_model(vocab_size):
    from aegis_sql.training.model import AegisLM, AegisLMConfig

    cfg = AegisLMConfig(vocab_size=vocab_size, d_model=96, n_layers=3, n_heads=4,
                        d_ff=256, max_seq_len=128, dropout=0.0)
    return AegisLM(cfg)


def test_model_forward_and_loss(tokenizer):
    model = _tiny_model(tokenizer.vocab_size)
    ids = torch.tensor([tokenizer.encode("SELECT COUNT(*) FROM TB_CTRT")[:64]])
    out = model(ids, labels=ids)
    assert out["logits"].shape[:2] == ids.shape
    assert out["loss"].item() > 0
    assert model.num_parameters() > 10_000


def test_generation_with_kv_cache_matches_without(tokenizer):
    torch.manual_seed(0)
    model = _tiny_model(tokenizer.vocab_size).eval()
    ids = torch.tensor([tokenizer.encode("SELECT")[:8]])
    a = model.generate(ids, max_new_tokens=12, temperature=0.0, use_cache=True)
    b = model.generate(ids, max_new_tokens=12, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), "KV cache changed greedy output"


def test_lora_is_identity_at_init_and_tiny(tokenizer):
    from aegis_sql.training.lora import apply_lora, mark_only_lora_trainable

    torch.manual_seed(0)
    model = _tiny_model(tokenizer.vocab_size).eval()
    ids = torch.tensor([tokenizer.encode("SELECT COUNT(*) FROM TB_CTRT")[:32]])
    with torch.no_grad():
        before = model(ids)["logits"].clone()

    wrapped = apply_lora(model, ["q_proj", "v_proj", "o_proj"], r=8, alpha=16, dropout=0.0)
    assert wrapped > 0
    with torch.no_grad():
        after = model(ids)["logits"]
    assert torch.allclose(before, after, atol=1e-6), "B must be zero-initialised"

    trainable, total = mark_only_lora_trainable(model)
    assert trainable / total < 0.05, f"LoRA trainable share too high: {trainable / total:.1%}"


@pytest.mark.slow
def test_sft_overfits_a_tiny_batch(tokenizer, tmp_path, settings):
    from aegis_sql.training.sft import SFTExample, SFTTrainer, build_prompt

    model = _tiny_model(tokenizer.vocab_size)
    examples = [
        SFTExample(
            prompt=build_prompt("실효된 계약 수", "TB_CTRT(CTRT_NO,CTRT_STAT_CD)"),
            target="SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_STAT_CD = '02'",
        )
    ] * 24
    cfg = settings.training.model_copy(update={
        "batch_size": 4, "epochs": 6, "lr": 3e-3, "max_seq_len": 128, "grad_accum": 1,
    })
    history = SFTTrainer(model, tokenizer, cfg, device="cpu").train(examples, output_dir=tmp_path)
    assert history["train_loss"][0] > history["train_loss"][-1], history["train_loss"]
    assert history["train_loss"][-1] < history["train_loss"][0] * 0.6


@pytest.mark.slow
def test_dpo_increases_reward_margin(tokenizer, settings):
    import copy

    from aegis_sql.training.dpo import DPOExample, DPOTrainer

    torch.manual_seed(0)
    policy = _tiny_model(tokenizer.vocab_size)
    reference = copy.deepcopy(policy)
    examples = [
        DPOExample(
            prompt="질문: 실효 계약 수\nSQL:",
            chosen=" SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_STAT_CD = '02'",
            rejected=" SELECT COUNT(*) FROM TB_CTRT WHERE CTRT_STAT_CD = '실효'",
        )
    ] * 8
    cfg = settings.training.model_copy(update={"batch_size": 2, "epochs": 4, "lr": 1e-3,
                                               "max_seq_len": 96, "grad_accum": 1})
    history = DPOTrainer(policy, reference, tokenizer, cfg, beta=0.1, device="cpu").train(examples)
    # The implicit reward margin is what actually has to move: the DPO loss can
    # fall while the policy stays indifferent between the two completions.
    assert history["margin_end"] > history["margin_start"], history
    assert history["final_acc"] >= 0.5
    assert history["margin"][-1] > history["margin"][0]
