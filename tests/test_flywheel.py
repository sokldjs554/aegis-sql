"""The data flywheel: schema-grounded sampling, back-translation, augmentation,
execution-based filtering and leakage-free splitting."""

from __future__ import annotations

import pytest

from aegis_sql.generation.skeleton import sql_skeleton


@pytest.fixture(scope="module")
def sampler(schema, profile, join_graph, glossary):
    from aegis_sql.flywheel.sql_sampler import SQLSampler

    return SQLSampler(schema, profile, join_graph, glossary, seed=20260824)


def test_sampled_sql_executes(sampler, executor):
    programs = sampler.sample(60, {"easy": 0.4, "medium": 0.4, "hard": 0.2})
    assert len(programs) >= 40
    failures = [(p.template_id, executor.execute(p.sql).error) for p in programs
                if not executor.execute(p.sql).ok]
    ratio = len(failures) / len(programs)
    assert ratio < 0.1, f"{ratio:.0%} of sampled SQL failed: {failures[:3]}"


def test_sampler_is_deterministic(schema, profile, join_graph, glossary):
    from aegis_sql.flywheel.sql_sampler import SQLSampler

    a = SQLSampler(schema, profile, join_graph, glossary, seed=7).sample(20, {"easy": 1.0})
    b = SQLSampler(schema, profile, join_graph, glossary, seed=7).sample(20, {"easy": 1.0})
    assert [p.sql for p in a] == [p.sql for p in b]


def test_sampler_covers_all_difficulties(sampler):
    programs = sampler.sample(120, {"easy": 0.34, "medium": 0.33, "hard": 0.33})
    assert {p.difficulty for p in programs} == {"easy", "medium", "hard"}
    assert len({p.template_id for p in programs}) >= 10


def test_back_translation_avoids_physical_names(sampler, schema, profile, glossary):
    from aegis_sql.flywheel.back_translate import BackTranslator

    bt = BackTranslator(schema, profile, glossary, llm=None, seed=1)
    for program in sampler.sample(25, {"easy": 0.5, "medium": 0.5}):
        question = bt.translate(program)
        assert question and len(question) > 5
        assert "TB_" not in question, question
        assert "_CD" not in question, question


def test_augmentation_preserves_numbers_and_dates(glossary):
    from aegis_sql.flywheel.augment import KoreanAugmenter

    aug = KoreanAugmenter(glossary, seed=3)
    original = "2025년 하반기에 체결된 계약 중 월납보험료가 20만원 이상인 건수를 알려줘"
    variants = aug.augment(original, 5)
    assert 1 <= len(variants) <= 5
    for v in variants:
        assert "2025" in v and "20만원" in v, v
        assert v != original


def test_augmentation_is_deterministic(glossary):
    from aegis_sql.flywheel.augment import KoreanAugmenter

    a = KoreanAugmenter(glossary, seed=11).augment("실효된 계약 건수를 알려줘", 4)
    b = KoreanAugmenter(glossary, seed=11).augment("실효된 계약 건수를 알려줘", 4)
    assert a == b


@pytest.mark.slow
def test_build_dataset_produces_leakage_free_splits(settings, tmp_path):
    from aegis_sql.flywheel.build_dataset import build, load_split

    stats = build(
        settings.model_copy(update={"flywheel": settings.flywheel.model_copy(
            update={"output_dir": str(tmp_path)})}),
        n_programs=200, augment_per_example=2, progress=False,
    )
    assert stats["counts"]["splits"]["train"] > 0 and stats["counts"]["splits"]["test"] > 0
    assert stats["leakage"]["train_test_overlap"] == 0

    train = load_split(tmp_path / "train.jsonl")
    test = load_split(tmp_path / "test.jsonl")
    train_skeletons = {sql_skeleton(p.sql) for p in train}
    test_skeletons = {sql_skeleton(p.sql) for p in test}
    assert not (train_skeletons & test_skeletons), "SQL skeleton leaked across splits"


@pytest.mark.slow
def test_every_generated_pair_executes(settings, tmp_path, executor):
    from aegis_sql.flywheel.build_dataset import build, load_split

    build(
        settings.model_copy(update={"flywheel": settings.flywheel.model_copy(
            update={"output_dir": str(tmp_path)})}),
        n_programs=120, augment_per_example=1, progress=False,
    )
    for pair in load_split(tmp_path / "train.jsonl")[:60]:
        assert executor.execute(pair.sql).ok, pair.sql
