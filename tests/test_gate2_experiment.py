import torch

from experiments.same_present_history import (
    build_gate2_models,
    classify_gate2,
    shuffle_history_within_query_scale,
)
from multiple_temporal_lenses.config import canonical_short_config
from multiple_temporal_lenses.tasks import make_same_present_pairs


def test_gate2_builds_fixed_state_models_without_transformer():
    models = build_gate2_models()
    assert set(models) == {
        "single_state",
        "raw_gated",
        "residual_gated",
        "raw_concat",
        "residual_concat",
        "selective_state",
    }
    assert all(model.resident_state_scalars == 36 for model in models.values())


def test_history_shuffle_preserves_present_query_and_targets_and_stays_within_query_scale():
    cfg, _, _ = canonical_short_config()
    a, _ = make_same_present_pairs(cfg, pair_count=96, seed=3001, split="heldout")
    shuffled = shuffle_history_within_query_scale(a, seed=9001)

    assert torch.equal(shuffled.sequence[:, -1], a.sequence[:, -1])
    assert torch.equal(shuffled.query, a.query)
    assert torch.equal(shuffled.target, a.target)
    assert torch.any(shuffled.sequence[:, :-1] != a.sequence[:, :-1])

    original_scale = a.query.argmax(dim=1)
    shuffled_scale = shuffled.query.argmax(dim=1)
    assert torch.equal(original_scale, shuffled_scale)


def test_gate2_classification_keeps_absolute_accuracy_bar_and_shuffle_effect_separate():
    assert classify_gate2(0.80, 0.55)["status"] == "PASS_HISTORY_DEPENDENCE"
    assert classify_gate2(0.74, 0.20)["status"] == "FAIL_HISTORY_DEPENDENCE"
    assert classify_gate2(0.90, 0.80)["status"] == "FAIL_HISTORY_DEPENDENCE"
