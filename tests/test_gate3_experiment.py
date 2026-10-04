import torch

from experiments.later_context_bends_meaning import (
    build_gate3_models,
    classify_gate3,
    make_supervised_later_context_batch,
)
from multiple_temporal_lenses.config import canonical_short_config


def test_gate3_builds_same_fixed_state_comparison_family():
    models = build_gate3_models()
    assert set(models) == {
        "single_state",
        "raw_gated",
        "residual_gated",
        "raw_concat",
        "residual_concat",
        "selective_state",
    }
    assert all(model.resident_state_scalars == 36 for model in models.values())


def test_two_probe_training_batch_is_one_batched_forward_with_equivalent_soft_targets():
    cfg, _, _ = canonical_short_config()
    batch = make_supervised_later_context_batch(cfg, batch_size=32, seed=123)
    assert batch.sequence.shape == (64, cfg.sequence_length, cfg.vocab_size)
    assert batch.query.shape == (64, 3)
    assert batch.target.shape == (64, cfg.vocab_size)
    assert torch.allclose(batch.target.sum(dim=1), torch.ones(64))
    assert torch.all((batch.target[:32] > 0).sum(dim=1) == 2)
    assert torch.all((batch.target[32:] > 0).sum(dim=1) == 1)
    assert torch.all(batch.query.argmax(dim=1) == 1)


def test_gate3_classification_requires_both_probability_revision_and_final_accuracy():
    assert classify_gate3(0.30, 0.85)["status"] == "PASS_REINTERPRETATION"
    assert classify_gate3(0.24, 0.90)["status"] == "FAIL_REINTERPRETATION"
    assert classify_gate3(0.40, 0.80)["status"] == "FAIL_REINTERPRETATION"

def test_single_state_shared_prefix_forward_matches_doubled_batch_outputs_and_gradients():
    from experiments.later_context_bends_meaning import single_state_two_probe_logits
    from multiple_temporal_lenses.models import SingleStateModel
    from multiple_temporal_lenses.train import classification_loss

    cfg, _, _ = canonical_short_config()
    source = make_supervised_later_context_batch(cfg, batch_size=8, seed=44)
    prefix = source.sequence[:8]
    full = source.sequence[8:]
    query = source.query[:8]
    target = source.target

    torch.manual_seed(9)
    standard = SingleStateModel(total_state_dim=36)
    optimized = SingleStateModel(total_state_dim=36)
    optimized.load_state_dict(standard.state_dict())

    standard_logits = standard(source.sequence, source.query).logits
    optimized_logits = single_state_two_probe_logits(optimized, prefix, full, query)
    assert torch.allclose(standard_logits, optimized_logits, atol=2e-6, rtol=1e-6)

    standard_loss = classification_loss(standard_logits, target)
    optimized_loss = classification_loss(optimized_logits, target)
    standard_loss.backward()
    optimized_loss.backward()
    for a, b in zip(standard.parameters(), optimized.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=5e-6, rtol=1e-5)

def test_single_state_shared_prefix_trainer_matches_standard_trainer_for_smoke_run():
    from experiments.later_context_bends_meaning import fit_single_state_shared_prefix
    from multiple_temporal_lenses.config import TrainConfig
    from multiple_temporal_lenses.models import SingleStateModel
    from multiple_temporal_lenses.train import fit_classifier, seed_everything

    cfg, _, _ = canonical_short_config()
    train_cfg = TrainConfig(lr=3e-3, batch_size=8, max_steps=2, patience=2)

    def factory(seed: int, split: str):
        assert split == "dev"
        return make_supervised_later_context_batch(cfg, 8, seed)

    seed_everything(77)
    standard = SingleStateModel(total_state_dim=36)
    optimized = SingleStateModel(total_state_dim=36)
    optimized.load_state_dict(standard.state_dict())

    a = fit_classifier(standard, factory, train_cfg, train_seed=77, validation_seed=88)
    b = fit_single_state_shared_prefix(optimized, cfg, train_cfg, train_seed=77, validation_seed=88)
    assert a.best_step == b.best_step
    assert abs(a.best_validation_loss - b.best_validation_loss) < 1e-6
    for key, value in standard.state_dict().items():
        assert torch.allclose(value, optimized.state_dict()[key], atol=1e-5, rtol=1e-5)
