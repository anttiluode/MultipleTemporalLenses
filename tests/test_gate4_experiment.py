import math
import torch

from experiments.explicit_history_controls import (
    build_stress_models,
    classify_routing_vs_selective,
    summarize_accuracy_by_scale,
)
from multiple_temporal_lenses.config import canonical_long_config
from multiple_temporal_lenses.metrics import explicit_history_scalar_proxy


def test_long_stress_lens_models_use_preregistered_2_20_200_timescales():
    bench, _, _ = canonical_long_config()
    models = build_stress_models(bench)
    assert set(models) == {"residual_gated", "raw_concat", "selective_state", "tiny_transformer"}
    expected = torch.tensor([math.exp(-1 / 2), math.exp(-1 / 20), math.exp(-1 / 200)])
    assert torch.allclose(models["residual_gated"].lens_bank.decays, expected, atol=1e-7)
    assert torch.allclose(models["raw_concat"].lens_bank.decays, expected, atol=1e-7)
    assert models["residual_gated"].resident_state_scalars == 36
    assert models["raw_concat"].resident_state_scalars == 36
    assert models["selective_state"].resident_state_scalars == 36


def test_transformer_history_proxy_reports_sequence_growth_not_fixed_state_match():
    bench, _, _ = canonical_long_config()
    transformer = build_stress_models(bench)["tiny_transformer"]
    assert explicit_history_scalar_proxy(transformer, 72) == 1752
    assert explicit_history_scalar_proxy(transformer, 208) == 5016


def test_selective_state_matching_gated_within_one_standard_error_narrows_routing_claim():
    out = classify_routing_vs_selective(
        gated_per_seed=[0.40, 0.42, 0.41, 0.43, 0.44],
        selective_per_seed=[0.39, 0.42, 0.42, 0.43, 0.43],
    )
    assert out["status"] == "SELECTIVE_STATE_EXPLAINS_RESULT"


def test_accuracy_by_scale_keeps_each_query_scale_separate():
    predictions = torch.tensor([1, 2, 3, 4, 0, 1])
    targets = torch.tensor([1, 0, 3, 4, 2, 1])
    scales = torch.tensor([0, 0, 1, 1, 2, 2])
    result = summarize_accuracy_by_scale(predictions, targets, scales, num_scales=3)
    assert result == {"0": 0.5, "1": 1.0, "2": 0.5}

def test_chunked_training_resume_matches_uninterrupted_training(tmp_path):
    from experiments.explicit_history_controls import run_training_chunk
    from multiple_temporal_lenses.config import TrainConfig, canonical_short_config
    from multiple_temporal_lenses.models import QueryGatedLensModel
    from multiple_temporal_lenses.tasks import make_query_timescale_batch
    from multiple_temporal_lenses.train import fit_classifier, seed_everything

    bench, _, _ = canonical_short_config()
    cfg = TrainConfig(lr=3e-3, batch_size=16, max_steps=6, patience=20)

    def factory(seed: int, split: str):
        return make_query_timescale_batch(bench, 16, seed, split)

    seed_everything(123)
    standard = QueryGatedLensModel(coordinate_mode="band")
    standard_result = fit_classifier(standard, factory, cfg, 123, 456)

    seed_everything(123)
    first = QueryGatedLensModel(coordinate_mode="band")
    ckpt = tmp_path / "resume.pt"
    partial = run_training_chunk(first, factory, cfg, 123, 456, ckpt, max_new_steps=3)
    assert partial["done"] is False

    resumed = QueryGatedLensModel(coordinate_mode="band")
    final = run_training_chunk(resumed, factory, cfg, 123, 456, ckpt, max_new_steps=3)
    assert final["done"] is True
    result = final["result"]
    assert result.train_losses == standard_result.train_losses
    assert result.validation_losses == standard_result.validation_losses
    assert result.best_step == standard_result.best_step
    for key, value in standard.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])

def test_chunked_training_does_not_advance_after_early_stop(tmp_path):
    from torch import nn
    from multiple_temporal_lenses.config import TrainConfig, canonical_short_config
    from multiple_temporal_lenses.models import ModelOutput
    from multiple_temporal_lenses.tasks import make_query_timescale_batch
    from experiments.explicit_history_controls import run_training_chunk

    class FlatModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.tensor(0.0))
        def forward(self, sequence, query):
            logits = torch.zeros(sequence.shape[0], 8) + self.dummy * 0.0
            return ModelOutput(logits=logits, diagnostics={})

    bench, _, _ = canonical_short_config()
    cfg = TrainConfig(lr=1e-3, batch_size=8, max_steps=20, patience=2)
    def factory(seed, split):
        return make_query_timescale_batch(bench, 8, seed, "dev")

    ckpt = tmp_path / "early.pt"
    model = FlatModel()
    first = run_training_chunk(model, factory, cfg, 1, 2, ckpt, max_new_steps=10)
    assert first["done"] is True
    first_next = first["next_step"]
    first_len = len(first["result"].train_losses)

    model2 = FlatModel()
    second = run_training_chunk(model2, factory, cfg, 1, 2, ckpt, max_new_steps=10)
    assert second["done"] is True
    assert second["next_step"] == first_next
    assert len(second["result"].train_losses) == first_len
