import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from multiple_temporal_lenses.config import TrainConfig, canonical_short_config
from multiple_temporal_lenses.models import ModelOutput, QueryGatedLensModel
from multiple_temporal_lenses.tasks import make_query_timescale_batch
from multiple_temporal_lenses.train import (
    FROZEN_SEEDS,
    classification_loss,
    evaluate_classifier,
    fit_classifier,
    model_metadata,
    seed_everything,
    write_receipt,
)


def _dev_factory(batch_size=32):
    bench, _, _ = canonical_short_config()

    def make(seed: int, split: str = "dev"):
        assert split == "dev"
        return make_query_timescale_batch(bench, batch_size, seed, split)

    return make


def _trained(seed: int):
    seed_everything(seed)
    model = QueryGatedLensModel(coordinate_mode="band")
    cfg = TrainConfig(lr=3e-3, batch_size=32, max_steps=20, patience=20)
    result = fit_classifier(
        model,
        _dev_factory(32),
        cfg,
        train_seed=seed,
        validation_seed=1999,
    )
    batch = _dev_factory(16)(1777, "dev")
    with torch.no_grad():
        logits = model(batch.sequence, batch.query).logits.clone()
    params = [parameter.detach().clone() for parameter in model.parameters()]
    return result, logits, params


def test_training_is_deterministic_for_same_seed_and_changes_with_seed():
    first, logits_first, params_first = _trained(123)
    second, logits_second, _ = _trained(123)

    assert first.train_losses == second.train_losses
    assert first.validation_losses == second.validation_losses
    assert first.best_step == second.best_step
    assert torch.equal(logits_first, logits_second)

    _, _, params_other_seed = _trained(124)
    assert any(
        not torch.equal(a, b) for a, b in zip(params_first, params_other_seed)
    )


def test_classification_loss_accepts_hard_and_soft_targets():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    hard = torch.tensor([0, 1])
    soft = torch.tensor([[0.5, 0.5], [0.25, 0.75]])

    hard_loss = classification_loss(logits, hard)
    soft_loss = classification_loss(logits, soft)

    assert hard_loss.ndim == 0
    assert soft_loss.ndim == 0
    assert hard_loss > 0
    assert soft_loss > 0


class ConstantLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))
        self.resident_state_scalars = 1

    def forward(self, sequence, query):
        logits = torch.zeros(sequence.shape[0], 8, device=sequence.device)
        logits = logits + self.dummy * 0.0
        return ModelOutput(logits=logits, diagnostics={})


def test_early_stopping_triggers_on_constant_validation_loss():
    model = ConstantLossModel()
    cfg = TrainConfig(lr=1e-3, batch_size=16, max_steps=30, patience=3)
    result = fit_classifier(
        model,
        _dev_factory(16),
        cfg,
        train_seed=1,
        validation_seed=2,
    )

    assert result.stopped_early is True
    assert result.best_step == 0
    assert len(result.train_losses) == 4
    assert len(result.validation_losses) == 4


@dataclass(frozen=True)
class _DummyBatch:
    sequence: torch.Tensor
    query: torch.Tensor
    target: torch.Tensor


def _frozen_contract_factory(calls: list[tuple[int, str]]):
    def make(seed: int, split: str):
        calls.append((seed, split))
        # Deliberately unrelated to the actual benchmark. This tests the frozen
        # seed/split contract without inspecting held-out benchmark outcomes.
        target = torch.tensor([seed % 8], dtype=torch.long)
        sequence = torch.zeros(1, 2, 8)
        query = torch.tensor([[1.0, 0.0, 0.0]])
        return _DummyBatch(sequence=sequence, query=query, target=target)

    return make


def test_frozen_evaluation_requires_exact_seeds_and_uses_heldout_split():
    calls: list[tuple[int, str]] = []
    model = ConstantLossModel()

    result = evaluate_classifier(
        model,
        _frozen_contract_factory(calls),
        seeds=FROZEN_SEEDS,
        kind="frozen",
    )

    assert [row["seed"] for row in result.per_seed] == list(FROZEN_SEEDS)
    assert calls == [(seed, "heldout") for seed in FROZEN_SEEDS]

    with pytest.raises(ValueError, match="exactly"):
        evaluate_classifier(
            model,
            _frozen_contract_factory([]),
            seeds=[3001],
            kind="frozen",
        )
    with pytest.raises(ValueError, match="exactly"):
        evaluate_classifier(
            model,
            _frozen_contract_factory([]),
            seeds=[2999, 3001, 3002, 3003, 3004],
            kind="frozen",
        )


def test_stress_evaluation_is_explicit_exception_to_frozen_seed_tuple():
    calls: list[tuple[int, str]] = []
    model = ConstantLossModel()

    result = evaluate_classifier(
        model,
        _frozen_contract_factory(calls),
        seeds=[4001, 4002],
        kind="stress",
    )

    assert len(result.per_seed) == 2
    assert calls == [(4001, "stress"), (4002, "stress")]


def test_receipt_seed_protection_metadata_and_hashes(tmp_path: Path):
    model = QueryGatedLensModel(coordinate_mode="band")
    metadata = model_metadata(model, sequence_length=72)
    common = dict(
        experiment="gate",
        config={"x": 1},
        metrics={"accuracy": 0.5},
        model_metadata=metadata,
    )

    development_path = tmp_path / "dev.json"
    write_receipt(
        development_path,
        seeds=[2999],
        kind="development",
        **common,
    )
    assert development_path.exists()

    with pytest.raises(ValueError, match="exactly"):
        write_receipt(
            tmp_path / "bad.json",
            seeds=[2999],
            kind="frozen",
            **common,
        )
    with pytest.raises(ValueError, match="below 3000"):
        write_receipt(
            tmp_path / "bad-dev.json",
            seeds=[3001],
            kind="development",
            **common,
        )

    frozen_path = tmp_path / "frozen.json"
    write_receipt(
        frozen_path,
        seeds=FROZEN_SEEDS,
        kind="frozen",
        **common,
    )
    data = json.loads(frozen_path.read_text(encoding="utf-8"))

    assert data["seeds"] == list(FROZEN_SEEDS)
    assert len(data["config_hash"]) == 64
    assert len(data["seeds_hash"]) == 64
    assert data["model_metadata"]["class"] == "QueryGatedLensModel"
    assert data["model_metadata"]["resident_state_scalars"] == 36
    assert data["model_metadata"]["parameter_count"] > 0
    assert data["python_version"]
    assert data["torch_version"]
    assert data["timestamp_utc"].endswith("+00:00")
