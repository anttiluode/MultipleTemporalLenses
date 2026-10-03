import math

import pytest
import torch
from torch import nn

from multiple_temporal_lenses.baselines import SelectiveStateBaseline, TinyCausalTransformer
from multiple_temporal_lenses.metrics import (
    count_parameters,
    explicit_history_scalar_proxy,
    lens_entropy,
    resident_state_scalars,
)
from multiple_temporal_lenses.models import QueryGatedLensModel, SingleStateModel


def test_count_parameters_matches_known_linear_module():
    layer = nn.Linear(3, 2, bias=True)
    assert count_parameters(layer) == 8


def test_resident_state_accounting_is_live_and_transformer_is_not_mislabeled_fixed_state():
    assert resident_state_scalars(SingleStateModel()) == 36
    assert resident_state_scalars(QueryGatedLensModel()) == 36
    assert resident_state_scalars(SelectiveStateBaseline(state_dim=36)) == 36
    assert resident_state_scalars(TinyCausalTransformer()) is None


def test_transformer_explicit_history_proxy_grows_with_sequence_length():
    model = TinyCausalTransformer(d_model=12, nhead=3, num_layers=1, ff_dim=24)
    short = explicit_history_scalar_proxy(model, 72)
    long = explicit_history_scalar_proxy(model, 208)

    assert short == 2 * (72 + 1) * 12
    assert long == 2 * (208 + 1) * 12
    assert long > short
    assert explicit_history_scalar_proxy(SelectiveStateBaseline(), 72) is None


def test_lens_entropy_handles_one_hot_and_uniform_distributions():
    one_hot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    uniform = torch.full((2, 3), 1.0 / 3.0)

    assert torch.allclose(lens_entropy(one_hot), torch.zeros(2), atol=1e-7)
    assert torch.allclose(
        lens_entropy(uniform),
        torch.full((2,), math.log(3.0)),
        atol=1e-6,
    )


def test_lens_entropy_rejects_nonprobability_rows():
    with pytest.raises(ValueError, match="probabilities"):
        lens_entropy(torch.tensor([[0.6, 0.6, -0.2]]))
