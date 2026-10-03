import math

import pytest
import torch

from multiple_temporal_lenses.lenses import (
    LeakyLensBank,
    decays_from_timescales,
    from_band_coordinates,
    impulse_kernel,
    to_coordinates,
)


def test_decays_follow_exp_minus_inverse_timescale_and_zero_limit():
    decays = decays_from_timescales([0.0, 2.0, 16.0])

    assert decays.shape == (3,)
    assert decays[0].item() == pytest.approx(0.0)
    assert decays[1].item() == pytest.approx(math.exp(-0.5))
    assert decays[2].item() == pytest.approx(math.exp(-1.0 / 16.0))


def test_two_step_recurrence_matches_hand_computation():
    bank = LeakyLensBank(state_dim=1, timescales=[2.0])
    sequence = torch.tensor([[[1.0], [0.5]]])

    state = bank.run(sequence)
    rho = math.exp(-0.5)
    after_first = (1.0 - rho) * 1.0
    expected = rho * after_first + (1.0 - rho) * 0.5

    assert state.shape == (1, 1, 1)
    assert state.item() == pytest.approx(expected)


def test_zero_timescale_tracks_current_event_exactly():
    bank = LeakyLensBank(state_dim=2, timescales=[0.0])
    sequence = torch.tensor([[[1.0, 0.0], [0.25, 0.75]]])

    state = bank.run(sequence)

    assert torch.equal(state[:, 0], sequence[:, -1])


def test_slower_lens_retains_impulse_longer_and_budget_is_exact():
    bank = LeakyLensBank(state_dim=12, timescales=[2.0, 16.0, 64.0])
    sequence = torch.zeros(1, 20, 12)
    sequence[:, 0, 0] = 1.0

    initial = bank.run(sequence[:, :1])
    state = bank.run(sequence)
    retention = state[0, :, 0] / initial[0, :, 0]

    assert bank.resident_state_scalars == 36
    assert 0.0 < retention[0] < retention[1] < retention[2] < 1.0


def test_band_coordinates_are_invertible_and_raw_is_identity():
    states = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0], [8.0, 13.0]],
            [[-1.0, 4.0], [2.0, 1.0], [2.5, -3.0]],
        ]
    )

    raw = to_coordinates(states, "raw")
    bands = to_coordinates(states, "band")
    reconstructed = from_band_coordinates(bands)

    assert torch.equal(raw, states)
    assert torch.allclose(reconstructed, states, atol=1e-6, rtol=0.0)


def test_invalid_timescale_and_shapes_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        decays_from_timescales([2.0, -1.0])
    with pytest.raises(ValueError, match="state_dim"):
        LeakyLensBank(state_dim=0, timescales=[2.0])

    bank = LeakyLensBank(state_dim=2, timescales=[2.0, 16.0])
    with pytest.raises(ValueError, match="shape"):
        bank.run(torch.zeros(3, 5))
    with pytest.raises(ValueError, match="state_dim"):
        bank.run(torch.zeros(3, 5, 4))


def test_band_coordinates_are_fast_minus_slow_residuals_with_slowest_tail():
    states = torch.tensor([[[10.0], [6.0], [2.0]]])
    bands = to_coordinates(states, "band")

    assert torch.equal(bands, torch.tensor([[[4.0], [4.0], [2.0]]]))
    assert torch.equal(from_band_coordinates(bands), states)


def test_impulse_kernel_matches_leaky_formula_and_exposes_signed_residual_bands():
    decays = torch.tensor([0.5, 0.95, 0.995])
    lags = [0, 2, 20, 200]
    raw = impulse_kernel(decays, lags, mode="raw")
    band = impulse_kernel(decays, lags, mode="band")

    expected_raw = torch.stack(
        [(1.0 - decays) * decays.pow(lag) for lag in lags], dim=0
    )
    assert torch.allclose(raw, expected_raw, atol=1e-7, rtol=0.0)

    expected_band = torch.stack(
        [raw[:, 0] - raw[:, 1], raw[:, 1] - raw[:, 2], raw[:, 2]], dim=1
    )
    assert torch.allclose(band, expected_band, atol=1e-7, rtol=0.0)

    assert torch.all(raw[0] > raw[-1])
    assert band[0, 0] > 0 and band[-1, 0] < 0
    assert band[0, 1] > 0 and band[-1, 1] < 0
