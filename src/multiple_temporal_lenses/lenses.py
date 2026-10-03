"""Fixed multiscale resident temporal states and coordinate transforms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn


def decays_from_timescales(timescales: Sequence[float]) -> Tensor:
    if len(timescales) == 0:
        raise ValueError("timescales must not be empty")
    values = [float(tau) for tau in timescales]
    if any(tau < 0.0 for tau in values):
        raise ValueError("timescales must be non-negative")
    decays = [0.0 if tau == 0.0 else math.exp(-1.0 / tau) for tau in values]
    return torch.tensor(decays, dtype=torch.float32)


class LeakyLensBank(nn.Module):
    def __init__(self, state_dim: int, timescales: Sequence[float]) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        self.state_dim = int(state_dim)
        self.num_lenses = len(timescales)
        self.register_buffer("decays", decays_from_timescales(timescales))

    @property
    def resident_state_scalars(self) -> int:
        return self.state_dim * self.num_lenses

    def run(self, encoded_sequence: Tensor) -> Tensor:
        if encoded_sequence.ndim != 3:
            raise ValueError("encoded_sequence must have shape [batch, time, state_dim]")
        if encoded_sequence.shape[-1] != self.state_dim:
            raise ValueError(
                f"encoded_sequence state_dim {encoded_sequence.shape[-1]} does not match {self.state_dim}"
            )
        batch_size = encoded_sequence.shape[0]
        state = encoded_sequence.new_zeros(
            (batch_size, self.num_lenses, self.state_dim)
        )
        rho = self.decays.to(
            device=encoded_sequence.device, dtype=encoded_sequence.dtype
        ).view(1, self.num_lenses, 1)
        one_minus_rho = 1.0 - rho
        for step in range(encoded_sequence.shape[1]):
            event = encoded_sequence[:, step].unsqueeze(1)
            state = rho * state + one_minus_rho * event
        return state


def to_coordinates(
    states: Tensor, mode: Literal["raw", "band"] = "raw"
) -> Tensor:
    """Expose raw lowpasses or fast-minus-slower temporal residual coordinates."""
    if states.ndim < 2:
        raise ValueError("states must include a lens dimension")
    if mode == "raw":
        return states
    if mode != "band":
        raise ValueError("mode must be 'raw' or 'band'")

    bands = torch.empty_like(states)
    if states.shape[-2] > 1:
        bands[..., :-1, :] = states[..., :-1, :] - states[..., 1:, :]
    bands[..., -1, :] = states[..., -1, :]
    return bands


def from_band_coordinates(bands: Tensor) -> Tensor:
    """Recover raw lowpasses from (fast-medium, ..., slowest) residuals."""
    if bands.ndim < 2:
        raise ValueError("bands must include a lens dimension")
    return torch.flip(
        torch.cumsum(torch.flip(bands, dims=[-2]), dim=-2), dims=[-2]
    )


def impulse_kernel(
    decays: Tensor, lags: Sequence[int], mode: Literal["raw", "band"] = "raw"
) -> Tensor:
    """Analytical impulse weights at selected lags for raw or residual coordinates.

    Returns shape [num_lags, num_lenses]. For a raw leaky state, an event at
    lag L has weight (1-rho)*rho**L. Residual coordinates use adjacent
    fast-minus-slower differences with the slowest lowpass retained as tail.
    """
    rho = torch.as_tensor(decays, dtype=torch.float32)
    if rho.ndim != 1 or rho.numel() == 0:
        raise ValueError("decays must be a non-empty 1D tensor")
    if torch.any((rho < 0.0) | (rho >= 1.0)):
        raise ValueError("decays must satisfy 0 <= rho < 1")
    lag_values = [int(lag) for lag in lags]
    if any(lag < 0 for lag in lag_values):
        raise ValueError("lags must be non-negative")

    lag_tensor = torch.tensor(
        lag_values, dtype=rho.dtype, device=rho.device
    ).view(-1, 1)
    raw = (1.0 - rho.view(1, -1)) * rho.view(1, -1).pow(lag_tensor)
    if mode == "raw":
        return raw
    if mode != "band":
        raise ValueError("mode must be 'raw' or 'band'")

    bands = torch.empty_like(raw)
    if raw.shape[1] > 1:
        bands[:, :-1] = raw[:, :-1] - raw[:, 1:]
    bands[:, -1] = raw[:, -1]
    return bands
