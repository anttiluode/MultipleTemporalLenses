"""Fixed multiscale resident temporal states and coordinate transforms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn


def decays_from_timescales(timescales: Sequence[float]) -> Tensor:
    """Convert non-negative time constants to leaky-state decay factors.

    A zero time constant is treated as the tau->0 limit, rho=0. Positive
    values use rho = exp(-1 / tau).
    """
    if len(timescales) == 0:
        raise ValueError("timescales must not be empty")
    values = [float(tau) for tau in timescales]
    if any(tau < 0.0 for tau in values):
        raise ValueError("timescales must be non-negative")
    decays = [0.0 if tau == 0.0 else math.exp(-1.0 / tau) for tau in values]
    return torch.tensor(decays, dtype=torch.float32)


class LeakyLensBank(nn.Module):
    """A fixed bank of leaky temporal states with heterogeneous timescales."""

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
        """Return final lens states for an encoded sequence shaped [B,T,D]."""
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
    """Expose lens states directly or as an invertible adjacent-difference basis."""
    if states.ndim < 2:
        raise ValueError("states must include a lens dimension")
    if mode == "raw":
        return states
    if mode != "band":
        raise ValueError("mode must be 'raw' or 'band'")

    bands = torch.empty_like(states)
    bands[..., 0, :] = states[..., 0, :]
    if states.shape[-2] > 1:
        bands[..., 1:, :] = states[..., 1:, :] - states[..., :-1, :]
    return bands


def from_band_coordinates(bands: Tensor) -> Tensor:
    """Recover raw lens states from the adjacent-difference band basis."""
    if bands.ndim < 2:
        raise ValueError("bands must include a lens dimension")
    return torch.cumsum(bands, dim=-2)
