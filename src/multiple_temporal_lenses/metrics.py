"""Accounting and readout diagnostics for Multiple Temporal Lenses."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters from live tensors."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def resident_state_scalars(model: nn.Module) -> int | None:
    """Return fixed resident-state size, or None for non-fixed-history models."""
    value = getattr(model, "resident_state_scalars", None)
    if value is None:
        return None
    if callable(value):
        value = value()
    return int(value)


def explicit_history_scalar_proxy(model: nn.Module, sequence_length: int) -> int | None:
    """Return a model-provided sequence-growing explicit-history proxy."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    method = getattr(model, "explicit_history_scalar_proxy", None)
    if method is None or not callable(method):
        return None
    return int(method(sequence_length))


def lens_entropy(weights: Tensor) -> Tensor:
    """Shannon entropy over the final dimension of valid probability rows."""
    if weights.ndim < 1 or weights.shape[-1] == 0:
        raise ValueError("weights must contain a probability dimension")
    if not torch.is_floating_point(weights):
        weights = weights.to(torch.float32)
    if torch.any(weights < 0) or torch.any(weights > 1):
        raise ValueError("weights must be probabilities in [0, 1]")
    totals = weights.sum(dim=-1)
    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=0.0):
        raise ValueError("weights must be probabilities that sum to 1")
    terms = torch.where(weights > 0, weights * torch.log(weights), torch.zeros_like(weights))
    return -terms.sum(dim=-1)
