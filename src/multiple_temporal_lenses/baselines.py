"""Selective-state and explicit-history controls for temporal-lens experiments."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .models import ModelOutput


class SelectiveStateBaseline(nn.Module):
    """A small input-selective diagonal recurrence, inspired by selective SSMs.

    This is deliberately called Mamba-like rather than Mamba: the current input
    controls per-state retention, but the implementation is only the minimal
    selective recurrence needed as an interpretation control.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 8,
        query_size: int = 3,
        state_dim: int = 36,
        query_dim: int = 12,
    ) -> None:
        super().__init__()
        if min(vocab_size, query_size, state_dim, query_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.vocab_size = int(vocab_size)
        self.query_size = int(query_size)
        self.state_dim = int(state_dim)
        self.query_dim = int(query_dim)

        self.retention_proj = nn.Linear(self.vocab_size, self.state_dim)
        self.candidate_proj = nn.Linear(self.vocab_size, self.state_dim)
        self.query_proj = nn.Linear(self.query_size, self.query_dim)
        self.classifier = nn.Linear(self.state_dim + self.query_dim, self.vocab_size)

    @property
    def resident_state_scalars(self) -> int:
        return self.state_dim

    def forward(self, sequence: Tensor, query: Tensor) -> ModelOutput:
        if sequence.ndim != 3 or sequence.shape[-1] != self.vocab_size:
            raise ValueError("sequence must have shape [batch, time, vocab_size]")
        if query.ndim != 2 or query.shape != (sequence.shape[0], self.query_size):
            raise ValueError(f"query must have shape [batch, {self.query_size}]")

        state = sequence.new_zeros((sequence.shape[0], self.state_dim))
        retention_steps = []
        for step in range(sequence.shape[1]):
            event = sequence[:, step]
            retention = torch.sigmoid(self.retention_proj(event))
            candidate = torch.tanh(self.candidate_proj(event))
            state = retention * state + (1.0 - retention) * candidate
            retention_steps.append(retention)

        retention_history = torch.stack(retention_steps, dim=1)
        q = self.query_proj(query)
        logits = self.classifier(torch.cat([state, q], dim=1))
        return ModelOutput(
            logits=logits,
            diagnostics={
                "hidden_state": state,
                "query_embedding": q,
                "retention": retention_history,
            },
        )


def _sinusoidal_positions(length: int, dim: int, *, device, dtype) -> Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    even_dims = torch.arange(0, dim, 2, device=device, dtype=dtype)
    scales = torch.exp(-math.log(10000.0) * even_dims / dim)
    angles = positions * scales.unsqueeze(0)
    encoding = torch.zeros((length, dim), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(angles)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding


class TinyCausalTransformer(nn.Module):
    """Explicit-history control with a final query token and causal attention."""

    def __init__(
        self,
        *,
        vocab_size: int = 8,
        query_size: int = 3,
        d_model: int = 12,
        nhead: int = 3,
        num_layers: int = 1,
        ff_dim: int = 24,
    ) -> None:
        super().__init__()
        if min(vocab_size, query_size, d_model, nhead, num_layers, ff_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.vocab_size = int(vocab_size)
        self.query_size = int(query_size)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.ff_dim = int(ff_dim)

        self.input_proj = nn.Linear(self.vocab_size, self.d_model, bias=False)
        self.query_proj = nn.Linear(self.query_size, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.classifier = nn.Linear(self.d_model, self.vocab_size)

    def explicit_history_scalar_proxy(self, sequence_length: int) -> int:
        """KV-like scalar proxy: K and V per token per layer, including query.

        PyTorch's encoder here does not use an inference KV cache; this quantity
        is only a consistent sequence-growing proxy for explicit token history.
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        return 2 * (sequence_length + 1) * self.d_model * self.num_layers

    def forward(self, sequence: Tensor, query: Tensor) -> ModelOutput:
        if sequence.ndim != 3 or sequence.shape[-1] != self.vocab_size:
            raise ValueError("sequence must have shape [batch, time, vocab_size]")
        if query.ndim != 2 or query.shape != (sequence.shape[0], self.query_size):
            raise ValueError(f"query must have shape [batch, {self.query_size}]")

        event_tokens = self.input_proj(sequence)
        query_token = self.query_proj(query).unsqueeze(1)
        tokens = torch.cat([event_tokens, query_token], dim=1)
        positions = _sinusoidal_positions(
            tokens.shape[1], self.d_model, device=tokens.device, dtype=tokens.dtype
        )
        tokens = tokens + positions.unsqueeze(0)

        causal_mask = torch.triu(
            torch.ones(
                tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        token_states = self.encoder(tokens, mask=causal_mask)
        query_state = token_states[:, -1]
        logits = self.classifier(query_state)
        return ModelOutput(
            logits=logits,
            diagnostics={"token_states": token_states, "query_state": query_state},
        )
