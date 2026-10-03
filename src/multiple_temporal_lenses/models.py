"""Primary fixed-state and query-conditioned models for core temporal-lens gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from .lenses import LeakyLensBank, to_coordinates


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    diagnostics: dict[str, Tensor]


class _TemporalLensBase(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 8,
        query_size: int = 3,
        state_dim: int = 12,
        timescales: Sequence[float] = (2.0, 16.0, 64.0),
        coordinate_mode: Literal["raw", "band"] = "band",
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or query_size <= 0 or state_dim <= 0:
            raise ValueError("vocab_size, query_size, and state_dim must be positive")
        if coordinate_mode not in ("raw", "band"):
            raise ValueError("coordinate_mode must be 'raw' or 'band'")
        if len(timescales) == 0:
            raise ValueError("timescales must not be empty")

        self.vocab_size = int(vocab_size)
        self.query_size = int(query_size)
        self.state_dim = int(state_dim)
        self.coordinate_mode = coordinate_mode
        self.num_lenses = len(tuple(timescales))

        # Bias-free on purpose: a zero event remains a zero event.
        self.encoder = nn.Linear(self.vocab_size, self.state_dim, bias=False)
        self.lens_bank = LeakyLensBank(self.state_dim, timescales)
        self.query_proj = nn.Linear(self.query_size, self.state_dim)

    @property
    def resident_state_scalars(self) -> int:
        return self.lens_bank.resident_state_scalars

    def compute_memory(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, vocab_size]")
        if sequence.shape[-1] != self.vocab_size:
            raise ValueError(
                f"sequence vocab size {sequence.shape[-1]} does not match {self.vocab_size}"
            )
        encoded = self.encoder(sequence)
        raw_states = self.lens_bank.run(encoded)
        coordinates = to_coordinates(raw_states, self.coordinate_mode)
        return raw_states, coordinates

    def _query_embedding(self, query: Tensor, batch_size: int) -> Tensor:
        if query.ndim != 2 or query.shape != (batch_size, self.query_size):
            raise ValueError(
                f"query must have shape [batch, {self.query_size}]"
            )
        return self.query_proj(query)


class QueryGatedLensModel(_TemporalLensBase):
    """Softmax routing over raw or temporal-residual coordinates."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.key_proj = nn.Linear(self.state_dim, self.state_dim)
        self.value_proj = nn.Linear(self.state_dim, self.state_dim)
        self.classifier = nn.Linear(2 * self.state_dim, self.vocab_size)

    def forward(self, sequence: Tensor, query: Tensor) -> ModelOutput:
        raw_states, coordinates = self.compute_memory(sequence)
        q = self._query_embedding(query, sequence.shape[0])
        keys = self.key_proj(coordinates)
        values = self.value_proj(coordinates)
        scores = torch.einsum("bkd,bd->bk", keys, q) / math.sqrt(self.state_dim)
        lens_weights = torch.softmax(scores, dim=1)
        context = torch.einsum("bk,bkd->bd", lens_weights, values)
        logits = self.classifier(torch.cat([context, q], dim=1))
        return ModelOutput(
            logits=logits,
            diagnostics={
                "raw_states": raw_states,
                "coordinates": coordinates,
                "query_embedding": q,
                "lens_weights": lens_weights,
            },
        )


class QueryConcatLensControl(_TemporalLensBase):
    """Strong control: query-conditioned MLP over all resident coordinates."""

    def __init__(self, *, hidden_dim: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if hidden_dim is None:
            # Match the gated reader's trainable parameter count as closely as an
            # integer hidden width permits, excluding the shared encoder/query map.
            gated_specific = (
                2 * (self.state_dim * self.state_dim + self.state_dim)
                + (2 * self.state_dim) * self.vocab_size
                + self.vocab_size
            )
            per_hidden = (
                self.num_lenses * self.state_dim
                + self.state_dim
                + 1
                + self.vocab_size
            )
            hidden_dim = max(1, round((gated_specific - self.vocab_size) / per_hidden))
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.reader_hidden_dim = int(hidden_dim)
        reader_input = self.num_lenses * self.state_dim + self.state_dim
        self.reader = nn.Sequential(
            nn.Linear(reader_input, self.reader_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.reader_hidden_dim, self.vocab_size),
        )

    def forward(self, sequence: Tensor, query: Tensor) -> ModelOutput:
        raw_states, coordinates = self.compute_memory(sequence)
        q = self._query_embedding(query, sequence.shape[0])
        flat_memory = coordinates.flatten(start_dim=1)
        logits = self.reader(torch.cat([flat_memory, q], dim=1))
        return ModelOutput(
            logits=logits,
            diagnostics={
                "raw_states": raw_states,
                "coordinates": coordinates,
                "query_embedding": q,
            },
        )


class SingleStateModel(nn.Module):
    """Matched resident-state control with one learned recurrent state."""

    def __init__(
        self,
        *,
        vocab_size: int = 8,
        query_size: int = 3,
        total_state_dim: int = 36,
        query_dim: int = 12,
    ) -> None:
        super().__init__()
        if min(vocab_size, query_size, total_state_dim, query_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.vocab_size = int(vocab_size)
        self.query_size = int(query_size)
        self.total_state_dim = int(total_state_dim)
        self.query_dim = int(query_dim)
        self.recurrence = nn.GRU(
            input_size=self.vocab_size,
            hidden_size=self.total_state_dim,
            batch_first=True,
        )
        self.query_proj = nn.Linear(self.query_size, self.query_dim)
        self.classifier = nn.Linear(
            self.total_state_dim + self.query_dim, self.vocab_size
        )

    @property
    def resident_state_scalars(self) -> int:
        return self.total_state_dim

    def forward(self, sequence: Tensor, query: Tensor) -> ModelOutput:
        if sequence.ndim != 3 or sequence.shape[-1] != self.vocab_size:
            raise ValueError("sequence must have shape [batch, time, vocab_size]")
        if query.ndim != 2 or query.shape != (sequence.shape[0], self.query_size):
            raise ValueError(
                f"query must have shape [batch, {self.query_size}]"
            )
        _, hidden = self.recurrence(sequence)
        state = hidden[-1]
        q = self.query_proj(query)
        logits = self.classifier(torch.cat([state, q], dim=1))
        return ModelOutput(
            logits=logits,
            diagnostics={"hidden_state": state, "query_embedding": q},
        )


# Compatibility name for the old plan. The corrected experiment uses the
# query-conditioned concat reader as the serious non-routing multiscale control.
UngatedLensModel = QueryConcatLensControl
