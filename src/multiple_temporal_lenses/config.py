"""Frozen configuration for the first Multiple Temporal Lenses gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EVAL_SEEDS = (3001, 3002, 3003, 3004, 3005)


@dataclass(frozen=True)
class BenchmarkConfig:
    sequence_length: int
    anchor_lags: tuple[int, int, int]
    vocab_size: int
    state_dim: int
    num_scales: int
    eval_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if any(lag <= 0 for lag in self.anchor_lags):
            raise ValueError("anchor_lags must be positive")
        if any(a >= b for a, b in zip(self.anchor_lags, self.anchor_lags[1:])):
            raise ValueError("anchor_lags must be strictly increasing")
        if max(self.anchor_lags) >= self.sequence_length:
            raise ValueError("every anchor lag must be smaller than sequence_length")
        if self.num_scales != len(self.anchor_lags):
            raise ValueError("num_scales must equal the number of anchor_lags")
        if self.vocab_size <= 0 or self.state_dim <= 0:
            raise ValueError("vocab_size and state_dim must be positive")


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int
    num_lenses: int
    coordinate_mode: Literal["raw", "band"] = "raw"

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.num_lenses <= 0:
            raise ValueError("state_dim and num_lenses must be positive")
        if self.coordinate_mode not in ("raw", "band"):
            raise ValueError("coordinate_mode must be 'raw' or 'band'")

    @property
    def resident_state_scalars(self) -> int:
        return self.state_dim * self.num_lenses


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-3
    batch_size: int = 128
    max_steps: int = 1000
    patience: int = 100

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.batch_size <= 0 or self.max_steps <= 0 or self.patience <= 0:
            raise ValueError("training counts must be positive")


def _canonical_triplet(
    *, sequence_length: int, anchor_lags: tuple[int, int, int]
) -> tuple[BenchmarkConfig, ModelConfig, TrainConfig]:
    benchmark = BenchmarkConfig(
        sequence_length=sequence_length,
        anchor_lags=anchor_lags,
        vocab_size=8,
        state_dim=12,
        num_scales=3,
        eval_seeds=EVAL_SEEDS,
    )
    model = ModelConfig(state_dim=12, num_lenses=3, coordinate_mode="raw")
    train = TrainConfig(lr=3e-3, batch_size=128, max_steps=1000, patience=100)
    return benchmark, model, train


def canonical_short_config() -> tuple[BenchmarkConfig, ModelConfig, TrainConfig]:
    """Return the frozen short-horizon Gate 0/1 configuration."""
    return _canonical_triplet(sequence_length=72, anchor_lags=(2, 16, 64))


def canonical_long_config() -> tuple[BenchmarkConfig, ModelConfig, TrainConfig]:
    """Return the preregistered long-horizon stress configuration."""
    return _canonical_triplet(sequence_length=208, anchor_lags=(2, 20, 200))
