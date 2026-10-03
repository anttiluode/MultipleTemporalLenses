"""Deterministic synthetic tasks for the first temporal-lens gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .config import BenchmarkConfig


@dataclass(frozen=True)
class SequenceBatch:
    sequence: Tensor
    query: Tensor
    target: Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LaterContextBatch:
    prefix_sequence: Tensor
    full_sequence: Tensor
    neutral_target: Tensor
    resolved_target: Tensor
    context: Tensor


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _lag_table(config: BenchmarkConfig, batch_size: int, seed: int, split: str) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.sequence_length == 72 and config.anchor_lags == (2, 16, 64):
        if split == "dev":
            choices = ((2, 3), (12, 16), (48, 56))
        elif split == "heldout":
            choices = ((4,), (20,), (64,))
        elif split == "stress":
            choices = tuple((lag,) for lag in config.anchor_lags)
        else:
            raise ValueError("split must be 'dev', 'heldout', or 'stress'")
    elif split == "stress":
        choices = tuple((lag,) for lag in config.anchor_lags)
    else:
        choices = tuple((lag,) for lag in config.anchor_lags)

    g = _generator(seed)
    columns = []
    for options in choices:
        option_tensor = torch.tensor(options, dtype=torch.long)
        indices = torch.randint(len(options), (batch_size,), generator=g)
        columns.append(option_tensor[indices])
    return torch.stack(columns, dim=1)


def make_query_timescale_batch(
    config: BenchmarkConfig,
    batch_size: int,
    seed: int,
    split: str,
) -> SequenceBatch:
    """Create a task where the query selects which temporal scale supplies the label."""
    lags = _lag_table(config, batch_size, seed, split)
    g = _generator(seed + 1)
    values = torch.randint(config.vocab_size, (batch_size, config.num_scales), generator=g)
    query_scale = torch.randint(config.num_scales, (batch_size,), generator=g)

    sequence = torch.zeros(
        batch_size, config.sequence_length, config.vocab_size, dtype=torch.float32
    )
    rows = torch.arange(batch_size)
    for scale in range(config.num_scales):
        positions = config.sequence_length - 1 - lags[:, scale]
        sequence[rows, positions, values[:, scale]] = 1.0

    query = torch.nn.functional.one_hot(
        query_scale, num_classes=config.num_scales
    ).to(torch.float32)
    target = values[rows, query_scale].to(torch.long)
    return SequenceBatch(
        sequence=sequence,
        query=query,
        target=target,
        metadata={"lags": lags, "values": values, "query_scale": query_scale},
    )


def make_same_present_pairs(
    config: BenchmarkConfig,
    pair_count: int,
    seed: int,
) -> tuple[SequenceBatch, SequenceBatch]:
    """Return paired trials with identical present/query and different relevant history."""
    a = make_query_timescale_batch(config, pair_count, seed, "dev")
    sequence_b = a.sequence.clone()
    target_b = a.target.clone()
    values_b = a.metadata["values"].clone()
    g = _generator(seed + 2)
    scales = a.metadata["query_scale"]

    offsets = torch.randint(1, config.vocab_size, (pair_count,), generator=g)
    new_values = (a.target + offsets) % config.vocab_size
    for row in range(pair_count):
        scale = int(scales[row])
        lag = int(a.metadata["lags"][row, scale])
        position = config.sequence_length - 1 - lag
        old_value = int(a.target[row])
        new_value = int(new_values[row])
        sequence_b[row, position, old_value] = 0.0
        sequence_b[row, position, new_value] = 1.0
        values_b[row, scale] = new_value
        target_b[row] = new_value

    b = SequenceBatch(
        sequence=sequence_b,
        query=a.query.clone(),
        target=target_b,
        metadata={
            "lags": a.metadata["lags"].clone(),
            "values": values_b,
            "query_scale": scales.clone(),
        },
    )
    return a, b


def make_later_context_batch(
    config: BenchmarkConfig,
    batch_size: int,
    seed: int,
) -> LaterContextBatch:
    """Create an ambiguous earlier event that later binary context resolves."""
    if config.vocab_size < 4:
        raise ValueError("later-context task requires vocab_size >= 4")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    g = _generator(seed)
    candidate_count = config.vocab_size - 2
    first = torch.randint(candidate_count, (batch_size,), generator=g) + 2
    offset = torch.randint(1, candidate_count, (batch_size,), generator=g)
    second = ((first - 2 + offset) % candidate_count) + 2
    context = torch.randint(2, (batch_size,), generator=g)

    prefix = torch.zeros(
        batch_size, config.sequence_length, config.vocab_size, dtype=torch.float32
    )
    ambiguous_lag = config.anchor_lags[1]
    ambiguous_pos = config.sequence_length - 1 - ambiguous_lag
    rows = torch.arange(batch_size)
    prefix[rows, ambiguous_pos, first] = 0.5
    prefix[rows, ambiguous_pos, second] = 0.5

    full = prefix.clone()
    full[rows, -1, context] = 1.0

    neutral = torch.zeros(batch_size, config.vocab_size, dtype=torch.float32)
    neutral[rows, first] = 0.5
    neutral[rows, second] = 0.5
    resolved = torch.where(context == 0, first, second).to(torch.long)

    return LaterContextBatch(
        prefix_sequence=prefix,
        full_sequence=full,
        neutral_target=neutral,
        resolved_target=resolved,
        context=context.to(torch.long),
    )
