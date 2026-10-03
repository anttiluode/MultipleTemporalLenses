"""Deterministic training, evaluation, and frozen-result receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
import copy
import hashlib
import json
import platform
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import EVAL_SEEDS, TrainConfig
from .metrics import (
    count_parameters,
    explicit_history_scalar_proxy,
    resident_state_scalars,
)

FROZEN_SEEDS = tuple(EVAL_SEEDS)
BatchFactory = Callable[[int, str], Any]


@dataclass(frozen=True)
class TrainResult:
    train_losses: list[float]
    validation_losses: list[float]
    best_step: int
    best_validation_loss: float
    stopped_early: bool


@dataclass(frozen=True)
class EvalResult:
    per_seed: list[dict[str, Any]]
    mean_accuracy: float
    mean_loss: float


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch CPU randomness."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def classification_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Cross entropy for hard class indices or full soft target distributions."""
    if logits.ndim < 2:
        raise ValueError("logits must have a class dimension")

    if target.dtype.is_floating_point:
        if target.shape != logits.shape:
            raise ValueError("soft targets must have the same shape as logits")
        if torch.any(target < 0):
            raise ValueError("soft targets must be non-negative")
        totals = target.sum(dim=-1)
        if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=0.0):
            raise ValueError("soft targets must sum to 1")
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    return F.cross_entropy(logits, target.long())


def fit_classifier(
    model: nn.Module,
    batch_factory: BatchFactory,
    train_config: TrainConfig,
    train_seed: int,
    validation_seed: int,
) -> TrainResult:
    """Train one classifier deterministically on development-only batches."""
    if int(train_seed) >= 3000 or int(validation_seed) >= 3000:
        raise ValueError("training and validation seeds must stay below 3000")

    seed_everything(train_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=0.0
    )
    validation = batch_factory(int(validation_seed), "dev")

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_loss = float("inf")
    best_step = -1
    best_state: dict[str, Tensor] | None = None
    stale_steps = 0
    stopped_early = False

    model.train()
    for step in range(train_config.max_steps):
        # Stable, development-only stream of fresh batches.
        batch_seed = int(train_seed) * 100_000 + step
        batch = batch_factory(batch_seed, "dev")

        optimizer.zero_grad(set_to_none=True)
        output = model(batch.sequence, batch.query)
        loss = classification_loss(output.logits, batch.target)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_output = model(validation.sequence, validation.query)
            validation_loss = float(
                classification_loss(validation_output.logits, validation.target)
                .detach()
                .cpu()
            )
        validation_losses.append(validation_loss)
        model.train()

        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= train_config.patience:
                stopped_early = True
                break

    if best_state is None:
        raise RuntimeError("training produced no model state")
    model.load_state_dict(best_state)
    model.eval()

    return TrainResult(
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_step=best_step,
        best_validation_loss=best_loss,
        stopped_early=stopped_early,
    )


def evaluate_classifier(
    model: nn.Module,
    batch_factory: BatchFactory,
    seeds: Sequence[int],
    *,
    kind: str = "frozen",
) -> EvalResult:
    """Evaluate with explicit development/frozen/stress seed contracts."""
    seed_tuple = tuple(int(seed) for seed in seeds)

    if kind == "frozen":
        if seed_tuple != FROZEN_SEEDS:
            raise ValueError(f"frozen evaluation requires exactly {FROZEN_SEEDS}")
        split = "heldout"
    elif kind == "development":
        if any(seed >= 3000 for seed in seed_tuple):
            raise ValueError("development evaluation seeds must stay below 3000")
        split = "dev"
    elif kind == "stress":
        if not seed_tuple:
            raise ValueError("stress evaluation requires at least one seed")
        split = "stress"
    else:
        raise ValueError("kind must be 'development', 'frozen', or 'stress'")

    per_seed: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for seed in seed_tuple:
            batch = batch_factory(seed, split)
            output = model(batch.sequence, batch.query)
            loss = float(classification_loss(output.logits, batch.target).cpu())

            # Accuracy is meaningful for hard targets. For soft targets, score
            # against the maximum-probability target only as a diagnostic.
            if batch.target.dtype.is_floating_point:
                expected = batch.target.argmax(dim=-1)
            else:
                expected = batch.target.long()
            accuracy = float(
                (output.logits.argmax(dim=-1) == expected).float().mean().cpu()
            )
            per_seed.append({"seed": seed, "accuracy": accuracy, "loss": loss})

    if per_seed:
        mean_accuracy = float(np.mean([row["accuracy"] for row in per_seed]))
        mean_loss = float(np.mean([row["loss"] for row in per_seed]))
    else:
        mean_accuracy = float("nan")
        mean_loss = float("nan")

    return EvalResult(
        per_seed=per_seed,
        mean_accuracy=mean_accuracy,
        mean_loss=mean_loss,
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_metadata(model: nn.Module, sequence_length: int | None = None) -> dict[str, Any]:
    """Report model budgets from live model attributes/tensors."""
    return {
        "class": model.__class__.__name__,
        "parameter_count": count_parameters(model),
        "resident_state_scalars": resident_state_scalars(model),
        "explicit_history_scalar_proxy": (
            explicit_history_scalar_proxy(model, sequence_length)
            if sequence_length is not None
            else None
        ),
    }


def write_receipt(
    path: str | Path,
    *,
    experiment: str,
    config: Any,
    metrics: Any,
    seeds: Sequence[int],
    model_metadata: Any,
    kind: str = "frozen",
) -> None:
    """Write a self-describing JSON receipt with mechanically protected seeds."""
    seed_list = [int(seed) for seed in seeds]

    if kind in {"frozen", "final"}:
        if tuple(seed_list) != FROZEN_SEEDS:
            raise ValueError(f"{kind} receipts require exactly {FROZEN_SEEDS}")
    elif kind == "development":
        if any(seed >= 3000 for seed in seed_list):
            raise ValueError("development receipt seeds must stay below 3000")
    elif kind == "stress":
        if not seed_list:
            raise ValueError("stress receipt requires at least one seed")
    else:
        raise ValueError(
            "kind must be 'development', 'frozen', 'final', or 'stress'"
        )

    config_json = _jsonable(config)
    payload = {
        "experiment": str(experiment),
        "kind": kind,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "config": config_json,
        "config_hash": _hash_json(config_json),
        "metrics": _jsonable(metrics),
        "seeds": seed_list,
        "seeds_hash": _hash_json(seed_list),
        "model_metadata": _jsonable(model_metadata),
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
