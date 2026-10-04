from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multiple_temporal_lenses.baselines import SelectiveStateBaseline
from multiple_temporal_lenses.config import canonical_short_config
from multiple_temporal_lenses.models import (
    QueryConcatLensControl,
    QueryGatedLensModel,
    SingleStateModel,
)
from multiple_temporal_lenses.tasks import SequenceBatch, make_later_context_batch
from multiple_temporal_lenses.train import TrainResult, classification_loss, fit_classifier, model_metadata, seed_everything, write_receipt

TRAIN_SEED = 101
VALIDATION_SEED = 202
EVAL_BATCH_SIZE = 512
RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "gate3_parts"
MODEL_NAMES = (
    "single_state",
    "raw_gated",
    "residual_gated",
    "raw_concat",
    "residual_concat",
    "selective_state",
)


def make_model(name: str) -> torch.nn.Module:
    factories = {
        "single_state": lambda: SingleStateModel(total_state_dim=36),
        "raw_gated": lambda: QueryGatedLensModel(coordinate_mode="raw"),
        "residual_gated": lambda: QueryGatedLensModel(coordinate_mode="band"),
        "raw_concat": lambda: QueryConcatLensControl(coordinate_mode="raw"),
        "residual_concat": lambda: QueryConcatLensControl(coordinate_mode="band"),
        "selective_state": lambda: SelectiveStateBaseline(state_dim=36),
    }
    if name not in factories:
        raise KeyError(name)
    return factories[name]()


def build_gate3_models() -> dict[str, torch.nn.Module]:
    return {name: make_model(name) for name in MODEL_NAMES}


def single_state_two_probe_logits(model: SingleStateModel, prefix_sequence: torch.Tensor, full_sequence: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Exact shared-prefix evaluation for the two Gate-3 probes."""
    if not torch.equal(prefix_sequence[:, :-1], full_sequence[:, :-1]):
        raise ValueError("prefix and full sequences must share all pre-context steps")
    _, shared_hidden = model.recurrence(prefix_sequence[:, :-1])
    _, prefix_hidden = model.recurrence(prefix_sequence[:, -1:], shared_hidden)
    _, full_hidden = model.recurrence(full_sequence[:, -1:], shared_hidden)
    q = model.query_proj(query)
    prefix_logits = model.classifier(torch.cat([prefix_hidden[-1], q], dim=1))
    full_logits = model.classifier(torch.cat([full_hidden[-1], q], dim=1))
    return torch.cat([prefix_logits, full_logits], dim=0)


def make_supervised_later_context_batch(config, batch_size: int, seed: int) -> SequenceBatch:
    """Pack prefix and resolved probes into one mathematically equivalent soft-target batch."""
    source = make_later_context_batch(config, batch_size=batch_size, seed=seed)
    query = torch.zeros(batch_size, config.num_scales, dtype=torch.float32)
    query[:, 1] = 1.0
    final_target = F.one_hot(source.resolved_target, num_classes=config.vocab_size).to(torch.float32)
    return SequenceBatch(
        sequence=torch.cat([source.prefix_sequence, source.full_sequence], dim=0),
        query=torch.cat([query, query], dim=0),
        target=torch.cat([source.neutral_target, final_target], dim=0),
        metadata={"probe_split": batch_size},
    )


def _factory(bench, batch_size: int):
    def make(seed: int, split: str):
        if split != "dev":
            raise ValueError("later-context training factory is development-only")
        return make_supervised_later_context_batch(bench, batch_size, seed)
    return make


def fit_single_state_shared_prefix(model: SingleStateModel, bench, train_config, train_seed: int, validation_seed: int) -> TrainResult:
    """Train the GRU with exact shared-prefix reuse for Gate 3's paired probes."""
    seed_everything(train_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=0.0)

    def build(seed: int):
        source = make_later_context_batch(bench, batch_size=train_config.batch_size, seed=seed)
        query = torch.zeros(train_config.batch_size, bench.num_scales, dtype=torch.float32)
        query[:, 1] = 1.0
        final_target = F.one_hot(source.resolved_target, num_classes=bench.vocab_size).to(torch.float32)
        target = torch.cat([source.neutral_target, final_target], dim=0)
        return source, query, target

    validation_source, validation_query, validation_target = build(int(validation_seed))
    train_losses = []
    validation_losses = []
    best_loss = float("inf")
    best_step = -1
    best_state = None
    stale_steps = 0
    stopped_early = False

    model.train()
    for step in range(train_config.max_steps):
        source, query, target = build(int(train_seed) * 100_000 + step)
        optimizer.zero_grad(set_to_none=True)
        logits = single_state_two_probe_logits(model, source.prefix_sequence, source.full_sequence, query)
        loss = classification_loss(logits, target)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_logits = single_state_two_probe_logits(
                model, validation_source.prefix_sequence, validation_source.full_sequence, validation_query
            )
            validation_loss = float(classification_loss(validation_logits, validation_target).detach().cpu())
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
    return TrainResult(train_losses, validation_losses, best_step, best_loss, stopped_early)


def classify_gate3(probability_shift: float, final_accuracy: float) -> dict[str, float | bool | str]:
    shift_ok = probability_shift >= 0.25
    accuracy_ok = final_accuracy > 0.80
    return {
        "mean_context_consistent_probability_shift": float(probability_shift),
        "final_accuracy": float(final_accuracy),
        "probability_shift_ok": bool(shift_ok),
        "final_accuracy_ok": bool(accuracy_ok),
        "status": "PASS_REINTERPRETATION" if shift_ok and accuracy_ok else "FAIL_REINTERPRETATION",
    }


def _score_seed(model, bench, seed: int, batch_size: int = EVAL_BATCH_SIZE) -> dict:
    batch = make_later_context_batch(bench, batch_size=batch_size, seed=seed)
    query = torch.zeros(batch_size, bench.num_scales, dtype=torch.float32)
    query[:, 1] = 1.0
    combined_sequence = torch.cat([batch.prefix_sequence, batch.full_sequence], dim=0)
    combined_query = torch.cat([query, query], dim=0)

    model.eval()
    with torch.no_grad():
        output = model(combined_sequence, combined_query)
        probabilities = torch.softmax(output.logits, dim=-1)
    prefix_prob, final_prob = probabilities[:batch_size], probabilities[batch_size:]
    rows = torch.arange(batch_size)
    candidate_mask = batch.neutral_target > 0
    prefix_candidate_mass = float((prefix_prob * candidate_mask).sum(dim=1).mean())
    final_candidate_mass = float((final_prob * candidate_mask).sum(dim=1).mean())
    prefix_consistent = float(prefix_prob[rows, batch.resolved_target].mean())
    final_consistent = float(final_prob[rows, batch.resolved_target].mean())
    final_accuracy = float((final_prob.argmax(dim=1) == batch.resolved_target).float().mean())
    return {
        "seed": int(seed),
        "prefix_candidate_probability_mass": prefix_candidate_mass,
        "final_candidate_probability_mass": final_candidate_mass,
        "prefix_context_consistent_probability": prefix_consistent,
        "final_context_consistent_probability": final_consistent,
        "context_consistent_probability_shift": final_consistent - prefix_consistent,
        "final_accuracy": final_accuracy,
    }


def run_model(name: str) -> dict:
    bench, _, train_cfg = canonical_short_config()
    seed_everything(TRAIN_SEED)
    model = make_model(name)
    if name == "single_state":
        fit = fit_single_state_shared_prefix(
            model, bench, train_cfg, train_seed=TRAIN_SEED, validation_seed=VALIDATION_SEED
        )
    else:
        fit = fit_classifier(
            model,
            _factory(bench, train_cfg.batch_size),
            train_cfg,
            train_seed=TRAIN_SEED,
            validation_seed=VALIDATION_SEED,
        )
    per_seed = [_score_seed(model, bench, seed) for seed in bench.eval_seeds]
    mean_shift = float(np.mean([row["context_consistent_probability_shift"] for row in per_seed]))
    mean_accuracy = float(np.mean([row["final_accuracy"] for row in per_seed]))
    return {
        "model": name,
        "training": {
            "best_step": fit.best_step,
            "best_validation_loss": fit.best_validation_loss,
            "stopped_early": fit.stopped_early,
        },
        "per_seed": per_seed,
        "mean_prefix_candidate_probability_mass": float(np.mean([row["prefix_candidate_probability_mass"] for row in per_seed])),
        "mean_final_candidate_probability_mass": float(np.mean([row["final_candidate_probability_mass"] for row in per_seed])),
        "mean_prefix_context_consistent_probability": float(np.mean([row["prefix_context_consistent_probability"] for row in per_seed])),
        "mean_final_context_consistent_probability": float(np.mean([row["final_context_consistent_probability"] for row in per_seed])),
        "mean_context_consistent_probability_shift": mean_shift,
        "mean_final_accuracy": mean_accuracy,
        "classification": classify_gate3(mean_shift, mean_accuracy),
        "model_metadata": model_metadata(model, bench.sequence_length),
    }


def write_part(name: str, payload: dict) -> Path:
    PARTS.mkdir(parents=True, exist_ok=True)
    path = PARTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def assemble() -> dict:
    bench, _, train_cfg = canonical_short_config()
    models = {}
    for name in MODEL_NAMES:
        path = PARTS / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        models[name] = json.loads(path.read_text(encoding="utf-8"))
    metrics = {
        "models": models,
        "primary_model": "residual_gated",
        "primary_classification": models["residual_gated"]["classification"],
        "best_final_accuracy_model": max(models, key=lambda n: models[n]["mean_final_accuracy"]),
        "largest_probability_shift_model": max(models, key=lambda n: models[n]["mean_context_consistent_probability_shift"]),
    }
    write_receipt(
        RESULTS / "gate3_later_context.json",
        experiment="gate3_later_context_bends_earlier_meaning",
        config={
            "benchmark": bench,
            "training": train_cfg,
            "train_seed": TRAIN_SEED,
            "validation_seed": VALIDATION_SEED,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "query_scale": 1,
            "training_probes": "prefix soft 50/50 target plus final one-hot target, packed into one equivalent soft-target batch",
            "criterion": "primary residual-gated probability shift >=0.25 and final accuracy >0.80",
        },
        metrics=metrics,
        seeds=bench.eval_seeds,
        model_metadata={name: payload["model_metadata"] for name, payload in models.items()},
        kind="frozen",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.dry:
        print(json.dumps({"models": list(MODEL_NAMES), "eval_batch_size": EVAL_BATCH_SIZE}, indent=2))
        return
    if args.model:
        payload = run_model(args.model)
        write_part(args.model, payload)
        print(json.dumps(payload["classification"], indent=2, sort_keys=True))
        return
    if args.assemble:
        metrics = assemble()
        print(json.dumps(metrics["primary_classification"], indent=2, sort_keys=True))
        return
    for name in MODEL_NAMES:
        write_part(name, run_model(name))
    metrics = assemble()
    print(json.dumps(metrics["primary_classification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
