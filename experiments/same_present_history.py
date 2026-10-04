from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from multiple_temporal_lenses.baselines import SelectiveStateBaseline
from multiple_temporal_lenses.config import canonical_short_config
from multiple_temporal_lenses.models import (
    QueryConcatLensControl,
    QueryGatedLensModel,
    SingleStateModel,
)
from multiple_temporal_lenses.tasks import SequenceBatch, make_query_timescale_batch, make_same_present_pairs
from multiple_temporal_lenses.train import fit_classifier, model_metadata, seed_everything, write_receipt

TRAIN_SEED = 101
VALIDATION_SEED = 202
PAIR_COUNT = 512
RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "gate2_parts"
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


def build_gate2_models() -> dict[str, torch.nn.Module]:
    return {name: make_model(name) for name in MODEL_NAMES}


def _query_factory(bench, batch_size: int):
    def make(seed: int, split: str):
        return make_query_timescale_batch(bench, batch_size, seed, split)
    return make


def shuffle_history_within_query_scale(batch: SequenceBatch, seed: int) -> SequenceBatch:
    """Break history-label pairing while preserving present/query and scale distribution."""
    g = torch.Generator().manual_seed(int(seed))
    shuffled = batch.sequence.clone()
    scales = batch.query.argmax(dim=1)
    for scale in range(batch.query.shape[1]):
        indices = torch.nonzero(scales == scale, as_tuple=False).flatten()
        if indices.numel() <= 1:
            continue
        perm = indices[torch.randperm(indices.numel(), generator=g)]
        shuffled[indices, :-1] = batch.sequence[perm, :-1]
    shuffled[:, -1] = batch.sequence[:, -1]
    return SequenceBatch(
        sequence=shuffled,
        query=batch.query.clone(),
        target=batch.target.clone(),
        metadata=dict(batch.metadata),
    )


def classify_gate2(paired_accuracy: float, shuffled_accuracy: float) -> dict[str, float | bool | str]:
    effect = float(paired_accuracy - shuffled_accuracy)
    absolute_ok = paired_accuracy > 0.75
    shuffle_ok = effect >= 0.15
    return {
        "paired_accuracy": float(paired_accuracy),
        "shuffled_accuracy": float(shuffled_accuracy),
        "history_effect": effect,
        "absolute_accuracy_ok": bool(absolute_ok),
        "shuffle_effect_ok": bool(shuffle_ok),
        "status": "PASS_HISTORY_DEPENDENCE" if absolute_ok and shuffle_ok else "FAIL_HISTORY_DEPENDENCE",
    }


def _memory_tensor(output) -> torch.Tensor | None:
    diagnostics = output.diagnostics
    for key in ("raw_states", "hidden_state", "query_state"):
        if key in diagnostics:
            return diagnostics[key].flatten(start_dim=1)
    return None


def _score_pair_seed(model, bench, seed: int, pair_count: int = PAIR_COUNT) -> dict:
    a, b = make_same_present_pairs(bench, pair_count=pair_count, seed=seed, split="heldout")
    shuffled_a = shuffle_history_within_query_scale(a, seed=seed + 50_000)
    shuffled_b = shuffle_history_within_query_scale(b, seed=seed + 60_000)

    model.eval()
    with torch.no_grad():
        out_a = model(a.sequence, a.query)
        out_b = model(b.sequence, b.query)
        out_sa = model(shuffled_a.sequence, shuffled_a.query)
        out_sb = model(shuffled_b.sequence, shuffled_b.query)

    pred_a = out_a.logits.argmax(dim=1)
    pred_b = out_b.logits.argmax(dim=1)
    pred_sa = out_sa.logits.argmax(dim=1)
    pred_sb = out_sb.logits.argmax(dim=1)
    paired_accuracy = float(torch.cat([pred_a == a.target, pred_b == b.target]).float().mean())
    shuffled_accuracy = float(torch.cat([pred_sa == a.target, pred_sb == b.target]).float().mean())

    rows = torch.arange(pair_count)
    margin_a = out_a.logits[rows, a.target] - out_a.logits[rows, b.target]
    margin_b = out_b.logits[rows, b.target] - out_b.logits[rows, a.target]
    pair_logit_margin = float(torch.cat([margin_a, margin_b]).mean())

    memory_a = _memory_tensor(out_a)
    memory_b = _memory_tensor(out_b)
    memory_l2 = None
    if memory_a is not None and memory_b is not None:
        memory_l2 = float(torch.linalg.vector_norm(memory_a - memory_b, dim=1).mean())

    lens_l1 = None
    if "lens_weights" in out_a.diagnostics and "lens_weights" in out_b.diagnostics:
        lens_l1 = float((out_a.diagnostics["lens_weights"] - out_b.diagnostics["lens_weights"]).abs().sum(dim=1).mean())

    return {
        "seed": int(seed),
        "paired_accuracy": paired_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "history_effect": paired_accuracy - shuffled_accuracy,
        "mean_pair_logit_margin": pair_logit_margin,
        "mean_memory_l2_separation": memory_l2,
        "mean_lens_weight_l1_difference": lens_l1,
    }


def run_model(name: str) -> dict:
    bench, _, train_cfg = canonical_short_config()
    seed_everything(TRAIN_SEED)
    model = make_model(name)
    fit = fit_classifier(
        model,
        _query_factory(bench, train_cfg.batch_size),
        train_cfg,
        train_seed=TRAIN_SEED,
        validation_seed=VALIDATION_SEED,
    )
    per_seed = [_score_pair_seed(model, bench, seed) for seed in bench.eval_seeds]
    paired = float(np.mean([row["paired_accuracy"] for row in per_seed]))
    shuffled = float(np.mean([row["shuffled_accuracy"] for row in per_seed]))
    classification = classify_gate2(paired, shuffled)
    return {
        "model": name,
        "training": {
            "best_step": fit.best_step,
            "best_validation_loss": fit.best_validation_loss,
            "stopped_early": fit.stopped_early,
        },
        "per_seed": per_seed,
        "mean_paired_accuracy": paired,
        "mean_shuffled_accuracy": shuffled,
        "mean_history_effect": paired - shuffled,
        "mean_pair_logit_margin": float(np.mean([row["mean_pair_logit_margin"] for row in per_seed])),
        "mean_memory_l2_separation": float(np.mean([row["mean_memory_l2_separation"] for row in per_seed if row["mean_memory_l2_separation"] is not None])),
        "mean_lens_weight_l1_difference": (
            float(np.mean([row["mean_lens_weight_l1_difference"] for row in per_seed if row["mean_lens_weight_l1_difference"] is not None]))
            if any(row["mean_lens_weight_l1_difference"] is not None for row in per_seed)
            else None
        ),
        "classification": classification,
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
    primary = models["residual_gated"]["classification"]
    metrics = {
        "models": models,
        "primary_model": "residual_gated",
        "primary_classification": primary,
        "best_paired_accuracy_model": max(models, key=lambda n: models[n]["mean_paired_accuracy"]),
    }
    write_receipt(
        RESULTS / "gate2_same_present_history.json",
        experiment="gate2_same_present_different_history",
        config={
            "benchmark": bench,
            "training": train_cfg,
            "train_seed": TRAIN_SEED,
            "validation_seed": VALIDATION_SEED,
            "pair_count_per_seed": PAIR_COUNT,
            "pair_split": "heldout",
            "shuffle_control": "permute history within query-scale groups while preserving present/query/labels",
            "criterion": "primary residual-gated paired accuracy >0.75 and >=0.15 above shuffled history",
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
        print(json.dumps({"models": list(MODEL_NAMES), "pair_count": PAIR_COUNT}, indent=2))
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
