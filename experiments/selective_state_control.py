from __future__ import annotations

import argparse
import json
from pathlib import Path

from multiple_temporal_lenses.baselines import SelectiveStateBaseline
from multiple_temporal_lenses.config import canonical_short_config
from multiple_temporal_lenses.tasks import make_query_timescale_batch
from multiple_temporal_lenses.train import (
    evaluate_classifier,
    fit_classifier,
    model_metadata,
    seed_everything,
    write_receipt,
)

TRAIN_SEED = 101
VALIDATION_SEED = 202
DEV_EVAL_SEEDS = (2101, 2102, 2103, 2104, 2105)
RAW_CONCAT_FROZEN_MEAN = 0.52578125
RAW_CONCAT_FROZEN_PER_SEED = (
    0.50390625,
    0.498046875,
    0.537109375,
    0.54296875,
    0.546875,
)
RESULTS = Path(__file__).resolve().parents[1] / "results"


def _factory(bench, batch_size: int):
    def make(seed: int, split: str):
        return make_query_timescale_batch(
            bench, batch_size=batch_size, seed=seed, split=split
        )
    return make


def run() -> dict:
    bench, _, train_cfg = canonical_short_config()
    seed_everything(TRAIN_SEED)
    model = SelectiveStateBaseline(state_dim=36)

    fit = fit_classifier(
        model,
        _factory(bench, train_cfg.batch_size),
        train_cfg,
        train_seed=TRAIN_SEED,
        validation_seed=VALIDATION_SEED,
    )
    dev = evaluate_classifier(
        model,
        _factory(bench, 512),
        DEV_EVAL_SEEDS,
        kind="development",
    )
    frozen = evaluate_classifier(
        model,
        _factory(bench, 512),
        bench.eval_seeds,
        kind="frozen",
    )

    selective_per_seed = tuple(row["accuracy"] for row in frozen.per_seed)
    wins = sum(
        selective > raw
        for selective, raw in zip(selective_per_seed, RAW_CONCAT_FROZEN_PER_SEED)
    )
    comparison = {
        "raw_concat_frozen_mean_accuracy": RAW_CONCAT_FROZEN_MEAN,
        "selective_state_frozen_mean_accuracy": frozen.mean_accuracy,
        "mean_delta_selective_minus_raw_concat": (
            frozen.mean_accuracy - RAW_CONCAT_FROZEN_MEAN
        ),
        "selective_state_seed_wins": wins,
        "status": (
            "SELECTIVE_STATE_MATCHES_OR_EXCEEDS_RAW_CONCAT"
            if frozen.mean_accuracy >= RAW_CONCAT_FROZEN_MEAN
            else "SELECTIVE_STATE_BELOW_RAW_CONCAT"
        ),
    }

    metrics = {
        "training": {
            "best_step": fit.best_step,
            "best_validation_loss": fit.best_validation_loss,
            "stopped_early": fit.stopped_early,
        },
        "development": {
            "mean_accuracy": dev.mean_accuracy,
            "mean_loss": dev.mean_loss,
            "per_seed": dev.per_seed,
        },
        "frozen": {
            "mean_accuracy": frozen.mean_accuracy,
            "mean_loss": frozen.mean_loss,
            "per_seed": frozen.per_seed,
        },
        "comparison_to_gate1_raw_concat": comparison,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_receipt(
        RESULTS / "gate4_selective_state_control.json",
        experiment="gate4_selective_state_control",
        config={
            "benchmark": bench,
            "training": train_cfg,
            "train_seed": TRAIN_SEED,
            "validation_seed": VALIDATION_SEED,
            "development_eval_seeds": DEV_EVAL_SEEDS,
            "state_budget": 36,
            "control_name": "SelectiveStateBaseline",
            "claim_boundary": (
                "Minimal input-selective diagonal recurrence; Mamba-like control, "
                "not an implementation of Mamba."
            ),
            "execution_note": (
                "Planned interpretation control run after Gate 1 with no "
                "hyperparameter changes or tuning against held-out results."
            ),
        },
        metrics=metrics,
        seeds=bench.eval_seeds,
        model_metadata=model_metadata(model, bench.sequence_length),
        kind="frozen",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.dry:
        bench, _, _ = canonical_short_config()
        model = SelectiveStateBaseline(state_dim=36)
        print(
            json.dumps(
                {
                    "model": model.__class__.__name__,
                    "resident_state_scalars": model.resident_state_scalars,
                    "eval_seeds": list(bench.eval_seeds),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    metrics = run()
    print(json.dumps(metrics["comparison_to_gate1_raw_concat"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
