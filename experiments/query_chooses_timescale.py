from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from multiple_temporal_lenses.config import canonical_short_config
from multiple_temporal_lenses.lenses import (
    LeakyLensBank,
    from_band_coordinates,
    impulse_kernel,
    to_coordinates,
)
from multiple_temporal_lenses.metrics import lens_entropy
from multiple_temporal_lenses.models import (
    QueryConcatLensControl,
    QueryGatedLensModel,
    SingleStateModel,
)
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
RESULTS = Path(__file__).resolve().parents[1] / "results"


def make_model(name: str) -> torch.nn.Module:
    factories = {
        "single_state": lambda: SingleStateModel(total_state_dim=36),
        "raw_gated": lambda: QueryGatedLensModel(coordinate_mode="raw"),
        "residual_gated": lambda: QueryGatedLensModel(coordinate_mode="band"),
        "raw_concat": lambda: QueryConcatLensControl(coordinate_mode="raw"),
        "residual_concat": lambda: QueryConcatLensControl(coordinate_mode="band"),
    }
    if name not in factories:
        raise KeyError(name)
    return factories[name]()


def build_primary_models() -> dict[str, torch.nn.Module]:
    """Build the frozen corrected Gate-1 comparison.

    The 2x2 multiscale factorial varies only coordinate basis and reader type:
    raw vs temporal residual, gated routing vs query-conditioned concat.
    """
    names = (
        "single_state",
        "raw_gated",
        "residual_gated",
        "raw_concat",
        "residual_concat",
    )
    return {name: make_model(name) for name in names}


def gate0_metrics(bench) -> dict:
    bank = LeakyLensBank(bench.state_dim, bench.anchor_lags)
    torch.manual_seed(17)
    states = torch.randn(8, 3, bench.state_dim)
    reconstructed = from_band_coordinates(to_coordinates(states, "band"))

    lags = [0, 2, 20, 200]
    raw = impulse_kernel(bank.decays, lags, "raw")
    residual = impulse_kernel(bank.decays, lags, "band")
    return {
        "resident_state_scalars": bank.resident_state_scalars,
        "timescales": list(bench.anchor_lags),
        "decays": bank.decays.tolist(),
        "diagnostic_lags": lags,
        "raw_impulse_weights": raw.tolist(),
        "residual_impulse_weights": residual.tolist(),
        "raw_recent_bias_all_lenses": bool(torch.all(raw[0] > raw[-1]).item()),
        "residual_sign_change": {
            "fast_minus_medium": bool((residual[0, 0] * residual[-1, 0] < 0).item()),
            "medium_minus_slow": bool((residual[0, 1] * residual[-1, 1] < 0).item()),
        },
        "residual_reconstruction_max_abs_error": float(
            (reconstructed - states).abs().max().item()
        ),
    }


def _factory(bench, batch_size: int):
    def make(seed: int, split: str):
        return make_query_timescale_batch(
            bench, batch_size=batch_size, seed=seed, split=split
        )
    return make


def _comparison(a: dict, b: dict) -> dict:
    wins = sum(
        x["accuracy"] > y["accuracy"]
        for x, y in zip(a["per_seed"], b["per_seed"])
    )
    return {
        "mean_delta": a["mean_accuracy"] - b["mean_accuracy"],
        "wins": wins,
        "clear": bool(a["mean_accuracy"] > b["mean_accuracy"] and wins >= 4),
    }


def classify_gate1(results: dict[str, dict]) -> dict:
    """Mechanically classify the corrected factorial before held-out evaluation."""
    comparisons = {
        "residual_gated_vs_raw_gated": _comparison(
            results["residual_gated"], results["raw_gated"]
        ),
        "residual_concat_vs_raw_concat": _comparison(
            results["residual_concat"], results["raw_concat"]
        ),
        "residual_gated_vs_residual_concat": _comparison(
            results["residual_gated"], results["residual_concat"]
        ),
        "residual_gated_vs_single_state": _comparison(
            results["residual_gated"], results["single_state"]
        ),
    }
    coordinate_effect = (
        comparisons["residual_gated_vs_raw_gated"]["clear"]
        or comparisons["residual_concat_vs_raw_concat"]["clear"]
    )
    routing_specific = (
        comparisons["residual_gated_vs_raw_gated"]["clear"]
        and comparisons["residual_gated_vs_residual_concat"]["clear"]
        and comparisons["residual_gated_vs_single_state"]["clear"]
    )
    if routing_specific:
        status = "ROUTING_SPECIFIC"
    elif coordinate_effect:
        status = "COORDINATE_ACCESSIBILITY"
    else:
        status = "NO_CLEAR_PRIMARY_ADVANTAGE"
    return {
        "status": status,
        "coordinate_effect": coordinate_effect,
        "routing_specific": routing_specific,
        "comparisons": comparisons,
    }


def _selection_by_scale(model, bench, seeds, split: str) -> dict:
    model.eval()
    entropy_by = {str(i): [] for i in range(3)}
    weights_by = {str(i): [] for i in range(3)}
    with torch.no_grad():
        for seed in seeds:
            batch = make_query_timescale_batch(bench, 512, seed, split)
            out = model(batch.sequence, batch.query)
            weights = out.diagnostics["lens_weights"]
            entropy = lens_entropy(weights)
            scale = batch.query.argmax(dim=-1)
            for i in range(3):
                mask = scale == i
                if mask.any():
                    entropy_by[str(i)].append(float(entropy[mask].mean().cpu()))
                    weights_by[str(i)].append(weights[mask].mean(dim=0).cpu())
    result = {}
    for i in range(3):
        key = str(i)
        mean_weights = torch.stack(weights_by[key]).mean(dim=0).tolist()
        result[key] = {
            "mean_entropy": sum(entropy_by[key]) / len(entropy_by[key]),
            "mean_weights": mean_weights,
        }
    return result


def run(mode: str) -> dict:
    bench, _, train_cfg = canonical_short_config()
    eval_seeds = bench.eval_seeds if mode == "frozen" else DEV_EVAL_SEEDS
    eval_kind = "frozen" if mode == "frozen" else "development"
    split = "heldout" if mode == "frozen" else "dev"
    RESULTS.mkdir(parents=True, exist_ok=True)

    g0 = gate0_metrics(bench)
    write_receipt(
        RESULTS / "gate0_state_mechanics.json",
        experiment="gate0_state_mechanics",
        config={"benchmark": bench, "basis": "raw_and_temporal_residual"},
        metrics=g0,
        seeds=[0],
        model_metadata={"mechanism": "LeakyLensBank"},
        kind="development",
    )

    results: dict[str, dict] = {}
    metadata: dict[str, dict] = {}
    trained: dict[str, torch.nn.Module] = {}

    for name in build_primary_models():
        seed_everything(TRAIN_SEED)
        model = make_model(name)
        fit = fit_classifier(
            model,
            _factory(bench, train_cfg.batch_size),
            train_cfg,
            train_seed=TRAIN_SEED,
            validation_seed=VALIDATION_SEED,
        )
        ev = evaluate_classifier(
            model,
            _factory(bench, 512),
            eval_seeds,
            kind=eval_kind,
        )
        results[name] = {
            "mean_accuracy": ev.mean_accuracy,
            "mean_loss": ev.mean_loss,
            "per_seed": ev.per_seed,
            "best_step": fit.best_step,
            "best_validation_loss": fit.best_validation_loss,
            "stopped_early": fit.stopped_early,
        }
        metadata[name] = model_metadata(model, bench.sequence_length)
        trained[name] = model

    for name in ("raw_gated", "residual_gated"):
        results[name]["selection_by_query_scale"] = _selection_by_scale(
            trained[name], bench, eval_seeds, split
        )

    classification = classify_gate1(results)
    metrics = {"models": results, "classification": classification}
    output = RESULTS / (
        "gate1_query_timescale.json"
        if mode == "frozen"
        else "gate1_query_timescale_dev.json"
    )
    write_receipt(
        output,
        experiment="gate1_query_chooses_timescale_corrected_factorial",
        config={
            "benchmark": bench,
            "training": train_cfg,
            "train_seed": TRAIN_SEED,
            "validation_seed": VALIDATION_SEED,
            "models": list(build_primary_models()),
            "decision_rule": "clear iff mean accuracy is greater and wins >=4/5 seeds",
        },
        metrics=metrics,
        seeds=eval_seeds,
        model_metadata=metadata,
        kind=eval_kind,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "frozen"], default="dev")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.dry:
        print(json.dumps({"models": list(build_primary_models())}, indent=2))
        return
    metrics = run(args.mode)
    print(json.dumps(metrics["classification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
