from experiments.query_chooses_timescale import (
    build_primary_models,
    classify_gate1,
    gate0_metrics,
)
from multiple_temporal_lenses.config import canonical_short_config


def test_builds_corrected_factorial_and_single_state_control():
    models = build_primary_models()
    assert set(models) == {
        "single_state",
        "raw_gated",
        "residual_gated",
        "raw_concat",
        "residual_concat",
    }
    assert all(model.resident_state_scalars == 36 for model in models.values())
    assert models["raw_gated"].coordinate_mode == "raw"
    assert models["residual_gated"].coordinate_mode == "band"
    assert models["raw_concat"].coordinate_mode == "raw"
    assert models["residual_concat"].coordinate_mode == "band"


def test_gate0_reports_exact_budget_invertibility_and_recent_bias():
    bench, _, _ = canonical_short_config()
    metrics = gate0_metrics(bench)
    assert metrics["resident_state_scalars"] == 36
    assert metrics["residual_reconstruction_max_abs_error"] < 1e-6
    assert metrics["raw_recent_bias_all_lenses"] is True
    assert metrics["residual_sign_change"]["fast_minus_medium"] is True
    assert metrics["residual_sign_change"]["medium_minus_slow"] is True


def test_classification_requires_four_of_five_seed_wins_for_clear_effect():
    def row(mean, per):
        return {"mean_accuracy": mean, "per_seed": [{"accuracy": x} for x in per]}

    results = {
        "single_state": row(.50, [.50] * 5),
        "raw_gated": row(.52, [.52] * 5),
        "residual_gated": row(.70, [.70] * 5),
        "raw_concat": row(.54, [.54] * 5),
        "residual_concat": row(.60, [.60] * 5),
    }
    out = classify_gate1(results)
    assert out["status"] == "ROUTING_SPECIFIC"
    assert out["comparisons"]["residual_gated_vs_raw_gated"]["wins"] == 5
    assert out["comparisons"]["residual_gated_vs_residual_concat"]["wins"] == 5
