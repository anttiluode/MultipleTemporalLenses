"""Analytical sanity check for raw leaky lowpasses and residual coordinates."""

from __future__ import annotations

import json

import torch

from multiple_temporal_lenses.lenses import decays_from_timescales, impulse_kernel


def _section(decays: torch.Tensor, lags: list[int]) -> dict:
    raw = impulse_kernel(decays, lags, mode="raw")
    band = impulse_kernel(decays, lags, mode="band")
    return {
        "decays": [float(x) for x in decays],
        "lags": lags,
        "raw_weights": raw.tolist(),
        "residual_weights": band.tolist(),
        "raw_recent_bias_all_lenses": bool(torch.all(raw[0] > raw[-1]).item()),
        "residual_sign_change": {
            "fast_minus_medium": bool((band[0, 0] * band[-1, 0] < 0).item()),
            "medium_minus_slow": bool((band[0, 1] * band[-1, 1] < 0).item()),
        },
    }


def build_diagnostic() -> dict:
    user_decays = torch.tensor([0.5, 0.95, 0.995], dtype=torch.float32)
    lags = [0, 2, 20, 200]
    user = _section(user_decays, lags)
    user["slow_lag200_over_lag2"] = float(
        user_decays[-1].pow(200) / user_decays[-1].pow(2)
    )

    canonical_timescales = [2.0, 20.0, 200.0]
    canonical_decays = decays_from_timescales(canonical_timescales)
    canonical = _section(canonical_decays, lags)
    canonical["timescales"] = canonical_timescales

    return {
        "claim_boundary": (
            "Raw leaky states are recent-biased lowpasses, not lag-specific bins. "
            "Residual coordinates are signed differences of overlapping lowpasses; "
            "they are invertible but are not disjoint rectangular age windows."
        ),
        "user_example": user,
        "canonical_long_timescales": canonical,
    }


def main() -> None:
    print(json.dumps(build_diagnostic(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
