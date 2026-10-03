import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_impulse_kernel_diagnostic_reports_recent_bias_and_residual_signs():
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    proc = subprocess.run(
        [sys.executable, str(repo / "experiments" / "impulse_kernel_diagnostic.py")],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)

    example = payload["user_example"]
    assert example["decays"] == pytest.approx([0.5, 0.95, 0.995])
    assert example["lags"] == [0, 2, 20, 200]
    assert example["slow_lag200_over_lag2"] == pytest.approx(0.995 ** 198)
    assert example["raw_recent_bias_all_lenses"] is True
    assert example["residual_sign_change"]["fast_minus_medium"] is True
    assert example["residual_sign_change"]["medium_minus_slow"] is True
