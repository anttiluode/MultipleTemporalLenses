import pytest

from multiple_temporal_lenses.config import (
    BenchmarkConfig,
    ModelConfig,
    canonical_long_config,
    canonical_short_config,
)


def test_canonical_short_configuration_is_frozen():
    benchmark, model, train = canonical_short_config()

    assert benchmark.sequence_length == 72
    assert benchmark.anchor_lags == (2, 16, 64)
    assert benchmark.vocab_size == 8
    assert benchmark.state_dim == 12
    assert benchmark.num_scales == 3
    assert benchmark.eval_seeds == (3001, 3002, 3003, 3004, 3005)
    assert all(lag < benchmark.sequence_length for lag in benchmark.anchor_lags)

    assert model.state_dim == 12
    assert model.num_lenses == 3
    assert model.coordinate_mode == "raw"
    assert model.resident_state_scalars == 36

    assert train.lr == pytest.approx(3e-3)
    assert train.batch_size == 128
    assert train.max_steps == 1000
    assert train.patience == 100


def test_canonical_long_configuration_uses_preregistered_stress_horizon():
    benchmark, model, train = canonical_long_config()

    assert benchmark.sequence_length == 208
    assert benchmark.anchor_lags == (2, 20, 200)
    assert benchmark.eval_seeds == (3001, 3002, 3003, 3004, 3005)
    assert all(lag < benchmark.sequence_length for lag in benchmark.anchor_lags)
    assert model.resident_state_scalars == 36
    assert train.max_steps == 1000


def test_benchmark_config_rejects_invalid_lag_geometry():
    with pytest.raises(ValueError, match="positive"):
        BenchmarkConfig(
            sequence_length=72,
            anchor_lags=(0, 16, 64),
            vocab_size=8,
            state_dim=12,
            num_scales=3,
            eval_seeds=(3001, 3002, 3003, 3004, 3005),
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        BenchmarkConfig(
            sequence_length=72,
            anchor_lags=(2, 2, 64),
            vocab_size=8,
            state_dim=12,
            num_scales=3,
            eval_seeds=(3001, 3002, 3003, 3004, 3005),
        )

    with pytest.raises(ValueError, match="sequence_length"):
        BenchmarkConfig(
            sequence_length=64,
            anchor_lags=(2, 16, 64),
            vocab_size=8,
            state_dim=12,
            num_scales=3,
            eval_seeds=(3001, 3002, 3003, 3004, 3005),
        )


def test_model_config_rejects_nonpositive_shape():
    with pytest.raises(ValueError):
        ModelConfig(state_dim=0, num_lenses=3, coordinate_mode="raw")
    with pytest.raises(ValueError):
        ModelConfig(state_dim=12, num_lenses=0, coordinate_mode="raw")
