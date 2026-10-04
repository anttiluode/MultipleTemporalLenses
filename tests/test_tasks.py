import torch

from multiple_temporal_lenses.config import canonical_long_config, canonical_short_config
from multiple_temporal_lenses.tasks import (
    make_later_context_batch,
    make_query_timescale_batch,
    make_same_present_pairs,
)


def test_query_timescale_batch_places_one_value_per_scale_and_target_matches_query():
    cfg, _, _ = canonical_short_config()
    batch = make_query_timescale_batch(cfg, batch_size=32, seed=17, split="dev")

    assert batch.sequence.shape == (32, cfg.sequence_length, cfg.vocab_size)
    assert batch.query.shape == (32, 3)
    assert torch.allclose(batch.query.sum(dim=1), torch.ones(32))

    for row in range(32):
        scale = int(batch.query[row].argmax())
        lag = int(batch.metadata["lags"][row, scale])
        value = int(batch.sequence[row, cfg.sequence_length - 1 - lag].argmax())
        assert int(batch.target[row]) == value
        assert int((batch.sequence[row].sum(dim=1) > 0).sum()) == 3
        assert torch.equal(batch.sequence[row, -1], torch.zeros(cfg.vocab_size))


def test_generation_is_seed_reproducible():
    cfg, _, _ = canonical_short_config()
    a = make_query_timescale_batch(cfg, 16, 123, "dev")
    b = make_query_timescale_batch(cfg, 16, 123, "dev")

    assert torch.equal(a.sequence, b.sequence)
    assert torch.equal(a.query, b.query)
    assert torch.equal(a.target, b.target)
    assert torch.equal(a.metadata["lags"], b.metadata["lags"])


def test_heldout_lags_are_disjoint_from_development_lags():
    cfg, _, _ = canonical_short_config()
    dev = make_query_timescale_batch(cfg, 512, 11, "dev")
    held = make_query_timescale_batch(cfg, 128, 3001, "heldout")

    for scale in range(3):
        assert set(held.metadata["lags"][:, scale].tolist()).isdisjoint(
            set(dev.metadata["lags"][:, scale].tolist())
        )


def test_same_present_pairs_have_identical_present_and_query_but_different_history_and_labels():
    cfg, _, _ = canonical_short_config()
    a, b = make_same_present_pairs(cfg, pair_count=24, seed=55)

    assert torch.equal(a.sequence[:, -1], b.sequence[:, -1])
    assert torch.equal(a.query, b.query)
    assert torch.all(a.target != b.target)
    assert torch.all((a.sequence[:, :-1] != b.sequence[:, :-1]).flatten(1).any(dim=1))


def test_same_present_pairs_can_use_frozen_heldout_lags_without_changing_present_contract():
    cfg, _, _ = canonical_short_config()
    a, b = make_same_present_pairs(cfg, pair_count=24, seed=3001, split="heldout")

    assert torch.equal(a.sequence[:, -1], b.sequence[:, -1])
    assert torch.equal(a.query, b.query)
    assert torch.all(a.target != b.target)
    for scale, expected_lag in enumerate((4, 20, 64)):
        assert set(a.metadata["lags"][:, scale].tolist()) == {expected_lag}


def test_later_context_keeps_ambiguous_event_fixed_and_context_resolves_target():
    cfg, _, _ = canonical_short_config()
    batch = make_later_context_batch(cfg, batch_size=20, seed=91)

    assert batch.prefix_sequence.shape == batch.full_sequence.shape
    diff = batch.prefix_sequence != batch.full_sequence
    assert torch.all(diff.flatten(2).any(dim=2).sum(dim=1) == 1)
    assert torch.allclose(batch.neutral_target.sum(dim=1), torch.ones(20))
    assert torch.all((batch.neutral_target > 0).sum(dim=1) == 2)
    assert torch.all(batch.resolved_target >= 0)
    assert torch.all(batch.resolved_target < cfg.vocab_size)


def test_long_config_exact_anchors_are_supported_without_tuning():
    cfg, _, _ = canonical_long_config()
    batch = make_query_timescale_batch(cfg, 8, 7, "stress")
    expected = torch.tensor(cfg.anchor_lags).repeat(8, 1)

    assert torch.equal(batch.metadata["lags"], expected)
