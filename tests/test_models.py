import torch

from multiple_temporal_lenses.models import (
    QueryConcatLensControl,
    QueryGatedLensModel,
    SingleStateModel,
)


def _example(batch=4, time=12, vocab=8):
    sequence = torch.zeros(batch, time, vocab)
    sequence[:, 1, 0] = 1.0
    sequence[:, 5, 1] = 1.0
    sequence[:, 9, 2] = 1.0
    query = torch.eye(3)[torch.arange(batch) % 3]
    return sequence, query


def _parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_primary_models_have_expected_shapes_and_36_scalar_resident_budget():
    sequence, query = _example()
    models = [
        SingleStateModel(),
        QueryGatedLensModel(coordinate_mode="raw"),
        QueryGatedLensModel(coordinate_mode="band"),
        QueryConcatLensControl(coordinate_mode="raw"),
        QueryConcatLensControl(coordinate_mode="band"),
    ]

    for model in models:
        output = model(sequence, query)
        assert output.logits.shape == (4, 8)
        assert model.resident_state_scalars == 36


def test_gated_and_concat_readers_receive_identical_resident_memory_when_encoder_is_copied():
    torch.manual_seed(4)
    sequence, query = _example()
    gated = QueryGatedLensModel(coordinate_mode="band")
    concat = QueryConcatLensControl(coordinate_mode="band")
    concat.encoder.load_state_dict(gated.encoder.state_dict())

    gated_out = gated(sequence, query)
    concat_out = concat(sequence, query)

    assert torch.equal(
        gated_out.diagnostics["raw_states"], concat_out.diagnostics["raw_states"]
    )
    assert torch.equal(
        gated_out.diagnostics["coordinates"], concat_out.diagnostics["coordinates"]
    )


def test_raw_and_residual_variants_differ_only_after_identical_physical_lowpass_states():
    torch.manual_seed(5)
    sequence, query = _example()
    raw = QueryGatedLensModel(coordinate_mode="raw")
    residual = QueryGatedLensModel(coordinate_mode="band")
    residual.encoder.load_state_dict(raw.encoder.state_dict())

    raw_out = raw(sequence, query)
    residual_out = residual(sequence, query)

    assert torch.equal(
        raw_out.diagnostics["raw_states"], residual_out.diagnostics["raw_states"]
    )
    assert not torch.equal(
        raw_out.diagnostics["coordinates"], residual_out.diagnostics["coordinates"]
    )


def test_gated_reader_exposes_normalized_query_dependent_lens_weights():
    model = QueryGatedLensModel(coordinate_mode="band")
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.weight[0, 0] = 1.0
        model.query_proj.weight.zero_()
        model.query_proj.bias.zero_()
        model.query_proj.weight[:3, :3] = torch.eye(3)
        model.key_proj.weight.copy_(torch.eye(model.state_dim))
        model.key_proj.bias.zero_()

    sequence = torch.zeros(1, 12, 8)
    sequence[:, 1, 0] = 1.0
    q0 = torch.tensor([[1.0, 0.0, 0.0]])
    q1 = torch.tensor([[0.0, 1.0, 0.0]])

    w0 = model(sequence, q0).diagnostics["lens_weights"]
    w1 = model(sequence, q1).diagnostics["lens_weights"]

    assert w0.shape == (1, 3)
    assert torch.allclose(w0.sum(dim=1), torch.ones(1))
    assert torch.allclose(w1.sum(dim=1), torch.ones(1))
    assert not torch.allclose(w0, w1)


def test_concat_reader_receives_query_but_has_no_softmax_selection_variable():
    model = QueryConcatLensControl(coordinate_mode="band")
    sequence = torch.zeros(1, 12, 8)
    q0 = torch.tensor([[1.0, 0.0, 0.0]])
    q1 = torch.tensor([[0.0, 1.0, 0.0]])

    out0 = model(sequence, q0)
    out1 = model(sequence, q1)

    assert "lens_weights" not in out0.diagnostics
    assert "lens_weights" not in out1.diagnostics
    assert not torch.allclose(out0.logits, out1.logits)


def test_concat_reader_is_parameter_matched_to_gated_reader_within_five_percent():
    gated = QueryGatedLensModel(coordinate_mode="band")
    concat = QueryConcatLensControl(coordinate_mode="band")
    gated_count = _parameter_count(gated)
    concat_count = _parameter_count(concat)

    assert abs(gated_count - concat_count) / gated_count <= 0.05
