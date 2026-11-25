import pytest
import torch

from climatereconstructionai.model.time_pooling import TimePooling


def test_fixed_modes_match_reference():
    h = torch.randn(2, 4, 3, 5, 6)
    assert torch.allclose(TimePooling(3, "mean")(h), h.mean(dim=1, keepdim=True))
    assert torch.allclose(TimePooling(3, "sum")(h), h.sum(dim=1, keepdim=True))
    assert torch.allclose(TimePooling(3, "last")(h), h[:, -1:, :, :, :])


def test_per_channel_modes():
    h = torch.randn(2, 4, 2, 5, 6)
    pool = TimePooling(2, "mean", channel_modes=["mean", "sum"])
    out = pool(h)
    assert out.shape == (2, 1, 2, 5, 6)
    assert torch.allclose(out[:, 0, 0], h[:, :, 0].mean(dim=1))
    assert torch.allclose(out[:, 0, 1], h[:, :, 1].sum(dim=1))


def test_attention_weights_normalized_per_channel_per_pixel():
    torch.manual_seed(0)
    h = torch.randn(2, 4, 3, 5, 6)
    pool = TimePooling(3, "attention")
    w = pool.weights(h)
    assert w.shape == h.shape
    assert torch.allclose(w.sum(dim=1), torch.ones(2, 3, 5, 6), atol=1e-6)
    # per-pixel scores: weights vary across space for a generic input
    assert w[0, :, 0].std() > 0


def test_attention_sum_weights_positive_and_unnormalized():
    torch.manual_seed(0)
    h = torch.randn(2, 4, 2, 5, 6)
    pool = TimePooling(2, channel_modes=["attention", "attention_sum"])
    w = pool.weights(h)
    assert torch.allclose(w[:, :, 0].sum(dim=1), torch.ones(2, 5, 6), atol=1e-6)
    assert (w[:, :, 1] > 0).all()
    assert w[:, :, 1].sum(dim=1).mean() > 1.0


def test_attention_score_receives_grad():
    torch.manual_seed(0)
    pool = TimePooling(2, "attention")
    h = torch.randn(1, 3, 2, 4, 5, requires_grad=True)
    pool(h).sum().backward()
    assert pool.score.weight.grad is not None
    assert h.grad is not None


def test_channel_modes_length_must_match():
    with pytest.raises(AssertionError):
        TimePooling(3, channel_modes=["mean", "sum"])


def test_unknown_mode_rejected():
    with pytest.raises(AssertionError):
        TimePooling(2, "median")


def test_apply_matches_full_weights_all_modes():
    torch.manual_seed(0)
    h = torch.randn(2, 4, 5, 6, 7)
    for modes in (["mean"] * 5, ["sum"] * 5, ["last"] * 5,
                  ["attention"] * 5, ["mean", "sum", "last", "attention", "attention_sum"]):
        pool = TimePooling(5, channel_modes=modes)
        w = pool.weights(h)
        expected = (h * w).sum(dim=1, keepdim=True)
        out = pool(h)
        assert torch.allclose(out, expected, atol=1e-6), modes


def test_apply_state_matches_full_weight_pooling():
    torch.manual_seed(0)
    h = torch.randn(2, 4, 3, 5, 6)
    s = torch.randn(2, 3, 5, 6)
    for modes in (["mean"] * 3, ["sum"] * 3, ["last"] * 3,
                  ["mean", "attention", "attention_sum"]):
        pool = TimePooling(3, channel_modes=modes)
        w = pool.weights(h)
        expected = (s.unsqueeze(1) * w).sum(dim=1)
        w_attn = pool.attention_weights(h)
        out = pool.apply_state(s, h.shape[1], w_attn)
        assert torch.allclose(out, expected, atol=1e-6), modes
