import numpy as np
import torch

from climatereconstructionai import config as cfg
from climatereconstructionai.model.bounds_scaler import (
    constrain_bounds,
    dewpoint_temperature_pairs,
)


def setup_cfg(**overrides):
    cfg.device = torch.device("cpu")
    cfg.weights = None
    cfg.global_padding = False
    cfg.attention = False
    cfg.skip_layers = 1
    cfg.lstm_steps = None
    cfg.gru_steps = None
    cfg.recurrent_steps = 0
    cfg.n_recurrent_steps = 1
    cfg.n_channel_steps = 1
    cfg.n_filters = None
    cfg.conv_factor = 8
    cfg.gnn_layers = 0
    cfg.time_pooling = "none"
    cfg.target_aggregation = "none"
    cfg.gnn_neighbors = 8
    cfg.masked_bn = False
    cfg.disable_first_bn = False
    cfg.decoder_upsample = "bilinear"
    cfg.n_target_data = 0
    cfg.out_channels = 1
    cfg.data_types = ["tas"]
    for key, value in overrides.items():
        setattr(cfg, key, value)


def test_dewpoint_temperature_pairs_detection():
    assert dewpoint_temperature_pairs(["2t", "dewpoint"]) == [(1, 0)]
    assert dewpoint_temperature_pairs(["dewpoint", "t2m"]) == [(0, 1)]
    assert dewpoint_temperature_pairs(["tdps", "tas"]) == [(0, 1)]
    assert dewpoint_temperature_pairs(["tp", "2r"]) == []
    assert dewpoint_temperature_pairs(["tas"]) == []


def test_scale_enforces_dewpoint_not_above_temperature():
    bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    binder = constrain_bounds(bounds, target_types=["2t", "dewpoint"])
    g = torch.Generator().manual_seed(0)
    output = torch.randn(2, 3, 2, 4, 5, generator=g)
    scaled = binder.scale(output.clone())
    assert torch.all(scaled[:, :, 1] <= scaled[:, :, 0])


def test_scale_without_temperature_dewpoint_pair_is_unchanged():
    bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    binder = constrain_bounds(bounds, target_types=["tp", "2r"])
    g = torch.Generator().manual_seed(0)
    output = torch.randn(2, 3, 2, 4, 5, generator=g)
    scaled = binder.scale(output.clone())
    assert torch.equal(scaled, output)


def test_constraint_applies_after_finite_bounds():
    bounds = np.array([[200.0, 320.0], [200.0, 320.0]])
    binder = constrain_bounds(bounds, target_types=["dewpoint", "t2m"])
    g = torch.Generator().manual_seed(1)
    raw = torch.randn(2, 1, 2, 4, 5, generator=g) * 10
    scaled = binder.scale(raw.clone())
    assert torch.all(scaled[:, :, 0] <= scaled[:, :, 1])
    assert torch.all(scaled >= 200.0) and torch.all(scaled <= 320.0)


def test_gradient_flows_through_constraint():
    bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    binder = constrain_bounds(bounds, target_types=["2t", "dewpoint"])
    raw = torch.randn(1, 1, 2, 4, 5, requires_grad=True)
    binder.scale(raw).sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_crainet_forward_enforces_constraint_for_multi_target():
    from climatereconstructionai.model.net import CRAINet

    setup_cfg(
        n_target_data=2,
        out_channels=2,
        data_types=["2t", "dewpoint"],
    )
    bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    model = CRAINet(
        img_size=(16, 16),
        enc_dec_layers=2,
        pool_layers=1,
        in_channels=1,
        out_channels=2,
        bounds=bounds,
    )
    g = torch.Generator().manual_seed(1)
    image = torch.randn(2, 1, 1, 16, 16, generator=g)
    mask = torch.ones(2, 1, 1, 16, 16)
    out = model(image, mask)
    assert out.shape == (2, 1, 2, 16, 16)
    assert torch.all(out[:, :, 1] <= out[:, :, 0])
    out.sum().backward()
