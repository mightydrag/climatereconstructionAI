import numpy as np
import torch

from climatereconstructionai import config as cfg
from climatereconstructionai.model.graph_conv_module import GraphConvBlock


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
    for key, value in overrides.items():
        setattr(cfg, key, value)


def test_graph_conv_shape():
    setup_cfg()
    block = GraphConvBlock(channels=4, neighbors=8)
    h = torch.randn(2, 4, 10, 12)
    out = block(h)
    assert out.shape == h.shape


def test_graph_conv_neighbors_4():
    setup_cfg()
    block = GraphConvBlock(channels=3, neighbors=4)
    h = torch.randn(1, 3, 8, 8)
    out = block(h)
    assert out.shape == h.shape


def test_graph_conv_translation_equivariant_in_longitude():
    setup_cfg(global_padding=True)
    block = GraphConvBlock(channels=2, neighbors=8)
    block.eval()
    h = torch.randn(1, 2, 6, 10)
    shift = 3
    out = block(h)
    out_shifted = block(torch.roll(h, shifts=shift, dims=3))
    assert torch.allclose(out_shifted, torch.roll(out, shifts=shift, dims=3), atol=1e-5)


def test_graph_conv_masked_node_does_not_propagate():
    setup_cfg()
    block = GraphConvBlock(channels=2, neighbors=8)
    block.eval()
    g = torch.Generator().manual_seed(0)
    h = torch.randn(1, 2, 8, 8, generator=g)
    mask = torch.ones(1, 2, 8, 8)
    mask[:, :, 3, 4] = 0.0  # node (3, 4) is a hole
    out_a = block(h, mask)
    h_mod = h.clone()
    h_mod[:, :, 3, 4] = 1e4  # changing the hole value must not affect other nodes
    out_b = block(h_mod, mask)
    keep = torch.ones(8, 8, dtype=torch.bool)
    keep[3, 4] = False
    assert torch.allclose(out_a[:, :, keep], out_b[:, :, keep], atol=1e-4)


def test_crainet_with_gnn_layers():
    from climatereconstructionai.model.net import CRAINet

    setup_cfg(gnn_layers=2, gnn_neighbors=8)
    bounds = np.array([[-np.inf, np.inf]])
    model = CRAINet(
        img_size=(16, 16),
        enc_dec_layers=2,
        pool_layers=1,
        in_channels=1,
        out_channels=1,
        bounds=bounds,
    )
    g = torch.Generator().manual_seed(1)
    image = torch.randn(2, 1, 1, 16, 16, generator=g)
    mask = (torch.rand(2, 1, 1, 16, 16, generator=g) > 0.2).float()
    out = model(image, mask)
    assert out.shape == (2, 1, 1, 16, 16)
    assert any(k.startswith("gnn.") for k in model.state_dict().keys())
    loss = out.sum()
    loss.backward()
    gnn_grads = [p.grad for k, p in model.named_parameters() if k.startswith("gnn.")]
    assert all(grad is not None for grad in gnn_grads)


def test_crainet_without_gnn_has_no_gnn_params():
    from climatereconstructionai.model.net import CRAINet

    setup_cfg(gnn_layers=0)
    bounds = np.array([[-np.inf, np.inf]])
    model = CRAINet(
        img_size=(16, 16),
        enc_dec_layers=2,
        pool_layers=1,
        in_channels=1,
        out_channels=1,
        bounds=bounds,
    )
    assert not any(k.startswith("gnn.") for k in model.state_dict().keys())


def test_crainet_gnn_with_lstm_steps():
    from climatereconstructionai.model.net import CRAINet

    setup_cfg(gnn_layers=2, lstm_steps=[1, 1], recurrent_steps=1, n_recurrent_steps=3)
    bounds = np.array([[-np.inf, np.inf]])
    model = CRAINet(
        img_size=(16, 16),
        enc_dec_layers=2,
        pool_layers=1,
        in_channels=1,
        out_channels=1,
        bounds=bounds,
    )
    g = torch.Generator().manual_seed(2)
    image = torch.randn(2, 3, 1, 16, 16, generator=g)
    mask = (torch.rand(2, 3, 1, 16, 16, generator=g) > 0.2).float()
    out = model(image, mask)
    assert out.shape == (2, 3, 1, 16, 16)
