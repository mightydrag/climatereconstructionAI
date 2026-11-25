import numpy as np
import torch
import torch.nn.functional as F

from climatereconstructionai import config as cfg
from climatereconstructionai.model.heads import (
    HurdleGammaHead,
    conserve_hurdle,
    hurdle_output_layout,
)
from climatereconstructionai.loss.hurdle_loss import HurdleGammaLoss, HurdleGammaLossValid
from climatereconstructionai.utils.sampling import correlated_noise, sample_hurdle, sample_members


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
    cfg.multi_gpus = False
    cfg.data_types = ["tas", "pr"]
    cfg.out_channels = 2
    cfg.hurdle_data_types = None
    cfg.hurdle_threshold = 0.1
    cfg.gt_channels = [0, 1]
    cfg.precip_log_transform = False
    cfg.normalize_data = False
    cfg.precip_log_names = ["pr"]
    for key, value in overrides.items():
        setattr(cfg, key, value)


def make_hurdle_data_dict(batch=2, channels=1, height=6, width=8, seed=0):
    # output holds 3 channels per hurdle variable: (logit_p, k, mu)
    g = torch.Generator().manual_seed(seed)
    output = torch.randn(batch, 3 * channels, height, width, generator=g)
    output[:, 1::3] = F.softplus(output[:, 1::3]) + 0.5  # positive k
    output[:, 2::3] = output[:, 2::3].clamp(max=3.0).exp()  # positive mu
    gt = torch.rand(batch, channels, height, width, generator=g).abs() * 5.0
    mask = (torch.rand(batch, channels, height, width, generator=g) > 0.3).float()
    return {"output": output, "gt": gt, "mask": mask, "loss_mask": None}


def test_hurdle_output_layout():
    layout = hurdle_output_layout(["tas", "pr", "hurs"], ["pr"])
    assert layout == [(0, False, 0, 1), (1, True, 1, 3), (2, False, 4, 1)]
    assert hurdle_output_layout(["tas"], None) == [(0, False, 0, 1)]


def test_hurdle_head_shapes_and_positivity():
    head = HurdleGammaHead(in_channels=8, n_hurdle=2)
    h = torch.randn(2, 8, 10, 12)
    logit_p, k, mu = head(h)
    assert logit_p.shape == (2, 2, 10, 12)
    assert k.shape == mu.shape == (2, 2, 10, 12)
    assert (k >= 0.5).all()
    assert (mu > 0).all()


def test_hurdle_head_log_space_residual():
    head = HurdleGammaHead(in_channels=4, n_hurdle=1)
    h = torch.randn(1, 4, 6, 6)
    coarse = torch.rand(1, 1, 6, 6) * 10.0
    _, _, mu_res = head(h, coarse_up=coarse)
    # exact check: mu_res = exp(log_mu + log1p(coarse))
    raw = head.conv(h).chunk(3, dim=1)[2]
    expected = torch.exp(raw + torch.log1p(coarse))
    assert torch.allclose(mu_res, expected, atol=1e-5)


def test_conserve_hurdle_matches_coarse_mean():
    g = torch.Generator().manual_seed(0)
    logit_p = torch.randn(2, 1, 16, 16, generator=g)
    k = torch.rand(2, 1, 16, 16, generator=g) + 1.0
    mu = torch.rand(2, 1, 16, 16, generator=g) * 5.0 + 0.5
    # coarse field at full resolution: correction round-trip is exact
    coarse_pr = torch.rand(2, 1, 16, 16, generator=g) * 3.0
    logit_p2, k2, mu2 = conserve_hurdle(logit_p, k, mu, coarse_pr, max_corr=1e6)
    assert torch.equal(logit_p2, logit_p) and torch.equal(k2, k)
    pooled = F.adaptive_avg_pool2d(torch.sigmoid(logit_p2) * mu2, coarse_pr.shape[-2:])
    assert torch.allclose(pooled, coarse_pr, atol=1e-5)


def test_conserve_hurdle_pooled_coarse_grid():
    g = torch.Generator().manual_seed(0)
    logit_p = torch.randn(2, 1, 16, 16, generator=g)
    k = torch.rand(2, 1, 16, 16, generator=g) + 1.0
    mu = torch.rand(2, 1, 16, 16, generator=g) * 5.0 + 0.5
    coarse_pr = torch.rand(2, 1, 4, 4, generator=g) * 3.0
    _, _, mu2 = conserve_hurdle(logit_p, k, mu, coarse_pr)
    pooled_before = F.adaptive_avg_pool2d(torch.sigmoid(logit_p) * mu, coarse_pr.shape[-2:])
    pooled = F.adaptive_avg_pool2d(torch.sigmoid(logit_p) * mu2, coarse_pr.shape[-2:])
    assert pooled.shape == coarse_pr.shape
    # upsampling the correction is only approximate, but it must reduce the
    # coarse-scale mismatch substantially
    err_before = (pooled_before - coarse_pr).abs().max()
    err_after = (pooled - coarse_pr).abs().max()
    assert err_after < 0.5 * err_before


def test_hurdle_loss_matches_manual_computation():
    setup_cfg(hurdle_data_types=["pr"], data_types=["pr"], out_channels=1)
    data_dict = make_hurdle_data_dict(batch=1, channels=1, seed=1)
    thr = 0.1
    loss = HurdleGammaLoss(threshold=thr)(data_dict)["hurdle_gamma"]

    output, gt, mask = data_dict["output"], data_dict["gt"], data_dict["mask"]
    logit_p, k, mu = output[:, 0:1], output[:, 1:2], output[:, 2:3]
    gt_ch = gt[:, 0:1]
    mask_ch = mask[:, 0:1]
    wet = (gt_ch > thr).float()
    hole = 1 - mask_ch
    bce = F.binary_cross_entropy_with_logits(logit_p, wet, reduction="none")
    occ = (hole * bce).sum() / hole.sum().clamp(min=1.0)
    gamma = torch.distributions.Gamma(k, k / mu.clamp_min(1e-6))
    nll = -gamma.log_prob(gt_ch.clamp_min(1e-6))
    w = hole * wet
    amt = (w * nll).sum() / w.sum().clamp(min=1.0)
    assert torch.allclose(loss, occ + amt)


def test_hurdle_loss_valid_variant_and_all_dry():
    setup_cfg(hurdle_data_types=["pr"], data_types=["pr"], out_channels=1)
    data_dict = make_hurdle_data_dict(batch=1, channels=1, seed=2)
    loss = HurdleGammaLossValid(threshold=0.1)(data_dict)["hurdle_gamma_valid"]
    assert torch.isfinite(loss) and loss > 0

    data_dict["gt"] = torch.zeros(1, 1, 6, 8)  # all dry
    loss_dry = HurdleGammaLoss(threshold=0.1)(data_dict)["hurdle_gamma"]
    assert torch.isfinite(loss_dry)


def test_hurdle_loss_denormalizes_log_precip_target():
    setup_cfg(
        hurdle_data_types=["pr"],
        data_types=["pr"],
        out_channels=1,
        precip_log_transform=True,
        normalize_data=True,
    )
    train_stats = {"mean": [0.5], "std": [0.8]}
    data_dict = make_hurdle_data_dict(batch=1, channels=1, seed=3)
    data_dict["gt"] = torch.rand(1, 1, 6, 8)  # normalized log1p target
    loss = HurdleGammaLoss(threshold=0.1, train_stats=train_stats)(data_dict)["hurdle_gamma"]

    output, gt, mask = data_dict["output"], data_dict["gt"], data_dict["mask"]
    logit_p, k, mu = output[:, 0:1], output[:, 1:2], output[:, 2:3]
    y = torch.expm1(gt[:, 0:1] * 0.8 + 0.5)  # raw target
    wet = (y > 0.1).float()
    hole = 1 - mask[:, 0:1]
    bce = F.binary_cross_entropy_with_logits(logit_p, wet, reduction="none")
    occ = (hole * bce).sum() / hole.sum().clamp(min=1.0)
    gamma = torch.distributions.Gamma(k, k / mu.clamp_min(1e-6))
    nll = -gamma.log_prob(y.clamp_min(1e-6))
    w = hole * wet
    amt = (w * nll).sum() / w.sum().clamp(min=1.0)
    assert torch.allclose(loss, occ + amt)


def test_hurdle_loss_gradient_flows():
    setup_cfg(hurdle_data_types=["pr"], data_types=["pr"], out_channels=1)
    data_dict = make_hurdle_data_dict(batch=1, channels=1, seed=4)
    data_dict["output"].requires_grad_(True)
    loss = HurdleGammaLoss(threshold=0.1)(data_dict)["hurdle_gamma"]
    loss.backward()
    assert data_dict["output"].grad is not None
    assert torch.isfinite(data_dict["output"].grad).all()


def test_hurdle_loss_ignored_without_hurdle_data_types():
    from climatereconstructionai.loss.get_loss import LossComputation

    setup_cfg(hurdle_data_types=None, data_types=["pr", "hurs"], out_channels=2)
    cfg.lambda_dict = {"hurdle_gamma": 6.0}
    cfg.bern_pos_weight = 1.0
    cfg.bern_threshold = 0.0
    cfg.gamma_shape = 1.0
    cfg.gamma_threshold = 0.0

    g = torch.Generator().manual_seed(5)
    img_mask = (torch.rand(2, 1, 2, 6, 8, generator=g) > 0.3).float()
    output = torch.randn(2, 1, 2, 6, 8, generator=g)
    gt = torch.rand(2, 1, 2, 6, 8, generator=g) + 0.5
    loss_dict = LossComputation(cfg.lambda_dict)(img_mask, None, output, gt)
    # hurdle head disabled: hurdle loss must not appear, old behavior unchanged
    assert "hurdle_gamma" not in loss_dict
    assert loss_dict["total"] == 0


def test_hurdle_loss_registered_when_enabled():
    from climatereconstructionai.loss.get_loss import LossComputation

    setup_cfg(hurdle_data_types=["pr"], data_types=["tas", "pr"], out_channels=2)
    cfg.lambda_dict = {"hurdle_gamma": 6.0}
    cfg.bern_pos_weight = 1.0
    cfg.bern_threshold = 0.0
    cfg.gamma_shape = 1.0
    cfg.gamma_threshold = 0.0

    g = torch.Generator().manual_seed(6)
    # output: tas (1 channel) + pr (3 hurdle channels)
    img_mask = (torch.rand(2, 1, 2, 6, 8, generator=g) > 0.3).float()
    output = torch.randn(2, 1, 4, 6, 8, generator=g)
    output[:, :, 2] = F.softplus(output[:, :, 2]) + 0.5
    output[:, :, 3] = output[:, :, 3].clamp(max=3.0).exp()
    gt = torch.rand(2, 1, 2, 6, 8, generator=g) * 5.0
    loss_dict = LossComputation(cfg.lambda_dict)(img_mask, None, output, gt)
    assert "hurdle_gamma" in loss_dict
    assert torch.isfinite(loss_dict["total"])
    assert abs(loss_dict["total"].item() - loss_dict["hurdle_gamma"]) < 1e-5


def test_correlated_noise_is_spatially_correlated():
    torch.manual_seed(0)
    z = correlated_noise((1, 1, 64, 64), sigma_px=6.0, device=torch.device("cpu"))
    assert z.shape == (1, 1, 64, 64)
    assert abs(z.std().item() - 1.0) < 0.1
    corr_z = (z[..., :-1, :] * z[..., 1:, :]).mean().item()
    torch.manual_seed(0)
    w = torch.randn(1, 1, 64, 64)
    corr_w = (w[..., :-1, :] * w[..., 1:, :]).mean().item()
    assert corr_z > corr_w + 0.2


def test_sample_hurdle_nonnegative_and_dry_when_logit_negative():
    g = torch.Generator().manual_seed(1)
    logit_p = torch.randn(2, 1, 12, 12, generator=g)
    k = torch.rand(2, 1, 12, 12, generator=g) + 1.0
    mu = torch.rand(2, 1, 12, 12, generator=g) * 4.0 + 0.5
    x = sample_hurdle(logit_p, k, mu, sigma_px=3.0)
    assert x.shape == logit_p.shape
    assert (x >= 0).all()

    dry = sample_hurdle(torch.full((1, 1, 8, 8), -20.0), k[:1, :, :8, :8], mu[:1, :, :8, :8])
    assert (dry == 0).all()


def test_sample_members_conserves_coarse_total():
    g = torch.Generator().manual_seed(2)
    # very high occurrence logit: every pixel wet, so members are strictly
    # positive and the per-member rescale can match a positive coarse total
    logit_p = torch.full((1, 1, 16, 16), 20.0)
    # large shape k: gamma samples stay away from zero so the per-member
    # rescale can match the coarse total almost exactly
    k = torch.rand(1, 1, 16, 16, generator=g) * 5.0 + 5.0
    mu = torch.rand(1, 1, 16, 16, generator=g) * 3.0 + 0.5
    coarse_pr = torch.rand(1, 1, 16, 16, generator=g) * 2.0  # full resolution: exact
    members = sample_members(logit_p, k, mu, n_members=5, sigma_px=4.0,
                             coarse_pr=coarse_pr, max_corr=1e6)
    assert members.shape == (5, 1, 1, 16, 16)
    assert (members > 0).all()
    # exact per-pixel match is impossible where a sampled amount is ~0 (the
    # correction denominator is clamped there); check per-member mean instead
    pooled_mean = members.reshape(5, 1, 16, 16).mean(dim=(-2, -1))
    coarse_mean = coarse_pr.mean(dim=(-2, -1))
    assert torch.allclose(pooled_mean, coarse_mean.expand_as(pooled_mean), rtol=1e-2)


def test_cranet_hurdle_head_channel_layout():
    from climatereconstructionai.model.net import CRAINet

    bounds = np.ones((2, 2)) * np.inf
    g = torch.Generator().manual_seed(0)
    inp = torch.randn(2, 1, 1, 32, 32, generator=g)
    mask = torch.ones(2, 1, 1, 32, 32)

    # hurdle disabled (default): old single-channel output
    setup_cfg(hurdle_data_types=None)
    model = CRAINet(img_size=(32, 32), enc_dec_layers=2, pool_layers=2,
                    in_channels=1, out_channels=2, bounds=bounds)
    assert model.hurdle_head is None
    out = model(inp, mask)
    assert out.shape == (2, 1, 2, 32, 32)

    # hurdle on pr: tas keeps 1 channel, pr gets 3 (logit, k, mu)
    setup_cfg(hurdle_data_types=["pr"])
    model = CRAINet(img_size=(32, 32), enc_dec_layers=2, pool_layers=2,
                    in_channels=1, out_channels=2, bounds=bounds)
    assert model.hurdle_head is not None
    out = model(inp, mask)
    assert out.shape == (2, 1, 4, 32, 32)
    k, mu = out[:, :, 2], out[:, :, 3]
    assert (k >= 0.5).all() and (mu > 0).all()


def test_set_lambdas_hurdle_presets():
    setup_cfg()  # also resets hurdle_data_types so later test files are unaffected
    cfg.lambda_loss = None
    for criterion, keys in [
        (13, {"valid", "hurdle_gamma", "tv", "prc", "style"}),
        (14, {"valid", "hurdle_gamma_valid", "tv_valid", "prc", "style"}),
    ]:
        cfg.loss_criterion = criterion
        cfg.set_lambdas()
        assert set(cfg.lambda_dict.keys()) == keys
        assert all(v > 0 for v in cfg.lambda_dict.values())
