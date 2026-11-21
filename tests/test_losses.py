import torch
import torch.nn.functional as F

from climatereconstructionai import config as cfg
from climatereconstructionai.loss.gamma_loss import GammaLoss, GammaLossValid, gamma_nll
from climatereconstructionai.loss.bern_gamma_loss import (
    BernoulliGammaLoss,
    BernoulliGammaLossValid,
)
from climatereconstructionai.loss.bernoulli_loss import BernoulliLoss


def make_data_dict(batch=2, channels=2, height=6, width=8, seed=0, positive_gt=False):
    g = torch.Generator().manual_seed(seed)
    output = torch.randn(batch, channels, height, width, generator=g)
    gt = torch.randn(batch, channels, height, width, generator=g)
    if positive_gt:
        gt = gt.abs() + 0.1
    mask = (torch.rand(batch, channels, height, width, generator=g) > 0.3).float()
    return {"output": output, "gt": gt, "mask": mask, "loss_mask": None}


def test_gamma_nll_matches_torch_distribution():
    shape = 2.5
    mean = 3.0
    y = torch.tensor([0.05, 0.5, 1.0, 4.0, 20.0])
    dist = torch.distributions.Gamma(concentration=shape, rate=shape / mean)
    expected = -dist.log_prob(y)
    result = gamma_nll(y, torch.full_like(y, mean), shape)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_gamma_loss_hole_and_wet_masking():
    data_dict = make_data_dict(batch=1, channels=1, positive_gt=True, seed=1)
    threshold, shape = 0.2, 1.7
    loss = GammaLoss(threshold=threshold, shape=shape)(data_dict)["gamma"]

    output, gt, mask = data_dict["output"], data_dict["gt"], data_dict["mask"]
    mean = F.softplus(output) + 1e-6
    wet = (gt > threshold).float()
    weight = (1 - mask) * wet
    expected = (weight * gamma_nll(gt, mean, shape)).sum() / weight.sum().clamp(min=1.0)
    assert torch.allclose(loss, expected)


def test_gamma_loss_valid_region():
    data_dict = make_data_dict(batch=1, channels=1, positive_gt=True, seed=2)
    threshold, shape = 0.0, 1.0
    loss = GammaLossValid(threshold=threshold, shape=shape)(data_dict)["gamma_valid"]

    output, gt, mask = data_dict["output"], data_dict["gt"], data_dict["mask"]
    mean = F.softplus(output) + 1e-6
    wet = (gt > threshold).float()
    weight = mask * wet
    expected = (weight * gamma_nll(gt, mean, shape)).sum() / weight.sum().clamp(min=1.0)
    assert torch.allclose(loss, expected)


def test_gamma_loss_channel_selection():
    data_dict = make_data_dict(channels=3, positive_gt=True, seed=3)
    loss_all = GammaLoss(threshold=0.0, shape=1.0)(data_dict)["gamma"]
    loss_ch1 = GammaLoss(threshold=0.0, shape=1.0, channels=[1])(data_dict)["gamma"]

    single = {k: (v[:, 1:2] if v is not None else None) for k, v in data_dict.items()}
    loss_single = GammaLoss(threshold=0.0, shape=1.0)(single)["gamma"]
    assert torch.allclose(loss_ch1, loss_single)

    per_channel = sum(
        GammaLoss(threshold=0.0, shape=1.0, channels=[c])(data_dict)["gamma"]
        for c in range(3)
    )
    assert torch.allclose(loss_all, per_channel)


def test_gamma_loss_all_dry_batch_is_finite():
    data_dict = make_data_dict(batch=1, channels=1, seed=4)
    data_dict["gt"] = -torch.rand(1, 1, 6, 8) - 1.0  # all below threshold
    loss = GammaLoss(threshold=0.0, shape=1.0)(data_dict)["gamma"]
    assert torch.isfinite(loss)


def test_bern_gamma_equals_bern_plus_gamma():
    data_dict = make_data_dict(channels=2, positive_gt=True, seed=5)
    pos_weight, threshold, shape = 3.0, 0.1, 2.0
    combined = BernoulliGammaLoss(
        pos_weight=pos_weight, threshold=threshold, shape=shape
    )(data_dict)["bern_gamma"]
    bern = BernoulliLoss(pos_weight, threshold)(data_dict)["bern"]
    gamma = GammaLoss(threshold=threshold, shape=shape)(data_dict)["gamma"]
    assert torch.allclose(combined, bern + gamma)


def test_bern_gamma_valid_variant():
    data_dict = make_data_dict(channels=2, positive_gt=True, seed=6)
    pos_weight, threshold, shape = 2.0, 0.0, 1.5
    combined = BernoulliGammaLossValid(
        pos_weight=pos_weight, threshold=threshold, shape=shape
    )(data_dict)["bern_gamma_valid"]
    assert torch.isfinite(combined)
    assert combined > 0


def test_loss_gradient_flows_to_output():
    data_dict = make_data_dict(channels=1, positive_gt=True, seed=7)
    data_dict["output"].requires_grad_(True)
    loss = BernoulliGammaLoss(threshold=0.0, shape=1.0)(data_dict)["bern_gamma"]
    loss.backward()
    assert data_dict["output"].grad is not None
    assert torch.isfinite(data_dict["output"].grad).all()


def test_select_channels_routing():
    cfg.data_types = ["tas", "pr", "hurs"]
    cfg.out_channels = 2
    from climatereconstructionai.loss.get_loss import select_channels

    assert select_channels(None) is None
    assert select_channels([]) is None
    assert select_channels(["hurs"]) == [1]
    assert select_channels(["pr", "hurs"]) == [0, 1]
    assert select_channels(["tas"]) == []


def test_loss_computation_with_new_losses():
    from climatereconstructionai.loss.get_loss import LossComputation

    cfg.device = torch.device("cpu")
    cfg.multi_gpus = False
    cfg.recurrent_steps = 0
    cfg.gt_channels = [0, 0]
    cfg.data_types = ["pr", "hurs"]
    cfg.out_channels = 2
    cfg.gamma_data_types = ["hurs"]
    cfg.bern_pos_weight = 2.0
    cfg.bern_threshold = 0.0
    cfg.gamma_shape = 1.5
    cfg.gamma_threshold = 0.0
    cfg.lambda_dict = {"valid": 1.0, "bern_gamma": 2.0, "gamma_valid": 3.0}

    g = torch.Generator().manual_seed(8)
    batch, channels, height, width = 2, 2, 6, 8
    img_mask = (torch.rand(batch, 1, channels, height, width, generator=g) > 0.3).float()
    output = torch.randn(batch, 1, channels, height, width, generator=g)
    gt = torch.rand(batch, 1, channels, height, width, generator=g) + 0.5

    loss_comp = LossComputation(cfg.lambda_dict)
    loss_dict = loss_comp(img_mask, None, output, gt)

    assert "bern_gamma" in loss_dict
    assert "gamma_valid" in loss_dict
    assert "total" in loss_dict
    assert torch.isfinite(loss_dict["total"])
    expected_total = loss_dict["valid"] + loss_dict["bern_gamma"] + loss_dict["gamma_valid"]
    assert abs(loss_dict["total"].item() - expected_total) < 1e-5


def test_channel_thresholds_precip_conversion():
    import numpy as np
    from climatereconstructionai.loss.get_loss import channel_thresholds

    cfg.data_types = ["tas", "pr"]
    cfg.out_channels = 2
    cfg.precip_log_transform = True
    cfg.normalize_data = True
    cfg.precip_log_names = ["tp", "pr"]
    train_stats = {"mean": [0.1, 0.5], "std": [1.0, 0.8]}

    # precip channel: raw 0.0 -> log1p(0)=0 -> z = (0 - 0.5) / 0.8 = -0.625
    thresholds = channel_thresholds(0.0, train_stats)
    assert thresholds[0] == 0.0  # non-precip channel unchanged (normalized space)
    assert abs(thresholds[1] - (np.log1p(0.0) - 0.5) / 0.8) < 1e-12

    # without log transform / normalization: unchanged
    cfg.precip_log_transform = False
    assert channel_thresholds(0.0, train_stats) == [0.0, 0.0]
    cfg.precip_log_transform = True
    cfg.normalize_data = False
    assert channel_thresholds(0.0, train_stats) == [0.0, 0.0]
    cfg.normalize_data = True

    # raw-unit threshold of 0.1 mm: log1p(0.1) then z-norm
    t = channel_thresholds(0.1, train_stats)
    assert abs(t[1] - (np.log1p(0.1) - 0.5) / 0.8) < 1e-12


def test_loss_accepts_per_channel_thresholds():
    data_dict = make_data_dict(channels=2, positive_gt=True, seed=11)
    loss_list = BernoulliLoss(1.0, [0.1, 0.2])(data_dict)["bern"]
    loss_ch = sum(
        BernoulliLoss(1.0, thr)(
            {k: (v[:, c:c + 1] if v is not None else None) for k, v in data_dict.items()}
        )["bern"]
        for c, thr in enumerate([0.1, 0.2])
    )
    assert torch.allclose(loss_list, loss_ch)


def test_set_lambdas_new_presets():
    cfg.lambda_loss = None
    for criterion, keys in [
        (9, {"valid", "bern_gamma", "tv", "prc", "style"}),
        (10, {"valid", "bern_gamma_valid", "tv_valid", "prc", "style"}),
        (11, {"valid", "gamma", "tv", "prc", "style"}),
    ]:
        cfg.loss_criterion = criterion
        cfg.set_lambdas()
        assert set(cfg.lambda_dict.keys()) == keys
        assert all(v > 0 for v in cfg.lambda_dict.values())
