import numpy as np
import pytest
import torch
import xarray as xr

from climatereconstructionai import config as cfg


def setup_cfg(**overrides):
    cfg.loop_random_seed = 3
    cfg.shuffle_masks = False
    cfg.n_target_data = 1
    cfg.out_channels = 1
    cfg.target_aggregation = "none"
    cfg.precip_log_transform = False
    cfg.normalize_data = False
    cfg.lazy_load = False
    cfg.dataset_name = None
    cfg.min_bounds = [-np.inf]
    cfg.max_bounds = [np.inf]
    cfg.data_types = ["tas", "pr"]
    cfg.channel_steps = None
    cfg.lstm_steps = None
    cfg.gru_steps = None
    cfg.gamma_data_types = None
    cfg.device = torch.device("cpu")
    for key, value in overrides.items():
        setattr(cfg, key, value)


def write_nc(path, var, data):
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data)},
        coords={
            "time": np.arange(data.shape[0]),
            "lat": np.arange(data.shape[1]),
            "lon": np.arange(data.shape[2]),
        },
    )
    ds.to_netcdf(path)


def make_loader(tmp_path, aggregation="sum", factor=8, n_days=3, time_steps=(1, 1)):
    from climatereconstructionai.utils.netcdfloader import NetCDFLoader

    rng = np.random.default_rng(0)
    height, width = 4, 6
    n_sub = n_days * factor
    tas_sub = rng.random((n_sub, height, width)).astype("float32")
    pr_day = rng.gamma(2.0, 1.0, size=(n_days, height, width)).astype("float32")
    mask_sub = np.ones((n_sub, height, width), dtype="float32")
    mask_day = np.ones((n_days, height, width), dtype="float32")

    train_dir = tmp_path / "train"
    mask_dir = tmp_path / "masks"
    train_dir.mkdir()
    mask_dir.mkdir()
    write_nc(train_dir / "tas.nc", "tas", tas_sub)
    write_nc(train_dir / "pr.nc", "pr", pr_day)
    write_nc(mask_dir / "tas_mask.nc", "tas", mask_sub)
    write_nc(mask_dir / "pr_mask.nc", "pr", mask_day)

    setup_cfg(target_aggregation=aggregation)
    loader = NetCDFLoader(
        str(tmp_path),
        ["tas.nc", "pr.nc"],
        str(mask_dir),
        ["tas_mask.nc", "pr_mask.nc"],
        "train",
        ["tas", "pr"],
        list(time_steps),
    )
    return loader, tas_sub, pr_day


def test_loader_sample_space_and_windows(tmp_path):
    factor = 8
    loader, tas_sub, pr_day = make_loader(
        tmp_path, aggregation="sum", factor=factor, n_days=3, time_steps=(1, 1)
    )
    assert loader.time_factor == factor
    assert len(loader) == 3

    image, mask, gt, index = loader[1]
    # window of T=3 sub-daily steps centered at day 1 -> steps 7, 8, 9
    assert image.shape == (3, 1, 4, 6)
    assert gt.shape == (3, 1, 4, 6)
    for t, sub_idx in enumerate((7, 8, 9)):
        assert np.allclose(image[t, 0].numpy(), tas_sub[sub_idx])
    # aggregated target is repeated over the whole window
    for t in range(3):
        assert np.allclose(gt[t, 0].numpy(), pr_day[1])
    assert index == 1


def test_loader_rejects_mixed_lengths_without_aggregation(tmp_path):
    from climatereconstructionai.utils.netcdfloader import NetCDFLoader

    rng = np.random.default_rng(0)
    train_dir = tmp_path / "train"
    mask_dir = tmp_path / "masks"
    train_dir.mkdir()
    mask_dir.mkdir()
    write_nc(train_dir / "tas.nc", "tas", rng.random((16, 4, 6)).astype("float32"))
    write_nc(train_dir / "pr.nc", "pr", rng.random((2, 4, 6)).astype("float32"))
    write_nc(mask_dir / "tas_mask.nc", "tas", np.ones((16, 4, 6), dtype="float32"))
    write_nc(mask_dir / "pr_mask.nc", "pr", np.ones((2, 4, 6), dtype="float32"))
    setup_cfg(target_aggregation="none")
    with pytest.raises(AssertionError):
        NetCDFLoader(
            str(tmp_path),
            ["tas.nc", "pr.nc"],
            str(mask_dir),
            ["tas_mask.nc", "pr_mask.nc"],
            "train",
            ["tas", "pr"],
            [1, 1],
        )


def test_prepare_data_dict_aggregation():
    from climatereconstructionai.loss.get_loss import prepare_data_dict

    setup_cfg()
    cfg.recurrent_steps = 1
    cfg.gt_channels = [0]
    cfg.out_channels = 1
    cfg.data_types = ["pr"]

    batch, steps, channels, height, width = 2, 3, 1, 4, 5
    output = torch.randn(batch, steps, channels, height, width)
    mask = torch.ones(batch, steps, channels, height, width)
    gt = torch.randn(batch, steps, channels, height, width)

    for agg, expected in [
        ("sum", output.sum(dim=1)),
        ("mean", output.mean(dim=1)),
        ("none", output[:, 1]),
    ]:
        cfg.target_aggregation = agg
        data_dict = prepare_data_dict(mask, None, output, gt, ["output", "gt", "mask"])
        assert data_dict["output"].shape == (batch, channels, height, width)
        assert torch.allclose(data_dict["output"], expected)


def test_loss_computation_with_aggregation():
    from climatereconstructionai.loss.get_loss import LossComputation

    setup_cfg(target_aggregation="sum", data_types=["pr"])
    cfg.multi_gpus = False
    cfg.recurrent_steps = 1
    cfg.gt_channels = [0]
    cfg.bern_pos_weight = 1.0
    cfg.bern_threshold = 0.0
    cfg.gamma_shape = 1.0
    cfg.gamma_threshold = 0.0
    cfg.lambda_dict = {"bern_gamma": 1.0}

    output = torch.randn(2, 3, 1, 4, 5)
    gt = torch.rand(2, 3, 1, 4, 5) + 0.2
    mask = torch.ones(2, 3, 1, 4, 5)

    loss_dict = LossComputation(cfg.lambda_dict)(mask, None, output, gt)
    assert "bern_gamma" in loss_dict
    assert torch.isfinite(loss_dict["total"])
