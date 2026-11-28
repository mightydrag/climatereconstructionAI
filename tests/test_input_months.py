import json

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from climatereconstructionai import config as cfg


def setup_cfg(**overrides):
    cfg.loop_random_seed = 3
    cfg.shuffle_masks = False
    cfg.n_target_data = 0
    cfg.out_channels = 1
    cfg.target_aggregation = "none"
    cfg.target_time_offset = 0
    cfg.time_pooling = "none"
    cfg.time_pooling_channel_modes = None
    cfg.precip_log_transform = False
    cfg.normalize_data = False
    cfg.lazy_load = False
    cfg.dataset_name = None
    cfg.min_bounds = [-np.inf]
    cfg.max_bounds = [np.inf]
    cfg.data_types = ["tas"]
    cfg.channel_steps = None
    cfg.lstm_steps = None
    cfg.gru_steps = None
    cfg.gamma_data_types = None
    cfg.input_months = None
    cfg.depth = None
    cfg.device = torch.device("cpu")
    for key, value in overrides.items():
        setattr(cfg, key, value)


def write_monthly_nc(path, var, values, start="2000-01-01"):
    n = len(values)
    time = pd.date_range(start, periods=n, freq="MS")
    data = np.stack(
        [np.full((2, 3), v, dtype="float32") for v in values]
    )
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": np.arange(2), "lon": np.arange(3)},
    )
    ds.to_netcdf(path)


def write_numeric_nc(path, var, n):
    data = np.stack([np.full((2, 3), float(t), dtype="float32") for t in range(n)])
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data)},
        coords={"time": np.arange(n), "lat": np.arange(2), "lon": np.arange(3)},
    )
    ds.to_netcdf(path)


def make_loader(tmp_path, input_months, lazy_load=False, n_files=1):
    from climatereconstructionai.utils.netcdfloader import NetCDFLoader

    train_dir = tmp_path / "train"
    mask_dir = tmp_path / "masks"
    train_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    if n_files == 1:
        write_monthly_nc(train_dir / "tas.nc", "tas", np.arange(24))
        write_monthly_nc(mask_dir / "tas_mask.nc", "tas", np.ones(24))
        img_names = ["tas.nc"]
        mask_names = ["tas_mask.nc"]
    else:
        write_monthly_nc(train_dir / "a.nc", "tas", np.arange(12))
        write_monthly_nc(
            train_dir / "b.nc", "tas", 100 + np.arange(12), start="2001-01-01"
        )
        write_monthly_nc(mask_dir / "a_mask.nc", "tas", np.ones(12))
        write_monthly_nc(
            mask_dir / "b_mask.nc", "tas", np.ones(12), start="2001-01-01"
        )
        with open(train_dir / "files.json", "w") as f:
            json.dump([str(train_dir / "a.nc"), str(train_dir / "b.nc")], f)
        with open(mask_dir / "masks.json", "w") as f:
            json.dump([str(mask_dir / "a_mask.nc"), str(mask_dir / "b_mask.nc")], f)
        img_names = ["files.json"]
        mask_names = ["masks.json"]

    setup_cfg(input_months=input_months, lazy_load=lazy_load)
    loader = NetCDFLoader(
        str(tmp_path),
        img_names,
        str(mask_dir),
        mask_names,
        "train",
        ["tas"],
        [0, 0],
    )
    return loader


def test_input_months_filters_time_axis(tmp_path):
    loader = make_loader(tmp_path, [5, 6, 7, 8, 9])
    # 24 monthly steps (2000-2001), May-Sep keeps 5 months per year
    assert len(loader) == 10
    assert sum(loader.img_length) == 10
    assert sum(loader.mask_length) == 10
    # kept original indices: 4..8 (2000) and 16..20 (2001)
    kept = [4, 5, 6, 7, 8, 16, 17, 18, 19, 20]
    for k, original_index in enumerate(kept):
        _, _, images, index = loader[k]
        assert index == k
        assert images.shape == (1, 1, 2, 3)
        assert images[0, 0, 0, 0].item() == float(original_index)


def test_input_months_multiple_files(tmp_path):
    loader = make_loader(tmp_path, [5, 6, 7, 8, 9], n_files=2)
    assert len(loader) == 10
    # first file (2000): kept values 4..8; second file (2001): kept values 104..108
    assert loader[0][2][0, 0, 0, 0].item() == 4.0
    assert loader[4][2][0, 0, 0, 0].item() == 8.0
    assert loader[6][2][0, 0, 0, 0].item() == 105.0
    assert loader[9][2][0, 0, 0, 0].item() == 108.0


def test_input_months_lazy_load(tmp_path):
    loader = make_loader(tmp_path, [5, 6, 7, 8, 9], lazy_load=True)
    assert len(loader) == 10
    assert loader[0][2][0, 0, 0, 0].item() == 4.0
    assert loader[9][2][0, 0, 0, 0].item() == 20.0


def test_input_months_undecodable_time_raises(tmp_path):
    from climatereconstructionai.utils.netcdfloader import NetCDFLoader

    train_dir = tmp_path / "train"
    mask_dir = tmp_path / "masks"
    train_dir.mkdir()
    mask_dir.mkdir()
    write_numeric_nc(train_dir / "tas.nc", "tas", 24)
    write_numeric_nc(mask_dir / "tas_mask.nc", "tas", 24)

    setup_cfg(input_months=[5, 6, 7, 8, 9])
    with pytest.raises(ValueError, match="--input-months"):
        NetCDFLoader(
            str(tmp_path),
            ["tas.nc"],
            str(mask_dir),
            ["tas_mask.nc"],
            "train",
            ["tas"],
            [0, 0],
        )


def test_input_months_no_match_raises(tmp_path):
    from climatereconstructionai.utils.netcdfloader import NetCDFLoader

    train_dir = tmp_path / "train"
    mask_dir = tmp_path / "masks"
    train_dir.mkdir()
    mask_dir.mkdir()
    # monthly data starting in May: months 5..10, so month 1 never occurs
    write_monthly_nc(train_dir / "tas.nc", "tas", np.arange(6), start="2000-05-01")
    write_monthly_nc(mask_dir / "tas_mask.nc", "tas", np.ones(6), start="2000-05-01")

    setup_cfg(input_months=[1])
    with pytest.raises(ValueError, match="No time steps"):
        NetCDFLoader(
            str(tmp_path),
            ["tas.nc"],
            str(mask_dir),
            ["tas_mask.nc"],
            "train",
            ["tas"],
            [0, 0],
        )


def test_no_input_months_keeps_everything(tmp_path):
    loader = make_loader(tmp_path, None)
    assert len(loader) == 24
    assert loader[23][2][0, 0, 0, 0].item() == 23.0


def test_input_months_filters_xrdss(tmp_path):
    from climatereconstructionai.utils.evaluation import get_xr_dss

    loader = make_loader(tmp_path, [5, 6, 7, 8, 9])
    xrdss_path = tmp_path / "xrdss.nc"
    write_monthly_nc(xrdss_path, "tas", np.arange(24))

    xr_dss = get_xr_dss([str(xrdss_path)], ["tas"])
    time_values = xr_dss[0][0]["time"].values
    assert len(time_values) == 10
    # sample indices of the filtered loader map onto the filtered time axis
    assert len(loader) == len(time_values)
    assert list(pd.DatetimeIndex(time_values).month) == [5, 6, 7, 8, 9] * 2
    assert str(time_values[0])[:10] == "2000-05-01"
    assert str(time_values[-1])[:10] == "2001-09-01"

    setup_cfg()
    xr_dss_full = get_xr_dss([str(xrdss_path)], ["tas"])
    assert len(xr_dss_full[0][0]["time"].values) == 24
