import json

from climatereconstructionai import config as cfg
from climatereconstructionai.utils.save_input_params import save_input_params


def test_save_input_params_writes_full_passed_args(tmp_path):
    cfg.passed_args = {"lr": 5e-5, "loss_criterion": 14, "hurdle_data_types": ["tp"]}
    cfg.n_channel_steps = 1
    save_input_params(cfg, dir=str(tmp_path))
    data = json.load(open(tmp_path / "input_params_summary.json"))
    assert data["loss_criterion"] == 14
    assert data["hurdle_data_types"] == ["tp"]
    assert data["lr"] == 5e-5
    assert data["n_channel_steps"] == 1


def test_save_input_params_legacy_fallback(tmp_path):
    saved = getattr(cfg, "passed_args", None)
    if saved is not None:
        del cfg.passed_args
    cfg.lr = 1e-4
    cfg.n_filters = None
    cfg.out_channels = 2
    cfg.encoding_layers = [5]
    cfg.pooling_layers = [0]
    cfg.n_channel_steps = 3
    try:
        save_input_params(cfg, dir=str(tmp_path))
        data = json.load(open(tmp_path / "input_params_summary.json"))
        assert data["out_channels"] == 2
        assert data["n_channel_steps"] == 3
        assert "loss_criterion" not in data
    finally:
        if saved is not None:
            cfg.passed_args = saved
