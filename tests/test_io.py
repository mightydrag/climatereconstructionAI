import torch
import torch.nn as nn

from climatereconstructionai.utils.io import load_model, remap_state_dict_keys


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.attention_module = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])


def make_ckpt(state_dict, label="best"):
    return {"labels": [label], label: {"n_iter": 7, "model": state_dict, "optimizer": {}}}


def test_remap_state_dict_keys_legacy_attention():
    state_dict = {
        "conv.weight": 1,
        "attention_0.weight": 2,
        "attention_1.bias": 3,
        "module.conv.bias": 4,
        "module.attention_0.bias": 5,
    }
    remapped = remap_state_dict_keys(state_dict)
    assert remapped == {
        "conv.weight": 1,
        "attention_module.0.weight": 2,
        "attention_module.1.bias": 3,
        "conv.bias": 4,
        "attention_module.0.bias": 5,
    }


def test_remap_state_dict_keys_keeps_current_attention_module():
    state_dict = {"attention_module.0.weight": 1, "module.attention_module.1.bias": 2}
    remapped = remap_state_dict_keys(state_dict)
    assert remapped == {"attention_module.0.weight": 1, "attention_module.1.bias": 2}


def test_load_model_legacy_attention_checkpoint():
    model = ToyNet()
    legacy_state_dict = {
        key.replace("attention_module.", "attention_"): value
        for key, value in model.state_dict().items()
    }
    ckpt_dict = make_ckpt(legacy_state_dict)
    n_iter = load_model(ckpt_dict, model, label="best")
    assert n_iter == 7
    for key, value in model.state_dict().items():
        assert torch.equal(ckpt_dict["best"]["model"][key], value)


def test_load_model_current_checkpoint_with_dataparallel_prefix():
    model = ToyNet()
    dp_state_dict = {f"module.{key}": value for key, value in model.state_dict().items()}
    ckpt_dict = make_ckpt(dp_state_dict)
    load_model(ckpt_dict, model, label="best")
    assert set(ckpt_dict["best"]["model"].keys()) == set(model.state_dict().keys())
