import json
import os
import types


def _is_config_value(value):
    return not isinstance(
        value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)
    )


def save_input_params(cfg, dir, filename="config.json"):
    """
    Save the full configuration to a JSON file in the experiment directory
    (snapshot directory for training, evaluation directory for evaluation).

    Parameters:
    cfg (module): The configuration module containing the parameters.
    dir (str): Directory in which the config file is saved.
    filename (str): The name of the config file to save.
    """
    config = {
        key: value
        for key, value in vars(cfg).items()
        if not key.startswith("_") and _is_config_value(value)
    }

    config_file = os.path.join(dir, filename)
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4, default=str)
