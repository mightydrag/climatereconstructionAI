import json
import os


def save_input_params(cfg, dir, filename="input_params_summary.json"):
    """
    Save the input parameters to a JSON file in the snapshot/evaluation directory.

    Writes the full set of explicitly passed CLI arguments (cfg.passed_args)
    when available, plus the derived channel layout; falls back to the small
    legacy summary otherwise.

    Parameters:
    cfg (module): The configuration module containing the parameters.
    dir (str): Directory to write the summary file into.
    filename (str): The name of the summary file to save.
    """
    passed = getattr(cfg, "passed_args", None)
    if passed:
        input_params = dict(passed)
    else:
        input_params = {
            "lr": cfg.lr,
            "n_filters": cfg.n_filters,
            "out_channels": cfg.out_channels,
            "encoding_layers": cfg.encoding_layers,
            "pooling_layers": cfg.pooling_layers,
        }
    input_params["n_channel_steps"] = getattr(cfg, "n_channel_steps", 1)

    summary_file = os.path.join(dir, filename)
    with open(summary_file, "w") as f:
        json.dump(input_params, f, indent=4, default=str)
