import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xarray as xr

from . import config as cfg
from .utils import skill_utils as su


def evaluate_skill(arg_file=None, prog_func=None):

    cfg.set_evaluate_args(arg_file, prog_func)
    if not os.path.exists(cfg.evaluation_dirs[0]):
        os.makedirs(cfg.evaluation_dirs[0])

    # Load the data
    # Read the JSON file to get the file paths
    print(f"Reading file paths from {cfg.data_root_dir}{cfg.data_names[0]}")
    with open(f"{cfg.data_root_dir}{cfg.data_names[0]}", "r") as f:
        file_paths = json.load(f)

    ensemble_files = file_paths["ensemble_files"]
    gt_files = file_paths["gt_files"]

    # Load the ensemble members as a single dataset
    ensemble_datasets = [xr.open_dataset(f) for f in ensemble_files]
    ds = xr.concat(ensemble_datasets, dim="ensemble")
    var = ds[cfg.data_types[0]].values

    gt_datasets = [xr.open_dataset(f) for f in gt_files]
    ds_gt = xr.concat(gt_datasets, dim="ensemble")
    gt = ds_gt[cfg.data_types[0]].values

    print(f"data var shape: {var.shape}, gt shape: {gt.shape}")

    # Evaluate full timeseries
    su.timeseries_skill(var, gt, cfg.evaluation_dirs[0], name="full")
    su.spatial_skill(var, gt, cfg.evaluation_dirs[0], name="full")

    # Evaluate argo era
    # argo era begins ~2000 --> 754-240=514
    argosplit = 514
    su.timeseries_skill(
        var[:, argosplit:], gt[:, argosplit:], cfg.evaluation_dirs[0], name="argo"
    )
    su.spatial_skill(
        var[:, argosplit:], gt[:, argosplit:], cfg.evaluation_dirs[0], name="argo"
    )

    # Evaluate preargo era
    su.timeseries_skill(
        var[:, :argosplit], gt[:, :argosplit], cfg.evaluation_dirs[0], name="preargo"
    )
    su.spatial_skill(
        var[:, :argosplit], gt[:, :argosplit], cfg.evaluation_dirs[0], name="preargo"
    )


if __name__ == "__main__":
    evaluate_skill()
