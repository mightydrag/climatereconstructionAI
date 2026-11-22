import os
import sys

from .model.net import CRAINet
from .utils.evaluation import (
    infill,
    get_batch_size,
    get_xr_dss,
    plot_ensemble_correlation_maps,
    plot_gridwise_correlation,
)
from .utils.io import load_ckpt, load_model
from torch.utils.data import DataLoader
from .utils.netcdfloader import NetCDFLoader, FiniteSampler
from .evaluate_pp_skill import evaluate_pp_skill
from .spei_calculation.spei_calc import compute_spei_from_surplus
import xarray as xr
import numpy as np
from . import config as cfg
import importlib.util
import os
from tqdm import tqdm


def store_encoding(ds):
    global encoding
    encoding = ds["time"].encoding
    return ds

def find_args_file(arg_file):
    if arg_file:
        return arg_file
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token in ("-f", "--load-from-file") and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--load-from-file="):
            return token.split("=", 1)[1]
    return None


def resolve_4a_reference_files():
    from .evaluate_4a import read_data_list

    n_target = cfg.n_target_data if cfg.n_target_data > 0 else 1
    files = []
    for entry in cfg.data_names[-n_target:]:
        path = entry if os.path.isabs(entry) else os.path.join(cfg.data_root_dir, entry)
        if os.path.isfile(path) and path.endswith((".nc", ".h5")):
            files.append(path)
        else:
            for f in read_data_list(path):
                if not os.path.isfile(f) and not os.path.isabs(f):
                    f = os.path.join(cfg.data_root_dir, f)
                files.append(f)
    return files


def run_4a_stats(arg_file=None):
    from . import evaluate_4a

    out_dir = getattr(cfg, "4a_output_dir", None) or os.path.join(
        cfg.evaluation_dirs[0], "4a-stats"
    )
    var = cfg.config_name if cfg.config_name not in (None, "None") else cfg.data_types[-1]
    path_nn = os.path.join(cfg.evaluation_dirs[0], "data")

    ref_files = resolve_4a_reference_files()
    if not ref_files:
        raise ValueError(
            "--run-4a-stats: no reference files could be resolved from --data-names"
        )

    years = {}
    for f in ref_files:
        y = evaluate_4a.extract_year(f)
        if y is not None:
            years[y] = f
    has_real_years = bool(years)
    if not has_real_years:
        years = {0: ref_files[-1]}
    ref_file0 = next(iter(years.values()))
    ref_pattern = os.path.basename(ref_file0)
    path_original = os.path.dirname(ref_file0)
    if has_real_years:
        year0 = evaluate_4a.extract_year(ref_file0)
        ref_pattern = ref_pattern.replace(str(year0), "{year}", 1)

    eval_name = cfg.eval_names[0]
    nn_pattern = eval_name + "_output.nc"
    year_nn = evaluate_4a.extract_year(eval_name)
    if year_nn is not None:
        nn_pattern = eval_name.replace(str(year_nn), "{year}", 1) + "_output.nc"

    years_avail = sorted(
        y
        for y in years
        if os.path.isfile(str(evaluate_4a.pattern_path(path_nn, nn_pattern, var, y)))
    )
    if not years_avail:
        print(
            f"4a stats: no NN output files matching '{nn_pattern}' found in "
            f"{path_nn} -> skipping"
        )
        return

    args_file = find_args_file(arg_file)
    if has_real_years and args_file and os.path.isfile(args_file):
        end_year = evaluate_4a.parse_args_file(args_file).get("end-year")
        if end_year not in (None, True) and years_avail[-1] < int(end_year):
            print(
                f"4a stats: NN outputs available only up to {years_avail[-1]}, "
                f"but the args file sets --end-year {end_year} -> skipping "
                "(rerun after the remaining years are evaluated)"
            )
            return

    mask_file = None
    if cfg.steady_masks:
        entry = cfg.steady_masks[0]
        mask_path = entry if os.path.isabs(entry) else os.path.join(cfg.mask_dir, entry)
        if os.path.isfile(mask_path) and mask_path.endswith((".nc", ".h5")):
            mask_file = mask_path
        elif os.path.isfile(mask_path):
            mask_list = evaluate_4a.read_data_list(mask_path)
            mask_file = mask_list[0] if mask_list else None

    argv = [
        "--path-original", path_original,
        "--path-nn", path_nn,
        "--var", var,
        "--year-min", str(years_avail[0]),
        "--year-max", str(years_avail[-1]),
        "--file-pattern-nn", nn_pattern,
        "--file-pattern-ref", ref_pattern,
        "--output-dir", out_dir,
    ]
    if mask_file:
        argv += ["--mask-file", mask_file]
    if getattr(cfg, "4a_deseasonalize", False):
        argv.append("--deseasonalize")

    print(f"Running 4a statistics analysis (output dir: {out_dir})")
    evaluate_4a.main(argv)


def resolve_model_name(model_dir, name):
    if name in ("best", "latest", "final") and not os.path.exists(
        f"{model_dir}/{name}.pth"
    ):
        candidates = [
            f[:-4]
            for f in os.listdir(model_dir)
            if f.endswith(".pth") and f[:-4].isdigit()
        ]
        if candidates:
            name = max(candidates, key=int)
    print(f"Using model:", name)
    return name

def evaluate(arg_file=None, prog_func=None):
    cfg.set_evaluate_args(arg_file, prog_func)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(f"{cfg.evaluation_dirs[0]}/images"):
        os.makedirs(f"{cfg.evaluation_dirs[0]}/images")
    if not os.path.exists(f"{cfg.evaluation_dirs[0]}/data"):
        os.makedirs(f"{cfg.evaluation_dirs[0]}/data")

    n_models = len(cfg.model_names)
    cfg.model_names = [resolve_model_name(cfg.model_dir, name) for name in cfg.model_names]

    eval_path = [f"{cfg.evaluation_dirs[0]}/data/{name}" for name in cfg.eval_names]
    output_names = {}
    count = 0
    for i_model in range(n_models):

        ckpt_dict = load_ckpt(
            f"{cfg.model_dir}/{cfg.model_names[i_model]}.pth",
            cfg.device,
        )

        if cfg.decoder_upsample is None:
            has_up_conv = any(
                ".up_conv." in key
                for label in ckpt_dict["labels"]
                for key in ckpt_dict[label]["model"]
            )
            cfg.decoder_upsample = "bilinear" if has_up_conv else "nearest"
            print(
                f"Auto-detected --decoder-upsample {cfg.decoder_upsample} "
                f"from checkpoint {cfg.model_names[i_model]}.pth"
            )

        if cfg.use_train_stats:
            data_stats = ckpt_dict["train_stats"]
        else:
            data_stats = None

        print(cfg.data_root_dir, cfg.data_names)
        dataset_val = NetCDFLoader(
            cfg.data_root_dir,
            cfg.data_names,
            cfg.mask_dir,
            cfg.mask_names,
            "infill",
            cfg.data_types,
            cfg.time_steps,
            data_stats,
        )
        xrdss = get_xr_dss(cfg.xrdss_paths, cfg.data_types)

        n_samples = len(dataset_val)
        print(n_samples)
        print(f"Shape of sample input: {dataset_val[0][0].shape}")

        if data_stats is None:
            if cfg.normalize_data:
                print("* Warning! Using mean and std from current data.")
                if cfg.n_target_data != 0:
                    print(
                        "* Warning! Mean and std from target data will be used to renormalize output."
                        " Mean and std from training data can be used with use_train_stats option."
                    )
            data_stats = {"mean": dataset_val.img_mean, "std": dataset_val.img_std}

        image_sizes = dataset_val.img_sizes
        if cfg.conv_factor is None:
            cfg.conv_factor = max(image_sizes[0])

        print(cfg.n_channel_steps)
        if len(image_sizes) - cfg.n_target_data > 1:
            model = CRAINet(
                img_size=image_sizes[0],
                enc_dec_layers=cfg.encoding_layers[0],
                pool_layers=cfg.pooling_layers[0],
                in_channels=cfg.n_channel_steps,
                out_channels=cfg.out_channels,
                fusion_img_size=image_sizes[1],
                fusion_enc_layers=cfg.encoding_layers[1],
                fusion_pool_layers=cfg.pooling_layers[1],
                fusion_in_channels=(len(image_sizes) - 1 - cfg.n_target_data)
                * cfg.n_channel_steps,
                bounds=dataset_val.bounds,
            ).to(cfg.device)
        else:
            model = CRAINet(
                img_size=image_sizes[0],
                enc_dec_layers=cfg.encoding_layers[0],
                pool_layers=cfg.pooling_layers[0],
                in_channels=cfg.n_channel_steps,
                out_channels=cfg.out_channels,
                bounds=dataset_val.bounds,
            ).to(cfg.device)

        for k in range(len(ckpt_dict["labels"])):
            count += 1
            label = ckpt_dict["labels"][k]
            load_model(ckpt_dict, model, label=label)
            model.eval()
            batch_size = get_batch_size(model.parameters(), n_samples, image_sizes)
            iterator_val = iter(
                DataLoader(
                    dataset_val,
                    batch_size=batch_size,
                    sampler=FiniteSampler(len(dataset_val)),
                    num_workers=0,
                )
            )
            infill(
                model,
                iterator_val,
                eval_path,
                output_names,
                data_stats,
                xrdss,
                count,
            )

    for name in output_names:
        if len(output_names[name]) == 1 and len(output_names[name][1]) == 1:
            os.rename(output_names[name][1][0], name + ".nc")
        elif not cfg.split_outputs:
            dss = []
            for i_model in output_names[name]:
                dss.append(
                    xr.open_mfdataset(
                        output_names[name][i_model],
                        preprocess=store_encoding,
                        autoclose=True,
                        combine="nested",
                        data_vars="minimal",
                        concat_dim="time",
                        chunks={},
                    )
                )
                dss[-1] = dss[-1].assign_coords({"member": i_model})

            if len(dss) == 1:
                ds = dss[-1].drop("member")
            else:
                ds = xr.concat(dss, dim="member")

            ds["time"].encoding = encoding
            ds["time"].encoding["original_shape"] = len(ds["time"])
            ds = ds.transpose("time", ...).reset_coords(drop=True)
            ds.to_netcdf(f"{name}.nc")

            for i_model in output_names[name]:
                for output_name in output_names[name][i_model]:
                    os.remove(output_name)
        # with --split-outputs the per-model/per-split files are kept as written

    # --- SPEI post-processing for CWB outputs (editable in-file) ---
    # Edit the following tuple to change the 1-based inclusive month range used per year
    spei_month_range = (1, 3)

    if cfg.data_types[0] == "CWB":
        names_to_process = [n for n in output_names if "output" in n.lower()]
        for name in tqdm(names_to_process, desc="Computing SPEI for outputs"):
            out_file = f"{name}.nc"
            if not os.path.exists(out_file):
                continue
            ds = xr.open_dataset(out_file)
            var = cfg.data_types[0]
            surplus = ds[var].squeeze()
            spei_da = compute_spei_from_surplus(surplus, month_range=spei_month_range)
            # Save SPEI under the variable name 'spei'
            out_ds = spei_da.to_dataset(name="spei")
            out_ds.to_netcdf(f"{name}_spei.nc")
            ds.close()

    if cfg.hindcast_eval == True:
        evaluate_pp_skill()

    if cfg.run_4a_stats:
        run_4a_stats(arg_file)


if __name__ == "__main__":
    evaluate()
