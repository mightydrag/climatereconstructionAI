#!/usr/bin/env python3
"""CERRA experiment workflow -- script version of cerra_experiment_config.ipynb.

Edit the SETTINGS block below (paths, region, splits, CONFIGS, MODEL_DEFAULTS,
selected, force_rewrite), then run:

    python3 cerra_experiment_config.py

The script
  1. validates that every required .nc file exists (writes nothing if any is
     missing),
  2. writes the JSON file lists under {ml_path}/data/...,
  3. derives the unique, deterministic (config-name, run-name) pair per
     experiment from a fingerprint of all experiment-defining settings
     (same settings -> same names -> re-run in place; any change -> new hash
     suffix -> new experiment),
  4. writes the train config (train-cnn-cerra-*.txt) and eval config
     (eval-cnn-cerra-*.txt) into input/levante/ and prints the submit commands.

Set force_rewrite = True in the settings to always rewrite the config files,
even when their content is unchanged.
"""


# =========================================================
# 1. General settings (shared by ALL configurations)
# =========================================================
ml_path = "/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions"
data_root = "/work/uo1075/u244085/cerra_data"

reg = "ger"
reg_sfx = f"_{reg}" if reg else ""
other = "remapcon_land"
other_sfx = f"_{other}" if other else ""
res = "2deg"
res_sfx = f"_{res}" if other else ""

# sub-daily (e.g. 3-hourly) input resolution for --target-aggregation runs;
# the sub-daily template keys (input_3h, mask_3h, ...) are derived from it
subdaily_res = "3h"
subdaily_sfx = f"_{subdaily_res}"
descriptor = "_lowres_values_filled_highres_grid"    # "" = no suffix in file/JSON names

# year of the single static files, e.g. orography/lsm stored as ..._2005.nc
static_year = 2005

# ---- binary masks ----
# one mask netCDF per year: {mask_folder}/cerra_dm_{var}_{year}{reg_sfx}_filled_mask.nc
mask_folder = f"{data_root}/mask"
mask_folder_test = f"{data_root}/test/mask"
steady_mask_year = 1985

# the binary-mask JSONs are written once, covering ALL years of all splits
splits = {
    "train": (1985, 2005),
    "val":  (2006, 2015),
    "test":   (2016, 2025),
}

# =========================================================
# 2. Experiment configurations
#    - key = experiment tag (used for selection and printed headers)
#    - "target" = target variable (defaults to the key if omitted);
#      must be a valid var_names key AND the name of the data folder
#    - "static_vars" defaults to [] if omitted
#    - "model" (optional) overrides MODEL_DEFAULTS for this config
# =========================================================
CONFIGS = {
    # original multi-variable experiment
    "relative_humidity-all_var": {
        "input_vars": ["relative_humidity", "t2m", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "relative_humidity",
    },
    "relative_humidity-all_var-atten": {
        "input_vars": ["relative_humidity", "t2m", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "relative_humidity",
        "model": {"attention":True},
    },
    # ---- sub-daily (3-hourly) inputs with a DAILY target ----
    # "input_template": "input_3h" selects the 3-hourly input/mask file layout.
    # The daily target is the aggregation of the 8 sub-daily steps; for
    # precipitation the daily value is the SUM OF THE PREVIOUS 8 steps, i.e. a
    # trailing window: --lstm-steps 7,0 makes day i cover sub-daily steps
    # 8i-7 .. 8i. Set "target_time_offset" if the daily stamp is not aligned
    # with step 8i (e.g. 06UTC-stamped daily sums with 00UTC-start 3-hourly
    # files -> offset 2). --target-aggregation sum for precipitation totals,
    # mean for daily means of the other variables. Pass the same flags at
    # evaluation (they are part of the generated eval config automatically).
    "tp-all_var-3h-daily": {
        "input_vars": ["tp", "t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "tp",
        "input_template": f"input_{subdaily_res}",
        "daily_input_vars": ["tp"],   # tp exists only daily -> daily input file
        "model": {"lr": 5e-5,
                  "conv-factor": 512,
                  "precip-log-transform": True,
                  "loss_criterion": 10,           # Bernoulli-Gamma on the valid region: pure downscaling, no holes to infill
                  "gamma_data_types": "tp",
                  "lstm_steps": [7, 0],           # trailing window: previous 8 steps
                  "target_aggregation": "sum",
                  "target_time_offset": 0,
                  "time_pooling": "attention",    # model maps the 8 steps to the daily total
                  "gnn_layers": 2},
    },
    "relative_humidity-all_var-3h-daily": {
        "input_vars": ["relative_humidity", "t2m", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "relative_humidity",
        "input_template": f"input_{subdaily_res}",
        "model": {"loss_criterion": 12,         # gamma on the valid region: pure downscaling
                  "gamma_data_types": "2r",
                  "min_bounds": [0], "max_bounds": [100],   # physical range of 2r
                  "lstm_steps": [7, 0],
                  "target_aggregation": "mean",
                  "target_time_offset": 0,
                  "time_pooling": "mean",         # daily mean directly from the window
                  "gnn_layers": 2},
    },
    # precipitation experiment: tp + drivers
    "tp_tciwv-all_var": {
        "input_vars": ["tciwv", "t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "tp",
        "model": {"lr": 5e-5,
                  "conv-factor": 512,
                  "precip-log-transform": True,
                  "loss_criterion": 8, },
    },
    "tp_tciwv-all_var-atten": {
        "input_vars": ["tciwv", "t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "tp",
        "model": {"lr": 5e-5,
                  "conv-factor": 512,
                  "precip-log-transform": True,
                  "loss_criterion": 8, 
                  "attention":True},
    },
    "tp-all_var": {
        "input_vars": ["tp", "t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "tp",
        "model": {"lr": 5e-5,
            "conv-factor": 512,
            "precip-log-transform": True,
            "loss_criterion": 8, },
    },
    "tp-all_var-atten": {
        "input_vars": ["tp", "t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "tp",
        "model": {"lr": 5e-5,
            "conv-factor": 512,
            "precip-log-transform": True,
            "loss_criterion": 8, 
            "attention":True},
    },
    # t2m + drivers
    "t2m-all_var": {
        "target": "t2m",
        "input_vars": ["t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
    },
    "t2m-all_var-atten": {
        "target": "t2m",
        "input_vars": ["t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "model": {"attention":True},
    },
    # ablations: single-input baselines
    "t2m-single": {
        "target": "t2m",
        "input_vars": ["t2m"],
        "static_vars": [],
    },
    "relative_humidity-single": {
        "target": "relative_humidity",
        "input_vars": ["relative_humidity"],
        "static_vars": [],
    },
    "tp-single": {
        "target": "tp",
        "input_vars": ["tp"],
        "static_vars": [],
    },
}

# =========================================================
# 2b. Model / run hyperparameters
#     EVERY key here (or in a config's "model" override) is written to the
#     generated config files as "--<key> <value>" (underscores -> dashes).
#     A boolean True becomes the bare flag (store_true); False omits it.
#     So any argument of climatereconstructionai/config.py can be set here.
#     Shared defaults; any key can be overridden per config via a
#     "model" dict inside the CONFIGS entry. All of these are part of
#     the experiment fingerprint: changing one yields a NEW run name
#     (a new experiment), never a silent overwrite of the old run.
# =========================================================
MODEL_DEFAULTS = {
    "device": "cuda",
    "n_threads": 64,
    "n_target_data": 1,
    "multi_gpus": True,   # -> --multi-gpus
    "lr": 5e-4,
    "out_channels": 1,
    "dataset_name": "cerra_ger_mult_file",
    "plot_results": "0,100",
    "shuffle_masks": True,
    "encoding_layers": "5",
    "pooling_layers": "0",
    "normalize_data": True,
    "decoder_upsample": "bilinear",
    "conv_factor": 128,
    "batch_size": 64,
    "loss_criterion": 7,
    "save_model_interval": 2500,
    "log_interval": 500,
    "eval_timesteps": "0,100",
    "max_iter": 100000,
    "attention": False,
    "precip-log-transform": False,
    "model_names": "best",      # eval only: which checkpoint(s) to evaluate
}

# ---- select which config to create the files for ----
# one key of CONFIGS -> only that config; None -> ALL configs
selected = "tp-all_var-atten"
if not selected in CONFIGS.keys():
    raise ValueError("WRONG CONFIG SELECTED")
# ---- where the generated train/eval config files are written ----
# the file-name prefixes match input/slurm-scripts/submit-train-cerra.sh and
# submit-eval-cerra.sh, which derive the slurm job name from the file name
config_out_dir = f"{ml_path}/input/levante"
train_prefix = "train-cnn-cerra-"
eval_prefix = "eval-cnn-cerra-"

# ---- force rewrite of the generated train/eval config files ----
# False (default): a file is only rewritten when its content changed
#                  (re-running the same experiment prints "unchanged")
# True:            always rewrite, even if an identical file already exists
force_rewrite = False



# =========================================================
# 2b. FILE-NAME LAYOUT -- change the base layout HERE ONLY
#
#     Placeholders usable in NC_TPL / JSON_TPL (filled via BASE):
#       {data_root} {mask_folder} {mask_folder_test} {var} {year}
#       {reg} {reg_sfx} {other_sfx} {res_sfx} {descriptor} {static_year}
#       {steady_mask_year} {selected} {split}
#     ANY of these may be used in ANY NC_TPL / JSON_TPL entry --
#     BASE supplies them all to every .format() call.
# =========================================================
NC_TPL = {
    "input":       "{data_root}/{var}/cerra_dm_{var}_{year}{reg_sfx}{descriptor}{res_sfx}.nc",
    "target":      "{data_root}/{var}/cerra_dm_{var}_{year}{other_sfx}{reg_sfx}.nc",
    "static":      "{data_root}/{var}/cerra_dm_{var}_{static_year}{other_sfx}{reg_sfx}.nc",
    "mask":        "{mask_folder}/cerra_dm_{var}_{year}{reg_sfx}_filled_mask.nc",
    "mask_static": "{mask_folder}/cerra_dm_{var}_{static_year}{reg_sfx}_filled_mask.nc",
    # same layout as "mask", but listed separately so the test-split masks
    # can point at a different folder if ever needed:
    # swap {mask_folder} -> {mask_folder_test} here to use the test masks
    "mask_test":   "{mask_folder}/cerra_dm_{var}_{year}{reg_sfx}_filled_mask.nc",
    "steady_mask": "{mask_folder}/cerra_dm_{var}_{steady_mask_year}{reg_sfx}_steady_mask.nc",
}

JSON_TPL = {
    "input":       "cerra_{var}{reg_sfx}{descriptor}_{split}.json",
    "static":      "cerra_{var}{reg_sfx}_{split}.json",
    "target":      "cerra_{var}{reg_sfx}{other_sfx}_target_{split}.json",
    "mask":        "cerra_{var}{reg_sfx}{other_sfx}_mask.json",
    "mask_test":   "cerra_{var}{reg_sfx}{other_sfx}_mask_test.json",
    "steady_mask": "cerra_{selected}_{reg_sfx}{other_sfx}_steady_mask.json",
}


# sub-daily input/mask templates for the chosen resolution (e.g. _3h files);
# adjust the patterns to the actual sub-daily CERRA file names
NC_TPL[f"input_{subdaily_res}"] = (
    "{data_root}/{var}/cerra_dm_{var}_{year}{reg_sfx}{descriptor}{res_sfx}"
    + subdaily_sfx + ".nc"
)
NC_TPL[f"mask_{subdaily_res}"] = (
    "{mask_folder}/cerra_dm_{var}_{year}{reg_sfx}_filled_mask" + subdaily_sfx + ".nc"
)
NC_TPL[f"mask_test_{subdaily_res}"] = NC_TPL[f"mask_{subdaily_res}"]
JSON_TPL[f"input_{subdaily_res}"] = (
    "cerra_{var}{reg_sfx}{descriptor}_{split}" + subdaily_sfx + ".json"
)
JSON_TPL[f"mask_{subdaily_res}"] = (
    "cerra_{var}{reg_sfx}{other_sfx}_mask" + subdaily_sfx + ".json"
)
JSON_TPL[f"mask_test_{subdaily_res}"] = (
    "cerra_{var}{reg_sfx}{other_sfx}_mask_test" + subdaily_sfx + ".json"
)


def _tpl_pair(input_tpl):
    """input template key -> (mask template key, mask-test template key)."""
    if input_tpl == "input":
        return "mask", "mask_test"
    res = input_tpl[len("input_"):]
    return f"mask_{res}", f"mask_test_{res}"


def _input_tpl_for(var, input_tpl, daily_input_vars):
    """Per-variable input template: vars in daily_input_vars use the daily
    'input'/'mask' templates even in a sub-daily (3h) experiment, so a
    daily-resolution file (e.g. daily precipitation) can be an input next to
    the 3-hourly ones without interpolation."""
    return "input" if var in daily_input_vars else input_tpl


# common placeholders, filled in once from the section-1 block
BASE = dict(
    data_root=data_root,
    mask_folder=mask_folder,
    mask_folder_test=mask_folder_test,
    reg=reg,
    reg_sfx=reg_sfx,
    other_sfx=other_sfx,
    res_sfx=res_sfx,
    descriptor=descriptor,
    static_year=static_year,
    steady_mask_year=steady_mask_year,
    selected=selected,
)

# evaluate-run templates: DERIVED from NC_TPL (defined after BASE!).
# {var} / {year} / {reg} stay LITERAL for the slurm scripts; every other
# placeholder ({data_root}, {descriptor}, {res_sfx}, {static_year}, ...)
# is filled in here from BASE.


def eval_tpl(nc_key):
    return (
        NC_TPL[nc_key]
        .replace("{var}", "{{var}}")
        .replace("{year}", "{{year}}")
        .replace("{reg_sfx}", "_{{reg}}" if reg else "")
        .format(**BASE)
    )


_EVAL_KEYS = {  # eval flag -> NC_TPL key
    "data":        "input",
    "static":      "static",
    "target":      "target",
    "mask":        "mask",
    "static_mask": "mask_static",
    "steady_mask": "steady_mask",
}
EVAL_TPL = {flag: eval_tpl(key) for flag, key in _EVAL_KEYS.items()}



import os
import json

# CERRA short names as reported by `cdo showname`
var_names = {
    "t2m":               "2t",
    "orography":         "orog",
    "land_sea_mask":     "lsm",
    "tp":                "tp",
    "relative_humidity": "2r",
    "surface_pressure":  "sp",
    "wind_speed":        "10si",
    "wind_direction":    "10wdir",
    "tciwv":             "tciwv",
}


# =========================================================
# 3. Helpers
# =========================================================
def check_files(target_var, input_vars, static_vars, input_tpl="input",
                daily_input_vars=()):
    """Return list of missing files for one configuration (empty = all present)."""
    mask_tpl, _ = _tpl_pair(input_tpl)
    missing = []
    for split, (start_year, end_year) in splits.items():
        if split == "test":
            continue  # test split: evaluated per year via file templates, no JSONs
        for year in range(start_year, end_year + 1):
            f_tgt = NC_TPL["target"].format(**BASE, var=target_var, year=year)
            if not os.path.isfile(f_tgt):
                missing.append(f_tgt)
            for var in input_vars:
                v_tpl = _input_tpl_for(var, input_tpl, daily_input_vars)
                v_mask_tpl = "mask" if var in daily_input_vars else mask_tpl
                f_in = NC_TPL[v_tpl].format(**BASE, var=var, year=year)
                if not os.path.isfile(f_in):
                    missing.append(f_in)
                # binary mask: one file per year, needed for every input var
                f_mask = NC_TPL[v_mask_tpl].format(**BASE, var=var, year=year)
                if not os.path.isfile(f_mask):
                    missing.append(f_mask)

    # static files: only one per variable needs to exist
    for var in static_vars:
        f_st = NC_TPL["static"].format(**BASE, var=var)
        if not os.path.isfile(f_st):
            missing.append(f_st)

    # steady mask: only one file, for the target variable
    f_steady = steady_mask_file(target_var)
    if not os.path.isfile(f_steady):
        missing.append(f_steady)
    return missing


def steady_mask_file(target_var):
    return NC_TPL["steady_mask"].format(**BASE, var=target_var)


def write_config(target_var, input_vars, static_vars, input_tpl="input",
                 daily_input_vars=()):
    """Write all JSON file lists for one configuration, return (flags, steady_mask_name)."""
    mask_tpl, mask_test_tpl = _tpl_pair(input_tpl)
    flags = {}

    for split, (start_year, end_year) in splits.items():
        years = range(start_year, end_year + 1)
        n_years = len(years)
        flags[split] = []

        # time-varying inputs: one file per year (daily file for daily_input_vars)
        for var in input_vars:
            v_tpl = _input_tpl_for(var, input_tpl, daily_input_vars)
            file_list_in = [
                NC_TPL[v_tpl].format(**BASE, var=var, year=year)
                for year in years
            ]
            path = os.path.join(ml_path, "data", split, var)
            os.makedirs(path, exist_ok=True)
            name_in = JSON_TPL[v_tpl].format(**BASE, var=var, split=split)
            with open(os.path.join(path, name_in), "w") as f:
                json.dump(file_list_in, f, indent=2)
            flags[split].append(f"{var}/{name_in}")
            print(f"{split}/{var}: {len(file_list_in)} input files -> {path}")

        # static variables: single file, repeated for every year so all lists
        # have the same length
        for var in static_vars:
            f_st = NC_TPL["static"].format(**BASE, var=var)
            file_list_st = [f_st] * n_years
            path = os.path.join(ml_path, "data", split, var)
            os.makedirs(path, exist_ok=True)
            name_st = JSON_TPL["static"].format(**BASE, var=var, split=split)
            with open(os.path.join(path, name_st), "w") as f:
                json.dump(file_list_st, f, indent=2)
            flags[split].append(f"{var}/{name_st}")
            print(f"{split}/{var}: 1 static file x {n_years} -> {path}")

        # target data: native 0.05 deg field of the target variable (always LAST)
        file_list_tgt = [
            NC_TPL["target"].format(**BASE, var=target_var, year=year)
            for year in years
        ]
        path = os.path.join(ml_path, "data", split, target_var)
        os.makedirs(path, exist_ok=True)
        name_tgt = JSON_TPL["target"].format(**BASE, var=target_var, split=split)
        with open(os.path.join(path, name_tgt), "w") as f:
            json.dump(file_list_tgt, f, indent=2)
        flags[split].append(f"{target_var}/{name_tgt}")
        print(f"{split}/{target_var}: {len(file_list_tgt)} target files -> {path}")

    # ---- binary masks: single JSON per input var, covering all years ----
    flags["mask"] = []
    tv_years = sorted(
        {y for s in ("train", "val") for y in range(splits[s][0], splits[s][1] + 1)}
    )
    for var in [*input_vars, *static_vars, target_var]:
        path = os.path.join(ml_path, "data", "masks", var)
        os.makedirs(path, exist_ok=True)
        # sub-daily inputs need sub-daily masks (same time axis as the inputs);
        # daily inputs and the target use the daily mask template
        if var in input_vars:
            v_mask_tpl = "mask" if var in daily_input_vars else mask_tpl
        else:
            v_mask_tpl = "mask"
        name_mask = JSON_TPL[v_mask_tpl].format(**BASE, var=var)
        if var in static_vars:
            # same logic as the static input files: one mask file
            # (from static_year) repeated so all lists have the same length
            f_mask = NC_TPL["mask_static"].format(**BASE, var=var)
            file_list_mask = [f_mask] * len(tv_years)
            print(f"mask: 1 static mask file x {len(tv_years)} -> {path}")
        else:
            file_list_mask = [
                NC_TPL[v_mask_tpl].format(**BASE, var=var, year=year)
                for year in tv_years
            ]
        with open(os.path.join(path, name_mask), "w") as f:
            json.dump(file_list_mask, f, indent=2)
        flags["mask"].append(f"{var}/{name_mask}")
        print(f"mask: {len(file_list_mask)} mask files ({tv_years[0]}-{tv_years[-1]}) -> {path}")

    # ---- binary masks for the test split only: separate folder, test years ----
    flags["mask_test"] = []
    test_start, test_end = splits["test"]
    test_years = range(test_start, test_end + 1)
    for var in [*input_vars, *static_vars, target_var]:
        path = os.path.join(ml_path, "data", "test", "masks", var)
        os.makedirs(path, exist_ok=True)
        if var in input_vars:
            v_mask_test_tpl = "mask_test" if var in daily_input_vars else mask_test_tpl
        else:
            v_mask_test_tpl = "mask_test"
        name_mask_test = JSON_TPL[v_mask_test_tpl].format(**BASE, var=var)
        if var in static_vars:
            f_mask = NC_TPL["mask_static"].format(**BASE, var=var)
            file_list_mask_test = [f_mask] * len(test_years)
        else:
            file_list_mask_test = [
                NC_TPL[v_mask_test_tpl].format(**BASE, var=var, year=year)
                for year in test_years
            ]
        with open(os.path.join(path, name_mask_test), "w") as f:
            json.dump(file_list_mask_test, f, indent=2)
        flags["mask_test"].append(f"{var}/{name_mask_test}")
        print(f"test mask: {len(file_list_mask_test)} mask file(s) "
              f"({test_years[0]}-{test_years[-1]}) -> {path}")

    # ---- steady mask (single file for the target channels, same folder) ----
    path = os.path.join(ml_path, "data", "masks", target_var)
    os.makedirs(path, exist_ok=True)
    name_steady = JSON_TPL["steady_mask"].format(**BASE)
    with open(os.path.join(path, name_steady), "w") as f:
        json.dump([steady_mask_file(target_var)], f, indent=2)
    print(f"steady mask: 1 file -> {path}")

    # steady mask copy for the evaluate run: in evaluate mode --steady-masks
    # resolves under --mask-dir, so it must exist in the test mask dir too
    path = os.path.join(ml_path, "data", "test", "masks", target_var)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name_steady), "w") as f:
        json.dump([steady_mask_file(target_var)], f, indent=2)
    print(f"steady mask (test): 1 file -> {path}")

    return flags, name_steady



# =========================================================
# 4. Main: validate everything, then write the JSON lists
# =========================================================
# ---- resolve the selection once, fail early on a bad name ----
if selected is None:
    run_configs = CONFIGS
elif selected in CONFIGS:
    run_configs = {selected: CONFIGS[selected]}
else:
    print(f"ERROR: selected config '{selected}' is not defined. Valid keys: {', '.join(CONFIGS)}")
    raise SystemExit(1)

# every configured variable needs a short name for --data-types

unknown = []
for key, cfg in run_configs.items():
    target_var = cfg.get("target", key)
    for v in cfg["input_vars"] + cfg.get("static_vars", []) + [target_var]:
        if v not in var_names:
            unknown.append(v)
if unknown:
    print(f"ERROR: no CERRA short name defined in var_names for: {sorted(set(unknown))}")
    raise SystemExit(1)

# ---- quick check: EVERY file of EVERY config must exist before we write anything ----
missing_by_config = {}
for key, cfg in run_configs.items():
    target_var = cfg.get("target", key)
    m = check_files(target_var, cfg["input_vars"], cfg.get("static_vars", []),
                    cfg.get("input_template", "input"),
                    cfg.get("daily_input_vars", []))
    if m:
        missing_by_config[key] = m

if missing_by_config:
    total = sum(len(m) for m in missing_by_config.values())
    print(f"ERROR: {total} file(s) missing -- no JSONs written:")
    for key, m in missing_by_config.items():
        print(f"  [{key}] {len(m)} missing:")
        for f in m:
            print(f"    {f}")
    raise SystemExit(1)

print("All files present. Writing JSONs...")

# results per selected config, used by the config-generation cell below
written = {}

for key, cfg in run_configs.items():
    target_var = cfg.get("target", key)
    input_vars = cfg["input_vars"]
    static_vars = cfg.get("static_vars", [])

    print()
    print(f"==================== config: {key} (target {target_var}) ====================")
    flags, name_steady = write_config(target_var, input_vars, static_vars,
                                      cfg.get("input_template", "input"),
                                      cfg.get("daily_input_vars", []))
    written[key] = {
        "cfg": cfg,
        "flags": flags,
        "name_steady": name_steady,
        "target_var": target_var,
    }

print()
print("done")



import hashlib


def model_params(cfg):
    """MODEL_DEFAULTS overridden by the config's own "model" dict.

    Keys are normalized (dashes -> underscores), so an override may use the
    CLI spelling ("conv-factor", "precip-log-transform") or the Python one
    ("conv_factor", "precip_log_transform").
    """
    m = {k.replace("-", "_"): v for k, v in MODEL_DEFAULTS.items()}
    m.update({k.replace("-", "_"): v for k, v in cfg.get("model", {}).items()})
    return m


def data_types(key, cfg):
    """--data-types list: input types, static types, target type (LAST)."""
    target_var = cfg.get("target", key)
    input_types = [var_names[v] for v in cfg["input_vars"]]
    static_types = [var_names[v] for v in cfg.get("static_vars", [])]
    return input_types + static_types + [var_names[target_var]]


def experiment_fingerprint(key, cfg):
    """Short hash over everything that defines the experiment.

    Same settings -> same hash -> same run name, so re-running an experiment
    reuses the same snapshot/evaluation directories. Any changed setting ->
    different hash -> new run name, so no run is ever overwritten silently.
    """
    spec = {
        "config_key": key,
        "target": cfg.get("target", key),
        "input_vars": cfg["input_vars"],
        "static_vars": cfg.get("static_vars", []),
        "model": model_params(cfg),
        "reg": reg,
        "other": other,
        "res": res,
        "descriptor": descriptor,
        "splits": {k: list(v) for k, v in splits.items()},
        "static_year": static_year,
        "steady_mask_year": steady_mask_year,
        "nc_tpl": NC_TPL,
        "json_tpl": JSON_TPL,
    }
    blob = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def experiment_names(key, cfg):
    """Return the unique (config_name, run_name) pair for one experiment.

    config_name = target variable, so the output paths
    evaluations/{config_name}/{run_name} match the 4a evaluation scripts,
    which derive the variable from the last --data-types entry.
    run_name    = readable tag (region, resolution, inputs, architecture,
    loss, upsampling, batch size) + fingerprint hash.
    """
    m = model_params(cfg)
    target_var = cfg.get("target", key)
    up_tag = {"bilinear": "bilin", "nearest": "near"}.get(
        m["decoder_upsample"], str(m["decoder_upsample"])
    )

    def num_tag(v):
        # encoding/pooling layers may be "5,3" (primary,secondary / fusion) or
        # a list [5, 3]; the run-name tag must be filesystem-safe
        if isinstance(v, (list, tuple)):
            return "-".join(str(x) for x in v)
        return str(v).replace(",", "-")

    el_tag = num_tag(m["encoding_layers"])
    po_tag = num_tag(m["pooling_layers"])
    h = experiment_fingerprint(key, cfg)
    config_name = target_var
    run_name = (
        f"cerra-{selected}-{reg}-{res}"
        f"-el{el_tag}-po{po_tag}-cf{m['conv_factor']}"
        f"-l{m['loss_criterion']}-{up_tag}-b{m['batch_size']}-{h}"
    )
    return config_name, run_name


def header_block(config_name, run_name):
    return [
        "--data-root-dir ./data/",
        "--mask-dir ./data/masks/",
        "--model-dir ./snapshots/",
        "--evaluation-dirs ./evaluations/",
        "--log-dir ./logs/",
        "",
        f"--config-name {config_name}",
        f"--run-name {run_name}",
        "",
    ]


def model_block(m):
    """Pass every model param through as a --flag line.

    key -> "--<key with dashes> value"; True booleans become the bare flag
    (store_true), False omits the line; lists are comma-joined; dicts are
    written as key,value pairs (e.g. lambda_loss). Anything set in
    MODEL_DEFAULTS or a per-config "model" dict ends up in the config files.
    """
    lines = []
    for key, val in m.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                lines.append(flag)
        elif isinstance(val, (list, tuple)):
            lines.append(f"{flag} {','.join(str(v) for v in val)}")
        elif isinstance(val, dict):
            lines.append(f"{flag} {','.join(f'{k},{v}' for k, v in val.items())}")
        else:
            lines.append(f"{flag} {val}")
        if key in ("dataset_name", "loss_criterion", "save_model_interval"):
            lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def render_train_config(key, entry, names):
    """Training config: JSON file lists, parsed by train --load-from-file."""
    cfg = entry["cfg"]
    flags = entry["flags"]
    m = model_params(cfg)
    config_name, run_name = names
    target_var = entry["target_var"]
    lines = header_block(config_name, run_name)
    lines += [
        f"--data-names {','.join(flags['train'])}",
        f"--mask-names {','.join(flags['mask'])}",
        f"--steady-masks {target_var}/{entry['name_steady']}",
        f"--val-names {','.join(flags['val'])}",
        f"--test-names {','.join(flags['test'])}",
        f"--xrdss-paths masks/{target_var}",
        f"--data-types {','.join(data_types(key, cfg))}",
        "",
    ]
    lines += model_block(m)
    return "\n".join(lines) + "\n"


def render_eval_config(key, entry, names):
    """Evaluation config: per-year file templates for the slurm eval scripts."""
    cfg = entry["cfg"]
    flags = entry["flags"]
    m = model_params(cfg)
    config_name, run_name = names
    test_start, test_end = splits["test"]
    # sub-daily experiments evaluate on the sub-daily input/mask templates
    input_tpl = cfg.get("input_template", "input")
    mask_tpl, _ = _tpl_pair(input_tpl)
    lines = header_block(config_name, run_name)
    lines += [
        f"--data-file-template {eval_tpl(input_tpl)}",
        f"--static-file-template {EVAL_TPL['static']}",
        f"--target-file-template {EVAL_TPL['target']}",
        f"--mask-file-template {eval_tpl(mask_tpl)}",
        f"--static-mask-template {EVAL_TPL['static_mask']}",
        f"--steady-mask-file {EVAL_TPL['steady_mask']}",
        f"--region {reg}",
    ]
    static_vars = cfg.get("static_vars", [])
    if static_vars:
        lines.append(f"--static-vars {','.join(var_names[v] for v in static_vars)}")
    lines += [
        f"--test-names {','.join(flags['test'])}",
        f"--data-types {','.join(data_types(key, cfg))}",
        f"--start-year {test_start}",
        f"--end-year {test_end}",
        "",
    ]
    lines += model_block(m)
    return "\n".join(lines) + "\n"


# =========================================================
# 5. Write one train + one eval config file per selected experiment
# =========================================================
os.makedirs(config_out_dir, exist_ok=True)

seen_names = {}
for key, entry in written.items():
    names = experiment_names(key, entry["cfg"])
    if names in seen_names:
        print(f"ERROR: configs '{seen_names[names]}' and '{key}' map to the same "
              f"(config-name, run-name) {names} -- rename one of them")
        raise SystemExit(1)
    seen_names[names] = key

    config_name, run_name = names
    file_tag = f"{run_name.removeprefix('cerra-')}"
    train_file = f"{train_prefix}{file_tag}.txt"
    eval_file = f"{eval_prefix}{file_tag}.txt"
    train_path = os.path.join(config_out_dir, train_file)
    eval_path = os.path.join(config_out_dir, eval_file)

    train_text = render_train_config(key, entry, names)
    eval_text = render_eval_config(key, entry, names)

    for path, text in ((train_path, train_text), (eval_path, eval_text)):
        if not force_rewrite and os.path.isfile(path) and open(path).read() == text:
            print(f"unchanged: {path}")
        else:
            with open(path, "w") as f:
                f.write(text)
            print(f"wrote:     {path}")

    print()
    print(f"==================== experiment: {key} ====================")
    print(f"config-name : {config_name}")
    print(f"run-name    : {run_name}")
    print()
    print("----- train config -----")
    print(train_text)
    print("----- eval config -----")
    print(eval_text)
    print("----- submit -----")
    print(f"./input/slurm-scripts/submit-train-cerra.sh {train_file}")
    print(f"./input/slurm-scripts/submit-eval-cerra.sh {eval_file}")
    print(f"./input/slurm-scripts/4a_submit_evaluation.sh --config {eval_file}")
    daily_inputs = entry["cfg"].get("daily_input_vars", [])
    if daily_inputs:
        in_tpl = entry['cfg'].get("input_template", "input")
        print(f"NOTE [{key}]: daily_input_vars={daily_inputs} are read from the DAILY "
              f"input template, all other inputs from '{in_tpl}'. "
              f"The eval config has a single --data-file-template ({eval_tpl(in_tpl)}); "
              f"make sure the eval scripts use the daily template for {daily_inputs}.")
    print()

