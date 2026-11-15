# %% [markdown]
# # CERRA train/eval config generator
#
# Run the cells top to bottom (Jupyter: *Run All*, or `python input/create_cerra_configs.py`).
#
# - Cell 1 holds **all** settings: paths, experiments, hyperparameters, output folder. Edit only that cell.
# - The script writes the JSON file lists under `{ml_path}/data/...` (as before) **and** the
#   train + eval `.txt` configs into `config_out_dir` (default `input/levante`).
# - File names, `--config-name` and `--run-name` are derived from the settings so every
#   experiment is uniquely identifiable; if a name is already taken by a *different*
#   config, a short content hash is appended automatically.

# %% [markdown]
# ## 1. Settings -- edit this cell only
# %%
import os
import json
import hashlib

# ---- paths --------------------------------------------------------------
ml_path = "/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions"
data_root = "/work/uo1075/u244085/cerra_data"

# repo checkout, used to resolve the default output folder (input/levante)
repo_root = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "__file__" in globals()
    else ml_path
)
config_out_dir = os.path.join(repo_root, "input", "levante")  # where the .txt configs are written

# set to False to skip the on-disk existence check (e.g. where the data is not mounted)
verify_files = True

# ---- region / grid / naming ---------------------------------------------
reg = "ger"
region_label = {"ger": "germany"}.get(reg, reg)  # label used inside --run-name
grid_tag = "0.05x2deg"  # highres x lowres grid, used in file names and --run-name
res = "2deg"  # lowres grid suffix on the input files
other = "remapcon_land"
descriptor = "_lowres_values_filled_highres_grid"

reg_sfx = f"_{reg}" if reg else ""
other_sfx = f"_{other}" if other else ""
res_sfx = f"_{res}" if res else ""

# ---- static files / masks / splits --------------------------------------
static_year = 2005  # year of the single static files (orography/lsm stored as ..._2005.nc)
steady_mask_year = 1985
mask_folder = f"{data_root}/mask"
mask_folder_test = f"{data_root}/test/mask"

splits = {
    "train": (1985, 2005),
    "val": (2006, 2015),
    "test": (2016, 2025),
}

# ---- experiments ----------------------------------------------------------
# key            = unique experiment id -> used as --config-name and in the file names
# "target"       = target variable (defaults to the key if omitted)
# "input_vars"   = time-varying inputs (one file per year)
# "static_vars"  = time-invariant inputs (one file each, defaults to [])
# "run"          = hyperparameters overriding RUN_DEFAULTS (all entries optional)
# "name_tag"     = override for the all-var/single tag in the names
# "name_suffix"  = extra tag appended to file/run names (e.g. "cfill", "log")
CONFIGS = {
    "relative_humidity": {
        "input_vars": ["relative_humidity", "t2m", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "run": {"lr": 5e-4, "loss_criterion": 0},
    },
    "tp_v2": {
        "input_vars": ["tciwv", "t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "tp",
        "run": {"lr": 5e-5, "loss_criterion": 6},
        "name_suffix": "cfill",
    },
    "tp": {
        "input_vars": ["tp", "t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "target": "tp",
        "run": {"lr": 5e-4, "loss_criterion": 5},
    },
    "t2m": {
        "input_vars": ["t2m", "relative_humidity", "surface_pressure", "wind_speed", "wind_direction"],
        "static_vars": ["orography"],
        "run": {
            "lr": 5e-4,
            "loss_criterion": 0,
            "encoding_layers": "5,5",
            "pooling_layers": "2,2",
            "conv_factor": 128,
        },
    },
    # ablations: single-input baselines
    "t2m_single": {
        "target": "t2m",
        "input_vars": ["t2m"],
        "run": {
            "lr": 5e-4,
            "loss_criterion": 0,
            "encoding_layers": "5,5",
            "pooling_layers": "2,2",
            "conv_factor": 128,
        },
    },
    "relative_humidity_single": {
        "target": "relative_humidity",
        "input_vars": ["relative_humidity"],
        "run": {"lr": 5e-4, "loss_criterion": 0},
    },
    "tp_single": {
        "input_vars": ["tp"],
        "target": "tp",
        "run": {"lr": 5e-4, "loss_criterion": 5},
    },
}

# one key of CONFIGS to process, or None for ALL configs
selected = "tp_v2"

# ---- default hyperparameters (each config's "run" dict overrides these) ---
RUN_DEFAULTS = {
    "device": "cuda",
    "n_threads": 64,
    "n_target_data": 1,
    "multi_gpus": True,
    "lr": 5e-5,
    "out_channels": 1,
    "dataset_name": "cerra_ger_mult_file",
    "plot_result": "0,100",
    "shuffle_masks": True,
    "encoding_layers": "5",
    "pooling_layers": "0",
    "normalize_data": True,
    "decoder_upsample": "bilinear",
    "conv_factor": 512,
    "batch_size": 64,
    "loss_criterion": 6,
    "save_model_interval": 2500,
    "log_interval": 500,
    "eval_timesteps": "0,100",
}

# %% [markdown]
# ## 2. File-name layout -- the single source of truth for .nc paths and JSON names
#
# `NC_TPL` defines the on-disk `.nc` layout; `JSON_TPL` the JSON list names under `ml_path`;
# `EVAL_TPL` (derived from `NC_TPL`) the templates for the per-year evaluate configs, where
# `{var}` / `{year}` / `{reg}` stay literal for the evaluate tool to resolve.
# %%
NC_TPL = {
    "input": "{data_root}/{var}/cerra_dm_{var}_{year}{reg_sfx}{descriptor}{res_sfx}.nc",
    "target": "{data_root}/{var}/cerra_dm_{var}_{year}{other_sfx}{reg_sfx}.nc",
    "static": "{data_root}/{var}/cerra_dm_{var}_{static_year}{other_sfx}{reg_sfx}.nc",
    "mask": "{mask_folder}/cerra_dm_{var}_{year}{reg_sfx}_filled_mask.nc",
    "mask_static": "{mask_folder}/cerra_dm_{var}_{static_year}{reg_sfx}_filled_mask.nc",
    # same layout as "mask", listed separately so test-split masks can point at
    # another folder: swap {mask_folder} -> {mask_folder_test} here
    "mask_test": "{mask_folder}/cerra_dm_{var}_{year}{reg_sfx}_filled_mask.nc",
    "steady_mask": "{mask_folder}/cerra_dm_{var}_{steady_mask_year}{reg_sfx}_steady_mask.nc",
}

JSON_TPL = {
    "input": "cerra_{var}{reg_sfx}{descriptor}_{split}.json",
    "static": "cerra_{var}{reg_sfx}_{split}.json",
    "target": "cerra_{var}{reg_sfx}{other_sfx}_target_{split}.json",
    "mask": "cerra_{var}{reg_sfx}{other_sfx}_mask.json",
    "mask_test": "cerra_{var}{reg_sfx}{other_sfx}_mask_test.json",
    "steady_mask": "cerra_{selected}_{reg_sfx}{other_sfx}_steady_mask.json",
}

# common placeholders, filled in once from the settings cell
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

# evaluate-run templates: DERIVED from NC_TPL; {var}/{year}/{reg} stay LITERAL
_EVAL_KEYS = {  # evaluate flag -> NC_TPL key
    "data": "input",
    "static": "static",
    "target": "target",
    "mask": "mask",
    "static_mask": "mask_static",
    "steady_mask": "steady_mask",
}
EVAL_TPL = {
    flag: NC_TPL[key]
    .replace("{var}", "{{var}}")
    .replace("{year}", "{{year}}")
    .replace("{reg_sfx}", "_{{reg}}" if reg else "")
    .format(**BASE)
    for flag, key in _EVAL_KEYS.items()
}

# CERRA short names as reported by `cdo showname`
var_names = {
    "t2m": "2t",
    "orography": "orog",
    "land_sea_mask": "lsm",
    "tp": "tp",
    "relative_humidity": "2r",
    "surface_pressure": "sp",
    "wind_speed": "10si",
    "wind_direction": "10wdir",
    "tciwv": "tciwv",
}

# %% [markdown]
# ## 3. JSON file-list helpers (check + write)
# %%
def check_files(target_var, input_vars, static_vars):
    """Return list of missing files for one configuration (empty = all present)."""
    missing = []
    for split, (start_year, end_year) in splits.items():
        if split == "test":
            continue  # test split: evaluated per year via file templates, no JSONs
        for year in range(start_year, end_year + 1):
            f_tgt = NC_TPL["target"].format(**BASE, var=target_var, year=year)
            if not os.path.isfile(f_tgt):
                missing.append(f_tgt)
            for var in input_vars:
                f_in = NC_TPL["input"].format(**BASE, var=var, year=year)
                if not os.path.isfile(f_in):
                    missing.append(f_in)
                # binary mask: one file per year, needed for every input var
                f_mask = NC_TPL["mask"].format(**BASE, var=var, year=year)
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


def write_config(target_var, input_vars, static_vars):
    """Write all JSON file lists for one configuration, return (flags, steady_mask_name)."""
    flags = {}

    for split, (start_year, end_year) in splits.items():
        years = range(start_year, end_year + 1)
        n_years = len(years)
        flags[split] = []

        # time-varying inputs: one file per year
        for var in input_vars:
            file_list_in = [
                NC_TPL["input"].format(**BASE, var=var, year=year)
                for year in years
            ]
            path = os.path.join(ml_path, "data", split, var)
            os.makedirs(path, exist_ok=True)
            name_in = JSON_TPL["input"].format(**BASE, var=var, split=split)
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

        # target data: native highres field of the target variable (always LAST)
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

    # ---- binary masks: single JSON per variable, covering all train+val years ----
    flags["mask"] = []
    tv_years = sorted(
        {y for s in ("train", "val") for y in range(splits[s][0], splits[s][1] + 1)}
    )
    for var in [*input_vars, *static_vars, target_var]:
        path = os.path.join(ml_path, "data", "masks", var)
        os.makedirs(path, exist_ok=True)
        name_mask = JSON_TPL["mask"].format(**BASE, var=var)
        if var in static_vars:
            # same logic as the static input files: one mask file
            # (from static_year) repeated so all lists have the same length
            f_mask = NC_TPL["mask_static"].format(**BASE, var=var)
            file_list_mask = [f_mask] * len(tv_years)
            print(f"mask: 1 static mask file x {len(tv_years)} -> {path}")
        else:
            file_list_mask = [
                NC_TPL["mask"].format(**BASE, var=var, year=year)
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
        name_mask_test = JSON_TPL["mask_test"].format(**BASE, var=var)
        if var in static_vars:
            f_mask = NC_TPL["mask_static"].format(**BASE, var=var)
            file_list_mask_test = [f_mask] * len(test_years)
        else:
            file_list_mask_test = [
                NC_TPL["mask_test"].format(**BASE, var=var, year=year)
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

# %% [markdown]
# ## 4. Train/eval config rendering -- names are derived, so experiments stay unique
# %%
_UPSAMPLE_SHORT = {"bilinear": "bilin", "nearest": "near"}


def exp_run(cfg):
    run = dict(RUN_DEFAULTS)
    run.update(cfg.get("run", {}))
    return run


def name_parts(key, cfg, run):
    """Return (target, run_name, train_stem, eval_stem) for one experiment."""
    target = cfg.get("target", key)
    slot = key if key != target else target  # keeps tp_v2/tp etc. distinguishable
    tag = cfg.get("name_tag") or ("all-var" if len(cfg["input_vars"]) > 1 else "single")
    suffix = cfg.get("name_suffix", "")
    sfx = f"-{suffix}" if suffix else ""
    el = str(run["encoding_layers"]).split(",")[0]
    up = _UPSAMPLE_SHORT.get(run["decoder_upsample"], run["decoder_upsample"] or "none")
    loss = run["loss_criterion"]
    run_name = f"cerra-{region_label}-{grid_tag}-el{el}-{tag}-l{loss}-{up}{sfx}"
    train_stem = f"train-cnn-cerra-{slot}-{reg}-{grid_tag}-el{el}-{tag}-l{loss}-{up}{sfx}"
    eval_stem = f"eval-cnn-cerra-{slot}-{reg}-el{el}-{tag}-l{loss}{sfx}"
    return target, run_name, train_stem, eval_stem


def _dir_block(mask_dir):
    return [
        "--data-root-dir ./data/",
        f"--mask-dir {mask_dir}",
        "--model-dir ./snapshots/",
        "--evaluation-dirs ./evaluations/",
        "--log-dir ./logs/",
        "",
    ]


def _model_block(run):
    lines = [
        f"--device {run['device']}",
        f"--n-threads {run['n_threads']}",
        f"--n-target-data {run['n_target_data']}",
    ]
    if run["multi_gpus"]:
        lines.append("--multiple-gpus")
    lines += [
        f"--lr {run['lr']:g}",
        f"--out-channels {run['out_channels']}",
        "",
        f"--dataset-name {run['dataset_name']}",
        f"--plot-result {run['plot_result']}",
    ]
    if run["shuffle_masks"]:
        lines.append("--shuffle-masks")
    lines += [
        f"--encoding-layers {run['encoding_layers']}",
        f"--pooling-layers {run['pooling_layers']}",
    ]
    if run["normalize_data"]:
        lines.append("--normalize-data")
    if run["decoder_upsample"]:
        lines.append(f"--decoder-upsample {run['decoder_upsample']}")
    lines += [
        f"--conv-factor {run['conv_factor']}",
        f"--batch-size {run['batch_size']}",
        "",
        f"--loss-criterion {run['loss_criterion']}",
        "",
        f"--save-model-interval {run['save_model_interval']}",
        f"--log-interval {run['log_interval']}",
        f"--eval-timesteps {run['eval_timesteps']}",
    ]
    return lines


def render_train_config(key, run, run_name, flags, name_steady, target_var, data_types):
    lines = _dir_block("./data/masks/")
    lines += [
        f"--config-name {key}",
        f"--run-name {run_name}",
        "",
        f"--data-names {','.join(flags['train'])}",
        f"--mask-names {','.join(flags['mask'])}",
        f"--steady-masks {target_var}/{name_steady}",
        f"--val-names {','.join(flags['val'])}",
        f"--test-names {','.join(flags['test'])}",
        f"--xrdss-paths masks/{target_var}",
        f"--data-types {','.join(data_types)}",
        "",
    ]
    lines += _model_block(run)
    return "\n".join(lines) + "\n"


def render_eval_config(key, run, run_name, data_types):
    test_start, test_end = splits["test"]
    lines = _dir_block("./data/test/masks/")
    lines += [
        f"--config-name {key}",
        f"--run-name {run_name}",
        "",
        f"--data-file-template {EVAL_TPL['data']}",
        f"--static-file-template {EVAL_TPL['static']}",
        f"--target-file-template {EVAL_TPL['target']}",
        f"--mask-file-template {EVAL_TPL['mask']}",
        f"--static-mask-template {EVAL_TPL['static_mask']}",
        f"--steady-mask-file {EVAL_TPL['steady_mask']}",
        f"--region {reg}",
        f"--data-types {','.join(data_types)}",
        f"--start-year {test_start}",
        f"--end-year {test_end}",
        "",
    ]
    lines += _model_block(run)
    return "\n".join(lines) + "\n"


def unique_config_path(stem, text):
    """Path for a config file: reuse if identical, else append a content hash."""
    os.makedirs(config_out_dir, exist_ok=True)
    path = os.path.join(config_out_dir, stem + ".txt")
    if not os.path.exists(path):
        return path
    if open(path).read() == text:
        return path  # identical config already there -> idempotent
    tag = hashlib.md5(text.encode()).hexdigest()[:6]
    return os.path.join(config_out_dir, f"{stem}-{tag}.txt")

# %% [markdown]
# ## 5. Run: check files -> write JSONs -> write train/eval configs
# %%
# ---- resolve the selection once, fail early on a bad name ----
if selected is None:
    run_configs = CONFIGS
elif selected in CONFIGS:
    run_configs = {selected: CONFIGS[selected]}
else:
    raise SystemExit(
        f"ERROR: selected config '{selected}' is not defined. Valid keys: {', '.join(CONFIGS)}"
    )

# every configured variable needs a short name for --data-types
unknown = []
for key, cfg in run_configs.items():
    target_var = cfg.get("target", key)
    for v in cfg["input_vars"] + cfg.get("static_vars", []) + [target_var]:
        if v not in var_names:
            unknown.append(v)
if unknown:
    raise SystemExit(
        f"ERROR: no CERRA short name defined in var_names for: {sorted(set(unknown))}"
    )

# ---- quick check: EVERY file of EVERY config must exist before writing anything ----
if verify_files:
    missing_by_config = {}
    for key, cfg in run_configs.items():
        m = check_files(cfg.get("target", key), cfg["input_vars"], cfg.get("static_vars", []))
        if m:
            missing_by_config[key] = m
    if missing_by_config:
        total = sum(len(m) for m in missing_by_config.values())
        print(f"ERROR: {total} file(s) missing -- no JSONs/configs written:")
        for key, m in missing_by_config.items():
            print(f"  [{key}] {len(m)} missing:")
            for f in m:
                print(f"    {f}")
        raise SystemExit(1)
    print("All files present. Writing JSONs and configs...")
else:
    print("verify_files=False -> skipping on-disk check. Writing JSONs and configs...")

os.makedirs(config_out_dir, exist_ok=True)
written = []

for key, cfg in run_configs.items():
    target_var = cfg.get("target", key)
    input_vars = cfg["input_vars"]
    static_vars = cfg.get("static_vars", [])
    BASE["selected"] = key  # per-config steady-mask JSON name

    flags, name_steady = write_config(target_var, input_vars, static_vars)
    run = exp_run(cfg)
    data_types = [var_names[v] for v in input_vars + static_vars + [target_var]]
    _, run_name, train_stem, eval_stem = name_parts(key, cfg, run)

    train_text = render_train_config(key, run, run_name, flags, name_steady, target_var, data_types)
    eval_text = render_eval_config(key, run, run_name, data_types)

    train_path = unique_config_path(train_stem, train_text)
    eval_path = unique_config_path(eval_stem, eval_text)
    with open(train_path, "w") as f:
        f.write(train_text)
    with open(eval_path, "w") as f:
        f.write(eval_text)
    written += [train_path, eval_path]

    print()
    print(f"==== {key} (target {target_var}) : config-name={key} run-name={run_name} ====")
    print(f"  train: {train_path}")
    print(f"  eval : {eval_path}")
    print(f"  submit train: ./input/slurm-scripts/submit-train-cerra.sh {os.path.basename(train_path)}")
    print(f"  submit eval : ./input/slurm-scripts/submit-eval-cerra.sh {os.path.basename(eval_path)}")

print()
print(f"done -> {len(written)} config file(s) written to {config_out_dir}")
