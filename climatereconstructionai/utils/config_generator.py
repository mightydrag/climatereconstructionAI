"""Generate CERRA JSON file lists and train/eval config files from one settings block.

The Jupyter workflow in ``notebooks/cerra_experiment_workflow.ipynb`` drives this
module: a single settings block defines paths, file-name templates and experiments.
The workflow checks that all referenced netCDF files exist, writes the JSON file
lists under ``<ml_path>/data/`` and renders the train/eval config files
(``--flag value`` lines, loaded via ``--load-from-file``) into the config directory
(default ``<ml_path>/input/levante/``).

Each experiment is identified by its ``--config-name`` / ``--run-name`` pair, which
is also the output path of the runs (``snapshots/``, ``logs/``, ``evaluations/``).
The pair is kept unique: rerunning the identical experiment is idempotent, while a
changed experiment reusing the same names gets a ``-rN`` suffix (or raises with
``on_duplicate="error"``), so outputs of different runs never mix.
"""

import json
import os
import re

VAR_NAMES = {
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

DEFAULT_SPLITS = {
    "train": (1985, 2005),
    "val": (2006, 2015),
    "test": (2016, 2025),
}

DEFAULT_NC_TPL = {
    "input": "{data_root}/{var}/cerra_dm_{var}_{year}{reg_sfx}{descriptor}{res_sfx}.nc",
    "target": "{data_root}/{var}/cerra_dm_{var}_{year}{other_sfx}{reg_sfx}.nc",
    "static": "{data_root}/{var}/cerra_dm_{var}_{static_year}{other_sfx}{reg_sfx}.nc",
    "mask": "{mask_folder}/cerra_dm_{var}_{year}{reg_sfx}_filled_mask.nc",
    "mask_static": "{mask_folder}/cerra_dm_{var}_{static_year}{reg_sfx}_filled_mask.nc",
    "mask_test": "{mask_folder}/cerra_dm_{var}_{year}{reg_sfx}_filled_mask.nc",
    "steady_mask": "{mask_folder}/cerra_dm_{var}_{steady_mask_year}{reg_sfx}_steady_mask.nc",
}

DEFAULT_JSON_TPL = {
    "input": "cerra_{var}{reg_sfx}{descriptor}_{split}.json",
    "static": "cerra_{var}{reg_sfx}_{split}.json",
    "target": "cerra_{var}{reg_sfx}{other_sfx}_target_{split}.json",
    "mask": "cerra_{var}{reg_sfx}{other_sfx}_mask.json",
    "mask_test": "cerra_{var}{reg_sfx}{other_sfx}_mask_test.json",
    "steady_mask": "cerra_{var}{reg_sfx}{other_sfx}_steady_mask.json",
}

DEFAULT_PATH_FLAGS = {
    "data-root-dir": "./data/",
    "mask-dir": "./data/masks/",
    "model-dir": "./snapshots/",
    "evaluation-dirs": "./evaluations/",
    "log-dir": "./logs/",
}

DEFAULT_MODEL_PARAMS = {
    "device": "cuda",
    "n-threads": 64,
    "n-target-data": 1,
    "multiple-gpus": True,
    "lr": "5e-4",
    "out-channels": 1,
    "dataset-name": "cerra_ger_mult_file",
    "plot-result": "0,100",
    "shuffle-masks": True,
    "encoding-layers": "5",
    "pooling-layers": "0",
    "normalize-data": True,
    "decoder-upsample": "bilinear",
    "conv-factor": 128,
    "batch-size": 64,
    "loss-criterion": 0,
    "save-model-interval": 2500,
    "log-interval": 500,
    "eval-timesteps": "0,100",
}

_EVAL_KEYS = {
    "data": "input",
    "static": "static",
    "target": "target",
    "mask": "mask",
    "static_mask": "mask_static",
    "steady_mask": "steady_mask",
}

_CONFIG_NAME_RE = re.compile(r"^--config-name\s+(\S+)\s*$", re.MULTILINE)
_RUN_NAME_RE = re.compile(r"^--run-name\s+(\S+)\s*$", re.MULTILINE)


def render_flag_lines(mapping):
    lines = []
    for key, value in mapping.items():
        if value is None or value is False:
            continue
        if value is True:
            lines.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"--{key} {','.join(str(v) for v in value)}")
        else:
            lines.append(f"--{key} {value}")
    return lines


def render_config_text(sections):
    blocks = ["\n".join(render_flag_lines(section)) for section in sections]
    return "\n\n".join(blocks) + "\n"


def train_config_filename(config_name, run_name):
    return f"train-cnn-{config_name}-{run_name}.txt"


def eval_config_filename(config_name, run_name):
    return f"eval-cnn-{config_name}-{run_name}.txt"


def scan_experiments(config_dir):
    """Map each (config-name, run-name) pair found in the config files to its files."""
    experiments = {}
    for fname in sorted(os.listdir(config_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(config_dir, fname)) as f:
            text = f.read()
        cn = _CONFIG_NAME_RE.search(text)
        rn = _RUN_NAME_RE.search(text)
        if cn and rn:
            experiments.setdefault((cn.group(1), rn.group(1)), []).append(fname)
    return experiments


class CerraConfigWorkflow:
    """Settings-driven generator for CERRA JSON lists and train/eval config files."""

    def __init__(self, data_root, ml_path, reg="", other="", res="", descriptor="",
                 mask_folder=None, mask_folder_test=None, static_year=2005,
                 steady_mask_year=1985, splits=None, nc_tpl=None, json_tpl=None,
                 var_names=None, configs=None):
        self.data_root = data_root
        self.ml_path = ml_path
        self.reg = reg
        self.other = other or ""
        self.res = res or ""
        self.descriptor = descriptor or ""
        self.mask_folder = mask_folder or os.path.join(data_root, "mask")
        self.mask_folder_test = mask_folder_test or self.mask_folder
        self.static_year = static_year
        self.steady_mask_year = steady_mask_year
        self.splits = dict(splits) if splits else dict(DEFAULT_SPLITS)
        self.var_names = dict(var_names) if var_names else dict(VAR_NAMES)
        self.configs = dict(configs) if configs else {}
        self.reg_sfx = f"_{reg}" if reg else ""
        self.other_sfx = f"_{self.other}" if self.other else ""
        self.res_sfx = f"_{self.res}" if self.res else ""
        self.nc_tpl = dict(DEFAULT_NC_TPL)
        if nc_tpl:
            self.nc_tpl.update(nc_tpl)
        self.json_tpl = dict(DEFAULT_JSON_TPL)
        if json_tpl:
            self.json_tpl.update(json_tpl)
        self.base = dict(
            data_root=self.data_root,
            mask_folder=self.mask_folder,
            mask_folder_test=self.mask_folder_test,
            reg=self.reg,
            reg_sfx=self.reg_sfx,
            other_sfx=self.other_sfx,
            res_sfx=self.res_sfx,
            descriptor=self.descriptor,
            static_year=self.static_year,
            steady_mask_year=self.steady_mask_year,
        )
        self.eval_tpl = {
            flag: self.nc_tpl[nc_key]
            .replace("{var}", "{{var}}")
            .replace("{year}", "{{year}}")
            .replace("{reg_sfx}", "_{{reg}}" if reg else "")
            .format(**self.base)
            for flag, nc_key in _EVAL_KEYS.items()
        }

    def resolve_config(self, key):
        if key not in self.configs:
            raise KeyError(f"unknown config '{key}'. Valid keys: {', '.join(self.configs)}")
        cfg = self.configs[key]
        target_var = cfg.get("target", key)
        input_vars = list(cfg["input_vars"])
        static_vars = list(cfg.get("static_vars", []))
        unknown = sorted({v for v in [*input_vars, *static_vars, target_var] if v not in self.var_names})
        if unknown:
            raise KeyError(f"no CERRA short name defined in var_names for: {unknown}")
        return target_var, input_vars, static_vars

    def nc_path(self, kind, var, year=None):
        return self.nc_tpl[kind].format(**self.base, var=var, year=year)

    def json_name(self, kind, var, split=None):
        return self.json_tpl[kind].format(**self.base, var=var, split=split)

    def steady_mask_file(self, target_var):
        return self.nc_path("steady_mask", target_var)

    def data_types(self, input_vars, static_vars, target_var):
        return [self.var_names[v] for v in [*input_vars, *static_vars, target_var]]

    def check_files(self, key, include_test=False):
        """Return the list of missing files for one config (empty = all present)."""
        target_var, input_vars, static_vars = self.resolve_config(key)
        missing = []
        splits = self.splits if include_test else {s: y for s, y in self.splits.items() if s != "test"}
        for split, (start_year, end_year) in splits.items():
            for year in range(start_year, end_year + 1):
                f_tgt = self.nc_path("target", target_var, year)
                if not os.path.isfile(f_tgt):
                    missing.append(f_tgt)
                for var in input_vars:
                    f_in = self.nc_path("input", var, year)
                    if not os.path.isfile(f_in):
                        missing.append(f_in)
                    f_mask = self.nc_path("mask", var, year)
                    if not os.path.isfile(f_mask):
                        missing.append(f_mask)
        for var in static_vars:
            for kind in ("static", "mask_static"):
                f_st = self.nc_path(kind, var)
                if not os.path.isfile(f_st):
                    missing.append(f_st)
        f_steady = self.steady_mask_file(target_var)
        if not os.path.isfile(f_steady):
            missing.append(f_steady)
        return missing

    def check_all(self, selected=None, include_test=False):
        """Return {config key: [missing files]} for the selected configs."""
        if selected is None:
            keys = list(self.configs)
        elif isinstance(selected, str):
            keys = [selected]
        else:
            keys = list(selected)
        missing = {}
        for key in keys:
            m = self.check_files(key, include_test=include_test)
            if m:
                missing[key] = m
        return missing

    def _entries(self, key):
        target_var, input_vars, static_vars = self.resolve_config(key)
        entries = {"splits": {}, "masks": [], "mask_test": [], "steady": None}
        for split, (start_year, end_year) in self.splits.items():
            years = list(range(start_year, end_year + 1))
            split_entries = []
            for var in input_vars:
                name = self.json_name("input", var, split)
                paths = [self.nc_path("input", var, y) for y in years]
                split_entries.append((f"{var}/{name}", f"data/{split}/{var}/{name}", paths))
            for var in static_vars:
                name = self.json_name("static", var, split)
                paths = [self.nc_path("static", var)] * len(years)
                split_entries.append((f"{var}/{name}", f"data/{split}/{var}/{name}", paths))
            name = self.json_name("target", target_var, split)
            paths = [self.nc_path("target", target_var, y) for y in years]
            split_entries.append((f"{target_var}/{name}", f"data/{split}/{target_var}/{name}", paths))
            entries["splits"][split] = split_entries
        tv_years = sorted(
            {y for s in ("train", "val") for y in range(self.splits[s][0], self.splits[s][1] + 1)}
        )
        test_years = list(range(self.splits["test"][0], self.splits["test"][1] + 1))
        for var in [*input_vars, *static_vars, target_var]:
            if var in static_vars:
                m_paths = [self.nc_path("mask_static", var)] * len(tv_years)
                mt_paths = [self.nc_path("mask_static", var)] * len(test_years)
            else:
                m_paths = [self.nc_path("mask", var, y) for y in tv_years]
                mt_paths = [self.nc_path("mask_test", var, y) for y in test_years]
            name = self.json_name("mask", var)
            entries["masks"].append((f"{var}/{name}", f"data/masks/{var}/{name}", m_paths))
            name_t = self.json_name("mask_test", var)
            entries["mask_test"].append((f"{var}/{name_t}", f"data/test/masks/{var}/{name_t}", mt_paths))
        steady_name = self.json_name("steady_mask", target_var)
        steady_path = self.steady_mask_file(target_var)
        entries["steady"] = (
            f"{target_var}/{steady_name}",
            f"data/masks/{target_var}/{steady_name}",
            [steady_path],
            f"data/test/masks/{target_var}/{steady_name}",
        )
        return target_var, input_vars, static_vars, entries, steady_name

    def build_file_lists(self, key):
        """Return (flags, steady_mask_name) without writing anything."""
        _, _, _, entries, steady_name = self._entries(key)
        flags = {split: [e[0] for e in entries["splits"][split]] for split in entries["splits"]}
        flags["mask"] = [e[0] for e in entries["masks"]]
        flags["mask_test"] = [e[0] for e in entries["mask_test"]]
        return flags, steady_name

    def write_json_lists(self, key):
        """Write all JSON file lists for one config; return (flags, steady_name, written)."""
        _, _, _, entries, steady_name = self._entries(key)
        written = []
        groups = [e for split_entries in entries["splits"].values() for e in split_entries]
        groups += entries["masks"] + entries["mask_test"]
        for _, rel, paths in groups:
            full = os.path.join(self.ml_path, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                json.dump(paths, f, indent=2)
            written.append(full)
        _, rel, paths, rel_test = entries["steady"]
        for rel_steady in (rel, rel_test):
            full = os.path.join(self.ml_path, rel_steady)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                json.dump(paths, f, indent=2)
            written.append(full)
        flags = {split: [e[0] for e in entries["splits"][split]] for split in entries["splits"]}
        flags["mask"] = [e[0] for e in entries["masks"]]
        flags["mask_test"] = [e[0] for e in entries["mask_test"]]
        return flags, steady_name, written

    def _sections(self, params, paths):
        path_section = dict(DEFAULT_PATH_FLAGS)
        if paths:
            path_section.update(paths)
        param_section = dict(DEFAULT_MODEL_PARAMS)
        if params:
            param_section.update(params)
        return path_section, param_section

    def render_train_config(self, key, config_name, run_name, flags=None, steady_name=None,
                            params=None, paths=None):
        target_var, input_vars, static_vars = self.resolve_config(key)
        if flags is None:
            flags, steady_name = self.build_file_lists(key)
        path_section, param_section = self._sections(params, paths)
        name_section = {"config-name": config_name, "run-name": run_name}
        data_section = {
            "data-names": ",".join(flags["train"]),
            "mask-names": ",".join(flags["mask"]),
            "steady-masks": f"{target_var}/{steady_name}",
            "val-names": ",".join(flags["val"]),
            "test-names": ",".join(flags["test"]),
            "xrdss-paths": f"masks/{target_var}",
            "data-types": ",".join(self.data_types(input_vars, static_vars, target_var)),
        }
        return render_config_text([path_section, name_section, data_section, param_section])

    def render_eval_config(self, key, config_name, run_name, params=None, paths=None,
                           model_names=None):
        target_var, input_vars, static_vars = self.resolve_config(key)
        test_start, test_end = self.splits["test"]
        path_section, param_section = self._sections(params, paths)
        name_section = {"config-name": config_name, "run-name": run_name}
        tpl_items = [
            ("data-file-template", self.eval_tpl["data"]),
            ("static-file-template", self.eval_tpl["static"]),
            ("target-file-template", self.eval_tpl["target"]),
            ("mask-file-template", self.eval_tpl["mask"]),
            ("static-mask-template", self.eval_tpl["static_mask"]),
            ("steady-mask-file", self.eval_tpl["steady_mask"]),
            ("region", self.reg),
        ]
        if static_vars:
            tpl_items.append(("static-vars", ",".join(self.var_names[v] for v in static_vars)))
        tpl_items += [
            ("data-types", ",".join(self.data_types(input_vars, static_vars, target_var))),
            ("start-year", test_start),
            ("end-year", test_end),
        ]
        if model_names:
            tpl_items.append(("model-names", model_names))
        return render_config_text([path_section, name_section, dict(tpl_items), param_section])

    def _output_dirs_exist(self, config_name, run_name):
        for d in ("snapshots", "evaluations", "logs"):
            if os.path.isdir(os.path.join(self.ml_path, d, config_name, run_name)):
                return True
        return False

    @staticmethod
    def _read(config_dir, fname):
        path = os.path.join(config_dir, fname)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    def write_experiment(self, key, run_name, config_name=None, config_dir=None,
                         train_params=None, eval_params=None, model_names=None,
                         on_duplicate="suffix", check_output_dirs=True, write_jsons=True,
                         paths=None):
        """Write JSON lists + train/eval configs for one experiment with a unique name."""
        if on_duplicate not in ("suffix", "error"):
            raise ValueError(f"on_duplicate must be 'suffix' or 'error', got {on_duplicate!r}")
        if not run_name:
            raise ValueError("run_name is required: together with config_name it identifies the experiment")
        config_name = config_name or key
        for name in (config_name, run_name):
            if "/" in name or os.sep in name:
                raise ValueError(f"invalid experiment name {name!r}: must not contain path separators")
        config_dir = config_dir or os.path.join(self.ml_path, "input", "levante")
        os.makedirs(config_dir, exist_ok=True)
        if eval_params is None:
            eval_params = train_params
        if write_jsons:
            flags, steady_name, written = self.write_json_lists(key)
        else:
            flags, steady_name = self.build_file_lists(key)
            written = []
        train_text = self.render_train_config(
            key, config_name, run_name, flags=flags, steady_name=steady_name,
            params=train_params, paths=paths,
        )
        eval_text = self.render_eval_config(
            key, config_name, run_name, params=eval_params, paths=paths, model_names=model_names,
        )
        existing = scan_experiments(config_dir)
        train_file = train_config_filename(config_name, run_name)
        eval_file = eval_config_filename(config_name, run_name)
        action = "written"
        pair = (config_name, run_name)
        if pair in existing or (check_output_dirs and self._output_dirs_exist(*pair)):
            checks = [
                set(existing.get(pair, [])) == {train_file, eval_file},
                self._read(config_dir, train_file) == train_text,
                self._read(config_dir, eval_file) == eval_text,
            ]
            if all(checks):
                action = "unchanged"
            elif on_duplicate == "error":
                raise FileExistsError(
                    f"experiment (config-name={config_name!r}, run-name={run_name!r}) already exists in "
                    f"{config_dir}: {existing.get(pair, [])}. Choose a different run_name or use "
                    "on_duplicate='suffix'."
                )
            else:
                i = 2
                while True:
                    candidate = (config_name, f"{run_name}-r{i}")
                    clashes = candidate in existing
                    if check_output_dirs:
                        clashes = clashes or self._output_dirs_exist(*candidate)
                    if not clashes:
                        break
                    i += 1
                run_name = f"{run_name}-r{i}"
                train_file = train_config_filename(config_name, run_name)
                eval_file = eval_config_filename(config_name, run_name)
                action = "suffixed"
        if action != "unchanged":
            with open(os.path.join(config_dir, train_file), "w") as f:
                f.write(train_text)
            with open(os.path.join(config_dir, eval_file), "w") as f:
                f.write(eval_text)
        target_var = self.resolve_config(key)[0]
        return {
            "config": key,
            "config_name": config_name,
            "run_name": run_name,
            "action": action,
            "train_file": os.path.join(config_dir, train_file),
            "eval_file": os.path.join(config_dir, eval_file),
            "train_file_name": train_file,
            "eval_file_name": eval_file,
            "json_files": written,
            "steady_mask": f"{target_var}/{steady_name}",
            "train_cmd": f"sbatch input/slurm-scripts/slurm-train-crai.sh {train_file}",
            "eval_cmd": f"CONFIG_FILE={eval_file} sbatch input/slurm-scripts/slurm-eval-crai.sh",
        }

    def run_all(self, experiments, config_dir=None, on_duplicate="suffix", include_test=True,
                check_output_dirs=True, write_jsons=True):
        """Check all files of all experiments first, then write them one by one.

        ``experiments`` maps a CONFIGS key to a dict with ``run_name`` and optional
        ``config_name``, ``train_params``, ``eval_params``, ``model_names`` and
        ``on_duplicate`` entries.
        """
        missing = self.check_all(selected=list(experiments), include_test=include_test)
        if missing:
            total = sum(len(m) for m in missing.values())
            lines = [f"{total} file(s) missing -- nothing written:"]
            for key, m in missing.items():
                lines.append(f"  [{key}] {len(m)} missing:")
                lines += [f"    {f}" for f in m]
            raise RuntimeError("\n".join(lines))
        results = []
        for key, spec in experiments.items():
            spec = dict(spec)
            results.append(self.write_experiment(
                key,
                run_name=spec.get("run_name") or key,
                config_name=spec.get("config_name"),
                config_dir=config_dir or spec.get("config_dir"),
                train_params=spec.get("train_params"),
                eval_params=spec.get("eval_params"),
                model_names=spec.get("model_names"),
                on_duplicate=spec.get("on_duplicate", on_duplicate),
                check_output_dirs=check_output_dirs,
                write_jsons=write_jsons,
            ))
        return results
