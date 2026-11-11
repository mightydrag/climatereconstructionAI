#!/usr/bin/env python3
"""
analysis.py
===========
Per-grid-cell evaluation of NN downscaling output against reference data.

Two input modes:
  * CONFIG-FILE MODE (--args-file, recommended): pass the arguments file of
    the reconstruction run ('--key value' lines). The script extracts:
      - variable             <- --config-name (fallback: --var)
      - NN output directory  <- --evaluation-dirs / --config-name /
                               --run-name / data
      - reference files      <- the LAST comma-separated JSON of --test-names
                               (the target data; years are read from the
                               file names)
      - cell mask            <- --steady-masks (first listed file)
    Relative paths in the args file / JSON lists are resolved against the
    location of the args file / JSON list.
  * MANUAL MODE: --path-original, --path-nn, --var, --year-min, --year-max
    (the behaviour of the previous version).

PASS 1 (per year, lazy, one .compute() per year):
  bias, MSE, error quantiles, explained variance, correlation,
  day-of-year climatological sums (if --deseasonalize), scalar stats.
PASS 2 (only with --deseasonalize, per year):
  per-cell day-of-year climatology (from ALL years) is removed from BOTH
  fields; EV / correlation are recomputed on the anomalies.

Outputs (into --output-dir):
  <var>_maps_<y0>-<y1>.nc, <var>_stats_<y0>-<y1>.txt, *.png

POINT / AREA STATISTICS (default: Hamburg at 53.5511N/9.9937E,
see --point-lat/--point-lon/--point-name; disable with --no-point-stats):
  the grid cell nearest to the point -- or, with --point-radius-km R > 0,
  ALL grid cells within R km (great-circle) of it -- is selected in the
  same per-year passes; pooled bias/RMSE/MAE/EV/correlation/R2 over those
  cells (and their deseasonalized counterparts) are added to the stats
  file, plus a time-series figure of the area-mean series
  (<var>_point_<name>_<y0>-<y1>_timeseries.png, or _area_... with radius).

DISTRIBUTION MATCHING:
  variance ratio (var NN / var ref) and fraction of gap closed
  (1 - MAE / MAE-of-climatology) per grid cell and pooled; ROC curves
  with AUC and Peirce skill score (PSS) for exceedance events at the
  reference quantiles given by --roc-quantiles (<var>_roc_<y0>-<y1>.png).

File-name patterns and --mask-file support two variable placeholders:
    {var}    -> the variable/config name, e.g. 'relative_humidity'
    {var_nc} -> the NetCDF variable name, e.g. '2r' (VAR_FILE_DEFAULTS;
                falls back to the config name)
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import dask

import matplotlib
matplotlib.use("Agg")  # headless: MUST come before pyplot import on compute nodes
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------
# variable name used in file names / config-name  ->  NetCDF variable name
# inside the data files. Used when --var-file is not given explicitly.
VAR_FILE_DEFAULTS = {
    "t2m":               "2t",
    "orography":         "orog",
    "land_sea_mask":     "lsm",
    "tp":                "tp",
    "relative_humidity": "2r",
    "surface_pressure":  "sp",
    "wind_speed":        "10si",
    "wind_direction":    "10wdir",
}
# generic fallback candidates if the mapping misses
VAR_CANDIDATES = ["2t", "t2m", "tas", "air_temperature"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NN downscaling evaluation")
    p.add_argument("--args-file", default=None,
                   help="Arguments file ('--key value' lines) of the "
                        "reconstruction run; drives variable, paths, years "
                        "and mask (recommended mode)")
    p.add_argument("--path-original", default=None,
                   help="[manual mode] reference data dir")
    p.add_argument("--path-nn", default=None,
                   help="[manual mode] NN output data dir")
    p.add_argument("--var", default=None,
                   help="Variable name used in file names "
                        "(config mode: fallback for --config-name)")
    p.add_argument("--var-file", default=None,
                   help="NetCDF variable name (default: auto-detected)")
    p.add_argument("--var-file-nn", default=None,
                   help="NetCDF variable name in the NN output files "
                        "(default: same as --var-file, else auto-detected)")
    p.add_argument("--year-min", type=int, default=None,
                   help="[manual mode] first year (config mode: optional filter)")
    p.add_argument("--year-max", type=int, default=None,
                   help="[manual mode] last year (config mode: optional filter)")
    p.add_argument("--output-dir", required=True, help="Results/figures directory")
    p.add_argument("--file-pattern-nn", default="cerra_dm_{var}_ger_{year}_crai_output.nc",
                   help="NN file name pattern relative to the NN output dir. "
                        "Tokens: {var} config name (e.g. 'relative_humidity'), "
                        "{var_nc} NetCDF var name (e.g. '2r'), {year}")
    p.add_argument("--file-pattern-ref", default="cerra_dm_{var}_ger_{year}.nc",
                   help="[manual mode] reference file name pattern; same tokens "
                        "as --file-pattern-nn")
    p.add_argument("--rename-nn", default="",
                   help="Optional coord renames for the NN file, "
                        "e.g. 'longitude:lon,latitude:lat,valid_time:time'")
    p.add_argument("--quantiles", default="0.05,0.25,0.5,0.75,0.95",
                   help="Comma-separated error quantiles per grid cell")
    p.add_argument("--quantile-stride", type=int, default=1,
                   help="Time sub-sampling stride for the quantile maps")
    p.add_argument("--qq-stride", type=int, default=10,
                   help="Time sub-sampling stride for the pooled QQ-plot samples")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale factor applied to bias/RMSE/quantile MAPS only")
    p.add_argument("--chunk-time", type=int, default=30)
    p.add_argument("--deseasonalize", action="store_true",
                   help="Remove the per-cell day-of-year climatology (from ALL "
                        "years) from both fields before computing EV/correlation")
    p.add_argument("--mask-file", default=None,
                   help="NetCDF mask file; may contain the {var} and {var_nc} "
                        "tokens (config mode: --steady-masks from "
                        "the args file is used unless this is given)")
    p.add_argument("--mask-var", default=None,
                   help="Mask variable name (default: the only/first data var)")
    p.add_argument("--mask-value", type=float, default=1.0,
                   help="Mask value that defines the selection (default: 1)")
    p.add_argument("--mask-mode", choices=("keep", "exclude"), default="keep",
                   help="'keep': keep ONLY cells where mask == value; "
                        "'exclude': NaN the cells where mask == value")
    p.add_argument("--point-name", default="Hamburg",
                   help="Name of the point location for the point statistics "
                        "(used in the stats file and figure names)")
    p.add_argument("--point-lat", type=float, default=53.5511,
                   help="Latitude of the point for the point statistics; the "
                        "nearest grid cell is used (default: Hamburg)")
    p.add_argument("--point-lon", type=float, default=9.9937,
                   help="Longitude of the point for the point statistics; the "
                        "nearest grid cell is used (default: Hamburg)")
    p.add_argument("--point-radius-km", type=float, default=0.0,
                   help="If > 0, pool the point statistics over ALL grid "
                        "cells within this radius (km, great-circle) of "
                        "--point-lat/--point-lon instead of only the nearest "
                        "cell (default 0: single nearest cell)")
    p.add_argument("--no-point-stats", action="store_true",
                   help="Disable the point statistics (default: enabled for "
                        "the --point-lat/--point-lon location)")
    p.add_argument("--roc-quantiles", default="0.5,0.9,0.95,0.99",
                   help="Reference quantiles defining the exceedance events "
                        "for the ROC/AUC/PSS analysis (comma-separated)")
    p.add_argument("--roc-max-points", type=int, default=200000,
                   help="Max number of pooled samples used for the ROC "
                        "curve / AUC computation")
    return p.parse_args()


# ------------------------------------------------------------------
# Arguments-file parsing ('--key value' lines, '#' comments)
# ------------------------------------------------------------------
def parse_args_file(path):
    """Parse an arguments file: one '--key value' pair per line.
    Lines starting with '#' are ignored (e.g. '#--xrdss-paths ...').
    Bare flags ('--normalize-data') are stored as True. Later occurrences
    of the same key win."""
    cfg = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if not tokens[0].startswith("--"):
            continue  # free text lines are ignored
        key = tokens[0][2:]
        if len(tokens) > 1 and not tokens[1].startswith("--"):
            cfg[key] = tokens[1]
        else:
            cfg[key] = True
    return cfg


def read_data_list(list_path):
    """Read a data-list JSON (or plain text file) -> list of file paths.

    Supports: JSON array of strings, JSON dict with 'files'/'data'/'paths'
    keys, or a text file with one path per line. Directories are expanded
    to their sorted *.nc contents. Relative paths resolve against the
    list file's location."""
    p = Path(list_path)
    if not p.is_file():
        raise FileNotFoundError(f"Data list not found: {p}")
    try:
        content = json.loads(p.read_text())
    except json.JSONDecodeError:
        entries = [l.strip() for l in p.read_text().splitlines()
                   if l.strip() and not l.strip().startswith("#")]
    else:
        if isinstance(content, str):
            content = [content]
        if isinstance(content, list):
            entries = [str(e) for e in content]
        elif isinstance(content, dict):
            for key in ("files", "data", "paths", "datasets"):
                if key in content:
                    entries = [str(e) for e in content[key]]
                    break
            else:
                raise ValueError(f"Unrecognised data-list format in {p} "
                                 f"(keys: {list(content)})")
        else:
            raise ValueError(f"Unrecognised data-list format in {p}")

    files = []
    for e in entries:
        ep = Path(e)
        if not ep.is_absolute():
            ep = p.parent / ep
        ep = ep.resolve()
        if ep.is_dir():
            files.extend(sorted(str(f) for f in ep.glob("*.nc")))
        else:
            files.append(str(ep))
    return files


def extract_year(path):
    """First 4-digit year (19xx/20xx) in the file name, else None."""
    m = re.findall(r"(?:19|20)\d{2}", Path(path).stem)
    return int(m[0]) if m else None


def subst_tokens(s, var, year=None):
    """Substitute the file-name tokens in a pattern/path string by plain
    string replacement (unlike str.format(), stray braces cannot crash the
    run; an unknown token stays literal and check_files reports it):
      {var}    -> the variable/config name, e.g. 'relative_humidity'
      {var_nc} -> the NetCDF variable name, e.g. '2r'
                  (VAR_FILE_DEFAULTS; falls back to the config name)
      {year}   -> the year (only when given; --mask-file has none)
    """
    s = str(s)
    s = s.replace("{var_nc}", VAR_FILE_DEFAULTS.get(str(var), str(var)))
    s = s.replace("{var}", str(var))
    if year is not None:
        s = s.replace("{year}", str(year))
    return s


def pattern_path(base_dir, pattern, var, year):
    """Join base_dir with the file-name pattern after token substitution."""
    return Path(base_dir) / subst_tokens(pattern, var, year)


def resolve_inputs(args):
    """Resolve variable, dirs, the {year: (nn_file, ref_file)} mapping, the
    mask file and the config name (None in manual mode). Config-file mode
    if --args-file is given, else manual."""
    if args.args_file:
        cfg = parse_args_file(args.args_file)
        base = Path(args.args_file).resolve().parent

        def cfg_get(key, default=None):
            v = cfg.get(key, default)
            return v if v is not True else default  # bare flag -> no value

        var = args.var or cfg_get("config-name")
        if not var:
            raise ValueError("Neither --var nor --config-name (in the args "
                             "file) is set")

        # ---- NN output directory: <evaluation-dirs>/<config-name>/<run-name>/data
        run_name = cfg_get("run-name")
        if not run_name:
            raise ValueError("--run-name is missing in the args file")
        eval_dir = str(cfg_get("evaluation-dirs", "evaluations")).split(",")[0].strip()
        eval_dir = Path(eval_dir)
        if not eval_dir.is_absolute():
            eval_dir = base / eval_dir
        config_name = cfg_get("config-name", var)
        path_nn = eval_dir / config_name / run_name / "data"
        print(f"args file : {args.args_file}")
        print(f"run name  : {run_name}")
        print(f"NN dir    : {path_nn}")

        # ---- reference files: LAST comma-separated --test-names JSON
        test_names = cfg_get("test-names")
        if not test_names:
            raise ValueError("--test-names is missing in the args file")
        target_json = [t for t in str(test_names).split(",") if t.strip()][-1].strip()
        target_p = Path(target_json)
        if not target_p.is_absolute():
            target_p = base / target_p
        ref_files = read_data_list(target_p)
        print(f"reference : {len(ref_files)} test files from {target_json}")

        files = {}
        for rf in ref_files:
            y = extract_year(rf)
            if y is None:
                print(f"WARNING: no year in the file name -> skipped: {rf}")
                continue
            files[y] = (pattern_path(path_nn, args.file_pattern_nn, var, y),
                        Path(rf))

        # optional year range filter
        if args.year_min is not None:
            files = {y: v for y, v in files.items() if y >= args.year_min}
        if args.year_max is not None:
            files = {y: v for y, v in files.items() if y <= args.year_max}
        if not files:
            raise ValueError("No reference files with a year in their name")

        path_original = str(Path(ref_files[0]).parent)

        # ---- mask: --steady-masks (first JSON -> first listed file)
        mask_file = args.mask_file
        steady = cfg_get("steady-masks")
        if not mask_file and steady:
            steady_json = [s for s in str(steady).split(",") if s.strip()][0].strip()
            sp = Path(steady_json)
            if not sp.is_absolute():
                sp = base / sp
            mask_list = read_data_list(sp)
            mask_file = mask_list[0] if mask_list else None
            if mask_list:
                print(f"steady mask: {mask_file}"
                      + (" (first of %d listed files)" % len(mask_list)
                         if len(mask_list) > 1 else ""))
        return var, path_original, str(path_nn), files, mask_file, config_name

    # ---------------- manual mode ----------------
    missing = [k for k, v in [("--path-original", args.path_original),
                              ("--path-nn", args.path_nn),
                              ("--var", args.var),
                              ("--year-min", args.year_min),
                              ("--year-max", args.year_max)] if v is None]
    if missing:
        raise ValueError("Missing required arguments (or pass --args-file): "
                         + ", ".join(missing))
    years = list(range(args.year_min, args.year_max + 1))
    files = {
        y: (pattern_path(args.path_nn, args.file_pattern_nn, args.var, y),
            pattern_path(args.path_original, args.file_pattern_ref, args.var, y))
        for y in years
    }
    return args.var, args.path_original, args.path_nn, files, args.mask_file, None


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def check_files(files):
    """Fail fast with a clear message if any input file is missing; for the
    first missing file, list what IS in its directory to help fix the
    file-name pattern."""
    missing = sorted({f for f in files if not Path(f).is_file()})
    if missing:
        msg = "Missing input files:\n  " + "\n  ".join(str(m) for m in missing)
        parent = Path(missing[0]).parent
        if parent.is_dir():
            present = sorted(f.name for f in parent.iterdir() if f.is_file())
            if present:
                msg += (f"\n\nFiles present in {parent} (first 5):\n  "
                        + "\n  ".join(present[:5])
                        + f"\n  ... adjust --file-pattern-nn if these do not match.")
        raise FileNotFoundError(msg)


def detect_var_file(args, ref_file):
    """Auto-detect the NetCDF variable name from the reference file."""
    if args.var_file:
        return args.var_file
    candidates = [VAR_FILE_DEFAULTS.get(args.var), args.var] + VAR_CANDIDATES
    candidates = [c for c in candidates if c]
    with xr.open_dataset(str(ref_file)) as ds:
        hit = [v for v in candidates if v in ds.data_vars]
        if not hit:
            raise ValueError(
                f"None of {candidates} found in {ref_file}.\n"
                f"Available variables: {list(ds.data_vars)}. "
                "Pass --var-file explicitly."
            )
    return hit[0]


def detect_var_file_nn(args, nn_file, var_file):
    """NetCDF variable name in the NN output file. The NN files may use a
    different name than the reference (e.g. 'prediction' instead of '2t').
    Order: --var-file-nn, then the reference name, then a single data var,
    then the usual candidates."""
    if args.var_file_nn:
        return args.var_file_nn
    with xr.open_dataset(nn_file) as ds:
        if var_file in ds.data_vars:
            return var_file
        if len(ds.data_vars) == 1:
            name = list(ds.data_vars)[0]
            print(f"NN variable: '{name}' (auto-detected; the reference "
                  f"file uses '{var_file}')")
            return name
        candidates = [VAR_FILE_DEFAULTS.get(args.var), args.var] + VAR_CANDIDATES
        hit = [c for c in candidates if c and c in ds.data_vars]
        if hit:
            print(f"NN variable: '{hit[0]}' (auto-detected; the reference "
                  f"file uses '{var_file}')")
            return hit[0]
        raise ValueError(
            f"Cannot determine the data variable in the NN file\n"
            f"  {nn_file}\n(the reference variable '{var_file}' is absent; "
            f"available: {list(ds.data_vars)}). Pass --var-file-nn explicitly."
        )


def ensure_datetime_time(da, file_ref):
    """Make sure the 'time' axis is a decoded datetime axis.

    Some preprocessing pipelines write 'time' as plain numbers without
    CF units, so xarray leaves it numeric and both
    groupby("time.dayofyear") and the .dt accessor crash. In that case
    the axis is rebuilt as a full calendar year starting Jan 1st: the
    year is taken from the file name and the step length is inferred
    from the number of timesteps (one year per file, no leap days)."""
    t = da["time"]
    if np.issubdtype(t.dtype, np.datetime64) or t.dtype == object:
        return da  # already datetime64 / cftime

    year = extract_year(file_ref)
    n = int(t.size)
    steps_per_day = None
    if year is not None:
        for days in (365, 366):
            if n % days == 0:
                steps_per_day = n // days
                break
    if steps_per_day is None:
        raise ValueError(
            f"The 'time' axis of\n  {file_ref}\nis not decoded (dtype "
            f"{t.dtype}, {n} steps) and cannot be rebuilt automatically "
            f"(need a year in the file name and a whole number of days "
            f"per year). Fix the time units in the data file."
        )
    freq = pd.Timedelta(hours=24 / steps_per_day)
    new_time = pd.date_range(f"{year}-01-01 00:00", periods=n, freq=freq)
    print(f"WARNING: numeric 'time' axis in {Path(file_ref).name} "
          f"(dtype {t.dtype}) -> rebuilt as {n} steps of {freq} starting "
          f"{year}-01-01. Check that this matches the data's calendar!")
    return da.assign_coords(time=new_time)


def load_pair(file_nn, file_ref, chunks, var_file, var_file_nn=None,
              keep_mask=None, rename_nn=""):
    """Open NN + reference, fix NN time axis, mask ocean, apply cell mask,
    align. var_file_nn is the NetCDF variable name in the NN file (may
    differ from the reference; see detect_var_file_nn)."""
    ref = xr.open_dataset(file_ref, chunks=chunks)[var_file]
    # make sure 'time' is a decoded datetime axis (needed for the
    # day-of-year climatology); rebuilds it if stored as plain numbers
    ref = ensure_datetime_time(ref, file_ref)

    # Open the NN file WITHOUT dask chunks first: its coordinate names may
    # differ (see --rename-nn), and chunk keys are only valid AFTER the
    # renames have been applied.
    nn_ds = xr.open_dataset(file_nn)

    # Optional coordinate renames on the NN file (dataset level, so they
    # also fix the time dimension name before chunking)
    if rename_nn:
        renames = dict(
            (pair.split(":") for pair in rename_nn.split(",") if ":" in pair)
        )
        renames = {k: v for k, v in renames.items()
                   if k in nn_ds.dims or k in nn_ds.coords}
        if renames:
            nn_ds = nn_ds.rename(renames)

    nn = nn_ds[var_file_nn or var_file]
    nn = nn.chunk({k: v for k, v in chunks.items() if k in nn.dims})

    if "time" not in nn.dims:
        raise ValueError(
            f"NN file\n  {file_nn}\nhas no 'time' dimension after the "
            f"renames (dims: {list(nn.dims)}). Check --rename-nn."
        )

    # The NN output time axis is wrong (constant year); overwrite with the
    # reference time axis. Clear error if the NN file is too short.
    n_ref = ref.sizes["time"]
    n_nn = nn.sizes["time"]
    if n_nn > n_ref:
        nn = nn.isel(time=slice(n_ref))
    elif n_nn < n_ref:
        raise ValueError(
            f"NN file\n  {file_nn}\nhas {n_nn} timesteps, but the reference\n"
            f"  {file_ref}\nhas {n_ref}. Cannot align the time axes."
        )
    nn["time"] = ref["time"]

    # Mask ocean / invalid cells using the reference NaN pattern
    nn = xr.where(np.isnan(ref), np.nan, nn)
    nn, ref = xr.align(nn, ref, join="exact")

    # Optional cell mask from a file (applied to BOTH fields, so all
    # metrics, the climatology and the pooled counts respect it)
    if keep_mask is not None:
        nn = nn.where(keep_mask)
        ref = ref.where(keep_mask)
    return nn, ref


# ------------------------------------------------------------------
# Optional cell mask from a file
# ------------------------------------------------------------------
def detect_spatial_names(ds):
    lat = next((d for d in ("lat", "latitude", "y")
                if d in ds.dims or d in ds.coords), None)
    lon = next((d for d in ("lon", "longitude", "x")
                if d in ds.dims or d in ds.coords), None)
    if lat is None or lon is None:
        raise ValueError(
            "Could not detect the spatial dims "
            f"(dims: {list(ds.dims)}, coords: {list(ds.coords)})."
        )
    return lat, lon


def build_keep_mask(args, mask_file, ref_file):
    """Load the mask file and return a (lat, lon) boolean DataArray of the
    cells to KEEP, aligned to the data grid. None if no mask file given."""
    if not mask_file:
        return None

    # fail fast with a clear message instead of a cryptic xarray
    # LRU-cache KeyError when the mask path is wrong
    if not Path(mask_file).is_file():
        hint = ""
        parent = Path(mask_file).parent
        if parent.is_dir():
            cand = sorted(parent.glob("*steady_mask*.nc"))
            if cand:
                hint = ("\nAvailable steady-mask files there:\n  "
                        + "\n  ".join(str(c) for c in cand[:10]))
        raise FileNotFoundError(
            f"Mask file not found: {mask_file}{hint}\n"
            "Pass the correct --mask-file / MASK_FILE (or fix the "
            "--steady-masks entry in the args file)."
        )

    with xr.open_dataset(mask_file) as mds:
        if args.mask_var:
            if args.mask_var not in mds.variables:
                raise ValueError(
                    f"Mask variable '{args.mask_var}' not found in "
                    f"{mask_file}. Available: {list(mds.data_vars)}."
                )
            m = mds[args.mask_var]
        elif len(mds.data_vars) == 1:
            m = mds[list(mds.data_vars)[0]]
        else:
            raise ValueError(
                f"Mask file {mask_file} contains several variables "
                f"({list(mds.data_vars)}). Pass --mask-var explicitly."
            )
        m = m.squeeze().load()  # drop singleton dims (e.g. time)

    m_lat, m_lon = detect_spatial_names(m)

    # the data grid (names + coordinate values)
    with xr.open_dataset(ref_file) as ds0:
        d_lat, d_lon = detect_spatial_names(ds0)
        d_lat_vals = ds0[d_lat].values
        d_lon_vals = ds0[d_lon].values

    # rename the mask coords onto the data's names
    if m_lat != d_lat:
        m = m.rename({m_lat: d_lat})
    if m_lon != d_lon:
        m = m.rename({m_lon: d_lon})

    # align to the data grid; nearest-neighbour if the grids differ
    same_grid = (
        m[d_lat].size == d_lat_vals.size
        and m[d_lon].size == d_lon_vals.size
        and np.allclose(m[d_lat].values, d_lat_vals)
        and np.allclose(m[d_lon].values, d_lon_vals)
    )
    if not same_grid:
        print("WARNING: mask grid differs from the data grid -> "
              "nearest-neighbour reindexing (check the alignment!)")
        m = m.reindex({d_lat: d_lat_vals, d_lon: d_lon_vals}, method="nearest")

    if args.mask_mode == "keep":
        keep = m == args.mask_value   # keep ONLY mask == value
    else:
        keep = m != args.mask_value   # NaN the cells with mask == value

    # Collapse any remaining non-spatial dimension (e.g. a time axis in
    # the mask file): a cell is kept only if the condition holds for ALL
    # steps, so the mask is True for EVERY time step of the data.
    extra_dims = sorted(set(keep.dims) - {d_lat, d_lon})
    for extra in extra_dims:
        print(f"Mask: reducing over extra dimension '{extra}' "
              f"(size {keep.sizes[extra]}) with .all() -> time-invariant mask")
        keep = keep.all(extra)
    keep = keep.squeeze()

    keep = keep.reset_coords(drop=True)
    n_keep = int(keep.sum())
    n_all = int(keep.size)
    print(f"Mask: mode={args.mask_mode}, value={args.mask_value} -> "
          f"{n_keep}/{n_all} cells kept ({100.0 * n_keep / max(n_all, 1):.1f}%)")
    return keep


# ------------------------------------------------------------------
# Point / area statistics, e.g. for Hamburg and its surroundings
# ------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (degrees in, numpy arrays allowed)."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = (np.radians(lat1), np.radians(lon1),
                              np.radians(lat2), np.radians(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def build_point_mask(ds, lat_name, lon_name, lat, lon, radius_km):
    """Boolean (lat, lon) DataArray of the grid cells used for the point /
    area statistics: ALL cells within radius_km of (lat, lon), or only the
    nearest cell when radius_km == 0."""
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values
    if radius_km and radius_km > 0:
        la2, lo2 = np.meshgrid(lat_vals, lon_vals, indexing="ij")
        mask = haversine_km(lat, lon, la2, lo2) <= radius_km
    else:
        mask = np.zeros((lat_vals.size, lon_vals.size), dtype=bool)
        mask[int(np.argmin(np.abs(lat_vals - lat))),
             int(np.argmin(np.abs(lon_vals - lon)))] = True
    return xr.DataArray(mask, dims=(lat_name, lon_name),
                        coords={lat_name: lat_vals, lon_name: lon_vals})


def pooled_stats(se, sse, sae, sr, ssr, sn, ssn, cnr, n):
    """Verification statistics pooled over n space-time points, computed
    from accumulated sums (bias=se/n, RMSE=sqrt(sse/n), MAE=sae/n, EV and
    correlation from the pooled moments). NaN when n < 2."""
    out = {"n": int(n), "bias": np.nan, "rmse": np.nan, "mae": np.nan,
           "ev": np.nan, "corr": np.nan, "r2": np.nan, "var_ratio": np.nan}
    if n < 2:
        return out
    var_e = (sse - se * se / n) / (n - 1)
    var_r = (ssr - sr * sr / n) / (n - 1)
    var_n = (ssn - sn * sn / n) / (n - 1)
    cov_nr = (cnr - sn * sr / n) / (n - 1)
    corr = np.nan
    if var_n > 0 and var_r > 0:
        corr = float(cov_nr / np.sqrt(var_n * var_r))
    out["bias"] = float(se / n)
    out["rmse"] = float(np.sqrt(sse / n))
    out["mae"] = float(sae / n)
    out["ev"] = float(1 - var_e / var_r) if var_r > 0 else np.nan
    out["corr"] = corr
    out["r2"] = corr ** 2 if np.isfinite(corr) else np.nan
    out["var_ratio"] = float(var_n / var_r) if var_r > 0 else np.nan
    return out


def roc_analysis(scores, events, threshold, max_points=200000):
    """ROC analysis for the binary `events` with continuous `scores`:
    AUC (rank / Mann-Whitney formula, ties averaged) and ROC curve points
    on a sub-sample of at most max_points points, plus the Peirce skill
    score PSS = TPR + TNR - 1 from the confusion matrix at `threshold`
    (all valid points). Returns None for degenerate events."""
    scores = np.asarray(scores, dtype="float64").ravel()
    events = np.asarray(events, dtype=bool).ravel()
    finite = np.isfinite(scores)
    scores = scores[finite]
    events = events[finite]
    n = scores.size
    n_pos = int(events.sum())
    n_neg = n - n_pos
    if n < 2 or n_pos == 0 or n_neg == 0:
        return None
    tp = int(np.sum(events & (scores > threshold)))
    fp = int(np.sum(~events & (scores > threshold)))
    fn = n_pos - tp
    tn = n_neg - fp
    pss = (tp / (tp + fn) + tn / (tn + fp) - 1.0
           if (tp + fn) > 0 and (tn + fp) > 0 else np.nan)
    if n > max_points:
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
        s_sub = scores[idx]
        e_sub = events[idx]
    else:
        s_sub, e_sub = scores, events
    s_sorted = np.sort(s_sub)
    e_sorted = e_sub[np.argsort(s_sub, kind="mergesort")].astype(np.int64)
    p_sub = int(e_sorted.sum())
    if p_sub == 0 or p_sub == s_sorted.size:  # degenerate sub-sample
        s_sorted = np.sort(scores)
        e_sorted = events[np.argsort(scores, kind="mergesort")].astype(np.int64)
        p_sub = n_pos
    uniq, inv, counts = np.unique(s_sorted, return_inverse=True,
                                  return_counts=True)
    starts = np.cumsum(counts) - counts
    avg_rank = starts + (counts + 1) / 2.0  # 1-based average rank per tie
    ranks = avg_rank[inv]
    auc = float((ranks[e_sorted == 1].sum() - p_sub * (p_sub + 1) / 2)
                / (p_sub * (s_sorted.size - p_sub)))
    thr = np.quantile(s_sub, np.linspace(0, 1, 201))
    cum_pos = np.concatenate([[0], np.cumsum(e_sorted)])
    k = np.searchsorted(s_sorted, thr, side="right")
    tp_gt = p_sub - cum_pos[k]
    fp_gt = (s_sorted.size - k) - tp_gt
    fpr = np.concatenate([[0.0], fp_gt / (s_sorted.size - p_sub), [1.0]])
    tpr = np.concatenate([[0.0], tp_gt / p_sub, [1.0]])
    return {"threshold": float(threshold), "auc": auc, "pss": float(pss),
            "fpr": fpr, "tpr": tpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.var_file = args.var_file or None
    args.var_file_nn = args.var_file_nn or None
    args.mask_file = args.mask_file or None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dask: use the CPUs Slurm gave us
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    dask.config.set(scheduler="threads", num_workers=n_workers)

    chunks = {"time": args.chunk_time}
    error_quantiles = [float(q) for q in args.quantiles.split(",")]

    # ---- resolve inputs (config file or manual) ------------------------
    var, path_original, path_nn, files, mask_file, config_name = resolve_inputs(args)
    # the mask path may contain {var}/{var_nc} tokens
    if mask_file and "{" in str(mask_file):
        mask_file = subst_tokens(mask_file, var)
        print(f"mask file : {mask_file}")
    years = sorted(files)
    tag = f"{var}_{years[0]}-{years[-1]}"
    print(f"=== Evaluation: {tag} ===")
    print(f"years     : {years}")
    print(f"config    : {config_name or 'manual mode'}")
    if not args.args_file:
        print(f"patterns  : nn='{args.file_pattern_nn}' "
              f"ref='{args.file_pattern_ref}'")

    # ---- sanity checks & auto-detection (fail fast, clear messages) ----
    check_files([f for pair in files.values() for f in pair])
    var_file = detect_var_file(args, files[years[0]][1])
    print(f"variable  : {var_file}")
    var_file_nn = detect_var_file_nn(args, str(files[years[0]][0]), var_file)
    # physical unit of the variable (for the plot metadata), if stored
    with xr.open_dataset(str(files[years[0]][1])) as ds0:
        unit = str(ds0[var_file].attrs.get("units", "")).strip()
    if unit:
        print(f"unit      : {unit}")

    # ---- deseasonalization settings -----------------------------------
    deseasonalize = args.deseasonalize
    if deseasonalize and len(years) < 2:
        print("WARNING: --deseasonalize with a single year is degenerate "
              "(each day is its own climatology -> zero anomalies). "
              "Skipping deseasonalization.")
        deseasonalize = False

    # ---- optional cell mask from a file -------------------------------
    keep_mask = build_keep_mask(args, mask_file, str(files[years[0]][1]))

    # ---- point / area (default Hamburg; all cells within radius) -------
    point_stats_on = not args.no_point_stats
    point_mask = None
    point_n_cells = 0
    point_cell = None  # (sel_lat, sel_lon) of the nearest cell (radius 0)
    if point_stats_on:
        with xr.open_dataset(str(files[years[0]][1])) as ds0:
            p_lat, p_lon = detect_spatial_names(ds0)
            point_mask = build_point_mask(ds0, p_lat, p_lon,
                                          args.point_lat, args.point_lon,
                                          args.point_radius_km)
        point_n_cells = int(point_mask.sum())
        if args.point_radius_km > 0:
            print(f"Area '{args.point_name}': {point_n_cells} grid cells "
                  f"within {args.point_radius_km} km of "
                  f"({args.point_lat}, {args.point_lon})")
        else:
            lat_vals = point_mask[p_lat].values
            lon_vals = point_mask[p_lon].values
            iy = int(np.argmin(np.abs(lat_vals - args.point_lat)))
            ix = int(np.argmin(np.abs(lon_vals - args.point_lon)))
            point_cell = (float(lat_vals[iy]), float(lon_vals[ix]))
            print(f"Point '{args.point_name}': requested "
                  f"({args.point_lat}, {args.point_lon}) -> grid cell "
                  f"({point_cell[0]:.4f}, {point_cell[1]:.4f})")
    # pooled moment accumulators (raw and deseasonalized) + area-mean series
    pt_keys = ("se", "sse", "sae", "sr", "ssr", "sn", "ssn", "cnr", "n")
    pt = dict.fromkeys(pt_keys, 0.0)
    pt_a = dict.fromkeys(pt_keys, 0.0)
    pt_nn_mean, pt_ref_mean = [], []
    pt_nn_mean_a, pt_ref_mean_a = [], []

    bias_maps, mse_maps, ev_maps, corr_maps = [], [], [], []
    var_ratio_maps, fgc_maps = [], []
    q_maps = {q: [] for q in error_quantiles}
    nn_qq_samples, ref_qq_samples = [], []
    sum_err = 0.0
    sum_sq = 0.0
    n_total = 0

    # keys match the stats_members entries ("doy_*") of pass 1
    doy_acc = {"doy_sum_ref": None, "doy_cnt_ref": None,
               "doy_sum_nn": None, "doy_cnt_nn": None}

    # ---------------- PASS 1: main loop over years (lazy) -------------
    for year in years:
        print(f"--- year {year} (pass 1) ---")
        # FIX (rev. 1): this call previously passed an undefined name
        # ('seasonalize') as a positional argument -> NameError.
        nn, ref = load_pair(str(files[year][0]), str(files[year][1]),
                            chunks, var_file, var_file_nn=var_file_nn,
                            keep_mask=keep_mask, rename_nn=args.rename_nn)
        err = nn - ref

        # per-grid-cell statistics (lazy)
        bias_y = err.mean("time")
        mse_y = (err ** 2).mean("time")
        ref_var = ref.var("time")
        ev_y = xr.where(ref_var > 1e-12, 1 - err.var("time") / ref_var, np.nan)

        nn_anom = nn - nn.mean("time")
        ref_anom = ref - ref.mean("time")
        corr_y = (nn_anom * ref_anom).mean("time") / (
            nn_anom.std("time") * ref_anom.std("time")
        )

        # distribution matching: variance ratio and fraction of gap closed
        # (gap = MAE of the no-skill climatology baseline around the ref)
        var_ratio_y = xr.where(ref_var > 1e-12, nn.var("time") / ref_var, np.nan)
        mad_ref_y = np.abs(ref_anom).mean("time")
        fgc_y = xr.where(mad_ref_y > 1e-12,
                         1 - np.abs(err).mean("time") / mad_ref_y, np.nan)

        err_q = err.isel(time=slice(None, None, args.quantile_stride))
        q_y = {}
        for q in error_quantiles:
            qm = err_q.quantile(q, dim="time")
            if "quantile" in qm.coords:
                qm = qm.drop_vars("quantile")
            q_y[q] = qm

        se = err.sum()
        sse = (err ** 2).sum()
        cnt = err.count()

        nn_sub = nn.isel(time=slice(None, None, args.qq_stride))
        ref_sub = ref.isel(time=slice(None, None, args.qq_stride))

        stats_members = {
            "bias": bias_y,
            "mse": mse_y,
            "ev": ev_y,
            "corr": corr_y,
            "var_ratio": var_ratio_y,
            "fgc": fgc_y,
            **{f"q{int(q * 100)}": q_y[q] for q in error_quantiles},
            "se": se,
            "sse": sse,
            "cnt": cnt,
            "nn_sub": nn_sub,
            "ref_sub": ref_sub,
        }
        # point/area pooled moments, computed in the same pass
        if point_stats_on:
            err_r = err.where(point_mask)
            ref_r = ref.where(point_mask)
            nn_r = nn.where(point_mask)
            stats_members["pt_se"] = err_r.sum()
            stats_members["pt_sse"] = (err_r ** 2).sum()
            stats_members["pt_sae"] = np.abs(err_r).sum()
            stats_members["pt_sr"] = ref_r.sum()
            stats_members["pt_ssr"] = (ref_r ** 2).sum()
            stats_members["pt_sn"] = nn_r.sum()
            stats_members["pt_ssn"] = (nn_r ** 2).sum()
            stats_members["pt_cnr"] = (nn_r * ref_r).sum()
            stats_members["pt_cnt"] = err_r.count()
            stats_members["pt_nn_mean"] = nn_r.mean(dim=[p_lat, p_lon])
            stats_members["pt_ref_mean"] = ref_r.mean(dim=[p_lat, p_lon])

        # day-of-year climatological sums (for pass 2), lazy
        # NOTE: group by the EXPLICIT day-of-year array, not the string
        # "time.dayofyear": string resolution only supports datetime64
        # time axes and fails for cftime (object) axes, e.g. calendar
        # "noleap" (AttributeError: 'IndexVariable' object has no
        # attribute 'dayofyear'). .dt.dayofyear supports both.
        if deseasonalize:
            doy = ref["time"].dt.dayofyear
            stats_members["doy_sum_ref"] = ref.groupby(doy).sum()
            stats_members["doy_cnt_ref"] = ref.groupby(doy).count()
            stats_members["doy_sum_nn"] = nn.groupby(doy).sum()
            stats_members["doy_cnt_nn"] = nn.groupby(doy).count()

        # one .compute() -> ONE pass over the data per year
        stats = xr.Dataset(stats_members).compute()

        bias_maps.append(stats["bias"])
        mse_maps.append(stats["mse"])
        ev_maps.append(stats["ev"])
        corr_maps.append(stats["corr"])
        var_ratio_maps.append(stats["var_ratio"])
        fgc_maps.append(stats["fgc"])
        for q in error_quantiles:
            q_maps[q].append(stats[f"q{int(q * 100)}"])
        sum_err += float(stats["se"])
        sum_sq += float(stats["sse"])
        n_total += int(stats["cnt"])

        nn_qq_samples.append(stats["nn_sub"].values.ravel().astype("float32"))
        ref_qq_samples.append(stats["ref_sub"].values.ravel().astype("float32"))
        if point_stats_on:
            for k, key in (("se", "pt_se"), ("sse", "pt_sse"),
                           ("sae", "pt_sae"), ("sr", "pt_sr"),
                           ("ssr", "pt_ssr"), ("sn", "pt_sn"),
                           ("ssn", "pt_ssn"), ("cnr", "pt_cnr"),
                           ("n", "pt_cnt")):
                pt[k] += float(stats[key])
            pt_nn_mean.append(stats["pt_nn_mean"])
            pt_ref_mean.append(stats["pt_ref_mean"])
        print(f"    valid points: {int(stats['cnt'])}")

        if deseasonalize:
            for k in doy_acc:
                v = stats[k].astype("float64")
                doy_acc[k] = v if doy_acc[k] is None else doy_acc[k] + v

    # ---------------- aggregate across years ----------------------------
    def mean_over_years(maps):
        return xr.concat(maps, dim="year").assign_coords(year=years).mean("year")

    scale = args.scale
    bias_map = mean_over_years(bias_maps) * scale
    mse_map = mean_over_years(mse_maps)
    rmse_map = np.sqrt(mse_map) * scale
    q_maps_mean = {q: mean_over_years(q_maps[q]) * scale for q in error_quantiles}
    ev_map = mean_over_years(ev_maps)
    corr_map = mean_over_years(corr_maps)
    r2_map = corr_map ** 2
    var_ratio_map = mean_over_years(var_ratio_maps)
    fgc_map = mean_over_years(fgc_maps)

    # ---------------- scalar stats (raw fields) --------------------------
    bias_scalar = sum_err / n_total
    rmse_scalar = np.sqrt(sum_sq / n_total)

    nn_all_raw = np.concatenate(nn_qq_samples)
    ref_all_raw = np.concatenate(ref_qq_samples)
    finite = np.isfinite(nn_all_raw) & np.isfinite(ref_all_raw)
    nn_all = nn_all_raw[finite]
    ref_all = ref_all_raw[finite]
    ev_scalar = 1 - np.var(nn_all - ref_all) / np.var(ref_all)
    r2_scalar = np.corrcoef(ref_all, nn_all)[0, 1] ** 2
    var_ref_all = np.var(ref_all)
    var_ratio_scalar = (float(np.var(nn_all) / var_ref_all)
                        if var_ref_all > 0 else np.nan)
    mae_scalar = float(np.mean(np.abs(nn_all - ref_all)))
    mad_ref_scalar = float(np.mean(np.abs(ref_all - ref_all.mean())))
    fgc_scalar = float(1 - mae_scalar / mad_ref_scalar) if mad_ref_scalar > 0 else np.nan

    print(f"Mean bias (NN - reference):    {bias_scalar:.4f}")
    print(f"Root Mean Squared Error (RMSE):{rmse_scalar:.4f}")
    print(f"Explained variance:           {ev_scalar:.4f}")
    print(f"R² (squared correlation):     {r2_scalar:.4f}")
    print(f"Variance ratio (NN/ref):      {var_ratio_scalar:.4f}")
    print(f"Fraction of gap closed:       {fgc_scalar:.4f}")

    # ROC / AUC / PSS for exceedance of reference quantiles (pooled samples)
    roc_results = []
    for q in (float(q) for q in args.roc_quantiles.split(",")):
        thr = float(np.quantile(ref_all, q))
        r = roc_analysis(nn_all, ref_all > thr, thr, args.roc_max_points)
        if r is None:
            print(f"ROC q={q:.2f}: degenerate event, skipped")
            continue
        r["q"] = q
        roc_results.append(r)
        print(f"ROC q={q:.2f} (threshold {thr:.4f}): "
              f"AUC={r['auc']:.4f}  PSS={r['pss']:.4f}")

    # ---------------- PASS 2: deseasonalized EV / correlation -----------
    ev_anom_map = corr_anom_map = r2_anom_map = None
    ev_anom_scalar = corr_anom_scalar = np.nan
    if deseasonalize:
        print("Building day-of-year climatology from all years ...")
        clim_ref = xr.where(doy_acc["doy_cnt_ref"] > 0,
                            doy_acc["doy_sum_ref"] / doy_acc["doy_cnt_ref"], np.nan)
        clim_nn = xr.where(doy_acc["doy_cnt_nn"] > 0,
                           doy_acc["doy_sum_nn"] / doy_acc["doy_cnt_nn"], np.nan)
        doy_acc = None  # free the accumulators

        ev_anom_maps, corr_anom_maps = [], []
        var_ratio_a_maps, fgc_a_maps = [], []
        # pooled scalar moments on anomalies
        # FIX: also accumulate the NN-anomaly moments (s_n, ss_n, c_nr) so
        # that the scalar correlation is corr(NN anomaly, ref anomaly).
        s_e = ss_e = s_r = ss_r = s_n = ss_n = c_nr = 0.0
        s_sae = s_sad = 0.0
        n_a = 0

        for year in years:
            print(f"--- year {year} (pass 2, deseasonalized) ---")
            nn, ref = load_pair(str(files[year][0]), str(files[year][1]),
                                chunks, var_file, var_file_nn=var_file_nn,
                                keep_mask=keep_mask, rename_nn=args.rename_nn)

            doy = ref["time"].dt.dayofyear
            # per-cell climatology indexed back onto the time axis
            ref_anom = ref - clim_ref.sel(dayofyear=doy)
            nn_anom = nn - clim_nn.sel(dayofyear=doy)
            err_a = nn_anom - ref_anom

            ref_var_a = ref_anom.var("time")
            ev_a = xr.where(ref_var_a > 1e-12,
                            1 - err_a.var("time") / ref_var_a, np.nan)

            nn_an = nn_anom - nn_anom.mean("time")
            ref_an = ref_anom - ref_anom.mean("time")
            corr_a = (nn_an * ref_an).mean("time") / (
                nn_an.std("time") * ref_an.std("time")
            )

            # distribution matching on anomalies
            var_ratio_a = xr.where(ref_var_a > 1e-12,
                                   nn_anom.var("time") / ref_var_a, np.nan)
            mad_ref_a = np.abs(ref_an).mean("time")
            fgc_a = xr.where(mad_ref_a > 1e-12,
                             1 - np.abs(err_a).mean("time") / mad_ref_a,
                             np.nan)

            # pooled moments for the scalar EV / correlation
            m = {
                "se": err_a.sum(),
                "sse": (err_a ** 2).sum(),
                "sr": ref_anom.sum(),
                "ssr": (ref_anom ** 2).sum(),
                "sn": nn_anom.sum(),
                "ssn": (nn_anom ** 2).sum(),
                "cnr": (nn_anom * ref_anom).sum(),
                "sae": np.abs(err_a).sum(),
                "sad": np.abs(ref_an).sum(),
                "cnt": err_a.count(),
            }
            # deseasonalized point/area pooled moments (same pass)
            if point_stats_on:
                err_a_r = err_a.where(point_mask)
                ref_a_r = ref_anom.where(point_mask)
                nn_a_r = nn_anom.where(point_mask)
                m["pt_se"] = err_a_r.sum()
                m["pt_sse"] = (err_a_r ** 2).sum()
                m["pt_sae"] = np.abs(err_a_r).sum()
                m["pt_sr"] = ref_a_r.sum()
                m["pt_ssr"] = (ref_a_r ** 2).sum()
                m["pt_sn"] = nn_a_r.sum()
                m["pt_ssn"] = (nn_a_r ** 2).sum()
                m["pt_cnr"] = (nn_a_r * ref_a_r).sum()
                m["pt_cnt"] = err_a_r.count()
                m["pt_nn_mean"] = nn_a_r.mean(dim=[p_lat, p_lon])
                m["pt_ref_mean"] = ref_a_r.mean(dim=[p_lat, p_lon])
            stats2 = xr.Dataset({"ev": ev_a, "corr": corr_a,
                                 "var_ratio_a": var_ratio_a, "fgc_a": fgc_a,
                                 **m}).compute()

            ev_anom_maps.append(stats2["ev"])
            corr_anom_maps.append(stats2["corr"])
            var_ratio_a_maps.append(stats2["var_ratio_a"])
            fgc_a_maps.append(stats2["fgc_a"])
            s_e += float(stats2["se"])
            ss_e += float(stats2["sse"])
            s_r += float(stats2["sr"])
            ss_r += float(stats2["ssr"])
            s_n += float(stats2["sn"])
            ss_n += float(stats2["ssn"])
            c_nr += float(stats2["cnr"])
            s_sae += float(stats2["sae"])
            s_sad += float(stats2["sad"])
            n_a += int(stats2["cnt"])
            if point_stats_on:
                for k, key in (("se", "pt_se"), ("sse", "pt_sse"),
                               ("sae", "pt_sae"), ("sr", "pt_sr"),
                               ("ssr", "pt_ssr"), ("sn", "pt_sn"),
                               ("ssn", "pt_ssn"), ("cnr", "pt_cnr"),
                               ("n", "pt_cnt")):
                    pt_a[k] += float(stats2[key])
                pt_nn_mean_a.append(stats2["pt_nn_mean"])
                pt_ref_mean_a.append(stats2["pt_ref_mean"])

        ev_anom_map = mean_over_years(ev_anom_maps)
        corr_anom_map = mean_over_years(corr_anom_maps)
        r2_anom_map = corr_anom_map ** 2
        var_ratio_anom_map = mean_over_years(var_ratio_a_maps)
        fgc_anom_map = mean_over_years(fgc_a_maps)

        # scalar EV / correlation from pooled anomaly moments
        var_e = (ss_e - s_e ** 2 / n_a) / (n_a - 1)
        var_r = (ss_r - s_r ** 2 / n_a) / (n_a - 1)
        var_n = (ss_n - s_n ** 2 / n_a) / (n_a - 1)
        cov_nr = (c_nr - s_n * s_r / n_a) / (n_a - 1)
        ev_anom_scalar = 1 - var_e / var_r
        corr_anom_scalar = cov_nr / np.sqrt(var_n * var_r)

        print(f"Anomaly explained variance:  {ev_anom_scalar:.4f}")
        print(f"Anomaly correlation:         {corr_anom_scalar:.4f}")
        var_ratio_anom_scalar = float(var_n / var_r) if var_r > 0 else np.nan
        fgc_anom_scalar = float(1 - s_sae / s_sad) if s_sad > 0 else np.nan
        print(f"Anomaly variance ratio:      {var_ratio_anom_scalar:.4f}")
        print(f"Anomaly fraction gap closed: {fgc_anom_scalar:.4f}")

    # ---------------- point / area statistics (e.g. Hamburg) ------------
    point_results = point_results_anom = None
    if point_stats_on:
        point_results = pooled_stats(pt["se"], pt["sse"], pt["sae"],
                                     pt["sr"], pt["ssr"], pt["sn"],
                                     pt["ssn"], pt["cnr"], pt["n"])
        if args.point_radius_km > 0:
            print(f"Area '{args.point_name}' ({point_n_cells} cells within "
                  f"{args.point_radius_km} km), raw fields:")
        else:
            print(f"Point '{args.point_name}' (grid cell "
                  f"{point_cell[0]:.4f}, {point_cell[1]:.4f}), raw fields:")
        print(f"  bias={point_results['bias']:.4f}  "
              f"RMSE={point_results['rmse']:.4f}  "
              f"MAE={point_results['mae']:.4f}")
        print(f"  EV={point_results['ev']:.4f}  "
              f"corr={point_results['corr']:.4f}  "
              f"R2={point_results['r2']:.4f}  "
              f"var_ratio={point_results['var_ratio']:.4f}  "
              f"(n={point_results['n']})")
        if point_results["n"] < 2:
            print(f"WARNING: '{args.point_name}' area has fewer than 2 "
                  "valid values (masked out?) -> statistics are NaN.")
        if deseasonalize:
            point_results_anom = pooled_stats(pt_a["se"], pt_a["sse"],
                                              pt_a["sae"], pt_a["sr"],
                                              pt_a["ssr"], pt_a["sn"],
                                              pt_a["ssn"], pt_a["cnr"],
                                              pt_a["n"])
            print(f"Area '{args.point_name}', deseasonalized:")
            print(f"  EV={point_results_anom['ev']:.4f}  "
                  f"corr={point_results_anom['corr']:.4f}  "
                  f"R2={point_results_anom['r2']:.4f}  "
                  f"bias={point_results_anom['bias']:.4f}  "
                  f"RMSE={point_results_anom['rmse']:.4f}  "
                  f"MAE={point_results_anom['mae']:.4f}  "
                  f"var_ratio={point_results_anom['var_ratio']:.4f}  "
                  f"(n={point_results_anom['n']})")

    # ---------------- save maps + stats ---------------------------------
    maps = {
        "bias": bias_map,
        "mse": mse_map,
        "rmse": rmse_map,
        "explained_variance": ev_map,
        "correlation": corr_map,
        "r2": r2_map,
        "variance_ratio": var_ratio_map,
        "fraction_gap_closed": fgc_map,
        **{f"error_q{int(q * 100):02d}": q_maps_mean[q] for q in error_quantiles},
    }
    if deseasonalize:
        maps.update({
            "ev_anomaly": ev_anom_map,
            "corr_anomaly": corr_anom_map,
            "r2_anomaly": r2_anom_map,
            "variance_ratio_anomaly": var_ratio_anom_map,
            "fraction_gap_closed_anomaly": fgc_anom_map,
        })
    maps_ds = xr.Dataset(maps)
    nc_path = out_dir / f"{tag}_maps.nc"
    maps_ds.to_netcdf(nc_path)
    print(f"Maps written to {nc_path}")

    stats_path = out_dir / f"{tag}_stats.txt"
    with stats_path.open("w") as f:
        f.write(f"Evaluation: {tag}\n")
        f.write(f"variable:       {var_file}\n")
        f.write(f"NN variable:    {var_file_nn}\n")
        f.write(f"reference dir:  {path_original}\n")
        f.write(f"NN dir:         {path_nn}\n")
        f.write(f"args file:      {args.args_file}\n")
        f.write(f"mask file:      {mask_file}\n")
        f.write(f"quantiles:      {args.quantiles}\n")
        f.write(f"map scale:      {args.scale}\n")
        f.write(f"deseasonalized: {deseasonalize}\n\n")
        f.write(f"Mean bias (NN - reference):     {bias_scalar:.6f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse_scalar:.6f}\n")
        f.write(f"Explained variance:            {ev_scalar:.6f}\n")
        f.write(f"R2 (squared correlation):      {r2_scalar:.6f}\n")
        f.write(f"Variance ratio (NN/ref):       {var_ratio_scalar:.6f}\n")
        f.write(f"Fraction of gap closed:        {fgc_scalar:.6f}\n")
        if deseasonalize:
            f.write(f"Anomaly explained variance:    {ev_anom_scalar:.6f}\n")
            f.write(f"Anomaly correlation:           {corr_anom_scalar:.6f}\n")
            f.write(f"Anomaly variance ratio:        {var_ratio_anom_scalar:.6f}\n")
            f.write(f"Anomaly fraction gap closed:   {fgc_anom_scalar:.6f}\n")
        if roc_results:
            f.write("\nROC/AUC/PSS (exceedance of reference quantiles):\n")
            for r in roc_results:
                f.write(f"  q={r['q']:.2f} threshold={r['threshold']:.6f}: "
                        f"AUC={r['auc']:.6f}  PSS={r['pss']:.6f}\n")
        f.write(f"Valid space-time points:        {n_total}\n")
        if point_stats_on:
            if args.point_radius_km > 0:
                f.write(f"\nArea statistics ({args.point_name}, "
                        f"{point_n_cells} cells within "
                        f"{args.point_radius_km} km of "
                        f"{args.point_lat}/{args.point_lon}), raw fields:\n")
            else:
                f.write(f"\nPoint statistics ({args.point_name}, requested "
                        f"{args.point_lat}/{args.point_lon}, grid cell "
                        f"{point_cell[0]:.4f}/{point_cell[1]:.4f}), "
                        f"raw fields:\n")
            f.write(f"  Area bias (NN - reference):  {point_results['bias']:.6f}\n")
            f.write(f"  Area RMSE:                   {point_results['rmse']:.6f}\n")
            f.write(f"  Area MAE:                    {point_results['mae']:.6f}\n")
            f.write(f"  Area explained variance:     {point_results['ev']:.6f}\n")
            f.write(f"  Area correlation:            {point_results['corr']:.6f}\n")
            f.write(f"  Area R2 (squared corr.):     {point_results['r2']:.6f}\n")
            f.write(f"  Area variance ratio:         {point_results['var_ratio']:.6f}\n")
            f.write(f"  Area valid space-time points:{point_results['n']:d}\n")
            if deseasonalize:
                f.write(f"Area statistics ({args.point_name}), deseasonalized:\n")
                f.write(f"  Area anomaly bias:           {point_results_anom['bias']:.6f}\n")
                f.write(f"  Area anomaly RMSE:           {point_results_anom['rmse']:.6f}\n")
                f.write(f"  Area anomaly MAE:            {point_results_anom['mae']:.6f}\n")
                f.write(f"  Area anomaly EV:             {point_results_anom['ev']:.6f}\n")
                f.write(f"  Area anomaly correlation:    {point_results_anom['corr']:.6f}\n")
                f.write(f"  Area anomaly R2:             {point_results_anom['r2']:.6f}\n")
                f.write(f"  Area anomaly variance ratio: {point_results_anom['var_ratio']:.6f}\n")
    print(f"Stats written to {stats_path}")

    # ---------------- figures ------------------------------------------
    ylab = f"{years[0]}-{years[-1]}"
    # metadata line shown in the title of EVERY figure:
    # variable, unit, time span and config name
    meta_line = (f"{var} ({var_file})" + (f" [{unit}]" if unit else "")
                 + f", {ylab}, config: {config_name or 'manual mode'}")

    def with_meta(title):
        return f"{title}\n{meta_line}"

    def map_and_hist(da, cmap, vmin, vmax, xlabel, unit_label, fname, title,
                     mean_line=True, extra_vlines=(), hist_title=None):
        fig, (ax_map, ax_hist) = plt.subplots(
            1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2, 1]}
        )
        da.plot(ax=ax_map, cmap=cmap, vmin=vmin, vmax=vmax,
                cbar_kwargs={"label": unit_label})
        ax_map.set_title(with_meta(title))
        vals = da.values.ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ax_hist.hist(vals, bins=51, color="C0", alpha=0.8)
            if mean_line:
                ax_hist.axvline(0, color="k", lw=1)
                ax_hist.axvline(vals.mean(), color="C3", ls="--",
                                label=f"mean = {vals.mean():.4f}\n"
                                      f"median = {np.median(vals):.4f}")
            for x, kw in extra_vlines:
                ax_hist.axvline(x, **kw)
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel(xlabel)
        ax_hist.set_ylabel("Number of grid cells")
        if hist_title:
            ax_hist.set_title(hist_title)
        if ax_hist.get_legend_handles_labels()[1]:
            ax_hist.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(out_dir / f"{tag}_{fname}.png", dpi=150)
        plt.close(fig)

    # --- MSE, bias, RMSE -------------------------------------------------
    map_and_hist(mse_map, "viridis", None, None, "MSE", "MSE", "mse",
                 f"Mean MSE per grid cell, {ylab}",
                 mean_line=False, hist_title="Distribution of per-cell MSE")

    vmax = float(np.nanmax(np.abs(bias_map.values)))
    map_and_hist(bias_map, "RdBu_r", -vmax, vmax,
                 "Bias (NN - reference)", "Bias (NN - reference)",
                 "bias", f"Mean bias, {ylab}")

    rmse_vals = rmse_map.values.ravel()
    rmse_vals = rmse_vals[np.isfinite(rmse_vals)]
    global_rmse = float(np.sqrt(np.mean(rmse_vals ** 2)))
    map_and_hist(rmse_map, "viridis", None, None, "RMSE", "RMSE", "rmse",
                 f"RMSE per grid cell, {ylab}", mean_line=False,
                 extra_vlines=[(global_rmse, {"color": "C3", "ls": "--", "lw": 1,
                                             "label": f"global RMSE = {global_rmse:.4f}"})],
                 hist_title="Distribution of per-cell RMSE")

    # --- error quantiles --------------------------------------------------
    vmax_q = max(float(np.nanmax(np.abs(m.values))) for m in q_maps_mean.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, q in enumerate(error_quantiles):
        vals = q_maps_mean[q].values.ravel()
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=51, range=(-vmax_q, vmax_q),
                histtype="step", lw=1.5, color=f"C{i}", label=f"q={q:.2f}")
    ax.axvline(0, color="k", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Error quantile value (NN - reference)")
    ax.set_ylabel("Number of grid cells")
    ax.set_title(with_meta(f"Per-cell error quantiles, {ylab}"))
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_dir / f"{tag}_error_quantiles_hist.png", dpi=150)
    plt.close(fig)

    for q in error_quantiles:
        qm = q_maps_mean[q]
        fig, (ax_map, ax_hist) = plt.subplots(
            1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2, 1]}
        )
        qm.plot(ax=ax_map, cmap="RdBu_r", vmin=-vmax_q, vmax=vmax_q,
                cbar_kwargs={"label": f"q{int(q * 100):02d} of error (NN - reference)"})
        ax_map.set_title(with_meta(f"Error quantile q={q:.2f} per grid cell, {ylab}"))
        vals = qm.values.ravel()
        vals = vals[np.isfinite(vals)]
        ax_hist.hist(vals, bins=51, color="C0", alpha=0.8)
        ax_hist.axvline(0, color="k", lw=1)
        ax_hist.axvline(np.median(vals), color="C3", ls="--", lw=1,
                        label=f"median = {np.median(vals):.4f}")
        ax_hist.set_xlim(-vmax_q, vmax_q)
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel(f"q{int(q * 100):02d} of error (NN - reference)")
        ax_hist.set_ylabel("Number of grid cells")
        ax_hist.set_title(f"Distribution of per-cell error quantile (q={q:.2f})")
        ax_hist.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(out_dir / f"{tag}_q{int(q * 100):02d}.png", dpi=150)
        plt.close(fig)

    # --- explained variance (raw) -----------------------------------------
    map_and_hist(ev_map, "RdBu_r", -1, 1,
                 "Explained variance", "Explained variance",
                 "explained_variance",
                 f"Explained variance per grid cell, {ylab}")

    # --- deseasonalized EV / correlation -----------------------------------
    if deseasonalize:
        map_and_hist(ev_anom_map, "RdBu_r", -1, 1,
                     "Explained variance (anomalies)",
                     "Explained variance (deseasonalized)",
                     "ev_anomaly",
                     f"Explained variance of deseasonalized data, {ylab}")
        map_and_hist(corr_anom_map, "RdBu_r", -1, 1,
                     "Correlation (anomalies)",
                     "Correlation (deseasonalized)",
                     "corr_anomaly",
                     f"Correlation of deseasonalized data, {ylab}")

    # --- distribution matching: variance ratio & fraction of gap closed ---
    vr_vals = var_ratio_map.values.ravel()
    vr_vals = vr_vals[np.isfinite(vr_vals)]
    dev = float(np.max(np.abs(vr_vals - 1))) if vr_vals.size else 1.0
    dev = min(max(dev, 0.5), 2.0)  # symmetric colour range around 1
    map_and_hist(var_ratio_map, "RdBu_r", 1 - dev, 1 + dev,
                 "Variance ratio (NN/ref)", "Variance ratio (NN/ref)",
                 "variance_ratio",
                 f"Variance ratio per grid cell, {ylab}")
    map_and_hist(fgc_map, "RdBu_r", -1, 1,
                 "Fraction of gap closed", "Fraction of gap closed",
                 "fraction_gap_closed",
                 f"Fraction of gap closed per grid cell, {ylab}")
    if deseasonalize:
        map_and_hist(var_ratio_anom_map, "RdBu_r", 1 - dev, 1 + dev,
                     "Variance ratio (anomalies)", "Variance ratio (NN/ref, deseasonalized)",
                     "variance_ratio_anomaly",
                     f"Variance ratio of deseasonalized data, {ylab}")
        map_and_hist(fgc_anom_map, "RdBu_r", -1, 1,
                     "Fraction of gap closed (anomalies)",
                     "Fraction of gap closed (deseasonalized)",
                     "fraction_gap_closed_anomaly",
                     f"Fraction of gap closed of deseasonalized data, {ylab}")

    # --- ROC curves for exceedance events (AUC / PSS) ----------------------
    if roc_results:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="no skill")
        for i, r in enumerate(roc_results):
            ax.plot(r["fpr"], r["tpr"], lw=1.5, color=f"C{i}",
                    label=f"q={r['q']:.2f}: AUC={r['auc']:.3f}, "
                          f"PSS={r['pss']:.3f}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(with_meta("ROC curves for exceedance events"))
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        plt.tight_layout()
        fig.savefig(out_dir / f"{tag}_roc.png", dpi=150)
        plt.close(fig)

    # --- pooled QQ plot -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    nn_sorted = np.sort(nn_all)
    ref_sorted = np.sort(ref_all)
    n_pts = min(20000, len(nn_sorted))
    idx = np.unique(np.linspace(0, len(nn_sorted) - 1, n_pts).astype(int))
    lims = [min(ref_sorted.min(), nn_sorted.min()), max(ref_sorted.max(), nn_sorted.max())]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.plot(ref_sorted[idx], nn_sorted[idx], ".", ms=2, color="C0", alpha=0.5)
    ax.set_xlabel("Reference quantiles")
    ax.set_ylabel("NN quantiles")
    ax.set_title(with_meta(f"Pooled QQ plot, {ylab}"))
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_dir / f"{tag}_qq.png", dpi=150)
    plt.close(fig)

    # --- distribution of values: NN vs reference --------------------------
    # Same pooled, stride-sub-sampled samples as the QQ plot (memory-safe
    # for many years). Left: normalized histograms overlaid; right: the
    # empirical CDFs, which make tail differences easy to see.
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(13, 5))
    lo = min(float(nn_all.min()), float(ref_all.min()))
    hi = max(float(nn_all.max()), float(ref_all.max()))
    bins = np.linspace(lo, hi, 51)
    ax_hist.hist(ref_all, bins=bins, density=True, alpha=0.6,
                 color="C1", label="Reference")
    ax_hist.hist(nn_all, bins=bins, density=True, alpha=0.6,
                 color="C0", label="NN")
    ax_hist.axvline(float(ref_all.mean()), color="C1", ls="--", lw=1,
                    label=f"ref mean = {float(ref_all.mean()):.4f}")
    ax_hist.axvline(float(nn_all.mean()), color="C0", ls="--", lw=1,
                    label=f"NN mean  = {float(nn_all.mean()):.4f}")
    ax_hist.set_xlabel(f"{var_file} value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(with_meta(f"Distribution of values, {ylab}"))
    ax_hist.legend(frameon=False)

    for data, color, lab in ((ref_all, "C1", "Reference"),
                             (nn_all, "C0", "NN")):
        xs = np.sort(data)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax_cdf.plot(xs, ys, color=color, lw=1.5, label=lab)
    ax_cdf.set_xlabel(f"{var_file} value")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_title(with_meta(f"Empirical CDF, {ylab}"))
    ax_cdf.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(out_dir / f"{tag}_value_distribution.png", dpi=150)
    plt.close(fig)

    # --- point / area time series (e.g. Hamburg and surroundings) ---------
    if point_stats_on:
        nn_pt_all = xr.concat(pt_nn_mean, dim="time")
        ref_pt_all = xr.concat(pt_ref_mean, dim="time")
        time_pt = nn_pt_all["time"]
        pname = args.point_name.lower().replace(" ", "_")
        if args.point_radius_km > 0:
            kind = "area"
            area_lbl = (f"{point_n_cells} cells within "
                        f"{args.point_radius_km} km, area mean")
        else:
            kind = "point"
            area_lbl = f"grid cell {point_cell[0]:.2f}, {point_cell[1]:.2f}"
        fig, (ax_ts, ax_err) = plt.subplots(2, 1, figsize=(13, 7),
                                            sharex=True)
        ax_ts.plot(time_pt, ref_pt_all, color="C1", lw=0.8, alpha=0.6,
                   label="Reference")
        ax_ts.plot(time_pt, nn_pt_all, color="C0", lw=0.8, alpha=0.6,
                   label="NN")
        ax_ts.set_ylabel(var_file)
        ax_ts.set_title(with_meta(f"{args.point_name} ({area_lbl}), {ylab}"))
        ax_ts.legend(frameon=False)
        err_pt = nn_pt_all - ref_pt_all
        ax_err.plot(time_pt, err_pt, color="C3", lw=0.8, alpha=0.7)
        ax_err.axhline(0, color="k", lw=1)
        ax_err.set_ylabel("Error (NN - reference, area mean)")
        ax_err.set_xlabel("time")
        plt.tight_layout()
        fig.savefig(out_dir / f"{tag}_{kind}_{pname}_timeseries.png",
                    dpi=150)
        plt.close(fig)

    print("All figures written.")


if __name__ == "__main__":
    main()