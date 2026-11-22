import argparse
import json
import warnings

import numpy as np

DEFAULT_PRECIP_NAMES = ("tp", "precip", "pr", "prate")
DEFAULT_N_QUANTILES = 101
_EPS = 1e-12

QDM_REFERENCE = (
    "Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of "
    "GCM precipitation by quantile mapping: How well do methods preserve changes "
    "in quantiles and extremes? Journal of Climate, 28(17), 6938-6959. "
    "https://doi.org/10.1175/JCLI-D-14-00754.1"
)


def _str_list(arg):
    return arg.split(",")


def _to_array_time(x, time):
    if hasattr(x, "values") and hasattr(x, "dims") and "time" in getattr(x, "dims", ()):
        if time is None:
            time = x["time"].values
        x = x.values
    return np.asarray(x, dtype=np.float64), time


def _group_indices(time, n_time, group_by_month):
    if not group_by_month:
        return np.zeros(n_time, dtype=int)
    if time is None:
        raise ValueError(
            "group_by_month=True requires a time coordinate (np.datetime64 array "
            "or xarray DataArray with a 'time' dimension)"
        )
    t = np.asarray(time)
    if not np.issubdtype(t.dtype, np.datetime64):
        raise ValueError("time must be a np.datetime64 array or pandas/xarray time index")
    return (t.astype("datetime64[M]").astype(int) % 12).astype(int)


def _interp_per_cell(x, xp, fp, clip=True, time_chunk=128, cell_chunk=4096):
    # x: (m, n) values, xp/fp: (n, q) per-cell non-decreasing grids.
    # Linear interpolation of fp at x per cell; NaN in x propagates to NaN.
    m, n = x.shape
    q = xp.shape[1]
    out = np.full((m, n), np.nan, dtype=np.float64)
    nan_mask = np.isnan(x)
    for t0 in range(0, m, time_chunk):
        t1 = min(t0 + time_chunk, m)
        for s in range(0, n, cell_chunk):
            e = min(s + cell_chunk, n)
            xc = x[t0:t1, s:e]
            xpc = xp[s:e]
            fpc = fp[s:e]
            if clip:
                xc = np.clip(xc, xpc[:, 0][None, :], xpc[:, -1][None, :])
            idx = (xc[:, :, None] >= xpc[None, :, :]).sum(axis=2)
            idx = np.clip(idx, 1, q - 1)
            mt, mc = xc.shape
            xp_b = np.broadcast_to(xpc[None, :, :], (mt, mc, q))
            fp_b = np.broadcast_to(fpc[None, :, :], (mt, mc, q))
            xp_lo = np.take_along_axis(xp_b, (idx - 1)[..., None], axis=2)[..., 0]
            xp_hi = np.take_along_axis(xp_b, idx[..., None], axis=2)[..., 0]
            fp_lo = np.take_along_axis(fp_b, (idx - 1)[..., None], axis=2)[..., 0]
            fp_hi = np.take_along_axis(fp_b, idx[..., None], axis=2)[..., 0]
            denom = xp_hi - xp_lo
            w = np.where(denom > 0, (xc - xp_lo) / np.where(denom > 0, denom, 1.0), 0.0)
            out[t0:t1, s:e] = np.where(
                nan_mask[t0:t1, s:e], np.nan, fp_lo + w * (fp_hi - fp_lo)
            )
    return out


class QuantileDeltaMapping:
    """Univariate quantile delta mapping (QDM) per grid cell and variable.

    Implements the QDM of Cannon et al. (2015, doi:10.1175/JCLI-D-14-00754.1):
    for a value x of the model/driving data, tau = F_mod_hist(x), and the
    corrected value is
      additive (interval) variables: F_ref_hist^-1(tau) + F_mod_fut^-1(tau) - F_mod_hist^-1(tau)
      multiplicative (ratio) variables: F_ref_hist^-1(tau) * F_mod_fut^-1(tau) / F_mod_hist^-1(tau)
    With mod_fut = mod_hist this reduces to empirical quantile mapping (EQM).

    Typical use case: a model trained on a reference dataset (e.g. CERRA) is
    driven with another dataset (e.g. MPI-GE). QDM is fitted on the overlap
    period of both datasets and maps the driving data into the reference space
    while preserving the driving data's own (e.g. scenario) changes.
    """

    def __init__(
        self,
        data_types,
        n_quantiles=DEFAULT_N_QUANTILES,
        group_by_month=False,
        ratio=None,
        precip_names=DEFAULT_PRECIP_NAMES,
    ):
        self.data_types = list(data_types)
        self.n_quantiles = int(n_quantiles)
        self.tau = (np.arange(self.n_quantiles) + 0.5) / self.n_quantiles
        self.group_by_month = bool(group_by_month)
        self.n_groups = 12 if self.group_by_month else 1
        self.precip_names = tuple(precip_names)
        if ratio is None:
            ratio = {}
        self.ratio = {
            dt: bool(ratio.get(dt, dt in self.precip_names)) for dt in self.data_types
        }
        self.n_lat = None
        self.n_lon = None
        self.q_ref = {}
        self.q_mod = {}
        self.q_fut = {}

    def _check_grid(self, shape):
        if self.n_lat is None:
            self.n_lat, self.n_lon = shape[1], shape[2]
        elif (self.n_lat, self.n_lon) != (shape[1], shape[2]):
            raise ValueError(
                "Spatial shape {} does not match the QDM grid ({}, {}). "
                "Regrid the data to the fitting grid first.".format(
                    shape[1:], self.n_lat, self.n_lon
                )
            )

    def _quantiles_per_group(self, x2d, groups):
        ncell = x2d.shape[1]
        out = np.full((self.n_groups, ncell, self.n_quantiles), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for g in range(self.n_groups):
                sel = x2d[groups == g]
                if sel.shape[0] == 0:
                    continue
                out[g] = np.nanquantile(sel, self.tau, axis=0).T
        return out

    def fit(self, data_type, ref, mod, time=None, mod_fut=None, fut_time=None):
        """Fit QDM transfer functions for one variable.

        ref, mod: (time, lat, lon) arrays (numpy or xarray) of the reference
        (e.g. CERRA) and the model/driving data (e.g. MPI-GE) over the
        calibration period. mod_fut: same for the application period (used for
        the QDM delta); if None, the calibration model data is used (EQM).
        """
        if data_type not in self.data_types:
            raise ValueError(
                "Unknown data type '{}'. Known: {}".format(data_type, self.data_types)
            )
        ref, time = _to_array_time(ref, time)
        mod, _ = _to_array_time(mod, None)
        if mod_fut is not None:
            mod_fut, fut_time = _to_array_time(mod_fut, fut_time)
        if ref.shape != mod.shape or ref.ndim != 3:
            raise ValueError(
                "ref and mod must have identical (time, lat, lon) shapes, got "
                "{} and {}".format(ref.shape, mod.shape)
            )
        self._check_grid(ref.shape)
        groups = _group_indices(time, ref.shape[0], self.group_by_month)
        r = ref.reshape(ref.shape[0], -1)
        m = mod.reshape(mod.shape[0], -1)
        if mod_fut is not None:
            f = mod_fut.reshape(mod_fut.shape[0], -1)
            f_groups = _group_indices(fut_time, mod_fut.shape[0], self.group_by_month)
        else:
            f, f_groups = m, groups
        self.q_ref[data_type] = self._quantiles_per_group(r, groups)
        self.q_mod[data_type] = self._quantiles_per_group(m, groups)
        self.q_fut[data_type] = self._quantiles_per_group(f, f_groups)
        return self

    def apply(self, data_type, x, time=None):
        """Apply QDM to one variable: x is (time, lat, lon), returns corrected."""
        if data_type not in self.q_ref:
            raise ValueError(
                "No fitted QDM for data type '{}'. Fitted: {}".format(
                    data_type, list(self.q_ref)
                )
            )
        x, time = _to_array_time(x, time)
        if x.ndim != 3:
            raise ValueError("x must be (time, lat, lon), got {}".format(x.shape))
        self._check_grid(x.shape)
        groups = _group_indices(time, x.shape[0], self.group_by_month)
        ncell = x.shape[1] * x.shape[2]
        x2d = x.reshape(x.shape[0], ncell)
        out = np.full_like(x2d, np.nan)
        tau_grid = np.broadcast_to(self.tau[None, :], (ncell, self.n_quantiles))
        for g in range(self.n_groups):
            sel = groups == g
            if not np.any(sel):
                continue
            xg = x2d[sel]
            q_mod_g = self.q_mod[data_type][g]
            q_ref_g = self.q_ref[data_type][g]
            q_fut_g = self.q_fut[data_type][g]
            tau = _interp_per_cell(xg, q_mod_g, tau_grid, clip=True)
            q_ref_at = _interp_per_cell(tau, tau_grid, q_ref_g, clip=True)
            q_mod_at = _interp_per_cell(tau, tau_grid, q_mod_g, clip=True)
            q_fut_at = _interp_per_cell(tau, tau_grid, q_fut_g, clip=True)
            if self.ratio[data_type]:
                denom = np.where(q_mod_at > _EPS, q_mod_at, 1.0)
                r = np.where(q_mod_at > _EPS, q_fut_at / denom, 1.0)
                corr = np.maximum(q_ref_at * r, 0.0)
            else:
                corr = q_ref_at + (q_fut_at - q_mod_at)
            out[sel] = corr
        return out.reshape(x.shape)

    def apply_channel(self, data_type, arr):
        """Alias kept for readability at call sites applying one loaded channel."""
        return self.apply(data_type, arr)

    def save(self, path):
        meta = {
            "data_types": self.data_types,
            "n_quantiles": self.n_quantiles,
            "group_by_month": self.group_by_month,
            "n_groups": self.n_groups,
            "n_lat": self.n_lat,
            "n_lon": self.n_lon,
            "ratio": self.ratio,
            "precip_names": list(self.precip_names),
            "reference": QDM_REFERENCE,
        }
        arrays = {"meta": np.array(json.dumps(meta)), "tau": self.tau}
        for dt in self.data_types:
            if dt in self.q_ref:
                arrays["{}__q_ref".format(dt)] = self.q_ref[dt]
                arrays["{}__q_mod".format(dt)] = self.q_mod[dt]
                arrays["{}__q_fut".format(dt)] = self.q_fut[dt]
        with open(path, "wb") as fh:
            np.savez(fh, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            qdm = cls(
                meta["data_types"],
                n_quantiles=meta["n_quantiles"],
                group_by_month=meta["group_by_month"],
                ratio=meta["ratio"],
                precip_names=meta.get("precip_names", DEFAULT_PRECIP_NAMES),
            )
            qdm.tau = npz["tau"]
            qdm.n_lat = meta["n_lat"]
            qdm.n_lon = meta["n_lon"]
            for dt in meta["data_types"]:
                key = "{}__q_ref".format(dt)
                if key in npz:
                    qdm.q_ref[dt] = npz[key]
                    qdm.q_mod[dt] = npz["{}__q_mod".format(dt)]
                    qdm.q_fut[dt] = npz["{}__q_fut".format(dt)]
        return qdm


def load_qdm(path):
    return QuantileDeltaMapping.load(path)


def _parse_period(period):
    if period is None:
        return None
    parts = period.split("-")
    if len(parts) == 1:
        return (parts[0], parts[0] + "-12-31")
    return (parts[0], parts[1] + "-12-31")


def _open_stack(files, var, period):
    import xarray as xr

    if isinstance(files, str):
        files = [files]
    da = xr.open_mfdataset(list(files), combine="by_coords")[var]
    if period is not None:
        da = da.sel(time=slice(period[0], period[1]))
    return da.load()


def fit_qdm_from_netcdf(
    ref_files,
    mod_files,
    data_types,
    out_path,
    mod_fut_files=None,
    hist_period=None,
    fut_period=None,
    n_quantiles=DEFAULT_N_QUANTILES,
    group_by_month=False,
    precip_names=DEFAULT_PRECIP_NAMES,
):
    """Fit QDM from NetCDF files and save the transfer functions to out_path (.npz).

    ref_files: reference dataset (e.g. CERRA) over the calibration period.
    mod_files: driving dataset (e.g. MPI-GE historical) over the same period.
    mod_fut_files: optional driving dataset of the application period (e.g.
    MPI-GE scenario) used for the QDM delta; defaults to mod_files (EQM).
    """
    hist_period = _parse_period(hist_period)
    fut_period = _parse_period(fut_period) if fut_period is not None else hist_period
    qdm = QuantileDeltaMapping(
        list(data_types),
        n_quantiles=n_quantiles,
        group_by_month=group_by_month,
        precip_names=precip_names,
    )
    for dt in data_types:
        ref = _open_stack(ref_files, dt, hist_period)
        mod = _open_stack(mod_files, dt, hist_period)
        mod_fut = _open_stack(mod_fut_files, dt, fut_period) if mod_fut_files else None
        qdm.fit(dt, ref, mod, mod_fut=mod_fut)
    qdm.save(out_path)
    return qdm


def apply_qdm_to_netcdf(in_path, out_path, qdm_path, data_types=None):
    """Apply a fitted QDM to all matching variables of a NetCDF file."""
    import xarray as xr

    qdm = load_qdm(qdm_path)
    ds = xr.open_dataset(in_path)
    out = ds.copy()
    for dt in data_types or qdm.data_types:
        if dt not in ds:
            continue
        da = ds[dt]
        if da.ndim != 3 or "time" not in da.dims:
            raise ValueError(
                "Variable '{}' must be (time, lat, lon) to apply QDM, got {}".format(
                    dt, da.dims
                )
            )
        time = da["time"].values if qdm.group_by_month else None
        out[dt] = (da.dims, qdm.apply(dt, da, time=time))
    history = out.attrs.get("history", "")
    out.attrs["history"] = (
        "Input bias-corrected with quantile delta mapping (QDM; {})\n{}".format(
            QDM_REFERENCE, history
        )
    )
    out.to_netcdf(out_path)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="quantile_mapping",
        description="Quantile delta mapping (QDM) bias correction of climate "
        "fields per grid cell (Cannon et al. 2015, doi:10.1175/JCLI-D-14-00754.1). "
        "Typical use: map driving data (e.g. MPI-GE) into the reference space "
        "(e.g. CERRA) a model was trained on.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fitp = sub.add_parser("fit", help="Fit QDM from NetCDF files and save a .npz")
    fitp.add_argument("--ref", required=True, help="Reference NetCDF file(s), comma separated (e.g. CERRA)")
    fitp.add_argument("--mod", required=True, help="Model/driving NetCDF file(s) to correct, comma separated (e.g. MPI-GE historical)")
    fitp.add_argument("--mod-fut", default=None, help="Driving NetCDF file(s) of the application period for the QDM delta (default: --mod, i.e. EQM)")
    fitp.add_argument("--vars", required=True, type=_str_list, help="Comma separated variable names")
    fitp.add_argument("--hist-period", default=None, help="Calibration period YYYY or YYYY-YYYY")
    fitp.add_argument("--fut-period", default=None, help="Application period YYYY or YYYY-YYYY (default: --hist-period)")
    fitp.add_argument("--n-quantiles", type=int, default=DEFAULT_N_QUANTILES)
    fitp.add_argument("--group-by-month", action="store_true", help="Fit separate transfer functions per calendar month")
    fitp.add_argument("--out", required=True, help="Output .npz file with the fitted transfer functions")

    app = sub.add_parser("apply", help="Apply a fitted QDM to a NetCDF file")
    app.add_argument("--in", dest="in_path", required=True, help="Input NetCDF file (e.g. MPI-GE)")
    app.add_argument("--out", required=True, help="Output NetCDF file with corrected fields")
    app.add_argument("--qdm", required=True, help="Fitted QDM .npz file (from 'fit')")
    app.add_argument("--vars", default=None, type=_str_list, help="Comma separated variables to correct (default: all in the .npz)")

    args = parser.parse_args(argv)
    if args.command == "fit":
        fit_qdm_from_netcdf(
            args.ref.split(","),
            args.mod.split(","),
            args.vars,
            args.out,
            mod_fut_files=args.mod_fut.split(",") if args.mod_fut else None,
            hist_period=args.hist_period,
            fut_period=args.fut_period,
            n_quantiles=args.n_quantiles,
            group_by_month=args.group_by_month,
        )
    else:
        apply_qdm_to_netcdf(args.in_path, args.out, args.qdm, data_types=args.vars)


if __name__ == "__main__":
    main()
