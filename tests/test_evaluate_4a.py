import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.evaluation


def _write_yearly_files(tmp_path, var="2t"):
    rng = np.random.default_rng(42)
    lat = np.linspace(46.0, 56.0, 21)
    lon = np.linspace(4.0, 16.0, 25)
    ref_dir = tmp_path / "ref"
    nn_dir = tmp_path / "nn"
    ref_dir.mkdir()
    nn_dir.mkdir()
    for year in (2016, 2017):
        time = np.arange(
            np.datetime64(f"{year}-01-01"),
            np.datetime64(f"{year}-01-01") + np.timedelta64(60, "D"),
        )
        base = 280.0 + 10.0 * np.sin(2 * np.pi * np.arange(60) / 60.0)
        field = base[:, None, None] + rng.normal(0.0, 1.0, (60, lat.size, lon.size))
        coords = {"time": time, "lat": lat, "lon": lon}
        ds = xr.Dataset({var: (("time", "lat", "lon"), field)}, coords=coords)
        ds.to_netcdf(ref_dir / f"cerra_dm_t2m_ger_{year}.nc")
        nn_field = field + 0.5 + rng.normal(0.0, 0.2, field.shape)
        ds_nn = xr.Dataset({var: (("time", "lat", "lon"), nn_field)}, coords=coords)
        ds_nn.to_netcdf(nn_dir / f"cerra_dm_t2m_ger_{year}_crai_output.nc")
    return ref_dir, nn_dir


def test_4a_analysis_manual_mode(tmp_path):
    from climatereconstructionai import evaluate_4a

    ref_dir, nn_dir = _write_yearly_files(tmp_path)
    out_dir = tmp_path / "out"
    evaluate_4a.main(
        [
            "--path-original", str(ref_dir),
            "--path-nn", str(nn_dir),
            "--var", "t2m",
            "--year-min", "2016",
            "--year-max", "2017",
            "--output-dir", str(out_dir),
        ]
    )
    stats = out_dir / "t2m_2016-2017_stats.txt"
    maps = out_dir / "t2m_2016-2017_maps.nc"
    assert stats.is_file()
    assert maps.is_file()
    text = stats.read_text()
    assert "Root Mean Squared Error" in text
    assert "Point statistics" in text
    with xr.open_dataset(maps) as ds:
        assert "bias" in ds
        assert "rmse" in ds
        assert "explained_variance" in ds
        assert "correlation" in ds


def test_4a_stats_flags_parsed(tmp_path):
    from climatereconstructionai import config as cfg

    arg_file = tmp_path / "eval.inp"
    arg_file.write_text(
        "--device cpu\n"
        "--dataset-name hadcrut-mod\n"
        f"--evaluation-dirs {tmp_path}/eval/\n"
        f"--log-dir {tmp_path}/logs/\n"
        f"--snapshot-dir {tmp_path}/snapshots/\n"
        "--run-4a-stats\n"
        f"--4a-output-dir {tmp_path}/4a\n"
        "--4a-deseasonalize\n"
    )
    cfg.set_evaluate_args(str(arg_file))
    assert cfg.run_4a_stats is True
    assert getattr(cfg, "4a_output_dir") == f"{tmp_path}/4a"
    assert getattr(cfg, "4a_deseasonalize") is True
