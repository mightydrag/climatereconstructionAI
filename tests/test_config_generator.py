import json
import os

import pytest

from climatereconstructionai.utils.config_generator import (
    CerraConfigWorkflow,
    eval_config_filename,
    scan_experiments,
    train_config_filename,
)

SPLITS = {"train": (2000, 2001), "val": (2002, 2002), "test": (2003, 2004)}

CONFIGS = {
    "t2m": {
        "target": "t2m",
        "input_vars": ["t2m", "relative_humidity"],
        "static_vars": ["orography"],
    },
    "tp_single": {"target": "tp", "input_vars": ["tp"], "static_vars": []},
}


def make_workflow(tmp_path):
    data_root = tmp_path / "cerra_data"
    ml_path = tmp_path / "repo"
    wf = CerraConfigWorkflow(
        data_root=str(data_root),
        ml_path=str(ml_path),
        reg="ger",
        other="remapcon_land",
        res="2deg",
        descriptor="_lowres_values_filled_highres_grid",
        static_year=2005,
        steady_mask_year=1985,
        splits=SPLITS,
        configs=CONFIGS,
    )
    return wf, data_root, ml_path


def create_all_files(wf):
    files = set()
    for key in wf.configs:
        target_var, input_vars, static_vars = wf.resolve_config(key)
        for split, (y0, y1) in wf.splits.items():
            for y in range(y0, y1 + 1):
                files.add(wf.nc_path("target", target_var, y))
                for v in input_vars:
                    files.add(wf.nc_path("input", v, y))
                    files.add(wf.nc_path("mask", v, y))
                    files.add(wf.nc_path("mask_test", v, y))
        for v in static_vars:
            files.add(wf.nc_path("static", v))
            files.add(wf.nc_path("mask_static", v))
        files.add(wf.steady_mask_file(target_var))
    for f in files:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "w").close()
    return files


def test_templates_and_paths(tmp_path):
    wf, data_root, _ = make_workflow(tmp_path)
    assert wf.nc_path("input", "t2m", 2000) == (
        f"{data_root}/t2m/cerra_dm_t2m_2000_ger_lowres_values_filled_highres_grid_2deg.nc"
    )
    assert wf.nc_path("target", "t2m", 2000) == f"{data_root}/t2m/cerra_dm_t2m_2000_remapcon_land_ger.nc"
    assert wf.nc_path("static", "orography") == f"{data_root}/orography/cerra_dm_orography_2005_remapcon_land_ger.nc"
    assert wf.nc_path("mask", "t2m", 2000) == f"{data_root}/mask/cerra_dm_t2m_2000_ger_filled_mask.nc"
    assert wf.steady_mask_file("t2m") == f"{data_root}/mask/cerra_dm_t2m_1985_ger_steady_mask.nc"
    tpl = wf.eval_tpl["data"]
    assert "{var}" in tpl and "{year}" in tpl and "_{reg}" in tpl
    assert str(data_root) in tpl
    assert "{data_root}" not in tpl


def test_check_files_and_json_lists(tmp_path):
    wf, _, ml_path = make_workflow(tmp_path)
    missing = wf.check_files("t2m", include_test=True)
    assert missing  # nothing created yet
    create_all_files(wf)
    assert wf.check_files("t2m", include_test=True) == []
    assert wf.check_all(include_test=True) == {}

    flags, steady_name, written = wf.write_json_lists("t2m")
    assert steady_name == "cerra_t2m_ger_remapcon_land_steady_mask.json"
    assert flags["train"][-1] == "t2m/cerra_t2m_ger_remapcon_land_target_train.json"
    assert flags["train"][0] == "t2m/cerra_t2m_ger_lowres_values_filled_highres_grid_train.json"
    assert flags["mask"][0] == "t2m/cerra_t2m_ger_remapcon_land_mask.json"

    with open(os.path.join(ml_path, "data/train/t2m/cerra_t2m_ger_lowres_values_filled_highres_grid_train.json")) as f:
        paths = json.load(f)
    assert len(paths) == 2
    assert paths[0].endswith("cerra_dm_t2m_2000_ger_lowres_values_filled_highres_grid_2deg.nc")

    with open(os.path.join(ml_path, "data/train/orography/cerra_orography_ger_train.json")) as f:
        static_paths = json.load(f)
    assert len(static_paths) == 2
    assert len(set(static_paths)) == 1

    with open(os.path.join(ml_path, "data/masks/t2m/cerra_t2m_ger_remapcon_land_mask.json")) as f:
        mask_paths = json.load(f)
    assert len(mask_paths) == 3  # train 2000-2001 + val 2002

    with open(os.path.join(ml_path, "data/test/masks/t2m/cerra_t2m_ger_remapcon_land_mask_test.json")) as f:
        test_mask_paths = json.load(f)
    assert len(test_mask_paths) == 2  # test 2003-2004

    steady_json = os.path.join(ml_path, "data/test/masks/t2m", steady_name)
    with open(steady_json) as f:
        assert json.load(f) == [wf.steady_mask_file("t2m")]
    assert len(written) > 0


def test_render_train_config(tmp_path):
    wf, _, _ = make_workflow(tmp_path)
    text = wf.render_train_config("t2m", "t2m", "my-run")
    assert "--config-name t2m\n" in text
    assert "--run-name my-run\n" in text
    assert "--data-root-dir ./data/\n" in text
    assert "--data-types 2t,2r,orog,2t\n" in text
    assert "--xrdss-paths masks/t2m\n" in text
    assert "--steady-masks t2m/cerra_t2m_ger_remapcon_land_steady_mask.json\n" in text
    data_names = [line for line in text.splitlines() if line.startswith("--data-names")][0]
    assert data_names.endswith("t2m/cerra_t2m_ger_remapcon_land_target_train.json")
    assert "--loss-criterion 0" in text
    text2 = wf.render_train_config("t2m", "t2m", "my-run", params={"loss-criterion": 6, "lr": "5e-5"})
    assert "--loss-criterion 6" in text2 and "--lr 5e-5" in text2


def test_render_eval_config(tmp_path):
    wf, data_root, _ = make_workflow(tmp_path)
    text = wf.render_eval_config("t2m", "t2m", "my-run")
    expected_data_tpl = (
        f"--data-file-template {data_root}/{{var}}/cerra_dm_{{var}}_{{year}}_{{reg}}"
        "_lowres_values_filled_highres_grid_2deg.nc"
    )
    assert expected_data_tpl in text
    expected_steady = f"--steady-mask-file {data_root}/mask/cerra_dm_{{var}}_1985_{{reg}}_steady_mask.nc"
    assert expected_steady in text
    assert "--region ger\n" in text
    assert "--static-vars orog\n" in text
    assert "--data-types 2t,2r,orog,2t\n" in text
    assert "--start-year 2003\n" in text
    assert "--end-year 2004\n" in text
    text_single = wf.render_eval_config("tp_single", "tp", "my-run")
    assert "--static-vars" not in text_single
    assert "--data-types tp,tp\n" in text_single


def test_write_experiment_uniqueness(tmp_path):
    wf, _, ml_path = make_workflow(tmp_path)
    create_all_files(wf)
    config_dir = tmp_path / "configs"

    r1 = wf.write_experiment("t2m", "exp1", config_dir=str(config_dir))
    assert r1["action"] == "written"
    assert r1["run_name"] == "exp1"
    assert os.path.isfile(r1["train_file"])
    assert os.path.isfile(r1["eval_file"])
    assert train_config_filename("t2m", "exp1") in os.listdir(config_dir)
    assert scan_experiments(str(config_dir)) == {("t2m", "exp1"): sorted(os.listdir(config_dir))}

    r2 = wf.write_experiment("t2m", "exp1", config_dir=str(config_dir))
    assert r2["action"] == "unchanged"
    assert r2["run_name"] == "exp1"
    assert sorted(os.listdir(config_dir)) == sorted([r1["train_file_name"], r1["eval_file_name"]])

    r3 = wf.write_experiment("t2m", "exp1", config_dir=str(config_dir), train_params={"lr": "1e-4"})
    assert r3["action"] == "suffixed"
    assert r3["run_name"] == "exp1-r2"
    assert eval_config_filename("t2m", "exp1-r2") in os.listdir(config_dir)

    with pytest.raises(FileExistsError):
        wf.write_experiment("t2m", "exp1", config_dir=str(config_dir),
                            train_params={"lr": "2e-4"}, on_duplicate="error")

    os.makedirs(os.path.join(ml_path, "snapshots", "t2m", "exp2"))
    r4 = wf.write_experiment("t2m", "exp2", config_dir=str(config_dir))
    assert r4["action"] == "suffixed"
    assert r4["run_name"] == "exp2-r2"


def test_run_all_missing_files(tmp_path):
    wf, _, _ = make_workflow(tmp_path)
    with pytest.raises(RuntimeError, match="missing"):
        wf.run_all({"t2m": {"run_name": "exp1"}}, config_dir=str(tmp_path / "configs"))
    assert not os.path.exists(tmp_path / "configs")

    create_all_files(wf)
    results = wf.run_all(
        {"t2m": {"run_name": "exp1"}, "tp_single": {"config_name": "tp", "run_name": "exp-tp"}},
        config_dir=str(tmp_path / "configs"),
    )
    assert [r["action"] for r in results] == ["written", "written"]
    assert results[1]["config_name"] == "tp"
    train_text = open(results[0]["train_file"]).read()
    assert "--run-name exp1" in train_text
