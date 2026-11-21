import numpy as np
import pytest

from climatereconstructionai.quantile_mapping import QuantileDeltaMapping, load_qdm


def _synthetic(n_time=4000, n_lat=4, n_lon=3, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-01") + np.timedelta64(n_time, "D"),
    )
    shape = (n_time, n_lat, n_lon)
    ref_pr = rng.gamma(2.0, 2.0, shape)
    mod_pr_hist = ref_pr * 1.3 + 0.4
    mod_pr_fut = mod_pr_hist * 1.2
    ref_tas = rng.normal(10.0, 5.0, shape)
    mod_tas_hist = ref_tas + 2.0
    mod_tas_fut = mod_tas_hist + 3.0
    return time, ref_pr, mod_pr_hist, mod_pr_fut, ref_tas, mod_tas_hist, mod_tas_fut


def _median_per_cell(x):
    return np.quantile(x, 0.5, axis=0)


def test_eqm_recovers_reference_distribution():
    _, ref, mod_hist, _, _, _, _ = _synthetic()
    qdm = QuantileDeltaMapping(["pr"])
    qdm.fit("pr", ref, mod_hist)  # mod_fut=None -> EQM
    corrected = qdm.apply("pr", mod_hist)
    assert corrected.shape == mod_hist.shape
    np.testing.assert_allclose(
        _median_per_cell(corrected), _median_per_cell(ref), rtol=0.05
    )
    q90_corr = np.quantile(corrected, 0.9, axis=0)
    q90_ref = np.quantile(ref, 0.9, axis=0)
    np.testing.assert_allclose(q90_corr, q90_ref, rtol=0.15)


def test_qdm_preserves_multiplicative_delta():
    _, ref, mod_hist, mod_fut, _, _, _ = _synthetic()
    qdm = QuantileDeltaMapping(["pr"])
    qdm.fit("pr", ref, mod_hist, mod_fut=mod_fut)
    # Feed the model's own historical quantiles as values: tau is recovered
    # exactly, so the correction must equal q_ref(tau) * q_fut(tau)/q_mod(tau).
    # With mod_fut = 1.2 * mod_hist, q_fut(tau) = 1.2 * q_mod(tau) exactly.
    n_lat, n_lon = ref.shape[1], ref.shape[2]
    x = qdm.q_mod["pr"][0].T.reshape(-1, n_lat, n_lon)
    corrected = qdm.apply("pr", x)
    expected = 1.2 * qdm.q_ref["pr"][0].T.reshape(-1, n_lat, n_lon)
    np.testing.assert_allclose(corrected, expected, rtol=1e-6, atol=1e-8)


def test_qdm_additive_variable():
    _, _, _, _, ref_tas, mod_tas_hist, mod_tas_fut = _synthetic()
    qdm = QuantileDeltaMapping(["tas"], ratio={"tas": False})
    qdm.fit("tas", ref_tas, mod_tas_hist, mod_fut=mod_tas_fut)
    # With mod_fut = mod_hist + 3, the correction must be q_ref(tau) + 3.
    n_lat, n_lon = ref_tas.shape[1], ref_tas.shape[2]
    x = qdm.q_mod["tas"][0].T.reshape(-1, n_lat, n_lon)
    corrected = qdm.apply("tas", x)
    expected = qdm.q_ref["tas"][0].T.reshape(-1, n_lat, n_lon) + 3.0
    np.testing.assert_allclose(corrected, expected, rtol=1e-6, atol=1e-8)


def test_nan_and_shape_preserved():
    _, ref, mod_hist, _, _, _, _ = _synthetic()
    mod_hist = mod_hist.copy()
    mod_hist[::7, 0, 0] = np.nan
    qdm = QuantileDeltaMapping(["pr"])
    qdm.fit("pr", ref, mod_hist)
    corrected = qdm.apply("pr", mod_hist)
    assert corrected.shape == mod_hist.shape
    assert np.all(np.isnan(corrected[::7, 0, 0]))
    assert np.all(~np.isnan(corrected[1::7, 0, 0]))


def test_monthly_grouping():
    time, ref, mod_hist, _, _, _, _ = _synthetic()
    qdm = QuantileDeltaMapping(["pr"], group_by_month=True)
    qdm.fit("pr", ref, mod_hist, time=time)
    corrected = qdm.apply("pr", mod_hist, time=time)
    assert corrected.shape == mod_hist.shape
    np.testing.assert_allclose(
        _median_per_cell(corrected), _median_per_cell(ref), rtol=0.1
    )


def test_save_load_roundtrip(tmp_path):
    _, ref, mod_hist, mod_fut, _, _, _ = _synthetic()
    qdm = QuantileDeltaMapping(["pr"])
    qdm.fit("pr", ref, mod_hist, mod_fut=mod_fut)
    path = str(tmp_path / "qdm.npz")
    qdm.save(path)
    qdm2 = load_qdm(path)
    np.testing.assert_allclose(
        qdm.apply("pr", mod_fut), qdm2.apply("pr", mod_fut), rtol=1e-12
    )


def test_grid_mismatch_raises():
    _, ref, mod_hist, _, _, _, _ = _synthetic()
    qdm = QuantileDeltaMapping(["pr"])
    qdm.fit("pr", ref, mod_hist)
    with pytest.raises(ValueError):
        qdm.apply("pr", mod_hist[:, :2, :])
