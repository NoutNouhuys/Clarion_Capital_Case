"""
Tests for Part B3 optimisation 2: volatility-normalised latency sizing.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from optimisations.vol_norm_latency_sizing import (
    vol_entry_size, vol_position_sizes, recent_vol_bps, sigma_at_times,
    calibrate_r_star, edge_decomposition,
)

R = 300.0  # a test R_star (bps * s)


def test_vol_entry_size_monotone_in_delay_and_sigma():
    d = np.array([1.0, 10.0, 100.0, 1000.0])
    assert np.all(np.diff(vol_entry_size(d, sigma_t=20.0, r_star=R)) <= 0)
    s = np.array([1.0, 10.0, 100.0, 1000.0])
    assert np.all(np.diff(vol_entry_size(50.0, sigma_t=s, r_star=R)) <= 0)


def test_vol_entry_size_invariant_when_capped():
    # When q < 1 the invariant q * d * sigma == R_star holds exactly.
    d = np.array([50.0, 100.0, 400.0]); s = np.array([20.0, 30.0, 50.0])
    q = vol_entry_size(d, s, R)
    assert np.all(q < 1.0)
    assert np.allclose(q * d * s, R, atol=1e-8)


def test_vol_entry_size_full_when_budget_not_binding():
    d = np.array([1.0, 5.0]); s = np.array([2.0, 3.0])   # d*s well below R
    assert np.allclose(vol_entry_size(d, s, R), 1.0)


def test_vol_entry_size_rejects_nonpositive_and_nonfinite():
    with pytest.raises(AssertionError):
        vol_entry_size(np.array([1.0, 0.0]), np.array([5.0, 5.0]), R)
    with pytest.raises(AssertionError):
        vol_entry_size(np.array([1.0, 2.0]), np.array([5.0, np.nan]), R)


def test_vol_exit_closes_full_open_position():
    for d in [0.3, 30.0, 600.0]:
        for s in [5.0, 50.0]:
            o, c = vol_position_sizes(d, s, R)
            assert c == o


def test_edge_decomposition_sanity_with_vol_sizing():
    rng = np.random.default_rng(1)
    pnl = rng.normal(size=150) * 4.0
    q = vol_entry_size(rng.uniform(10.0, 800.0, 150),
                       rng.uniform(5.0, 60.0, 150), R)
    fp, av, net = edge_decomposition(q, pnl)
    assert net == pytest.approx(fp - av)


def _synthetic_m1(n=200, seed=2):
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 1e-3, size=n))
    return pd.DataFrame({"close": close}, index=idx)


def test_sigma_at_times_is_look_ahead_free():
    # sigma at a signal time must use only bars strictly before it: it equals the
    # shifted rolling std at the previous bar and differs from the look-ahead
    # version that includes the signal bar.
    m1 = _synthetic_m1()
    vol = recent_vol_bps(m1, window=60)
    t_signal = m1.index[150]
    sigma = sigma_at_times(np.array([t_signal], dtype="datetime64[ns]"), m1, vol)[0]

    assert sigma == pytest.approx(vol.to_numpy()[149])           # bar strictly < t
    ret_bps = m1["close"].pct_change() * 1e4
    lookahead = ret_bps.rolling(60).std().to_numpy()[150]        # includes bar 150
    assert not np.isclose(sigma, lookahead)


def test_calibrate_r_star_positive_and_scales():
    vol = recent_vol_bps(_synthetic_m1(), window=60)
    r = calibrate_r_star(vol)
    assert r > 0.0
    assert r == pytest.approx(30.0 * np.median(vol.dropna()))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
