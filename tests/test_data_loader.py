"""Tests for the M1 -> H1 resampling aggregation."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from simulator.data_loader import resample_h1


def test_resample_ohlc_aggregation():
    idx = pd.date_range("2020-01-02 10:00", periods=120, freq="1min")
    m1 = pd.DataFrame({
        "open": list(range(120)),
        "high": [i + 1 for i in range(120)],
        "low": [i - 1 for i in range(120)],
        "close": list(range(120)),
        "tickvol": [1] * 120,
        "vol": [0] * 120,
        "spread_pts": [2.0] * 120,
        "spread_price": [0.02] * 120,
    }, index=idx)

    h1 = resample_h1(m1)
    assert len(h1) == 2

    first = h1.iloc[0]                 # first hour = minutes 0..59
    assert first["open"] == 0          # first
    assert first["close"] == 59        # last
    assert first["high"] == 60         # max high = 59 + 1
    assert first["low"] == -1          # min low = 0 - 1
    assert first["tickvol"] == 60      # summed
    assert first["spread_price"] == pytest.approx(0.02)  # averaged


def test_resample_drops_empty_hours():
    # two bars three hours apart -> only the two populated hours survive
    idx = pd.to_datetime(["2020-01-02 10:00", "2020-01-02 13:30"])
    m1 = pd.DataFrame({
        "open": [100, 200], "high": [101, 201], "low": [99, 199],
        "close": [100, 200], "tickvol": [1, 1], "vol": [0, 0],
        "spread_pts": [1.0, 1.0], "spread_price": [0.01, 0.01],
    }, index=idx)
    h1 = resample_h1(m1)
    assert len(h1) == 2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
