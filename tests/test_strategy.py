"""Tests for the reference strategy: no look-ahead and correct entry triggering."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from simulator.reference_strategy import _add_indicators, run


def make_h1(highs, lows, closes, start="2020-01-01 00:00"):
    n = len(highs)
    idx = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({
        "open": closes, "high": highs, "low": lows, "close": closes,
        "tickvol": [100] * n, "vol": [0] * n,
        "spread_pts": [1] * n, "spread_price": [0.01] * n,
    }, index=idx)


def test_channel_excludes_current_bar():
    """chan_high at bar i must be the max of the PREVIOUS 20 bars, not incl. i."""
    highs = list(range(1, 41))            # strictly increasing -> easy to check
    lows = [h - 0.5 for h in highs]
    h1 = make_h1(highs, lows, closes=highs)
    ind = _add_indicators(h1)
    i = 25
    assert ind["chan_high"].iloc[i] == pytest.approx(max(highs[i - 20:i]))
    assert ind["chan_high"].iloc[i] != pytest.approx(highs[i])  # would be look-ahead


def test_breakout_triggers_long_entry():
    highs = [100] * 22
    lows = [99] * 22
    closes = [100] * 22
    closes[21] = 105            # close breaks above the 20-bar channel high (100)
    highs[21] = 105
    trades = run(make_h1(highs, lows, closes), "X")
    assert any(t.direction == "long" for t in trades)


def test_no_entry_without_breakout():
    n = 30
    trades = run(make_h1([100] * n, [99] * n, [100] * n), "X")
    assert trades == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
