"""
Unit + integration tests for the friction stack.

Synthetic data isolates each mechanism; one integration test on real SPXUSD
data locks the accounting identity, cost monotonicity, and determinism.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from simulator.frictions import (
    _locate_fill, _tranche_fill, level1_costs, compute_fills,
    apply_frictions, FrictionConfig, BPS,
)
from simulator.reference_strategy import Trade
from simulator.data_loader import load_m1, resample_h1
from simulator import reference_strategy as strat


def make_m1(start, opens, highs, lows, closes, tickvols, spreads):
    idx = pd.date_range(start, periods=len(opens), freq="1min")
    n = len(opens)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "tickvol": tickvols, "vol": [0] * n,
        "spread_pts": spreads, "spread_price": spreads,
    }, index=idx)


# --- Level 2: sub-minute pricing -------------------------------------------
def test_locate_fill_interpolates_open_to_close():
    m1 = make_m1("2020-01-02 10:00", [100, 100], [106, 106], [100, 100],
                 [106, 106], [100, 100], [0.1, 0.1])
    mid, pos = _locate_fill(m1, m1.index[0] + pd.Timedelta(seconds=30))
    assert pos == 0
    assert mid == pytest.approx(103.0)  # 100 + 0.5 * (106 - 100)


def test_locate_fill_gap_uses_next_traded_bar():
    idx = pd.to_datetime(["2020-01-02 10:00", "2020-01-02 10:05"])
    m1 = pd.DataFrame({
        "open": [100, 200], "high": [100, 200], "low": [100, 200],
        "close": [100, 200], "tickvol": [100, 100], "vol": [0, 0],
        "spread_pts": [0.1, 0.1], "spread_price": [0.1, 0.1],
    }, index=idx)
    mid, pos = _locate_fill(m1, pd.Timestamp("2020-01-02 10:02:00"))
    assert pos == 1
    assert mid == pytest.approx(200.0)  # next available bar's open


def test_flat_path_has_no_slippage():
    # Flat price -> instantaneous fills equal the theoretical prices.
    n = 70
    m1 = make_m1("2020-01-02 11:00", [100] * n, [100] * n, [100] * n,
                 [100] * n, [100] * n, [0.1] * n)
    t = Trade("X", pd.Timestamp("2020-01-02 10:00"), "long", 100.0, 90.0,
              pd.Timestamp("2020-01-02 11:00"), 100.0, "signal")
    out = compute_fills(t, m1, lat_entry_s=0.0, lat_exit_s=0.0)
    assert out["entry_l2"] == pytest.approx(100.0)
    assert out["exit_l2"] == pytest.approx(100.0)


# --- Level 2: gap-aware stop fills -----------------------------------------
def test_stop_gap_through_fills_worse_than_stop():
    m1 = make_m1("2020-01-02 11:00",
                 opens=[100, 100, 100, 100, 100, 98],
                 highs=[100, 100, 100, 100, 100, 98],
                 lows=[100, 100, 100, 100, 100, 98],
                 closes=[100, 100, 100, 100, 100, 98],
                 tickvols=[100] * 6, spreads=[0.1] * 6)
    t = Trade("X", pd.Timestamp("2020-01-02 10:00"), "long", 100.0, 99.0,
              pd.Timestamp("2020-01-02 11:00"), 99.0, "stop")
    out = compute_fills(t, m1, lat_entry_s=0.0, lat_exit_s=0.0)
    assert out["exit_l2"] == pytest.approx(98.0)  # gapped below the stop


def test_stop_never_fills_better_than_stop():
    m1 = make_m1("2020-01-02 11:00",
                 opens=[100, 100, 100, 100, 100, 99.5],
                 highs=[100, 100, 100, 100, 100, 99.5],
                 lows=[100, 100, 100, 100, 100, 99.0],   # touches stop intrabar
                 closes=[100, 100, 100, 100, 100, 99.5],
                 tickvols=[100] * 6, spreads=[0.1] * 6)
    t = Trade("X", pd.Timestamp("2020-01-02 10:00"), "long", 100.0, 99.0,
              pd.Timestamp("2020-01-02 11:00"), 99.0, "stop")
    out = compute_fills(t, m1, lat_entry_s=0.0, lat_exit_s=0.0)
    assert out["exit_l2"] == pytest.approx(99.0)  # clamped to the stop


# --- Level 3: tranched partial fills ---------------------------------------
def test_tranche_thick_liquidity_equals_l2():
    m1 = make_m1("2020-01-02 10:00", [100, 110], [100, 110], [100, 110],
                 [100, 110], [100, 100], [0.1, 0.1])
    v = _tranche_fill(m1, 0, first_price=100.0, median_tickvol=100, max_minutes=10)
    assert v == pytest.approx(100.0)  # clears fully in minute 0


def test_tranche_thin_liquidity_blends_later_minutes():
    m1 = make_m1("2020-01-02 10:00", [100, 110], [100, 110], [100, 110],
                 [100, 110], [50, 100], [0.1, 0.1])
    v = _tranche_fill(m1, 0, first_price=100.0, median_tickvol=100, max_minutes=10)
    # 0.5 @ 100 (minute 0) + 0.5 @ mid(110,110)=110 -> 105
    assert v == pytest.approx(105.0)


# --- Level 1: spread + commission ------------------------------------------
def test_level1_costs():
    entry = pd.Timestamp("2020-01-02 10:00")
    exit_ = pd.Timestamp("2020-01-02 15:00")
    h1 = pd.DataFrame({"spread_price": [0.2, 0.4]}, index=[entry, exit_])
    t = Trade("X", entry, "long", 100.0, 98.0, exit_, 110.0, "signal")
    sc, cc = level1_costs(t, h1, FrictionConfig(commission_bps_per_side=0.5))
    assert sc == pytest.approx(0.3)                  # 0.2/2 + 0.4/2
    assert cc == pytest.approx(0.5 * BPS * (100 + 110))


# --- Integration on real data ----------------------------------------------
@pytest.fixture(scope="module")
def spx():
    m1 = load_m1("SPXUSD")
    h1 = resample_h1(m1)
    trades = strat.run(h1, "SPXUSD")
    return m1, h1, trades


def test_accounting_identity_and_monotonicity(spx):
    m1, h1, trades = spx
    df = apply_frictions(trades, h1, level=3, m1=m1)

    # cost decomposition + net-PnL identity
    assert np.allclose(
        df["total_cost"],
        df["spread_cost"] + df["commission_cost"] + df["slippage_cost"] + df["partial_cost"],
    )
    assert np.allclose(df["net_pnl"], df["level0_pnl"] - df["total_cost"])

    # one-sided / non-negative costs
    assert (df["spread_cost"] >= 0).all()
    assert (df["commission_cost"] >= 0).all()
    assert (df["partial_cost"] >= -1e-9).all()  # L3 charged one-sided

    # per-trade monotonicity where it must hold (L2 slippage may be favorable)
    l1 = df["level0_pnl"] - df["spread_cost"] - df["commission_cost"]
    l2 = l1 - df["slippage_cost"]
    l3 = l2 - df["partial_cost"]
    assert (l1 <= df["level0_pnl"] + 1e-9).all()  # spread+commission never help
    assert (l3 <= l2 + 1e-9).all()                # partial fills never help


def test_symmetric_partial_allows_favorable(spx):
    m1, h1, trades = spx
    one = apply_frictions(trades, h1, level=3,
                          config=FrictionConfig(partial_one_sided=True), m1=m1)
    sym = apply_frictions(trades, h1, level=3,
                          config=FrictionConfig(partial_one_sided=False), m1=m1)
    # one-sided is a non-negative penalty; symmetric admits favorable tranches
    assert (one["partial_cost"] >= -1e-9).all()
    assert (sym["partial_cost"] < -1e-9).any()
    # symmetric total cost is never higher (it can only give some cost back)
    assert sym["total_cost"].sum() <= one["total_cost"].sum() + 1e-6


def test_determinism_under_seed(spx):
    m1, h1, trades = spx
    a = apply_frictions(trades, h1, level=3, m1=m1)["total_cost"].sum()
    b = apply_frictions(trades, h1, level=3, m1=m1)["total_cost"].sum()
    assert a == b


def test_level2_requires_m1(spx):
    _, h1, trades = spx
    with pytest.raises(ValueError):
        apply_frictions(trades, h1, level=2, m1=None)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
