"""
Tests for the Part B3 optimisation: local stop execution.

The optimisation drops the REPLICATION delay for STOP exits only (a stop level
is known in advance, so no signal has to travel master -> follower first), but
keeps the follower's own physical order-to-fill execution latency -- drawn
independently from the SAME lognormal distribution the master's own exit
latency uses. Signal/eod exits keep their normal replication delay. All fill
pricing (gap handling, stop clamp) is reused unchanged from frictions.py.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from simulator.frictions import FrictionConfig, ONE_HOUR, compute_fills, _find_stop_breach
from simulator.reference_strategy import Trade
from simulator.data_loader import load_m1, resample_h1, TICK_SIZE
from simulator import reference_strategy as strat
from replication import delay_models as dm
from replication.followers import (
    run_master, _price_follower, _follower_latency, ReplicationConfig, SIGNAL_SIDE,
)
from optimisations.local_stop_execution import (
    stop_exit_mask, localise_exit_latency, price_local_stop_follower, run_local_stop,
    draw_follower_stop_latency,
)


# ---------------------------------------------------------------------------
# Synthetic helpers (controlled fill-time semantics)
# ---------------------------------------------------------------------------
def _stub(reason: str) -> Trade:
    return Trade(symbol="X", entry_time=pd.Timestamp("2021-01-01 00:00"),
                 direction="long", signal_price=100.0, stop_price=95.0,
                 exit_time=pd.Timestamp("2021-01-01 03:00"), exit_price=99.0,
                 exit_reason=reason)


def _synth_stop_scenario(direction: str = "long"):
    """A single trade whose stop is gapped through at a known breach minute
    (03:10). Flat 100 everywhere else. Returns (trade, h1, m1, breach_minute)."""
    base = pd.Timestamp("2021-03-01 00:00")
    entry_time = base
    exit_time = base + pd.Timedelta(hours=3)
    # as_unit("ns"): pd.date_range on exact-minute timestamps infers datetime64[us]
    # by default, but real M1 data (load_m1) is datetime64[ns]. Fractional-second
    # latency draws (nanosecond precision) can't losslessly downcast into a [us]
    # index, so force [ns] here to match production resolution.
    idx = pd.date_range(base + ONE_HOUR, base + pd.Timedelta(hours=5), freq="min").as_unit("ns")
    m1 = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                       "tickvol": 100.0}, index=idx)
    breach_min = base + pd.Timedelta(hours=3, minutes=10)
    if direction == "long":
        stop = 95.0
        m1.loc[breach_min, ["open", "high", "low", "close"]] = [94.0, 100.0, 94.0, 100.0]
    else:
        stop = 105.0
        m1.loc[breach_min, ["open", "high", "low", "close"]] = [106.0, 106.0, 100.0, 100.0]
    trade = Trade(symbol="TEST", entry_time=entry_time, direction=direction,
                  signal_price=100.0, stop_price=stop, exit_time=exit_time,
                  exit_price=stop, exit_reason="stop")
    h1 = pd.DataFrame({"spread_price": [0.0, 0.0]}, index=[entry_time, exit_time])
    return trade, h1, m1, breach_min


# ---------------------------------------------------------------------------
# 8. Stop exit: exit-fill time == breach + L_own (fix; supersedes old test 1)
#
# Change vs. the old test 1 (`test_stop_exit_fill_time_is_breach_no_dexit`):
# that test forced d_exit = 0.0 and asserted exit_fill_time == breach exactly.
# The fix replaces that 0.0 with an independent lognormal draw
# (`draw_follower_stop_latency`, same distribution as the master's own exit
# latency), so exit_fill_time is now `breach + L_own` with L_own > 0 almost
# surely -- the fill is no longer exactly at the breach minute.
# ---------------------------------------------------------------------------
def test_stop_exit_fill_time_uses_own_latency_not_dexit():
    trade, h1, m1, breach_min = _synth_stop_scenario("long")
    fcfg = FrictionConfig()
    breach = _find_stop_breach(
        m1, trade.exit_time, trade.direction, trade.stop_price,
        not_before=trade.entry_time + ONE_HOUR + pd.Timedelta(seconds=0.5),
    )
    assert breach == breach_min

    l_own = float(draw_follower_stop_latency(1, fcfg)[0])
    assert l_own > 0.0   # lognormal draw: essentially always strictly positive

    out = compute_fills(trade, m1, lat_entry_s=0.5, lat_exit_s=l_own)
    assert out["exit_fill_time"] == breach + pd.Timedelta(seconds=l_own)
    assert out["exit_fill_time"] != breach   # no longer fills exactly AT the breach

    # Vast-majority check across a larger batch of (synthetic) stop trades.
    batch = draw_follower_stop_latency(200, fcfg)
    assert (batch > 0).mean() > 0.99


# ---------------------------------------------------------------------------
# Signal exit: exit-fill time == exit_time + 1h + d_exit (unchanged)
# ---------------------------------------------------------------------------
def test_signal_exit_fill_time_keeps_dexit():
    trade, h1, m1, _ = _synth_stop_scenario("long")
    sig = Trade(symbol="TEST", entry_time=trade.entry_time, direction="long",
                signal_price=100.0, stop_price=trade.stop_price,
                exit_time=trade.exit_time, exit_price=100.0, exit_reason="signal")

    out = compute_fills(sig, m1, lat_entry_s=0.5, lat_exit_s=7.0)
    assert out["exit_fill_time"] == sig.exit_time + ONE_HOUR + pd.Timedelta(seconds=7.0)

    # the localiser must NOT touch a non-stop exit delay, regardless of stop_latency
    mask = stop_exit_mask([sig])
    assert not mask[0]
    stop_latency = np.array([999.0])   # deliberately absurd: must be ignored (not a stop)
    assert localise_exit_latency(np.array([7.0]), mask, stop_latency)[0] == 7.0


def test_localise_uses_stop_latency_only_for_stops():
    trades = [_stub("stop"), _stub("signal"), _stub("stop"), _stub("eod")]
    mask = stop_exit_mask(trades)
    assert list(mask) == [True, False, True, False]
    d = np.array([10.0, 20.0, 30.0, 40.0])
    stop_latency = np.array([1.5, 2.5, 3.5, 4.5])
    out = localise_exit_latency(d, mask, stop_latency)
    # stops (idx 0, 2) take their own latency draw; signal/eod (idx 1, 3) keep d_exit
    assert list(out) == [1.5, 20.0, 3.5, 40.0]


# ---------------------------------------------------------------------------
# Real-data fixtures and a one-follower helper
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spx():
    m1 = load_m1("SPXUSD")
    h1 = resample_h1(m1)
    trades = strat.run(h1, "SPXUSD")
    return m1, h1, trades


def _follower_latencies(trades, rcfg, median_s, lat_e, lat_x, f=0):
    z_e, z_x = dm.standard_normals(rcfg.n_followers, len(trades), rcfg.seed)
    d_e, d_x = dm.follower_delays(median_s, z_e, z_x, rcfg.sigma)
    return _follower_latency(SIGNAL_SIDE, d_e[f], d_x[f], lat_e, lat_x)


# ---------------------------------------------------------------------------
# 3. Stop level is the master's, not recomputed from the follower's entry
# ---------------------------------------------------------------------------
def test_stop_level_is_masters_not_recomputed(spx):
    m1, h1, trades = spx
    fcfg = FrictionConfig(); rcfg = ReplicationConfig(n_followers=6)
    _, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    is_stop = stop_exit_mask(trades)
    # Two very different entry delays must clamp stop fills to the SAME master
    # stop_price -- proof the follower never derives its own stop from its entry.
    for median_s in (3.0, 600.0):
        fe, fx = _follower_latencies(trades, rcfg, median_s, lat_e, lat_x)
        loc = price_local_stop_follower(trades, h1, m1, fcfg, fe, fx, is_stop)
        for t, fill in zip(trades, loc["exit_fill"].to_numpy()):
            if t.exit_reason != "stop":
                continue
            if t.direction == "long":
                assert fill <= t.stop_price + 1e-9
            else:
                assert fill >= t.stop_price - 1e-9


# ---------------------------------------------------------------------------
# 4. Entry side (and non-stop exits) unchanged vs the baseline follower
# ---------------------------------------------------------------------------
def test_entry_side_unchanged(spx):
    m1, h1, trades = spx
    fcfg = FrictionConfig(); rcfg = ReplicationConfig(n_followers=4)
    _, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    fe, fx = _follower_latencies(trades, rcfg, 120.0, lat_e, lat_x)
    is_stop = stop_exit_mask(trades)

    base = _price_follower(trades, h1, m1, fcfg, fe, fx)
    loc = price_local_stop_follower(trades, h1, m1, fcfg, fe, fx, is_stop)

    assert (base["entry_fill_time"].to_numpy() == loc["entry_fill_time"].to_numpy()).all()
    assert np.allclose(base["entry_fill"].to_numpy(), loc["entry_fill"].to_numpy())
    # non-stop exits keep their normal delay -> identical to the baseline
    ns = ~is_stop
    assert (base.loc[ns, "exit_fill_time"].to_numpy()
            == loc.loc[ns, "exit_fill_time"].to_numpy()).all()


# ---------------------------------------------------------------------------
# 5. Every exit closes the full position (one populated exit fill per trade)
# ---------------------------------------------------------------------------
def test_all_exits_close_full_position(spx):
    m1, h1, trades = spx
    fcfg = FrictionConfig(); rcfg = ReplicationConfig(n_followers=3)
    _, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    fe, fx = _follower_latencies(trades, rcfg, 120.0, lat_e, lat_x)
    loc = price_local_stop_follower(trades, h1, m1, fcfg, fe, fx, stop_exit_mask(trades))
    assert len(loc) == len(trades)
    assert loc["exit_fill_time"].notna().all()
    assert loc["exit_fill"].notna().all()


# ---------------------------------------------------------------------------
# 6. Sanity: local stop does not worsen mean decay on most assets at 120s
# ---------------------------------------------------------------------------
def test_decay_not_worse_on_most_assets_at_120s():
    fcfg = FrictionConfig()
    rcfg = ReplicationConfig(n_followers=8, delay_grid_s=(120.0,))
    improved = 0
    for sym in TICK_SIZE:
        m1 = load_m1(sym); h1 = resample_h1(m1)
        trades = strat.run(h1, symbol=sym)[:120]
        out = run_local_stop(trades, h1, m1, fcfg, rcfg)
        row = out.iloc[0]
        if row["mean_decay_after_bps"] <= row["mean_decay_before_bps"] + 1e-9:
            improved += 1
    assert improved >= 3   # sanity, not a hard requirement


# ---------------------------------------------------------------------------
# 7. Follower stop fill is never better than the stop level (same clamp as
# master). Unchanged in structure vs. the old test 7: `_clamp` in frictions.py
# depends only on the fill price located at exit_fill_time, never on the
# latency value that produced that time, so the min/max-vs-stop_price
# guarantee holds identically whether exit_fill_time is breach + 0 (old) or
# breach + L_own (fixed). Kept as a regression test for that guarantee.
# ---------------------------------------------------------------------------
def test_follower_stop_fill_never_better_than_stop(spx):
    m1, h1, trades = spx
    fcfg = FrictionConfig(); rcfg = ReplicationConfig(n_followers=8)
    _, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    is_stop = stop_exit_mask(trades)
    assert is_stop.any(), "need at least one stop exit to exercise the clamp"

    z_e, z_x = dm.standard_normals(rcfg.n_followers, len(trades), rcfg.seed)
    d_e, d_x = dm.follower_delays(120.0, z_e, z_x, rcfg.sigma)
    for f in range(rcfg.n_followers):
        fe, fx = _follower_latency(SIGNAL_SIDE, d_e[f], d_x[f], lat_e, lat_x)
        exit_fill = price_local_stop_follower(
            trades, h1, m1, fcfg, fe, fx, is_stop)["exit_fill"].to_numpy()
        for i, t in enumerate(trades):
            if t.exit_reason != "stop":
                continue
            if t.direction == "long":
                assert exit_fill[i] <= t.stop_price + 1e-9
            else:
                assert exit_fill[i] >= t.stop_price - 1e-9


# ---------------------------------------------------------------------------
# 9. The follower's own stop latency is independent of the master's lat_x
# ---------------------------------------------------------------------------
def test_stop_latency_independent_of_master_lat_x(spx):
    m1, h1, trades = spx
    fcfg = FrictionConfig()
    _, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    stop_latency = draw_follower_stop_latency(len(trades), fcfg)

    assert not np.array_equal(stop_latency, lat_x)
    # sample check: not even elementwise-close on a prefix slice
    sample = min(50, len(trades))
    assert not np.allclose(stop_latency[:sample], lat_x[:sample])


# ---------------------------------------------------------------------------
# 10. Reproducibility: the same seed reproduces the same draw / same fills
# ---------------------------------------------------------------------------
def test_stop_latency_reproducible_across_calls(spx):
    m1, h1, trades = spx
    fcfg = FrictionConfig(); rcfg = ReplicationConfig(n_followers=3)

    l1 = draw_follower_stop_latency(len(trades), fcfg)
    l2 = draw_follower_stop_latency(len(trades), fcfg)
    assert np.array_equal(l1, l2)

    _, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    fe, fx = _follower_latencies(trades, rcfg, 120.0, lat_e, lat_x)
    is_stop = stop_exit_mask(trades)

    a = price_local_stop_follower(trades, h1, m1, fcfg, fe, fx, is_stop)
    b = price_local_stop_follower(trades, h1, m1, fcfg, fe, fx, is_stop)
    assert (a["exit_fill_time"].to_numpy() == b["exit_fill_time"].to_numpy()).all()
    assert np.allclose(a["exit_fill"].to_numpy(), b["exit_fill"].to_numpy())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
