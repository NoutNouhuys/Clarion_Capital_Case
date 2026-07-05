"""
Tests for Part C: the pre-trade risk gate (gate/latency_model.py,
gate/policies.py, gate/evaluate.py).
"""

import sys
import pathlib
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from simulator.frictions import FrictionConfig, ONE_HOUR
from simulator.reference_strategy import Trade
from replication.followers import run_master, net_bps_per_trade

from gate.latency_model import (
    draw_gate_z, gate_latency_ms, draw_uninformed_reject_flags,
    GATE_MEDIAN_GRID_MS,
)
from gate.policies import (
    resolve_trade_outcome, informed_reject_flags, POLICY_H, POLICY_S,
)
from gate.evaluate import (
    baseline_counterfactual, price_gated, evaluate_configuration,
    check_covid_warmup, cluster_stress_days,
)


# ---------------------------------------------------------------------------
# Synthetic pricing fixture: one long trade whose entry fill sits inside a
# linearly-trending first minute bar, so a millisecond-scale entry-latency
# shift (gate g_i / tau) maps to a distinct, deterministic fill price. The
# exit sits in a flat region so it is insensitive to the (randomly drawn)
# master exit latency.
# ---------------------------------------------------------------------------
def _trend_scenario():
    entry_time = pd.Timestamp("2021-06-01 00:00")
    exit_time = entry_time + ONE_HOUR
    trade = Trade(symbol="TEST", entry_time=entry_time, direction="long",
                  signal_price=100.0, stop_price=50.0,
                  exit_time=exit_time, exit_price=110.0, exit_reason="signal")
    idx = pd.date_range(entry_time + ONE_HOUR, entry_time + pd.Timedelta(hours=4),
                        freq="min").as_unit("ns")
    m1 = pd.DataFrame({"open": 100.6, "high": 100.6, "low": 100.6, "close": 100.6,
                       "tickvol": 100.0}, index=idx)
    entry_bar = entry_time + ONE_HOUR
    m1.loc[entry_bar, ["open", "high", "low", "close"]] = [100.0, 100.6, 100.0, 100.6]
    h1 = pd.DataFrame({"spread_price": [0.0, 0.0]}, index=[entry_time, exit_time])
    return [trade], h1, m1


# ---------------------------------------------------------------------------
# 1. g_i(m) = m * exp(sigma * z_i): empirical median matches the regime
# ---------------------------------------------------------------------------
def test_gate_latency_empirical_median_matches_regime():
    z = draw_gate_z(200_000)
    for median_ms in GATE_MEDIAN_GRID_MS:
        g = gate_latency_ms(z, median_ms)
        assert abs(np.median(g) - median_ms) / median_ms < 0.02


# ---------------------------------------------------------------------------
# 2. CRN: z_i, g_i(30), g_i(150), r_i are identical regardless of tau/policy
# ---------------------------------------------------------------------------
def test_crn_z_g_r_reproducible_and_independent_of_tau_policy():
    n = 50
    z1 = draw_gate_z(n)
    z2 = draw_gate_z(n)
    assert np.array_equal(z1, z2)

    g30_a = gate_latency_ms(z1, 30.0)
    g30_b = gate_latency_ms(z2, 30.0)
    assert np.array_equal(g30_a, g30_b)
    g150 = gate_latency_ms(z1, 150.0)
    assert not np.array_equal(g30_a, g150)   # different regime -> different g_i

    r1 = draw_uninformed_reject_flags(n)
    r2 = draw_uninformed_reject_flags(n)
    assert np.array_equal(r1, r2)

    # g_i/r_i are computed once and simply fed into resolve_trade_outcome --
    # evaluating under different (tau, policy) never re-derives them.
    for tau_ms in (10.0, 100.0):
        for policy in (POLICY_H, POLICY_S):
            resolve_trade_outcome(g30_a, tau_ms, policy, r1)
            assert np.array_equal(g30_a, gate_latency_ms(z1, 30.0))
            assert np.array_equal(r1, draw_uninformed_reject_flags(n))


# ---------------------------------------------------------------------------
# 3. Policy H timeout: no fill, no PnL contribution
# ---------------------------------------------------------------------------
def test_policy_h_timeout_no_fill_no_pnl():
    g_ms = np.array([500.0])
    outcome = resolve_trade_outcome(g_ms, tau_ms=50.0, policy=POLICY_H,
                                    reject_flags=np.array([False]))
    assert outcome["blocked"][0]
    assert outcome["size"][0] == 0.0

    baseline_bps = np.array([12.3])
    gated_bps = np.array([999.0])   # arbitrary/unused: size=0 must zero it out
    metrics = evaluate_configuration(
        np.array(["2021-01-01"], dtype="datetime64[ns]"),
        baseline_bps, gated_bps, outcome, covid_windows=[], own_windows=[])
    assert metrics["gated_pnl_bps"] == 0.0
    assert metrics["net_impact_bps"] == -12.3


# ---------------------------------------------------------------------------
# 4. Policy S timeout: exact 0.5x size, priced at signal_time + tau, not g_i
# ---------------------------------------------------------------------------
def test_policy_s_timeout_half_size_priced_at_tau_not_g():
    trades, h1, m1 = _trend_scenario()
    fcfg = FrictionConfig()
    _, baseline_bps, lat_e, lat_x = baseline_counterfactual(trades, h1, m1, fcfg)

    g_ms = np.array([50_000.0])    # 50s: far above tau, and far past the
                                    # trending entry bar (flat region beyond)
    tau_ms = 100.0                 # 100ms: inside the trending entry bar
    outcome = resolve_trade_outcome(g_ms, tau_ms, POLICY_S,
                                    reject_flags=np.array([False]))
    assert outcome["timeout"][0]
    assert not outcome["blocked"][0]
    assert outcome["size"][0] == 0.5
    assert outcome["entry_latency_ms"][0] == tau_ms   # NOT g_ms

    bps_at_tau = price_gated(trades, h1, m1, fcfg, outcome["entry_latency_ms"], lat_e, lat_x)
    bps_at_g = price_gated(trades, h1, m1, fcfg, g_ms, lat_e, lat_x)
    assert not np.isclose(bps_at_tau[0], bps_at_g[0])


# ---------------------------------------------------------------------------
# 5. Reject logic identical for Policy H and Policy S (timely branch only)
# ---------------------------------------------------------------------------
def test_reject_logic_identical_for_policy_h_and_s():
    g_ms = np.array([5.0, 500.0, 5.0, 500.0])
    tau_ms = 50.0
    reject_flags = np.array([True, False, False, True])
    out_h = resolve_trade_outcome(g_ms, tau_ms, POLICY_H, reject_flags)
    out_s = resolve_trade_outcome(g_ms, tau_ms, POLICY_S, reject_flags)
    timely = ~out_h["timeout"]
    assert timely.any()
    assert (out_h["blocked"][timely] == out_s["blocked"][timely]).all()


# ---------------------------------------------------------------------------
# 6. A timeout trade is never rejected via r_i, even if r_i says reject
# ---------------------------------------------------------------------------
def test_timeout_never_blocked_by_reject_flag():
    g_ms = np.array([500.0])
    tau_ms = 50.0
    outcome = resolve_trade_outcome(g_ms, tau_ms, POLICY_S, reject_flags=np.array([True]))
    assert outcome["timeout"][0]
    assert not outcome["blocked"][0]
    assert outcome["size"][0] == 0.5


# ---------------------------------------------------------------------------
# 7. Baseline counterfactual has no gate delay or rejects
# ---------------------------------------------------------------------------
def test_baseline_has_no_gate_delay_or_rejects():
    trades, h1, m1 = _trend_scenario()
    fcfg = FrictionConfig()
    baseline_df, baseline_bps, lat_e, lat_x = baseline_counterfactual(trades, h1, m1, fcfg)
    ref_df, ref_lat_e, ref_lat_x = run_master(trades, h1, m1, fcfg)
    assert np.array_equal(lat_e, ref_lat_e)
    assert np.array_equal(lat_x, ref_lat_x)
    assert np.allclose(net_bps_per_trade(baseline_df), net_bps_per_trade(ref_df))
    assert np.allclose(baseline_bps, net_bps_per_trade(ref_df))


# ---------------------------------------------------------------------------
# 8. net impact == 0 when g_i = 0 for every trade and no rejects
# ---------------------------------------------------------------------------
def test_net_impact_zero_when_no_gate_delay_and_no_rejects():
    trades, h1, m1 = _trend_scenario()
    fcfg = FrictionConfig()
    _, baseline_bps, lat_e, lat_x = baseline_counterfactual(trades, h1, m1, fcfg)

    g_ms = np.zeros(1)
    outcome = resolve_trade_outcome(g_ms, tau_ms=1000.0, policy=POLICY_H,
                                    reject_flags=np.zeros(1, dtype=bool))
    assert not outcome["timeout"].any() and not outcome["blocked"].any()

    gated_bps = price_gated(trades, h1, m1, fcfg, outcome["entry_latency_ms"], lat_e, lat_x)
    gated_pnl = outcome["size"] * gated_bps
    assert np.allclose(gated_pnl, baseline_bps)

    metrics = evaluate_configuration(
        np.array([trades[0].entry_time], dtype="datetime64[ns]"),
        baseline_bps, gated_bps, outcome, covid_windows=[], own_windows=[])
    assert abs(metrics["net_impact_bps"]) < 1e-9


# ---------------------------------------------------------------------------
# 9. With g_i > 0 but nothing times out and no rejects, net impact equals
# exactly the repricing effect of g_i on each (approved) trade.
# ---------------------------------------------------------------------------
def test_net_impact_equals_repricing_effect_when_approved_with_delay():
    trades, h1, m1 = _trend_scenario()
    fcfg = FrictionConfig()
    _, baseline_bps, lat_e, lat_x = baseline_counterfactual(trades, h1, m1, fcfg)

    g_ms = np.array([250.0])   # inside the trending entry bar
    outcome = resolve_trade_outcome(g_ms, tau_ms=1000.0, policy=POLICY_H,
                                    reject_flags=np.zeros(1, dtype=bool))
    assert not outcome["timeout"].any()

    gated_bps = price_gated(trades, h1, m1, fcfg, outcome["entry_latency_ms"], lat_e, lat_x)
    expected_effect = gated_bps - baseline_bps
    assert abs(expected_effect[0]) > 1e-9   # the shift must actually move the fill

    metrics = evaluate_configuration(
        np.array([trades[0].entry_time], dtype="datetime64[ns]"),
        baseline_bps, gated_bps, outcome, covid_windows=[], own_windows=[])
    assert abs(metrics["net_impact_bps"] - expected_effect.sum()) < 1e-9


# ---------------------------------------------------------------------------
# 10. foregone/avoided classification matches the sign of baseline PnL
# ---------------------------------------------------------------------------
def test_foregone_avoided_classification_matches_baseline_sign():
    baseline_bps = np.array([5.0, -3.0, 2.0, -7.0])
    gated_bps = np.zeros(4)
    outcome = {
        "timeout":        np.array([False, False, False, False]),
        "blocked":        np.array([True, True, False, True]),
        "scaled_timeout": np.zeros(4, dtype=bool),
        "size":           np.array([0.0, 0.0, 1.0, 0.0]),
    }
    metrics = evaluate_configuration(
        np.array(["2021-01-01"] * 4, dtype="datetime64[ns]"),
        baseline_bps, gated_bps, outcome, covid_windows=[], own_windows=[])
    assert metrics["n_foregone"] == 1
    assert metrics["bps_foregone"] == 5.0
    assert metrics["n_avoided"] == 2
    assert metrics["bps_avoided"] == 10.0


# ---------------------------------------------------------------------------
# 11. Policy-S timeout trades are excluded from foregone/avoided but still
# contribute to gated_pnl_bps and bps_scaled_timeout_repricing_impact.
# ---------------------------------------------------------------------------
def test_scaled_timeout_excluded_from_foregone_avoided():
    baseline_bps = np.array([10.0])
    gated_bps = np.array([3.0])   # priced at tau; 0.5x applied downstream
    outcome = {
        "timeout":        np.array([True]),
        "blocked":        np.array([False]),
        "scaled_timeout": np.array([True]),
        "size":           np.array([0.5]),
    }
    metrics = evaluate_configuration(
        np.array(["2021-01-01"], dtype="datetime64[ns]"),
        baseline_bps, gated_bps, outcome, covid_windows=[], own_windows=[])
    assert metrics["n_foregone"] == 0
    assert metrics["n_avoided"] == 0
    assert metrics["n_scaled_timeout"] == 1
    assert metrics["gated_pnl_bps"] == 1.5
    assert metrics["bps_scaled_timeout_repricing_impact"] == pytest.approx(1.5 - 5.0)


# ---------------------------------------------------------------------------
# 12. Informed rejection selects the lowest 2% PER INSTRUMENT
# ---------------------------------------------------------------------------
def test_informed_rejection_per_instrument():
    n = 100
    rng = np.random.default_rng(0)
    baseline_bps = np.concatenate([
        rng.normal(0, 1, n),     # instrument A: small PnL scale
        rng.normal(0, 50, n),    # instrument B: large PnL scale
    ])
    symbols = np.array(["A"] * n + ["B"] * n)
    flags = informed_reject_flags(baseline_bps, symbols, rate=0.02)

    assert flags[:n].sum() == 2
    assert flags[n:].sum() == 2
    a_threshold = np.sort(baseline_bps[:n])[2]
    assert baseline_bps[:n][flags[:n]].max() <= a_threshold
    b_threshold = np.sort(baseline_bps[n:])[2]
    assert baseline_bps[n:][flags[n:]].max() <= b_threshold


# ---------------------------------------------------------------------------
# Stress-window precedence: covid + own_stress + rest reconciles to
# net_impact_bps even when the two window definitions OVERLAP.
# ---------------------------------------------------------------------------
def test_stress_window_columns_reconcile_to_net_impact_with_overlap():
    n = 5
    entry_times = pd.to_datetime([
        "2021-01-01",  # inside COVID only
        "2021-01-02",  # inside COVID AND own-stress (the overlap case)
        "2021-01-10",  # inside own-stress only
        "2021-02-01",  # inside neither
        "2021-01-03",  # inside COVID only
    ])
    baseline_bps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gated_bps = np.array([0.5, 1.0, 2.5, 3.0, 4.0])
    outcome = {
        "timeout":        np.zeros(n, dtype=bool),
        "blocked":        np.zeros(n, dtype=bool),
        "scaled_timeout": np.zeros(n, dtype=bool),
        "size":           np.ones(n),
    }
    covid_windows = [(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-05"))]
    own_windows = [(pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-10"))]  # overlaps COVID

    metrics = evaluate_configuration(
        entry_times, baseline_bps, gated_bps, outcome, covid_windows, own_windows)

    total = (metrics["covid_net_impact_bps"] + metrics["own_stress_net_impact_bps"]
             + metrics["rest_net_impact_bps"])
    assert abs(total - metrics["net_impact_bps"]) < 1e-9

    # the overlapping day (Jan 2) must be attributed to COVID, not own-stress
    expected_covid = (0.5 - 1.0) + (1.0 - 2.0) + (4.0 - 5.0)   # Jan1, Jan2, Jan3
    expected_own = (2.5 - 3.0)                                  # Jan10 only
    assert abs(metrics["covid_net_impact_bps"] - expected_covid) < 1e-9
    assert abs(metrics["own_stress_net_impact_bps"] - expected_own) < 1e-9


# ---------------------------------------------------------------------------
# 13. COVID window warm-up check
# ---------------------------------------------------------------------------
def test_covid_warmup_warns_on_insufficient_history():
    idx = pd.date_range("2020-02-10", periods=5, freq="1h")
    h1 = pd.DataFrame({"open": 1.0}, index=idx)
    with pytest.warns(RuntimeWarning):
        ok = check_covid_warmup(h1)
    assert not ok


def test_covid_warmup_ok_with_enough_history():
    idx = pd.date_range("2020-01-01", "2020-02-15", freq="1h")
    h1 = pd.DataFrame({"open": 1.0}, index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = check_covid_warmup(h1)
    assert ok


# ---------------------------------------------------------------------------
# 14. Stress-day clustering: merge within 5 days, drop windows < 3 days
# ---------------------------------------------------------------------------
def test_cluster_stress_days_merges_and_drops_short_windows():
    days = pd.to_datetime([
        "2021-01-01", "2021-01-03", "2021-01-05",   # span 5d -> kept
        "2021-02-01",                                 # span 1d -> dropped
        "2021-03-01", "2021-03-02", "2021-03-03",     # span 3d -> kept
    ])
    windows = cluster_stress_days(days, gap_days=5, min_span_days=3)
    assert windows == [
        (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-05")),
        (pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-03")),
    ]



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
