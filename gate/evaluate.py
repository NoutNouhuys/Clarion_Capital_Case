"""
Part C: pre-trade risk gate evaluation.

Every master ENTRY order must pass through a risk gate before it reaches the
market. The gate has a stochastic decision latency g_i (gate/latency_model.py)
and a hard timeout budget tau. If the gate answers within tau it approves or
rejects the order (gate/policies.resolve_trade_outcome); if it does not, a
timeout policy (H = hard block, S = fail-safe half-size) decides what happens
instead (gate/policies.py).

GATE SCOPE (deliberate design choice, not an omission): the gate sits ONLY on
entries. Risk-reducing EXITS -- stop-exits and signal-exits that close a
position -- always execute at the realized position size and never pass
through the gate. A gate that could also block or delay an exit would be
gating risk REDUCTION, which defeats the point of a pre-trade risk control.

FIXED TRADE STREAM (deliberate design choice, with a stated limitation): the
trade stream -- which signals fire, when, on which instrument -- is exactly
the Part A/B reference-strategy stream (simulator.reference_strategy.run) and
does NOT change with the gate's decisions. If an entry is blocked (Policy-H
timeout, or a timely reject), the strategy's own state machine behaves, for
the purpose of determining the NEXT signal, as if the position had been
taken: "one position per symbol at a time" is enforced against the fixed
Part A/B stream, not against what a specific gate configuration actually
executed. The gate is a filter/overlay on top of a fixed stream, not an input
that feeds back into the strategy's own state.
    Why: without a fixed trade stream there is no clean paired comparison
    across the 40 gate configurations -- Common Random Numbers require "trade
    i" to mean the same trade in every configuration.
    Limitation: in reality a blocked entry would change the strategy's own
    position state and could therefore let a different next signal through
    (e.g. a channel breakout that the fixed stream's "already in position"
    assumption would have suppressed). This is not modeled here; the net
    performance impact reported below is conditional on the fixed Part A/B
    stream, not on a fully gate-aware re-simulation of the strategy.

REPRICING (section 5 of the design spec): an approved, timely order is priced
by the EXISTING Level-2 friction pipeline (simulator.frictions / _price_
follower) with its ENTRY ARRIVAL time shifted from signal_time to
signal_time + g_i -- "before the Level-2 friction simulator does its work",
i.e. the master's own physical entry latency (drawn once via
replication.followers.run_master, the same lat_entry used by the ungated
baseline) still applies ON TOP of the shift, unchanged. Effective entry
latency fed into the pipeline is therefore `g_i + lat_entry_master`
(seconds), the fill-side-style ADDITIVE composition
(replication.followers._follower_latency, FILL_SIDE mode: lat =
L_master + d) rather than Part B's signal-side REPLACEMENT. This is not
merely a reading of the prose: it is forced by section 10's own zero-effect
sanity test -- with g_i = 0 the gated fill must equal the baseline fill
EXACTLY, which only holds if the master's own nonzero physical latency is
still present in both. A Policy-S timeout is priced the same way with tau in
place of g_i: effective entry latency = `tau + lat_entry_master` (NOT
`g_i + lat_entry_master` -- using the real, tau-exceeding g_i would mean
waiting for the actual answer, i.e. no timeout behaviour at all), at half
size. Exit latency is always the trade's own Level-2 master exit latency,
unaffected by the gate, matching the "exits do not go through the gate"
scope rule above.
    Fractional-size check (also required by section 5): the Level-2 pipeline
    used here (simulator.frictions.apply_frictions, level=2) has no minimum-
    lot or tickvol-based sizing at all -- commission is a bps rate of
    notional (simulator/frictions.py:126-130) and Level-2 has no partial-fill
    tranching (that is Level 3, not used here). Every per-trade cost is
    linear in position size, so applying the Policy-S 0.5x scale AFTER
    pricing (multiplying the priced bps by `size`, exactly the pattern
    optimisations/vol_norm_latency_sizing.py already uses for entry-only
    sizing) is exact, not an approximation. Exercised by test_gate.py::
    test_scaled_timeout_excluded_from_foregone_avoided, which asserts
    gated_pnl_bps == size * priced_bps exactly for a Policy-S timeout trade.

NET PERFORMANCE IMPACT (section 6): defined as the SUM, over every trade in
the stream (not just the blocked ones), of (gated PnL - baseline PnL) in bps
of that trade's own entry notional. Approved trades still cost bps via the
g_i repricing (identical mechanism to Part B's follower decay), so a
definition that only looks at blocked trades would silently drop that term.

STRESS WINDOWS (section 8): reported as three columns -- covid_net_impact_bps,
own_stress_net_impact_bps, rest_net_impact_bps -- that PARTITION the trade
stream and therefore sum exactly to net_impact_bps. The two stress
definitions (fixed 2020-02-15..2020-04-15 COVID window, and each
instrument's own top-5%-realized-vol clustered windows) are not disjoint in
practice -- the COVID crash IS most instruments' own top-vol period, so a
naive "within COVID" / "within own-stress" / "outside both" split would
double-count the overlap and the three columns would NOT sum to
net_impact_bps. Precedence rule: a trade inside BOTH windows is attributed to
covid_net_impact_bps only (own_stress_net_impact_bps is defined as "own-stress
AND NOT covid"); rest_net_impact_bps is the complement of the union of the
two, as before. This is a stated modelling choice (COVID takes precedence as
the more specific, named event), not derived from the spec text.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simulator.frictions import FrictionConfig
from simulator.reference_strategy import Trade
from replication.followers import run_master, net_bps_per_trade, _price_follower

from gate.latency_model import (
    draw_gate_z, gate_latency_ms, draw_uninformed_reject_flags,
    GATE_MEDIAN_GRID_MS, DEFAULT_TAU_GRID_MS,
)
from gate.policies import (
    resolve_trade_outcome, informed_reject_flags,
    POLICY_H, POLICY_S, POLICIES, UNINFORMED, INFORMED, REJECTION_MODES,
)

COVID_START = pd.Timestamp("2020-02-15")
COVID_END = pd.Timestamp("2020-04-15")
WARMUP_BARS = 20  # Donchian(20) lookback; ATR(14) needs fewer, so 20 binds.

STRESS_VOL_WINDOW = 20     # daily bars, analogue of opt1's intraday VOL_WINDOW
STRESS_TOP_PCT = 0.05
STRESS_CLUSTER_GAP_DAYS = 5
STRESS_MIN_SPAN_DAYS = 3


# ---------------------------------------------------------------------------
# Baseline counterfactual (section 6): full size, Level-2 friction, no gate.
# Computed ONCE per trade and reused across the full 40-configuration grid.
# ---------------------------------------------------------------------------
def baseline_counterfactual(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame, fcfg: FrictionConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (baseline_df, baseline_pnl_bps, lat_entry_master, lat_exit_master).
    `lat_exit_master` is reused, unchanged, for every gated pricing pass -- exits
    never go through the gate (see module docstring, GATE SCOPE)."""
    baseline_df, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    return baseline_df, net_bps_per_trade(baseline_df), lat_e, lat_x


# ---------------------------------------------------------------------------
# Gated pricing for one configuration
# ---------------------------------------------------------------------------
def price_gated(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame, fcfg: FrictionConfig,
    gate_latency_ms_arr: np.ndarray, lat_e_master: np.ndarray, lat_x_master: np.ndarray,
) -> np.ndarray:
    """Per-trade net bps with entry latency `gate_latency_ms_arr/1000 +
    lat_e_master` (additive, see module docstring REPRICING section: g_i for
    an approved timely order, tau for a Policy-S timeout, arbitrary/unused
    where size will be zeroed downstream). Exit latency is always the
    unmodified master exit latency -- exits are not gated."""
    fe = gate_latency_ms_arr / 1000.0 + lat_e_master
    fdf = _price_follower(trades, h1, m1, fcfg, fe, lat_x_master)
    return net_bps_per_trade(fdf)


# ---------------------------------------------------------------------------
# Section 8a: fixed COVID stress window
# ---------------------------------------------------------------------------
def check_covid_warmup(h1: pd.DataFrame) -> bool:
    """Warn (do not silently proceed) if there is not enough H1 history before
    COVID_START for the reference strategy's Donchian(20)/ATR(14) warm-up."""
    pre = h1.loc[: COVID_START]
    ok = len(pre) >= WARMUP_BARS
    if not ok:
        warnings.warn(
            f"COVID window warm-up check failed: only {len(pre)} H1 bars "
            f"before {COVID_START.date()}, need >= {WARMUP_BARS} bars for "
            f"Donchian(20)/ATR(14) to be defined at the first in-window signal.",
            RuntimeWarning,
        )
    return ok


# ---------------------------------------------------------------------------
# Section 8b: instrument-own, data-driven stress windows
# ---------------------------------------------------------------------------
def daily_realized_vol(m1: pd.DataFrame, window: int = STRESS_VOL_WINDOW) -> pd.Series:
    """Daily analogue of opt1's recent_vol_bps (optimisations/vol_norm_latency_
    sizing.py): rolling std of close-to-close daily returns in bps, shifted by
    one day so day T's value uses only days strictly before T."""
    daily_close = m1["close"].resample("1D").last().dropna()
    ret_bps = daily_close.pct_change() * 1e4
    return ret_bps.rolling(window).std().shift(1)


def top_stress_days(vol: pd.Series, pct: float = STRESS_TOP_PCT) -> pd.DatetimeIndex:
    """Top `pct` share of days by realized vol (NaN warm-up days dropped)."""
    v = vol.dropna()
    if v.empty:
        return pd.DatetimeIndex([])
    thresh = v.quantile(1.0 - pct)
    return v.index[v >= thresh].sort_values()


def cluster_stress_days(
    days, gap_days: int = STRESS_CLUSTER_GAP_DAYS, min_span_days: int = STRESS_MIN_SPAN_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Chain-merge days within `gap_days` calendar days of the previous day in
    the running cluster into one window; drop windows spanning fewer than
    `min_span_days` calendar days."""
    days = sorted(pd.DatetimeIndex(days))
    if not days:
        return []
    clusters = [[days[0]]]
    for d in days[1:]:
        if (d - clusters[-1][-1]).days <= gap_days:
            clusters[-1].append(d)
        else:
            clusters.append([d])
    windows = []
    for c in clusters:
        start, end = c[0], c[-1]
        span_days = (end - start).days + 1
        if span_days >= min_span_days:
            windows.append((start, end))
    return windows


def instrument_stress_windows(m1: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    vol = daily_realized_vol(m1)
    days = top_stress_days(vol)
    return cluster_stress_days(days)


def _in_any_window(times: np.ndarray, windows: list[tuple]) -> np.ndarray:
    times = pd.to_datetime(times)
    mask = np.zeros(len(times), dtype=bool)
    for start, end in windows:
        end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)  # end date inclusive
        mask |= (times >= pd.Timestamp(start)) & (times < end_excl)
    return mask


# ---------------------------------------------------------------------------
# Section 6/9: per-configuration classification + metrics
# ---------------------------------------------------------------------------
def evaluate_configuration(
    entry_times: np.ndarray,
    baseline_bps: np.ndarray,
    gated_bps: np.ndarray,
    outcome: dict,
    covid_windows: list[tuple],
    own_windows: list[tuple],
) -> dict:
    """Aggregate metrics for one (median, tau, policy, rejection_mode) x
    instrument configuration, given the per-trade baseline/gated bps and the
    policy-resolved per-trade outcome (gate/policies.resolve_trade_outcome)."""
    size = outcome["size"]
    blocked = outcome["blocked"]
    scaled_timeout = outcome["scaled_timeout"]
    timeout = outcome["timeout"]

    gated_pnl = size * gated_bps
    diff = gated_pnl - baseline_bps

    foregone = blocked & (baseline_bps > 0.0)
    avoided = blocked & (baseline_bps <= 0.0)

    covid_mask = _in_any_window(entry_times, covid_windows)
    own_mask = _in_any_window(entry_times, own_windows) & ~covid_mask  # precedence: COVID wins
    rest_mask = ~(covid_mask | own_mask)
    # covid_mask, own_mask, rest_mask now partition every trade exactly once,
    # so covid_net_impact_bps + own_stress_net_impact_bps + rest_net_impact_bps
    # == net_impact_bps by construction (see module docstring, STRESS WINDOWS).

    row = {
        "timeout_rate":            float(timeout.mean()),
        "baseline_pnl_bps":        float(baseline_bps.sum()),
        "gated_pnl_bps":           float(gated_pnl.sum()),
        "net_impact_bps":          float(diff.sum()),
        "n_foregone":              int(foregone.sum()),
        "bps_foregone":            float(baseline_bps[foregone].sum()),
        "n_avoided":               int(avoided.sum()),
        "bps_avoided":             float((-baseline_bps[avoided]).sum()),
        "covid_net_impact_bps":    float(diff[covid_mask].sum()),
        "own_stress_net_impact_bps": float(diff[own_mask].sum()),
        "rest_net_impact_bps":     float(diff[rest_mask].sum()),
    }
    n_scaled = int(scaled_timeout.sum())
    row["n_scaled_timeout"] = n_scaled
    row["bps_scaled_timeout_repricing_impact"] = float(
        (gated_pnl[scaled_timeout] - 0.5 * baseline_bps[scaled_timeout]).sum()
    )
    # Policy H never produces scaled-timeout trades; the caller (evaluate_symbol)
    # overrides both fields to NaN for Policy-H rows per section 9 of the spec.
    return row


# ---------------------------------------------------------------------------
# Full grid driver
# ---------------------------------------------------------------------------
@dataclass
class GateConfig:
    median_grid_ms: tuple[float, ...] = GATE_MEDIAN_GRID_MS
    tau_grid_ms: tuple[float, ...] = DEFAULT_TAU_GRID_MS
    policies: tuple[str, ...] = POLICIES
    rejection_modes: tuple[str, ...] = REJECTION_MODES


def evaluate_symbol(
    symbol: str, trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame,
    fcfg: FrictionConfig, gcfg: GateConfig,
) -> pd.DataFrame:
    """Evaluates the full grid for one instrument's fixed trade stream. Draws
    z_i and r_i ONCE (shared across the whole grid, section 7 CRN), and the
    baseline counterfactual ONCE (section 6)."""
    check_covid_warmup(h1)

    n = len(trades)
    entry_times = np.array([t.entry_time for t in trades], dtype="datetime64[ns]")
    symbols_arr = np.full(n, symbol)

    baseline_df, baseline_bps, lat_e, lat_x = baseline_counterfactual(trades, h1, m1, fcfg)

    z = draw_gate_z(n)
    r_uninformed = draw_uninformed_reject_flags(n)
    r_informed = informed_reject_flags(baseline_bps, symbols_arr)  # baseline first (section 7)

    covid_windows = [(COVID_START, COVID_END)]
    own_windows = instrument_stress_windows(m1)

    rows = []
    for median_ms in gcfg.median_grid_ms:
        g_ms = gate_latency_ms(z, median_ms)
        for tau_ms in gcfg.tau_grid_ms:
            for policy in gcfg.policies:
                for rejection_mode in gcfg.rejection_modes:
                    reject_flags = r_uninformed if rejection_mode == UNINFORMED else r_informed
                    outcome = resolve_trade_outcome(g_ms, tau_ms, policy, reject_flags)
                    gated_bps = price_gated(trades, h1, m1, fcfg,
                                            outcome["entry_latency_ms"], lat_e, lat_x)
                    metrics = evaluate_configuration(
                        entry_times, baseline_bps, gated_bps, outcome,
                        covid_windows, own_windows,
                    )
                    if policy == POLICY_H:
                        metrics["n_scaled_timeout"] = float("nan")
                        metrics["bps_scaled_timeout_repricing_impact"] = float("nan")
                    rows.append({
                        "instrument": symbol, "gate_median_ms": median_ms,
                        "tau_ms": tau_ms, "policy": policy,
                        "rejection_mode": rejection_mode, **metrics,
                    })
    return pd.DataFrame(rows)


def evaluate_grid(
    data: dict[str, dict[str, pd.DataFrame]], trades_by_symbol: dict[str, list[Trade]],
    fcfg: FrictionConfig, gcfg: GateConfig,
) -> pd.DataFrame:
    """Evaluates the full 40-configuration grid for every instrument in `data`."""
    frames = []
    for symbol, trades in trades_by_symbol.items():
        h1, m1 = data[symbol]["h1"], data[symbol]["m1"]
        frames.append(evaluate_symbol(symbol, trades, h1, m1, fcfg, gcfg))
    return pd.concat(frames, ignore_index=True)


COLUMN_ORDER = [
    "instrument", "gate_median_ms", "tau_ms", "policy", "rejection_mode",
    "timeout_rate", "baseline_pnl_bps", "gated_pnl_bps", "net_impact_bps",
    "n_foregone", "bps_foregone", "n_avoided", "bps_avoided",
    "n_scaled_timeout", "bps_scaled_timeout_repricing_impact",
    "covid_net_impact_bps", "own_stress_net_impact_bps", "rest_net_impact_bps",
]


if __name__ == "__main__":
    import pathlib
    from simulator.data_loader import load_all, TICK_SIZE
    from simulator import reference_strategy as strat

    fcfg = FrictionConfig()
    gcfg = GateConfig()
    data = load_all()
    trades_by_symbol = {
        sym: strat.run(data[sym]["h1"], symbol=sym) for sym in TICK_SIZE
    }
    out = evaluate_grid(data, trades_by_symbol, fcfg, gcfg)[COLUMN_ORDER]

    results = pathlib.Path("results"); results.mkdir(exist_ok=True)
    out.to_csv(results / "partC_gate_policy_grid.csv", index=False)
    print(f"wrote results/partC_gate_policy_grid.csv ({len(out)} rows)")
