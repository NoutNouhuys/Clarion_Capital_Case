"""
Part B3 diagnostic: entry-stop anchoring check for local stop execution.

Records, as a reproducible artefact, the per-trade diagnostic used to explain
the negative mean_decay_after_bps observed for XAUUSD (and, at the extremes,
ETHUSD) in `results/partB_local_stop.csv`. The stop level a follower inherits is
anchored to the MASTER's entry (master entry +/- 2xATR), while each follower's
own entry is independently delayed and lands at a different point on the same
M1 price path. At large replication delay, a follower can enter after the
market has already moved toward that fixed, copied stop level, so its own
entry-to-stop distance is smaller than the master's -- it has less room left
to lose before the identical stop triggers.

This script does NOT modify `local_stop_execution.py`. It only reuses the
existing pricing functions (`run_master`, `price_local_stop_follower`) to read
off each account's own realized entry fill price and compare it against the
shared, master-anchored stop level.

For each asset x delay in the grid, two groups are reported:
    control     -- all stop-exit trades (unconditional population)
    conditional -- the subset of those trades where the local-stop-execution
                   follower's decay_after_bps is negative (follower appears to
                   beat the master)

The control row additionally reports the SHARE of negative-decay trades and
the mean magnitude of the negative vs. non-negative subsets. This tests, per
asset, whether the aggregate mean_decay_after only turns negative for XAUUSD
because XAUUSD has a larger SHARE of negative-decay trades than the other
three assets, or because its negative trades are individually LARGER in
magnitude (or both) -- rather than assuming either mechanism.

Output: results/partB_anchoring_diagnostic.csv
    symbol, median_delay_s, group, n_trades, pct_follower_closer,
    mean_master_distance, mean_follower_distance, mean_distance_gap,
    pct_negative_decay, mean_negative_decay_bps, mean_positive_decay_bps
    (the last three are populated on the control row only; NaN on conditional,
    since they describe the split of the control population itself)
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from simulator.frictions import FrictionConfig
from simulator.reference_strategy import Trade
from replication import delay_models as dm
from replication.followers import (
    run_master, net_bps_per_trade, _follower_latency, ReplicationConfig, SIGNAL_SIDE,
)
from optimisations.local_stop_execution import (
    stop_exit_mask, price_local_stop_follower, draw_follower_stop_latency,
)

# Diagnostic subset of the B3 delay grid, extendable to the full grid
# (0.3, 3, 30, 120, 600) if needed -- these two are where local stop
# execution's decay-after turns negative for XAUUSD.
DEFAULT_DIAGNOSTIC_DELAYS = (120.0, 600.0)


def _entry_stop_distances(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame, fcfg: FrictionConfig,
    rcfg: ReplicationConfig, median_s: float, f: int,
) -> pd.DataFrame:
    """Per stop-exit trade: master vs. follower entry-to-stop distance (price
    points) and the local-stop-execution decay_after_bps, for one follower at
    one median delay."""
    n = len(trades)
    is_stop = stop_exit_mask(trades)

    master_df, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    m_bps = net_bps_per_trade(master_df)

    z_e, z_x = dm.standard_normals(rcfg.n_followers, n, rcfg.seed)
    d_e_all, d_x_all = dm.follower_delays(median_s, z_e, z_x, rcfg.sigma)
    fe, fx = _follower_latency(SIGNAL_SIDE, d_e_all[f], d_x_all[f], lat_e, lat_x)

    stop_latency = draw_follower_stop_latency(n, fcfg)
    fdf_a = price_local_stop_follower(trades, h1, m1, fcfg, fe, fx, is_stop, stop_latency)
    a_bps = net_bps_per_trade(fdf_a)

    rows = []
    for i, t in enumerate(trades):
        if not is_stop[i]:
            continue
        m_entry = master_df["entry_fill"].iloc[i]
        f_entry = fdf_a["entry_fill"].iloc[i]
        stop = t.stop_price
        m_dist = abs(m_entry - stop)
        f_dist = abs(f_entry - stop)
        rows.append({
            "trade_idx": i,
            "decay_after_bps": float(m_bps[i] - a_bps[i]),
            "master_distance": m_dist,
            "follower_distance": f_dist,
            "follower_closer": f_dist < m_dist,
        })
    return pd.DataFrame(rows)


def _summarize_group(df: pd.DataFrame) -> dict:
    return {
        "n_trades":              int(len(df)),
        "pct_follower_closer":   float(df["follower_closer"].mean() * 100.0) if len(df) else float("nan"),
        "mean_master_distance":  float(df["master_distance"].mean()) if len(df) else float("nan"),
        "mean_follower_distance": float(df["follower_distance"].mean()) if len(df) else float("nan"),
        "mean_distance_gap":     float((df["master_distance"] - df["follower_distance"]).mean()) if len(df) else float("nan"),
    }


def _negative_share_stats(dist_df: pd.DataFrame) -> dict:
    """Share and magnitude of negative-decay trades within the full stop-trade
    population, to test the two candidate 'why only XAUUSD in aggregate'
    explanations: a larger negative SHARE, a larger negative MAGNITUDE, or
    both."""
    neg = dist_df[dist_df["decay_after_bps"] < 0.0]
    pos = dist_df[dist_df["decay_after_bps"] >= 0.0]
    return {
        "pct_negative_decay":     float(len(neg) / len(dist_df) * 100.0) if len(dist_df) else float("nan"),
        "mean_negative_decay_bps": float(neg["decay_after_bps"].mean()) if len(neg) else float("nan"),
        "mean_positive_decay_bps": float(pos["decay_after_bps"].mean()) if len(pos) else float("nan"),
    }


def anchoring_diagnostic(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame,
    fcfg: FrictionConfig, rcfg: ReplicationConfig,
    delays=DEFAULT_DIAGNOSTIC_DELAYS, f: int = 0,
) -> pd.DataFrame:
    """One row per (median_delay_s, group) with control (all stop trades) vs.
    conditional (decay_after_bps < 0) entry-to-stop distance statistics. The
    control row also carries the negative-decay share/magnitude split."""
    rows = []
    for median_s in delays:
        dist_df = _entry_stop_distances(trades, h1, m1, fcfg, rcfg, median_s, f)
        control = dist_df
        conditional = dist_df[dist_df["decay_after_bps"] < 0.0]
        share_stats = _negative_share_stats(dist_df)
        empty_share = {k: float("nan") for k in share_stats}

        for group_name, group_df, extra in (
            ("control", control, share_stats),
            ("conditional", conditional, empty_share),
        ):
            stats = _summarize_group(group_df)
            rows.append({"median_delay_s": median_s, "group": group_name, **stats, **extra})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pathlib
    from simulator.data_loader import load_m1, resample_h1, TICK_SIZE
    from simulator import reference_strategy as strat

    fcfg = FrictionConfig(); rcfg = ReplicationConfig()
    frames = []
    for sym in TICK_SIZE:
        m1 = load_m1(sym); h1 = resample_h1(m1); trades = strat.run(h1, symbol=sym)
        out = anchoring_diagnostic(trades, h1, m1, fcfg, rcfg)
        out.insert(0, "symbol", sym)
        frames.append(out)

    results = pathlib.Path("results"); results.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(
        results / "partB_anchoring_diagnostic.csv", index=False)
    print("wrote results/partB_anchoring_diagnostic.csv")
