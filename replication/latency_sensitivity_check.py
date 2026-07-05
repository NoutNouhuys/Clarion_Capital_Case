"""
Part B3 diagnostic: latency-sensitivity check for local stop execution.

Records, as a reproducible artefact, the comparison already used to justify
the fix to `optimisations/local_stop_execution.py`: an EARLIER version of that
module localised stop exits by setting the post-breach exit latency to
`0.0` (`exit_fill_time = breach + 0`), which silently assumed a follower
faster than the master ever is. The fix replaced that `0.0` with an
independent draw from the SAME lognormal distribution the master's own exit
latency uses (`draw_follower_stop_latency`).

This script does NOT modify `local_stop_execution.py`. The current (fixed)
module no longer contains a zero-latency code path, so the zero-latency
variant is reproduced here as a minimal, clearly-labelled local helper
(`_zero_latency_stop_follower`) that mirrors the old behaviour exactly, for
comparison purposes only. All actual pricing is still delegated to the
existing, unmodified `_price_follower` -- nothing about the fill/gap/clamp
mechanics is reimplemented.

Output: results/partB_latency_sensitivity.csv
    symbol, median_delay_s,
    mean_decay_after_zero_latency, mean_decay_after_drawn_latency,
    abs_difference_bps
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
    run_master, net_bps_per_trade, _price_follower, _follower_latency,
    ReplicationConfig, SIGNAL_SIDE,
)
from optimisations.local_stop_execution import (
    stop_exit_mask, price_local_stop_follower, draw_follower_stop_latency,
)


# ---------------------------------------------------------------------------
# Reproduction of the OLD (pre-fix) zero-latency localisation, for comparison
# only. Mirrors the original `localise_exit_latency` / `price_local_stop_
# follower` mechanism exactly (same np.where shape, same delegation to
# `_price_follower`), just with 0.0 in place of the current independent draw.
# ---------------------------------------------------------------------------
def _localise_exit_latency_zero(d_exit: np.ndarray, is_stop: np.ndarray) -> np.ndarray:
    return np.where(is_stop, 0.0, d_exit)


def _zero_latency_stop_follower(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame, fcfg: FrictionConfig,
    fe: np.ndarray, fx: np.ndarray, is_stop: np.ndarray,
) -> pd.DataFrame:
    fx_local = _localise_exit_latency_zero(fx, is_stop)
    return _price_follower(trades, h1, m1, fcfg, fe, fx_local)


# ---------------------------------------------------------------------------
# Driver: both variants, same CRN draws, per asset x delay grid
# ---------------------------------------------------------------------------
def run_latency_sensitivity(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame,
    fcfg: FrictionConfig, rcfg: ReplicationConfig,
) -> pd.DataFrame:
    """One row per median delay. Both variants are priced from the same
    per-follower entry/exit replication draws and the same
    `draw_follower_stop_latency` seed, so the only difference between them is
    the post-breach exit latency assumption (0.0 vs an independent lognormal
    draw) -- isolating exactly the effect the fix addressed."""
    n = len(trades)
    is_stop = stop_exit_mask(trades)

    master_df, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    m_bps = net_bps_per_trade(master_df)

    stop_latency = draw_follower_stop_latency(n, fcfg)

    z_e, z_x = dm.standard_normals(rcfg.n_followers, n, rcfg.seed)
    F = rcfg.n_followers

    rows = []
    for median_s in rcfg.delay_grid_s:
        d_e_all, d_x_all = dm.follower_delays(median_s, z_e, z_x, rcfg.sigma)
        zero_decay = np.empty(F)
        drawn_decay = np.empty(F)

        for f in range(F):
            fe, fx = _follower_latency(SIGNAL_SIDE, d_e_all[f], d_x_all[f], lat_e, lat_x)

            fdf_zero = _zero_latency_stop_follower(trades, h1, m1, fcfg, fe, fx, is_stop)
            fdf_drawn = price_local_stop_follower(
                trades, h1, m1, fcfg, fe, fx, is_stop, stop_latency)

            zero_decay[f] = float(np.mean(m_bps - net_bps_per_trade(fdf_zero)))
            drawn_decay[f] = float(np.mean(m_bps - net_bps_per_trade(fdf_drawn)))

        mean_zero = float(zero_decay.mean())
        mean_drawn = float(drawn_decay.mean())
        rows.append({
            "median_delay_s":                 median_s,
            "mean_decay_after_zero_latency":  mean_zero,
            "mean_decay_after_drawn_latency": mean_drawn,
            "abs_difference_bps":             abs(mean_zero - mean_drawn),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pathlib
    from simulator.data_loader import load_m1, resample_h1, TICK_SIZE
    from simulator import reference_strategy as strat

    fcfg = FrictionConfig(); rcfg = ReplicationConfig()
    frames = []
    for sym in TICK_SIZE:
        m1 = load_m1(sym); h1 = resample_h1(m1); trades = strat.run(h1, symbol=sym)
        out = run_latency_sensitivity(trades, h1, m1, fcfg, rcfg)
        out.insert(0, "symbol", sym)
        frames.append(out)

    results = pathlib.Path("results"); results.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(
        results / "partB_latency_sensitivity.csv", index=False)
    print("wrote results/partB_latency_sensitivity.csv")
