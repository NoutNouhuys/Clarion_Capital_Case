"""
Master / follower replication (Part B).

The master is the Part-A Level-2 simulator (spread + commission + slippage). A
follower re-prices the *same* trade stream through the *same* friction path, with
a different latency:

    signal-side : lat = d                 (follower acts on the master's SIGNAL)
    fill-side   : lat = L_master + d      (follower acts only once the master's
                                           FILL is confirmed)

`d` is the follower replication delay (delay_models.follower_delays); `L_master`
is the master's own per-trade execution latency. The structural difference
between the two modes is exactly `L_master`: a fill-side follower can never be
faster than the master, whereas a signal-side follower can (both start from the
same signal). On minute bars that ~0.5 s offset is usually invisible -- it only
matters when it tips a fill across an M1 boundary (the "bar-crossing" rate).

Level 2 only: the assignment says "treat the Level-2 simulator as a master".
Metric: net PnL per round trip in bps of notional (cross-asset comparable).
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from simulator.frictions import (
    apply_frictions, _draw_latencies, FrictionConfig, BPS,
)
from simulator.reference_strategy import Trade
from replication import delay_models as dm

MASTER = "master"
SIGNAL_SIDE = "signal"
FILL_SIDE = "fill"


@dataclass
class ReplicationConfig:
    n_followers: int = 25
    sigma: float = dm.DEFAULT_SIGMA
    seed: int = 2024
    delay_grid_s: tuple[float, ...] = field(default_factory=lambda: dm.DELAY_GRID_S)


# ---------------------------------------------------------------------------
# Core pricing
# ---------------------------------------------------------------------------
def net_bps_per_trade(df: pd.DataFrame) -> np.ndarray:
    """Per-round-trip net PnL in bps of entry notional."""
    return (df["net_pnl"] / df["signal_price"].abs() / BPS).to_numpy()


def run_master(trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame,
               fcfg: FrictionConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Level-2 master run. Returns (per-trade df, L_master_entry, L_master_exit)."""
    lat_e, lat_x = _draw_latencies(len(trades), fcfg)
    df = apply_frictions(trades, h1, level=2, config=fcfg, m1=m1,
                         lat_entry=lat_e, lat_exit=lat_x)
    return df, lat_e, lat_x


def _follower_latency(mode: str, d_entry: np.ndarray, d_exit: np.ndarray,
                      lat_e: np.ndarray, lat_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mode == SIGNAL_SIDE:
        return d_entry, d_exit
    if mode == FILL_SIDE:
        return lat_e + d_entry, lat_x + d_exit
    raise ValueError(f"unknown mode {mode!r}")


def run_followers(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame,
    fcfg: FrictionConfig, rcfg: ReplicationConfig,
    median_s: float, mode: str,
    lat_e: np.ndarray, lat_x: np.ndarray,
    z_entry: np.ndarray, z_exit: np.ndarray,
) -> np.ndarray:
    """
    Price all `n_followers` followers for one median delay and one mode.
    Returns an array of shape (n_followers,): each follower's mean net bps.
    """
    d_entry_all, d_exit_all = dm.follower_delays(median_s, z_entry, z_exit, rcfg.sigma)
    scores = np.empty(rcfg.n_followers)
    for f in range(rcfg.n_followers):
        fe, fx = _follower_latency(mode, d_entry_all[f], d_exit_all[f], lat_e, lat_x)
        df = apply_frictions(trades, h1, level=2, config=fcfg, m1=m1,
                             lat_entry=fe, lat_exit=fx)
        scores[f] = net_bps_per_trade(df).mean()
    return scores


# ---------------------------------------------------------------------------
# Subpart 1: order-level timing diagnostics (signal-side vs fill-side)
# ---------------------------------------------------------------------------
def _price_follower(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame, fcfg: FrictionConfig,
    fe: np.ndarray, fx: np.ndarray,
) -> pd.DataFrame:
    """One follower's full per-trade frictions df (carries realized fill times)."""
    return apply_frictions(trades, h1, level=2, config=fcfg, m1=m1,
                           lat_entry=fe, lat_exit=fx)


def _order_diag(master_df: pd.DataFrame, foll_df: pd.DataFrame,
                m1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-order (entries AND exits/stops) timing diagnostics of a follower vs the
    master, read straight from the realized fill timestamps (so the
    latency-dependent stop breach is already resolved -- no timing is re-derived).

    Returns three per-order arrays (length 2*n_trades): lag in seconds
    (follower minus master fill time), a faster-than-master flag (lag < 0), and a
    bar-crossing flag (follower and master fills sit in different M1 bars).
    """
    idx = m1.index
    lags, faster, cross = [], [], []
    for col in ("entry_fill_time", "exit_fill_time"):
        mt = pd.to_datetime(master_df[col]); ft = pd.to_datetime(foll_df[col])
        lag = (ft - mt).dt.total_seconds().to_numpy()
        pos_m = idx.searchsorted(mt.to_numpy(), side="right") - 1
        pos_f = idx.searchsorted(ft.to_numpy(), side="right") - 1
        lags.append(lag); faster.append(lag < 0.0); cross.append(pos_m != pos_f)
    return (np.concatenate(lags), np.concatenate(faster), np.concatenate(cross))


# ---------------------------------------------------------------------------
# Subpart 1 driver: signal-side vs fill-side across the delay grid
# ---------------------------------------------------------------------------
def signal_vs_fill_diagnostics(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame,
    fcfg: FrictionConfig, rcfg: ReplicationConfig,
) -> pd.DataFrame:
    """One row per (median delay, mode) with mean follower decay (bps vs master),
    mean lag vs master (s), % of orders faster than the master, and the
    order-level bar-crossing rate (entries and exits)."""
    master_df, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    master_bps = net_bps_per_trade(master_df).mean()
    z_entry, z_exit = dm.standard_normals(rcfg.n_followers, len(trades), rcfg.seed)

    rows = []
    for median_s in rcfg.delay_grid_s:
        d_e_all, d_x_all = dm.follower_delays(median_s, z_entry, z_exit, rcfg.sigma)
        for mode in (SIGNAL_SIDE, FILL_SIDE):
            bps, lag_all, fast_all, cross_all = [], [], [], []
            for f in range(rcfg.n_followers):
                fe, fx = _follower_latency(mode, d_e_all[f], d_x_all[f], lat_e, lat_x)
                fdf = _price_follower(trades, h1, m1, fcfg, fe, fx)
                bps.append(net_bps_per_trade(fdf).mean())
                lag, fast, cross = _order_diag(master_df, fdf, m1)
                lag_all.append(lag); fast_all.append(fast); cross_all.append(cross)
            mode_bps = float(np.mean(bps))
            rows.append({
                "median_delay_s":   median_s,
                "mode":             mode,
                "master_bps":       master_bps,
                "mode_mean_bps":    mode_bps,
                "decay_bps":        master_bps - mode_bps,
                "mean_lag_s":       float(np.concatenate(lag_all).mean()),
                "pct_faster":       float(np.concatenate(fast_all).mean() * 100.0),
                "order_bar_cross":  float(np.concatenate(cross_all).mean() * 100.0),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from simulator.data_loader import load_m1, resample_h1, TICK_SIZE
    from simulator import reference_strategy as strat

    fcfg = FrictionConfig()
    rcfg = ReplicationConfig()
    frames = []
    for sym in TICK_SIZE:
        m1 = load_m1(sym); h1 = resample_h1(m1); trades = strat.run(h1, symbol=sym)
        out = signal_vs_fill_diagnostics(trades, h1, m1, fcfg, rcfg)
        out.insert(0, "symbol", sym)
        frames.append(out)

    import pathlib
    outpath = pathlib.Path("results/partB_signal_vs_fill.csv")
    outpath.parent.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(outpath, index=False)
    print(f"wrote {outpath} ({sum(len(f) for f in frames)} rows)")
