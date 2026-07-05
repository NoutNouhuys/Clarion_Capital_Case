"""
Volatility-normalised latency sizing --- second B3 optimisation.

Rule: q(d, sigma_t) = min(1.0, R_star / (d * sigma_t))

Why this improves on pure latency-decay sizing (optimisation 1):
The same delay is not equally costly in all market conditions.
Follower-master price divergence during a delay is proportional to
both the delay length and short-horizon volatility. This rule targets
the product d * sigma_t directly, keeping delay-volatility exposure
constant rather than delay alone.

Invariant: q(d, sigma_t) * d * sigma_t = R_star whenever q < 1.
A copy that is twice as delay-risky receives half the size.

R_star calibration: 30.0 * median(recent_vol_bps) per asset,
computed globally over the full dataset. This gives full size at a
normal volatility state and 30-second delay, matching the timing
anchor from the bar-crossing diagnostics. No PnL fitting.

Look-ahead: sigma_t uses only M1 bars strictly before the signal
time (rolling window with shift=1). Verified by assertion.

No q_min. No per-instrument tuning beyond the per-asset median vol
that enters R_star. Exits always close the full outstanding position.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
import numpy as np
import pandas as pd

from simulator.frictions import FrictionConfig
from simulator.reference_strategy import Trade
from replication import delay_models as dm
from replication.followers import (
    run_master, net_bps_per_trade, _price_follower,
    _follower_latency, ReplicationConfig, SIGNAL_SIDE,
)

ONE_HOUR = pd.Timedelta(hours=1)
VOL_WINDOW = 60          # M1 bars
R_STAR_DELAY_ANCHOR = 30.0   # seconds: full size at normal vol and 30s delay

# Delay above which a copy is treated as data corruption rather than a stale but
# real fill. Generous by design: on clean data it never binds, so it does not
# shape q -- it only guards against non-physical inputs.
DATA_QUALITY_MAX_DELAY_S = 86_400.0  # 1 day


# ---------------------------------------------------------------------------
# Shared sizing helpers (data-quality guard + master-edge decomposition)
# ---------------------------------------------------------------------------
def apply_delay_cap(delay_s, max_delay_s):
    """Data-quality guard, NOT an optimisation parameter: clip implausible /
    corrupt delays before sizing. Kept separate from the sizing rule on purpose."""
    return np.minimum(delay_s, max_delay_s)


def edge_decomposition(q, trade_pnl_bps):
    """Decompose the edge given up by copying at size q < 1.

    trade_pnl_bps should be the *follower's own* PnL per trade, not the
    master's — the follower never had master fills, so weighting (1-q)
    against master outcomes overstates foregone edge.

    Foregone positive edge: (1-q) of each winning trade you underweight.
    Avoided losing exposure: (1-q) of each losing trade you underweight.
    Net edge given up = foregone positive - avoided losing."""
    foregone_positive_edge = ((1 - q) * np.maximum(trade_pnl_bps, 0)).sum()
    avoided_losing_exposure = ((1 - q) * np.maximum(-trade_pnl_bps, 0)).sum()
    net_edge_given_up = foregone_positive_edge - avoided_losing_exposure

    lhs = foregone_positive_edge + avoided_losing_exposure
    rhs = ((1 - q) * np.abs(trade_pnl_bps)).sum()
    assert abs(lhs - rhs) < 1e-8

    return foregone_positive_edge, avoided_losing_exposure, net_edge_given_up


# ---------------------------------------------------------------------------
# Volatility (look-ahead free) and R_star calibration
# ---------------------------------------------------------------------------
def recent_vol_bps(m1: pd.DataFrame, window: int = VOL_WINDOW) -> pd.Series:
    """Rolling std of M1 returns in bps, excluding the current bar (shift=1), so
    the value at timestamp T uses only returns strictly before T."""
    ret_bps = m1["close"].pct_change() * 1e4          # (c_t - c_{t-1})/c_{t-1} * 1e4
    return ret_bps.rolling(window).std().shift(1)


def calibrate_r_star(vol_series: pd.Series) -> float:
    """R_star = 30 * global median recent vol (bps). One number per asset."""
    return R_STAR_DELAY_ANCHOR * float(np.median(vol_series.dropna()))


def sigma_at_times(signal_times: np.ndarray, m1: pd.DataFrame,
                   vol_series: pd.Series) -> np.ndarray:
    """Volatility known at each signal time, using only M1 bars strictly before
    it. Asserts the look-ahead-free property. NaN (warm-up) falls back to the
    global median vol so it is size-neutral."""
    idx = m1.index
    signal_times = np.asarray(signal_times, dtype="datetime64[ns]")
    pos = idx.searchsorted(signal_times, side="left") - 1   # last bar strictly before
    assert (pos >= 0).all(), "signal precedes the first M1 bar"
    # LOOK-AHEAD GUARD: the bar feeding sigma_t must lie strictly before the signal.
    assert (idx.to_numpy()[pos] < signal_times).all(), "look-ahead in sigma_t"
    sigma = vol_series.to_numpy()[pos]
    med = float(np.median(vol_series.dropna()))
    return np.where(np.isfinite(sigma), sigma, med)


def trade_sigma_bps(trades: list[Trade], m1: pd.DataFrame,
                    vol_series: pd.Series) -> np.ndarray:
    """Per-trade sigma_t (bps) at the signal time (= entry H1 bar close)."""
    signal_times = np.array([t.entry_time + ONE_HOUR for t in trades],
                            dtype="datetime64[ns]")
    return sigma_at_times(signal_times, m1, vol_series)


# ---------------------------------------------------------------------------
# The sizing rule and its exit invariant
# ---------------------------------------------------------------------------
def vol_entry_size(delay_s, sigma_t, r_star: float):
    """q(d, sigma_t) = min(1, R_star / (d * sigma_t)), entries only. No q_min."""
    d = np.atleast_1d(delay_s); s = np.atleast_1d(sigma_t)
    assert np.all(np.isfinite(d)) and np.all(np.isfinite(s))
    assert np.all(d > 0) and np.all(s > 0)
    q = np.minimum(1.0, r_star / (delay_s * sigma_t))
    return q


def vol_position_sizes(delay_s, sigma_t, r_star: float):
    """Open/close sizes for a copied trade. The exit closes the full open
    position, so close_size == open_size for every (delay, sigma) --- no
    exit-side scaling."""
    q = vol_entry_size(delay_s, sigma_t, r_star)
    return q, q  # (open, close)


# ---------------------------------------------------------------------------
# Shared metrics from one follower pricing pass
# ---------------------------------------------------------------------------
def _metrics(m: np.ndarray, B: np.ndarray, Q: np.ndarray) -> dict:
    """Decay before/after, absolute follower PnL, and gross edge decomposition
    for a size matrix Q (n_foll x n_trades) given master bps m and follower bps
    matrix B."""
    n = m.shape[0]
    before_f = (m[None, :] - B).mean(axis=1)          # per-follower baseline decay
    after_f = (m[None, :] - Q * B).mean(axis=1)        # per-follower sized decay
    pnl_full_f = B.mean(axis=1)                        # follower PnL at q = 1
    pnl_sized_f = (Q * B).mean(axis=1)                 # follower PnL under sizing
    avg_q = float(Q.mean())
    worst_after = float(after_f.max())

    pos_b = np.maximum(B, 0.0)
    neg_b = np.maximum(-B, 0.0)
    fp = ((1 - Q) * pos_b).sum(axis=1) / n             # per-follower, per-trade bps
    av = ((1 - Q) * neg_b).sum(axis=1) / n
    # edge-decomposition identity (per follower)
    assert np.allclose(fp + av, ((1 - Q) * np.abs(B)).sum(axis=1) / n, atol=1e-8)
    # NET identity: the net of the decomposition IS the decay improvement with
    # opposite sign (both equal mean((1-Q)B)). The gross split fp/av carries the
    # information; the net is reported for the identity, not as a separate fact.
    assert np.allclose(fp - av, after_f - before_f, atol=1e-8)

    return {
        "avg_copied_size":            avg_q,
        "master_pnl_bps":             float(m.mean()),
        "follower_pnl_full_bps":      float(pnl_full_f.mean()),
        "follower_pnl_sized_bps":     float(pnl_sized_f.mean()),
        "mean_decay_before_bps":      float(before_f.mean()),
        "mean_decay_after_bps":       float(after_f.mean()),
        "std_decay_before_bps":       float(before_f.std(ddof=1)),
        "std_decay_after_bps":        float(after_f.std(ddof=1)),
        "worst_decay_before_bps":     float(before_f.max()),
        "worst_decay_after_bps":      worst_after,
        "worst_decay_after_scaled_bps": worst_after * avg_q,
        "foregone_positive_edge_bps": float(fp.mean()),
        "avoided_losing_exposure_bps": float(av.mean()),
        "net_edge_given_up_bps":      float((fp - av).mean()),
        "net_improvement_bps":        float(before_f.mean() - after_f.mean()),
    }


def _price_grid(trades, h1, m1, fcfg, rcfg):
    """Price the master and all followers on the full delay grid once. Returns
    (m_bps, {median: (B, d_entry_all)})."""
    n = len(trades)
    master_df, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    m = net_bps_per_trade(master_df)
    z_e, z_x = dm.standard_normals(rcfg.n_followers, n, rcfg.seed)
    grid = {}
    for median_s in rcfg.delay_grid_s:
        d_e_all, d_x_all = dm.follower_delays(median_s, z_e, z_x, rcfg.sigma)
        B = np.empty((rcfg.n_followers, n))
        for f in range(rcfg.n_followers):
            fe, fx = _follower_latency(SIGNAL_SIDE, d_e_all[f], d_x_all[f], lat_e, lat_x)
            B[f] = net_bps_per_trade(_price_follower(trades, h1, m1, fcfg, fe, fx))
        grid[median_s] = (B, d_e_all)
    return m, grid


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
@dataclass
class VolSizingConfig:
    max_delay_s: float = DATA_QUALITY_MAX_DELAY_S


def run_vol_norm_sizing(trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame,
                        fcfg: FrictionConfig, rcfg: ReplicationConfig,
                        vcfg: VolSizingConfig) -> tuple[pd.DataFrame, float]:
    """Full-grid vol-normalised sizing metrics. Returns (df, R_star)."""
    vol_series = recent_vol_bps(m1)
    r_star = calibrate_r_star(vol_series)
    sigma_t = trade_sigma_bps(trades, m1, vol_series)
    m, grid = _price_grid(trades, h1, m1, fcfg, rcfg)

    rows = []
    for median_s, (B, d_e_all) in grid.items():
        d = apply_delay_cap(d_e_all, vcfg.max_delay_s)
        Q = vol_entry_size(d, sigma_t[None, :], r_star)
        row = {"median_delay_s": median_s, "R_star": r_star}
        row.update(_metrics(m, B, Q))
        rows.append(row)
    return pd.DataFrame(rows), r_star


if __name__ == "__main__":
    import pathlib
    from simulator.data_loader import load_m1, resample_h1, TICK_SIZE
    from simulator import reference_strategy as strat

    fcfg = FrictionConfig(); rcfg = ReplicationConfig()
    vcfg = VolSizingConfig()
    vol_frames = []
    for sym in TICK_SIZE:
        m1 = load_m1(sym); h1 = resample_h1(m1); trades = strat.run(h1, symbol=sym)
        vdf, r_star = run_vol_norm_sizing(trades, h1, m1, fcfg, rcfg, vcfg)
        vdf.insert(0, "symbol", sym)
        vol_frames.append(vdf)

    results = pathlib.Path("results"); results.mkdir(exist_ok=True)
    all_vol = pd.concat(vol_frames, ignore_index=True)
    all_vol.to_csv(results / "partB_vol_norm_sizing.csv", index=False)
    print(f"wrote results/partB_vol_norm_sizing.csv ({len(all_vol)} rows)")
