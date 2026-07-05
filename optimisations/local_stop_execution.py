"""
Part B3 optimisation: local stop execution.

Baseline replication delays *every* follower order by the replication delay d,
including the exit of a stop. But a stop is not a discretionary signal that has
to travel from master to follower: the stop level is known in advance (it is the
master's 2xATR level, fixed at entry), so a follower can arm that level locally
and be filled the moment its own M1 path touches it -- no signal round-trip, no
post-breach delay. Signal exits (opposite breakout, EOD) still have to be copied
from the master and keep the normal exit delay.

Rule (this optimisation):

    exit_reason == "stop"  ->  d_exit = L_own   (the follower's OWN independent
                                order-to-fill execution latency at the breach,
                                NOT zero)
    otherwise              ->  d_exit unchanged (normal replication delay)

What "d_exit = 0" would get wrong: the optimisation removes the REPLICATION
delay on a stop breach (no signal has to travel master -> follower first,
because the stop level is known in advance), but it must NOT also remove the
follower's own physical order-to-fill latency -- the time between the local
path touching the stop and the follower's own fill being confirmed. The master
carries exactly that same kind of latency on every exit (see
FrictionConfig.latency_median_s / latency_sigma and `_draw_latencies` in
frictions.py). Setting the follower's stop-exit latency to 0.0 would silently
assume a follower that is always faster than the master ever is (median exit
latency 0.5s), which is not a claim this optimisation makes. Instead, `L_own`
is drawn from the SAME lognormal distribution as the master's exit latency,
but from an INDEPENDENT RNG stream (own seed offset, see
STOP_LATENCY_SEED_OFFSET) -- same friction, independent draw.

Everything else is deliberately unchanged:

  * Stop LEVEL stays the master's (`trade.stop_price`, = master entry +/- 2xATR).
    The follower does NOT recompute a stop from its own delayed entry price.
  * The stop-breach scan still starts only AFTER the follower's own delayed
    entry fill (`not_before = entry_time + 1h + d_entry`), exactly as the
    baseline does -- a follower that enters late cannot be stopped on price
    action it was not yet in the market for.
  * Entry side is untouched (entry delay d_entry as-is).
  * Sizing is untouched (q is not applied here; decay is measured at unit size).

Pricing reuse: the ONLY change is the value of the per-trade exit latency handed
to the existing pricing engine. Replacing the replication-delayed d_exit with
`L_own` for a stop trade makes `compute_fills` produce
`exit_fill_time = breach + L_own`, and the fill price then comes from the SAME
`_locate_fill` + stop `_clamp` path the master uses (frictions.py). We do not
re-implement any fill/gap/clamp logic here; we call `_price_follower` with a
localised exit-latency vector.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, replace as _dc_replace
import numpy as np
import pandas as pd

from simulator.frictions import FrictionConfig, ONE_HOUR, _find_stop_breach, _draw_latencies
from simulator.reference_strategy import Trade
from replication import delay_models as dm
from replication.followers import (
    run_master, net_bps_per_trade, _price_follower,
    _follower_latency, ReplicationConfig, SIGNAL_SIDE,
)

STOP = "stop"

# Fixed offset applied to fcfg.seed for the follower's own stop-exit latency
# draw. Isolates its RNG stream from both the master's own lat_x draw
# (run_master -> _draw_latencies(n, fcfg), unmodified seed) and the
# replication-delay CRN stream (delay_models.standard_normals(rcfg.seed)).
# Fixed (not random) so the draw is reproducible run to run.
STOP_LATENCY_SEED_OFFSET = 9173


# ---------------------------------------------------------------------------
# The optimisation: localise the stop exit (drop the REPLICATION delay only)
# ---------------------------------------------------------------------------
def stop_exit_mask(trades: list[Trade]) -> np.ndarray:
    """Boolean per-trade mask of stop exits (the only exits that get localised)."""
    return np.array([t.exit_reason == STOP for t in trades], dtype=bool)


def draw_follower_stop_latency(n: int, fcfg: FrictionConfig) -> np.ndarray:
    """Independent draw of the follower's own order-to-fill latency at a
    localised stop breach, from the SAME lognormal distribution the master's
    own exit latency uses (`_draw_latencies`, `FrictionConfig.latency_median_s`
    / `latency_sigma`) but on an INDEPENDENT RNG stream (seed offset by
    STOP_LATENCY_SEED_OFFSET) -- reuses `_draw_latencies` itself (same mu/sigma
    computation) rather than re-deriving the lognormal math. Deterministic in
    (n, fcfg.seed), so it is identical across followers, across the delay grid,
    and between repeated before/after runs (CRN)."""
    stream_cfg = _dc_replace(fcfg, seed=fcfg.seed + STOP_LATENCY_SEED_OFFSET)
    _, lat_x_own = _draw_latencies(n, stream_cfg)
    return lat_x_own


def localise_exit_latency(
    d_exit: np.ndarray, is_stop: np.ndarray, stop_latency: np.ndarray,
) -> np.ndarray:
    """Replace the exit delay with the follower's own execution latency for stop
    exits only; leave signal/eod exits on their normal replication delay. This
    is the *entire* mechanism of the optimisation: it changes only
    `exit_fill_time` inside `compute_fills` (breach + L_own instead of
    breach + d_exit), never the way the fill price is subsequently located or
    clamped."""
    return np.where(is_stop, stop_latency, d_exit)


def price_local_stop_follower(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame, fcfg: FrictionConfig,
    fe: np.ndarray, fx: np.ndarray, is_stop: np.ndarray,
    stop_latency: np.ndarray | None = None,
) -> pd.DataFrame:
    """One follower priced with local stop execution. Thin wrapper: it localises
    the exit-latency vector for stop trades (replication delay -> the
    follower's own independent execution latency) and defers *all* pricing to
    the existing `_price_follower` (same engine, same gap/clamp path as the
    master). `stop_latency` may be precomputed and shared across followers
    (CRN); if omitted it is drawn here via `draw_follower_stop_latency`."""
    if stop_latency is None:
        stop_latency = draw_follower_stop_latency(len(trades), fcfg)
    fx_local = localise_exit_latency(fx, is_stop, stop_latency)
    return _price_follower(trades, h1, m1, fcfg, fe, fx_local)


# ---------------------------------------------------------------------------
# Divergence diagnostics: reuse the master's own breach finder
# ---------------------------------------------------------------------------
def _entry_fill_times(trades: list[Trade], lat_entry: np.ndarray) -> list[pd.Timestamp]:
    """Realized entry fill time per trade, identical to compute_fills:
    entry_time + 1h + entry latency (seconds)."""
    return [t.entry_time + ONE_HOUR + pd.Timedelta(seconds=float(l))
            for t, l in zip(trades, lat_entry)]


def _breach_times(trades: list[Trade], m1: pd.DataFrame,
                  entry_fill_times: list[pd.Timestamp],
                  is_stop: np.ndarray) -> list[pd.Timestamp | None]:
    """Per-trade stop-breach minute (None for non-stop trades, or when no touch is
    found in the exit hour). Reuses the master's `_find_stop_breach` verbatim, with
    the given side's own delayed entry as the `not_before` gate."""
    out: list[pd.Timestamp | None] = []
    for i, t in enumerate(trades):
        if not is_stop[i]:
            out.append(None)
            continue
        out.append(_find_stop_breach(
            m1, t.exit_time, t.direction, t.stop_price,
            not_before=entry_fill_times[i],
        ))
    return out


# ---------------------------------------------------------------------------
# Driver: before/after decay + slippage + divergence, per median delay
# (single asset; one row per delay -- same shape as the Part-B diagnostics)
# ---------------------------------------------------------------------------
@dataclass
class LocalStopConfig:
    """No tunable parameters: local stop execution is a mechanism, not a fitted
    rule (d_exit -> the follower's own execution latency for stops). Kept for
    interface symmetry with the other optimisation drivers."""
    pass


def run_local_stop(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame,
    fcfg: FrictionConfig, rcfg: ReplicationConfig,
) -> pd.DataFrame:
    """One row per median delay for a single asset. 'before' = baseline follower
    (normal replication delay on every exit); 'after' = local stop execution
    (stop exits use the follower's own independent execution latency instead of
    the replication delay). Both are priced from the SAME common-random-number
    draws so the before/after comparison carries no sampling noise.

    Sign convention for divergent_exit_pnl_bps: POSITIVE = follower worse off than
    the master (per-trade bps, averaged over followers)."""
    n = len(trades)
    is_stop = stop_exit_mask(trades)
    pct_stop = float(is_stop.mean() * 100.0)

    master_df, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    m_bps = net_bps_per_trade(master_df)

    # Master breach minute per stop trade -- fixed across followers.
    master_entry_ft = _entry_fill_times(trades, lat_e)
    m_breach = _breach_times(trades, m1, master_entry_ft, is_stop)

    # Follower's own stop-exit execution latency: one draw, independent of the
    # master's lat_x and of the replication-delay CRN below, reused across every
    # follower and every median_s so the before/after comparison isolates the
    # optimisation's effect (dropping the replication delay only).
    stop_latency = draw_follower_stop_latency(n, fcfg)

    z_e, z_x = dm.standard_normals(rcfg.n_followers, n, rcfg.seed)
    F = rcfg.n_followers

    rows = []
    for median_s in rcfg.delay_grid_s:
        d_e_all, d_x_all = dm.follower_delays(median_s, z_e, z_x, rcfg.sigma)
        before = np.empty(F); after = np.empty(F)
        slip_before = np.empty(F); slip_after = np.empty(F)
        div_count = np.empty(F); div_pnl = np.empty(F)

        for f in range(F):
            fe, fx = _follower_latency(SIGNAL_SIDE, d_e_all[f], d_x_all[f], lat_e, lat_x)

            fdf_b = _price_follower(trades, h1, m1, fcfg, fe, fx)          # baseline
            fdf_a = price_local_stop_follower(
                trades, h1, m1, fcfg, fe, fx, is_stop, stop_latency)
            b_bps = net_bps_per_trade(fdf_b)
            a_bps = net_bps_per_trade(fdf_a)

            before[f] = float(np.mean(m_bps - b_bps))
            after[f] = float(np.mean(m_bps - a_bps))
            slip_before[f] = float(fdf_b["exit_slip_bps"].mean())
            slip_after[f] = float(fdf_a["exit_slip_bps"].mean())

            # Divergent stop exits of the localised (after) follower vs the master.
            f_breach = _breach_times(trades, m1, _entry_fill_times(trades, fe), is_stop)
            cnt = 0
            pnl = 0.0
            for i in range(n):
                if not is_stop[i]:
                    continue
                mb, fb = m_breach[i], f_breach[i]
                phantom = (fb is not None) and (mb is None)        # follower stopped, master not
                later = (fb is not None) and (mb is not None) and (fb > mb)  # follower's breach later
                if phantom or later:
                    cnt += 1
                    pnl += float(m_bps[i] - a_bps[i])              # +ve => follower worse
            div_count[f] = cnt
            div_pnl[f] = pnl / n                                   # per-trade bps

        rows.append({
            "median_delay_s":            median_s,
            "pct_stop_exits":            pct_stop,
            "mean_decay_before_bps":     float(before.mean()),
            "mean_decay_after_bps":      float(after.mean()),
            "std_decay_before_bps":      float(before.std(ddof=1)),
            "std_decay_after_bps":       float(after.std(ddof=1)),
            "worst_decay_before_bps":    float(before.max()),
            "worst_decay_after_bps":     float(after.max()),
            "mean_exit_slip_before_bps": float(slip_before.mean()),
            "mean_exit_slip_after_bps":  float(slip_after.mean()),
            "divergent_exit_count":      float(div_count.mean()),
            "divergent_exit_pnl_bps":    float(div_pnl.mean()),
            "net_improvement_bps":       float(before.mean() - after.mean()),
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
        out = run_local_stop(trades, h1, m1, fcfg, rcfg)
        out.insert(0, "symbol", sym)
        frames.append(out)

    results = pathlib.Path("results"); results.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(results / "partB_local_stop.csv", index=False)
    print("wrote results/partB_local_stop.csv")
