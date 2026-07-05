"""
Friction models applied to the reference strategy's trade stream.

The strategy produces Level-0 fills (signal-bar close). Each level makes
execution more realistic; levels are cumulative.

    Level 0 - Frictionless: fill at the signal-bar close, no costs. (baseline)

    Level 1 - Spread + commission:
        spread     : realized spread_price from the data, half crossed per side
                     (entry spread/2 + exit spread/2 -> ~one full spread / round trip)
        commission : 0.5 bps of notional, charged per side

    Level 2 - Slippage + latency:
        The signal is only known once the H1 bar CLOSES, so the earliest
        executable price is the M1 path of the FOLLOWING hour. Every order is
        delayed by a master execution latency and then priced from the actual
        M1 bars (the within-hour path is OBSERVED at minute resolution).

        The only MODELED (not observed) piece is the sub-minute fill price:
        with OHLC we cannot see the tick path inside a minute, so we linearly
        interpolate open->close by the elapsed fraction of the minute. This is
        the minimum-assumption estimate of the price at a time offset; it
        ignores intra-minute high/low excursions (a stated limitation).

        Stops are not magic: they TRIGGER when the M1 path first touches the
        stop level and FILL at the stop or worse (gap-aware), never better.
        Orders sent into a closed/gapped market fill at the next traded bar.

        L2 isolates the mid-price drift (delay + latency). Spread/commission are
        charged once at L1, so the bid/ask cross is never double-counted.

    Level 3 - Partial fills (tranched execution, liquidity-aware):
        A 1-unit order does not always clear in one minute. We tranche it across
        consecutive M1 bars, filling a fraction (tickvol_t / median_tickvol) of
        the order per minute (capped at 1). This calibrates the order size to
        "one median-liquidity minute": in normal/thick liquidity it clears in the
        first minute and L3 == L2; in THIN/volatile minutes it tranches over
        several minutes, with later tranches priced at their (worse, later) M1
        mids. The effective fill is the size-weighted average of the tranches.

        MODELED piece = partial-fill sizing (the tickvol->fraction map), since
        OHLC has no order book. A `max_fill_minutes` cap forces completion at the
        last price (we assume the order always eventually clears; true residual
        non-fill / tracking error is noted as an extension, not modeled).

        CONSERVATIVE side: tranching is charged one-sided -- it can only worsen
        the round-trip PnL, never improve it (partial_cost = max(0, pnl_L2 -
        pnl_L3)). Reason: a *favorable* tranche outcome (e.g. catching a crash
        bounce) would require passive limit orders carrying non-fill risk that we
        do not credit, and our mids ignore market impact (out of scope). Charging
        only the adverse side keeps L3 a genuine friction and avoids a free lunch
        in exactly the stress windows where partial fills truly hurt.

Slippage is logged in basis points against the original signal timestamp
(signal_price). Costs are reported in price points (PnL units) and in bps
relative to the entry notional, so they compare across assets.
"""

from __future__ import annotations

# Allow running as a script (python3 simulator/frictions.py) as well as -m.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
import numpy as np
import pandas as pd

from simulator.reference_strategy import Trade

BPS = 1e-4
ONE_HOUR = pd.Timedelta(hours=1)
ONE_MINUTE = pd.Timedelta(minutes=1)


@dataclass
class FrictionConfig:
    # --- Level 1 ---
    commission_bps_per_side: float = 0.5
    # --- Level 2: master execution latency (seconds), lognormal, right-skewed ---
    # Justification: a non-colocated systematic path = decision compute + network
    # round-trip + broker ack/match. Sub-second median with an occasional
    # multi-second tail. On minute bars a sub-second latency lands in the first
    # post-signal minute, so the DELAY (acting one bar late) dominates the drift.
    latency_median_s: float = 0.5
    latency_sigma: float = 0.5
    seed: int = 42
    # --- Level 3: partial-fill tranching ---
    # The order clears in one minute at median liquidity; below median it tranches.
    # max_fill_minutes is a completion vangnet for very thin minutes.
    max_fill_minutes: int = 30
    # partial_one_sided=True (default): conservative adverse-selection penalty --
    # tranching can only WORSEN the round trip (a favorable late tranche would
    # require passive limit orders whose non-fill risk we do not credit). Set
    # False for a SYMMETRIC robustness check where late tranches may land
    # favorably too; used only to show the main conclusion is not an artefact of
    # the one-sided charge.
    partial_one_sided: bool = True


def _sign(direction: str) -> int:
    return 1 if direction == "long" else -1


# ===========================================================================
# LEVEL 0 - Frictionless baseline
#   Fill at the signal-bar close, no costs. "The lie every backtest tells."
# ===========================================================================
def level0_pnl_points(trade: Trade) -> float:
    """Frictionless round-trip PnL in price points (1 unit)."""
    return _sign(trade.direction) * (trade.exit_price - trade.signal_price)


# ===========================================================================
# LEVEL 1 - Spread + commission
#   spread     : realized spread_price, half crossed per side (~1 spread / RT)
#   commission : 0.5 bps of notional, per side
# ===========================================================================
def level1_costs(trade: Trade, h1: pd.DataFrame, config: FrictionConfig) -> tuple[float, float]:
    """Returns (spread_cost, commission_cost) for one trade, in price points."""
    sp_entry = float(h1.at[trade.entry_time, "spread_price"])
    sp_exit = float(h1.at[trade.exit_time, "spread_price"])
    spread_cost = sp_entry / 2 + sp_exit / 2
    commission_cost = (
        config.commission_bps_per_side
        * BPS
        * (abs(trade.signal_price) + abs(trade.exit_price))
    )
    return spread_cost, commission_cost


# ===========================================================================
# LEVEL 2/3 helpers - price fills from the real M1 path
# ===========================================================================
def _draw_latencies(n: int, config: FrictionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-trade entry/exit master latencies (seconds), seeded & reproducible."""
    rng = np.random.default_rng(config.seed)
    mu = np.log(config.latency_median_s)  # lognormal median = exp(mu)
    lat_entry = rng.lognormal(mean=mu, sigma=config.latency_sigma, size=n)
    lat_exit = rng.lognormal(mean=mu, sigma=config.latency_sigma, size=n)
    return lat_entry, lat_exit


def _locate_fill(m1: pd.DataFrame, fill_time: pd.Timestamp) -> tuple[float, int]:
    """
    Price an instantaneous fill at `fill_time` from the M1 path.

    If `fill_time` falls inside a minute bar, linearly interpolate open->close by
    the elapsed fraction (the sub-minute model). If the market is gapped/closed
    (no bar covers `fill_time`), fill at the open of the next traded bar.
    Returns (mid_price, bar_position) where bar_position is the iloc index of the
    minute the fill landed in (used as the start of tranched execution at L3).
    """
    idx = m1.index
    pos = idx.searchsorted(fill_time, side="right") - 1  # last bar start <= fill_time
    if pos >= 0:
        bar_start = idx[pos]
        elapsed = (fill_time - bar_start).total_seconds()
        if 0 <= elapsed < 60:
            bar = m1.iloc[pos]
            f = elapsed / 60.0
            mid = bar["open"] + f * (bar["close"] - bar["open"])
            return float(mid), int(pos)
    # gap / closed market -> fill at the next available traded bar's open
    nxt = idx.searchsorted(fill_time, side="left")
    if nxt < len(idx):
        return float(m1.iloc[nxt]["open"]), int(nxt)
    return float(m1.iloc[-1]["close"]), len(m1) - 1  # no data after: last close


def _tranche_fill(
    m1: pd.DataFrame, start_pos: int, first_price: float,
    median_tickvol: float, max_minutes: int,
) -> float:
    """
    Size-weighted fill price when a 1-unit order is tranched across M1 bars.

    Fraction fillable in minute m = min(1, tickvol_m / median_tickvol). The
    trigger minute is priced at `first_price` (the L2 sub-minute price); later
    minutes at their M1 mid (open+close)/2. Any residual after `max_minutes`
    clears at the last price (forced completion).
    """
    o = m1["open"].to_numpy()
    c = m1["close"].to_numpy()
    tv = m1["tickvol"].to_numpy()
    n = len(m1)

    remaining = 1.0
    weighted = 0.0
    last_price = first_price
    for step in range(max_minutes):
        j = start_pos + step
        if j >= n:
            break
        cap = min(1.0, tv[j] / median_tickvol) if median_tickvol > 0 else 1.0
        price = first_price if step == 0 else (o[j] + c[j]) / 2.0
        fill = min(remaining, cap)
        weighted += fill * price
        remaining -= fill
        last_price = price
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:  # forced completion of the residual
        weighted += remaining * last_price
    return float(weighted)


def _find_stop_breach(
    m1: pd.DataFrame, hour_start: pd.Timestamp, direction: str,
    stop: float, not_before: pd.Timestamp,
) -> pd.Timestamp | None:
    """First M1 bar within [hour_start, hour_start+1h) (and >= not_before) whose
    path touches the stop level. Returns its timestamp, or None if not found."""
    window = m1.loc[hour_start: hour_start + ONE_HOUR - ONE_MINUTE]
    window = window[window.index >= not_before]
    if window.empty:
        return None
    if direction == "long":
        hit = window.index[window["low"] <= stop]
    else:
        hit = window.index[window["high"] >= stop]
    return hit[0] if len(hit) else None


def compute_fills(
    trade: Trade, m1: pd.DataFrame, lat_entry_s: float, lat_exit_s: float,
    tranche: bool = False, median_tickvol: float = 0.0, max_minutes: int = 30,
) -> dict:
    """
    Realistic entry/exit fills from the M1 path.

    Always returns the Level-2 instantaneous fills (entry_l2/exit_l2). When
    `tranche` is True, also returns the Level-3 tranched fills (entry_l3/exit_l3).
    Stop exits clamp the effective fill to the stop-or-worse (never better).
    """
    # ENTRY: signal known at H1 close (= entry_time + 1h); fill after latency.
    entry_fill_time = trade.entry_time + ONE_HOUR + pd.Timedelta(seconds=lat_entry_s)
    entry_l2, entry_pos = _locate_fill(m1, entry_fill_time)

    # EXIT trigger time
    if trade.exit_reason == "stop":
        breach = _find_stop_breach(
            m1, trade.exit_time, trade.direction, trade.stop_price,
            not_before=entry_fill_time,
        )
        if breach is None:  # defensive: H1 said stop but M1 slice missing
            breach = trade.exit_time + ONE_HOUR
        exit_fill_time = breach + pd.Timedelta(seconds=lat_exit_s)
    else:  # signal / eod exit: known at the exit H1 bar close
        exit_fill_time = trade.exit_time + ONE_HOUR + pd.Timedelta(seconds=lat_exit_s)
    exit_l2_raw, exit_pos = _locate_fill(m1, exit_fill_time)

    def _clamp(price: float) -> float:
        if trade.exit_reason != "stop":
            return price
        return min(trade.stop_price, price) if trade.direction == "long" \
            else max(trade.stop_price, price)

    out = {
        "entry_l2": entry_l2, "exit_l2": _clamp(exit_l2_raw),
        # Realized fill timestamps (already resolve the latency-dependent stop
        # breach). Exposed so replication diagnostics can diff follower vs master
        # fill times over entries and exits without re-deriving the timing.
        "entry_fill_time": entry_fill_time, "exit_fill_time": exit_fill_time,
    }
    if tranche:
        entry_l3 = _tranche_fill(m1, entry_pos, entry_l2, median_tickvol, max_minutes)
        exit_l3_raw = _tranche_fill(m1, exit_pos, exit_l2_raw, median_tickvol, max_minutes)
        out["entry_l3"] = entry_l3
        out["exit_l3"] = _clamp(exit_l3_raw)
    return out


# ===========================================================================
# Orchestration - stack the levels onto the trade stream
# ===========================================================================
def apply_frictions(
    trades: list[Trade],
    h1: pd.DataFrame,
    level: int = 1,
    config: FrictionConfig | None = None,
    m1: pd.DataFrame | None = None,
    lat_entry: np.ndarray | None = None,
    lat_exit: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Returns a per-trade DataFrame with cost attribution and net PnL.

    Latency injection (Part B): if `lat_entry`/`lat_exit` (per-trade seconds) are
    supplied they are used verbatim instead of the internal master draw. This is
    the single hook that lets a follower account reuse the exact same pricing
    path with a different latency (e.g. signal-side = follower delay; fill-side =
    master latency + follower delay). When omitted, the master latency is drawn
    from `config` as before, so default behaviour is unchanged.

    Columns (points unless noted):
        level0_pnl        frictionless PnL
        spread_cost       half-spread crossed on entry + exit          (L1)
        commission_cost   0.5 bps/side notional                        (L1)
        slippage_cost     delay + latency mid-price drift              (L2)
        partial_cost      extra cost from tranched / partial fills     (L3)
        total_cost        spread + commission + slippage + partial
        net_pnl           level0_pnl - total_cost
        cost_bps          total_cost / signal_price, in bps
        entry/exit_fill   realistic fill prices for the active level
        entry/exit_slip_bps  per-fill slippage vs theoretical, in bps
    """
    if config is None:
        config = FrictionConfig()
    if level >= 2 and m1 is None:
        raise ValueError("Level >= 2 requires the M1 DataFrame (m1=...).")

    median_tickvol = 0.0
    if level >= 2 and (lat_entry is None or lat_exit is None):
        lat_entry, lat_exit = _draw_latencies(len(trades), config)
    if level >= 3:
        median_tickvol = float(m1["tickvol"].median())

    rows = []
    for i, t in enumerate(trades):
        l0 = level0_pnl_points(t)
        sgn = _sign(t.direction)

        spread_cost = commission_cost = slippage_cost = partial_cost = 0.0
        entry_fill, exit_fill = t.signal_price, t.exit_price
        entry_fill_time = exit_fill_time = pd.NaT

        if level >= 1:
            spread_cost, commission_cost = level1_costs(t, h1, config)

        if level >= 2:
            f = compute_fills(
                t, m1, lat_entry[i], lat_exit[i],
                tranche=(level >= 3),
                median_tickvol=median_tickvol,
                max_minutes=config.max_fill_minutes,
            )
            # Level 2 attribution: drift of the instantaneous fills vs theory.
            pnl_l2 = sgn * (f["exit_l2"] - f["entry_l2"])
            slippage_cost = l0 - pnl_l2
            entry_fill, exit_fill = f["entry_l2"], f["exit_l2"]
            entry_fill_time, exit_fill_time = f["entry_fill_time"], f["exit_fill_time"]

            if level >= 3:
                # Level 3 attribution: tranched fills vs the L2 instantaneous fills.
                # Default is the conservative adverse-selection penalty (charged
                # one-sided: tranching can only worsen the round trip). With
                # partial_one_sided=False the charge is symmetric -- late tranches
                # may also land favorably -- a robustness variant, not the default.
                pnl_l3 = sgn * (f["exit_l3"] - f["entry_l3"])
                raw_partial = pnl_l2 - pnl_l3
                partial_cost = max(0.0, raw_partial) if config.partial_one_sided else raw_partial
                if partial_cost != 0.0:
                    entry_fill, exit_fill = f["entry_l3"], f["exit_l3"]

        # Per-fill slippage vs theoretical (Level-0) prices, adverse-positive.
        inv = 1.0 / abs(t.signal_price) / BPS
        entry_slip_bps = sgn * (entry_fill - t.signal_price) * inv
        exit_slip_bps = sgn * (t.exit_price - exit_fill) * inv

        total_cost = spread_cost + commission_cost + slippage_cost + partial_cost
        rows.append({
            "symbol":          t.symbol,
            "direction":       t.direction,
            "entry_time":      t.entry_time,
            "exit_time":       t.exit_time,
            "exit_reason":     t.exit_reason,
            "signal_price":    t.signal_price,
            "exit_price":      t.exit_price,
            "entry_fill":      entry_fill,
            "exit_fill":       exit_fill,
            "entry_fill_time": entry_fill_time,
            "exit_fill_time":  exit_fill_time,
            "level0_pnl":      l0,
            "spread_cost":     spread_cost,
            "commission_cost": commission_cost,
            "slippage_cost":   slippage_cost,
            "partial_cost":    partial_cost,
            "total_cost":      total_cost,
            "net_pnl":         l0 - total_cost,
            "cost_bps":        total_cost / abs(t.signal_price) / BPS,
            "entry_slip_bps":  entry_slip_bps,
            "exit_slip_bps":   exit_slip_bps,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate cost/PnL summary for a per-trade frictions DataFrame."""
    inv_notional = 1.0 / df["signal_price"].abs()
    return {
        "n_trades":           len(df),
        "level0_pnl_points":  df["level0_pnl"].sum(),
        "net_pnl_points":     df["net_pnl"].sum(),
        "spread_cost_points": df["spread_cost"].sum(),
        "commission_points":  df["commission_cost"].sum(),
        "slippage_points":    df["slippage_cost"].sum(),
        "partial_points":     df["partial_cost"].sum(),
        "total_cost_points":  df["total_cost"].sum(),
        "avg_cost_bps":       df["cost_bps"].mean(),
        "avg_spread_bps":     (df["spread_cost"] * inv_notional / BPS).mean(),
        "avg_commission_bps": (df["commission_cost"] * inv_notional / BPS).mean(),
        "avg_slippage_bps":   (df["slippage_cost"] * inv_notional / BPS).mean(),
        "avg_partial_bps":    (df["partial_cost"] * inv_notional / BPS).mean(),
        "avg_entry_slip_bps": df["entry_slip_bps"].mean(),
        "avg_exit_slip_bps":  df["exit_slip_bps"].mean(),
    }


if __name__ == "__main__":
    from simulator.data_loader import load_m1, resample_h1, TICK_SIZE
    from simulator import reference_strategy as strat

    cfg = FrictionConfig()

    print("Part A - friction waterfall per asset (Level 0 -> Level 3)")
    print("(average cost per round-trip trade, in bps of entry notional)\n")
    header = (f"{'asset':7s} {'n':>5s} | {'spread':>7s} {'commis':>7s} "
              f"{'slippage':>8s} {'partial':>8s} | {'TOTAL':>7s} | dominant")
    print(header)
    print("-" * len(header))

    for sym in TICK_SIZE:
        m1 = load_m1(sym)
        h1 = resample_h1(m1)
        trades = strat.run(h1, symbol=sym)
        s = summarize(apply_frictions(trades, h1, level=3, config=cfg, m1=m1))

        parts = {
            "spread": s["avg_spread_bps"],
            "commission": s["avg_commission_bps"],
            "slippage": s["avg_slippage_bps"],
            "partial": s["avg_partial_bps"],
        }
        dominant = max(parts, key=parts.get)
        print(f"{sym:7s} {s['n_trades']:5d} | "
              f"{s['avg_spread_bps']:7.3f} {s['avg_commission_bps']:7.3f} "
              f"{s['avg_slippage_bps']:8.3f} {s['avg_partial_bps']:8.3f} | "
              f"{s['avg_cost_bps']:7.3f} | {dominant}")

    print("\nNote: PnL in price points is not comparable across assets (different")
    print("price scales); bps of notional is. The full per-asset waterfall figure")
    print("and CSVs are produced by engine.py.")
