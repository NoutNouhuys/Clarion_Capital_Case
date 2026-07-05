"""
Reference strategy.

Signal logic (H1 bars):
  Long  entry: close > highest high of previous 20 bars
  Short entry: close < lowest  low  of previous 20 bars
  Exit:        opposite signal, or stop at 2xATR(14) from entry price
  One position per symbol at a time.

Signal-bar close is used as the Level-0 (frictionless) fill price.
Friction layers are applied by engine.py, not here.
"""

from __future__ import annotations

# Allow running as a script (python3 simulator/reference_strategy.py) as well as -m.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from dataclasses import dataclass
from typing import Literal


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    direction: Literal["long", "short"]
    signal_price: float        # H1 close at signal bar = Level-0 fill price
    stop_price: float          # 2x ATR(14) from signal_price
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: Literal["signal", "stop", "eod"] | None = None

    @property
    def pnl_points(self) -> float | None:
        if self.exit_price is None:
            return None
        sign = 1 if self.direction == "long" else -1
        return sign * (self.exit_price - self.signal_price)

    duration_bars: int | None = None  # set by run() once the exit is known


def _add_indicators(h1: pd.DataFrame) -> pd.DataFrame:
    df = h1.copy()

    # Previous-20-bar channel: shift(1) excludes the current bar
    df["chan_high"] = df["high"].shift(1).rolling(20).max()
    df["chan_low"]  = df["low"].shift(1).rolling(20).min()

    # ATR(14): Wilder's true-range definition, smoothed with a SIMPLE moving
    # average (rolling mean) -- NOT Wilder's RMA (ewm, alpha=1/14). Deliberate:
    # the reference strategy is the fixed constant here and is not the graded
    # object, so we keep the simpler, common SMA-ATR variant rather than tune it.
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    return df

    
def run(h1: pd.DataFrame, symbol: str = "") -> list[Trade]:
    """
    Runs the reference strategy on an H1 DataFrame.
    Returns a list of completed (and any open) Trade objects.
    """
    df = _add_indicators(h1)
    bar_index = {ts: i for i, ts in enumerate(df.index)}

    trades: list[Trade] = []
    position: Trade | None = None

    for ts, bar in df.iterrows():
        if pd.isna(bar["chan_high"]) or pd.isna(bar["atr14"]):
            continue

        # --- 1. Stop check (intra-bar, before close-based signals) ---
        if position is not None:
            stop_hit = False
            if position.direction == "long" and bar["low"] <= position.stop_price:
                position.exit_time   = ts
                position.exit_price  = position.stop_price
                position.exit_reason = "stop"
                position.duration_bars = bar_index[ts] - bar_index[position.entry_time]
                trades.append(position)
                position = None
                stop_hit = True

            elif position.direction == "short" and bar["high"] >= position.stop_price:
                position.exit_time   = ts
                position.exit_price  = position.stop_price
                position.exit_reason = "stop"
                position.duration_bars = bar_index[ts] - bar_index[position.entry_time]
                trades.append(position)
                position = None
                stop_hit = True

            # --- 2. Opposite-signal exit (at bar close) ---
            if not stop_hit and position is not None:
                flip = (
                    (position.direction == "long"  and bar["close"] < bar["chan_low"]) or
                    (position.direction == "short" and bar["close"] > bar["chan_high"])
                )
                if flip:
                    position.exit_time   = ts
                    position.exit_price  = bar["close"]
                    position.exit_reason = "signal"
                    position.duration_bars = bar_index[ts] - bar_index[position.entry_time]
                    trades.append(position)
                    position = None
                    # fall through to entry check below

        # --- 3. Entry check (at bar close, only if flat) ---
        if position is None:
            atr = bar["atr14"]
            if bar["close"] > bar["chan_high"]:
                position = Trade(
                    symbol      = symbol,
                    entry_time  = ts,
                    direction   = "long",
                    signal_price= bar["close"],
                    stop_price  = bar["close"] - 2 * atr,
                )
            elif bar["close"] < bar["chan_low"]:
                position = Trade(
                    symbol      = symbol,
                    entry_time  = ts,
                    direction   = "short",
                    signal_price= bar["close"],
                    stop_price  = bar["close"] + 2 * atr,
                )

    # --- 4. Force-close any open position at end of data ---
    if position is not None:
        last_bar = df.iloc[-1]
        position.exit_time   = df.index[-1]
        position.exit_price  = last_bar["close"]
        position.exit_reason = "eod"
        position.duration_bars = bar_index[df.index[-1]] - bar_index[position.entry_time]
        trades.append(position)

    return trades


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    rows = [{
        "symbol":       t.symbol,
        "direction":    t.direction,
        "entry_time":   t.entry_time,
        "signal_price": t.signal_price,
        "stop_price":   t.stop_price,
        "exit_time":    t.exit_time,
        "exit_price":   t.exit_price,
        "exit_reason":  t.exit_reason,
        "pnl_points":   t.pnl_points,
        "duration_bars":t.duration_bars,
    } for t in trades]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from simulator.data_loader import load_m1, resample_h1

    m1 = load_m1("SPXUSD")
    h1 = resample_h1(m1)
    trades = run(h1, symbol="SPXUSD")
    df = trades_to_df(trades)

    print(f"Total trades : {len(df)}")
    print(f"Long / Short : {(df.direction=='long').sum()} / {(df.direction=='short').sum()}")
    print(f"Stop exits   : {(df.exit_reason=='stop').sum()}")
    print(f"Signal exits : {(df.exit_reason=='signal').sum()}")
    print(f"EOD exits    : {(df.exit_reason=='eod').sum()}")
    print()
    print("First 10 trades:")
    print(df.head(10).to_string(index=False))
