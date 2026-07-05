"""
Loads MetaTrader M1 CSVs and resamples to H1.

SPREAD column is in points (tick-size units), not basis points.
Tick sizes: SPXUSD=0.01, all others=0.001.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

TICK_SIZE = {
    "SPXUSD": 0.01,
    "USDJPY": 0.001,
    "XAUUSD": 0.001,
    "ETHUSD": 0.001,
}

_FILE_PREFIX = {
    "SPXUSD": "SPXUSD_M1",
    "USDJPY": "USDJPY.l_M1",
    "XAUUSD": "XAUUSD.l_M1",
    "ETHUSD": "ETHUSD_M1",
}


def _find_csv(symbol: str) -> Path:
    prefix = _FILE_PREFIX[symbol]
    matches = list(DATA_DIR.glob(f"{prefix}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV found for {symbol} in {DATA_DIR}")
    return matches[0]


def load_m1(symbol: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by UTC timestamp with columns:
      open, high, low, close, tickvol, vol, spread_pts, spread_price

    spread_price = spread_pts * tick_size  (in price units)
    """
    path = _find_csv(symbol)
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S"
    ).dt.as_unit("ns")  # ns resolution so sub-second fill latency stays comparable
    df = df.drop(columns=["date", "time"]).set_index("timestamp")
    df.rename(columns={"spread": "spread_pts"}, inplace=True)
    df["spread_price"] = df["spread_pts"] * TICK_SIZE[symbol]
    return df


def resample_h1(m1: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples M1 bars to H1 using standard OHLC aggregation.
    spread_pts/spread_price are averaged; tickvol and vol are summed.
    Empty hours (no ticks) are dropped.
    """
    agg = {
        "open":         "first",
        "high":         "max",
        "low":          "min",
        "close":        "last",
        "tickvol":      "sum",
        "vol":          "sum",
        "spread_pts":   "mean",
        "spread_price": "mean",
    }
    h1 = m1.resample("1h").agg(agg)
    return h1.dropna(subset=["open"])


def load_all(symbols: list[str] | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Returns {symbol: {"m1": df_m1, "h1": df_h1}} for each symbol.
    Defaults to all four instruments.
    """
    if symbols is None:
        symbols = list(TICK_SIZE.keys())
    result = {}
    for sym in symbols:
        m1 = load_m1(sym)
        result[sym] = {"m1": m1, "h1": resample_h1(m1)}
    return result


if __name__ == "__main__":
    data = load_all()
    for sym, frames in data.items():
        m1, h1 = frames["m1"], frames["h1"]
        print(f"\n{sym}")
        print(f"  M1 : {len(m1):>8,} bars  {m1.index[0]} -> {m1.index[-1]}")
        print(f"  H1 : {len(h1):>8,} bars  {h1.index[0]} -> {h1.index[-1]}")
        print(f"  spread_price  mean={m1['spread_price'].mean():.5f}  max={m1['spread_price'].max():.5f}")
