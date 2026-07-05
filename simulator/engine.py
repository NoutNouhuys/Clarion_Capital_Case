"""
Part A engine - the reproducible deliverable.

Runs the reference strategy + the Level 0->3 friction stack over all four
assets, then writes:
    results/partA_summary.csv          per-asset cost & PnL summary
    results/partA_trades_<SYMBOL>.csv  per-trade detail (fills, slippage, costs)
    results/partA_waterfall.png        per-asset Level 0->3 decay waterfall

It prints a summary table and is deterministic under the FrictionConfig seed.

    python simulator/engine.py            # all assets, default config
    python simulator/engine.py --symbols SPXUSD ETHUSD
"""

from __future__ import annotations

# Allow running as a script (python3 simulator/engine.py) as well as -m.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render straight to file
import matplotlib.pyplot as plt
import pandas as pd

from simulator.data_loader import load_m1, resample_h1, TICK_SIZE
from simulator import reference_strategy as strat
from simulator.frictions import apply_frictions, summarize, FrictionConfig

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Friction components shown in the waterfall, in cumulative order.
COMPONENTS = [
    ("spread",     "avg_spread_bps"),
    ("commission", "avg_commission_bps"),
    ("slippage",   "avg_slippage_bps"),
    ("partial",    "avg_partial_bps"),
]


def run_asset(symbol: str, config: FrictionConfig) -> tuple[pd.DataFrame, dict]:
    """Run strategy + full friction stack for one symbol; return (trades_df, summary)."""
    m1 = load_m1(symbol)
    h1 = resample_h1(m1)
    trades = strat.run(h1, symbol=symbol)
    df = apply_frictions(trades, h1, level=3, config=config, m1=m1)
    s = summarize(df)

    # Cumulative PnL by level, in price points (derived from the single L3 run).
    s["L0_points"] = s["level0_pnl_points"]
    s["L1_points"] = s["L0_points"] - s["spread_cost_points"] - s["commission_points"]
    s["L2_points"] = s["L1_points"] - s["slippage_points"]
    s["L3_points"] = s["L2_points"] - s["partial_points"]  # == net_pnl_points

    parts = {name: s[key] for name, key in COMPONENTS}
    s["symbol"] = symbol
    s["dominant"] = max(parts, key=parts.get)
    return df, s


def make_waterfall(summaries: list[dict], path: Path) -> None:
    """2x2 grid of per-asset Level 0->3 execution-cost waterfalls (bps / round trip)."""
    n = len(summaries)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for ax, s in zip(axes.flat, summaries):
        labels = [name for name, _ in COMPONENTS]
        values = [s[key] for _, key in COMPONENTS]
        cum = 0.0
        for label, val in zip(labels, values):
            ax.bar(label, val, bottom=cum,
                   color="tab:red" if val >= 0 else "tab:green")
            cum += val
        ax.bar("TOTAL", cum, color="tab:blue")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{s['symbol']}  (total {cum:.2f} bps, dom: {s['dominant']})")
        ax.set_ylabel("execution cost (bps / round trip)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes.flat[n:]:  # hide unused panels
        ax.set_visible(False)

    fig.suptitle("Part A - Execution reality gap: Level 0 -> 3 cost waterfall per asset",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Part A execution-gap engine")
    ap.add_argument("--symbols", nargs="+", default=list(TICK_SIZE),
                    help="symbols to run (default: all four)")
    ap.add_argument("--seed", type=int, default=FrictionConfig.seed)
    args = ap.parse_args()

    config = FrictionConfig(seed=args.seed)
    RESULTS.mkdir(exist_ok=True)

    summaries = []
    for sym in args.symbols:
        df, s = run_asset(sym, config)
        df.to_csv(RESULTS / f"partA_trades_{sym}.csv", index=False)
        summaries.append(s)

    summary_cols = ["symbol", "n_trades", "L0_points", "L1_points", "L2_points", "L3_points",
                    "avg_spread_bps", "avg_commission_bps", "avg_slippage_bps",
                    "avg_partial_bps", "avg_cost_bps", "dominant"]
    summary_df = pd.DataFrame(summaries)[summary_cols]
    summary_df.to_csv(RESULTS / "partA_summary.csv", index=False)

    make_waterfall(summaries, RESULTS / "partA_waterfall.png")
    for sym in args.symbols:
        print(f"wrote results/partA_trades_{sym}.csv")
    print(f"wrote results/partA_summary.csv ({len(summary_df)} rows)")
    print("wrote results/partA_waterfall.png")


if __name__ == "__main__":
    main()
