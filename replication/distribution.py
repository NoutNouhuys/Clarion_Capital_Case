"""
Part B, subpart 2: follower performance distribution vs the master.

B1 established that on M1 data signal-side and fill-side replication differ by at
most ~0.06 bps, so the distribution analysis uses *signal-side* replication as the
main case (fill-side is produced in the output files and is economically
indistinguishable). For each median delay on the grid we re-price all 25
followers, express each follower's outcome as its decay vs the master (in bps),
and summarise the *distribution* -- not just the mean.

The message is that replication delay is less a stable average tax than a
dispersion and tail-risk amplifier: the mean decay can be small (or even
favourable at sub-second delays), while the spread across followers and the
worst-case follower grow monotonically with the delay.

Outputs:
    results/partB_distribution.csv         per (asset, median) distribution stats
    results/partB_eth_distribution.png     ETHUSD decay vs delay (mean, p10-p90, worst)
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
    run_master, run_followers, net_bps_per_trade, ReplicationConfig, SIGNAL_SIDE,
)


def follower_decay_distribution(
    trades: list[Trade], h1: pd.DataFrame, m1: pd.DataFrame,
    fcfg: FrictionConfig, rcfg: ReplicationConfig,
) -> tuple[float, dict[float, np.ndarray]]:
    """Signal-side per-follower decay (bps vs master) at each median delay.

    Returns (master_bps, {median_s: decay_array of shape (n_followers,)}), where
    decay = master_bps - follower_mean_bps (positive = worse than the master)."""
    master_df, lat_e, lat_x = run_master(trades, h1, m1, fcfg)
    master_bps = net_bps_per_trade(master_df).mean()
    z_e, z_x = dm.standard_normals(rcfg.n_followers, len(trades), rcfg.seed)
    out: dict[float, np.ndarray] = {}
    for median_s in rcfg.delay_grid_s:
        foll_bps = run_followers(trades, h1, m1, fcfg, rcfg, median_s,
                                 SIGNAL_SIDE, lat_e, lat_x, z_e, z_x)
        out[median_s] = master_bps - foll_bps
    return master_bps, out


def distribution_stats(symbol: str, master_bps: float,
                       decays: dict[float, np.ndarray]) -> pd.DataFrame:
    """One row per median delay: mean/std/p10/p90/worst decay and the share of
    followers worse than the master."""
    rows = []
    for median_s, dec in decays.items():
        rows.append({
            "symbol":        symbol,
            "median_delay_s": median_s,
            "master_bps":    master_bps,
            "mean_decay_bps": float(dec.mean()),
            "std_decay_bps": float(dec.std(ddof=1)),
            "p10_decay_bps": float(np.percentile(dec, 10)),
            "p90_decay_bps": float(np.percentile(dec, 90)),
            "worst_decay_bps": float(dec.max()),   # largest decay = worst follower
            "share_worse_than_master": float((dec > 0.0).mean()),
        })
    return pd.DataFrame(rows)


def plot_eth_distribution(stats: pd.DataFrame, path) -> None:
    """ETHUSD decay vs median delay: mean line, p10-p90 band, worst-follower line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = stats[stats["symbol"] == "ETHUSD"].sort_values("median_delay_s")
    x = s["median_delay_s"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Master sits at zero by construction: this is INCREMENTAL replication decay
    # relative to the Level-2 master, not total execution cost (the master already
    # carries Part-A Level-2 frictions).
    ax.axhline(0, color="black", lw=0.8, label="Level-2 master benchmark")
    ax.fill_between(x, s["p10_decay_bps"], s["p90_decay_bps"],
                    color="tab:blue", alpha=0.20, label="p10-p90 across followers")
    ax.plot(x, s["mean_decay_bps"], "o-", color="tab:blue", label="mean follower decay")
    ax.plot(x, s["worst_decay_bps"], "s--", color="tab:red", label="worst follower")
    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_xlabel("median replication delay (seconds, log scale)")
    ax.set_ylabel("incremental decay vs Level-2 master (bps / round trip)")
    ax.set_title("ETHUSD - incremental follower replication decay vs delay")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import pathlib
    from simulator.data_loader import load_m1, resample_h1, TICK_SIZE
    from simulator import reference_strategy as strat

    fcfg = FrictionConfig(); rcfg = ReplicationConfig()
    frames = []
    for sym in TICK_SIZE:
        m1 = load_m1(sym); h1 = resample_h1(m1); trades = strat.run(h1, symbol=sym)
        master_bps, decays = follower_decay_distribution(trades, h1, m1, fcfg, rcfg)
        stats = distribution_stats(sym, master_bps, decays)
        frames.append(stats)

    allstats = pd.concat(frames, ignore_index=True)
    results = pathlib.Path("results"); results.mkdir(exist_ok=True)
    allstats.to_csv(results / "partB_distribution.csv", index=False)
    print(f"wrote results/partB_distribution.csv ({len(allstats)} rows)")
    plot_eth_distribution(allstats, results / "partB_eth_distribution.png")
    print("wrote results/partB_eth_distribution.png")
