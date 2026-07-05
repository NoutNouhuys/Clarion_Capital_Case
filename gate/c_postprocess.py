"""
Part C post-processing: figures/tables derived from the already-computed
results/partC_gate_policy_grid.csv, plus the stress-window dates that
gate/evaluate.py computes internally but does not persist on its own.

Reuses gate/evaluate.instrument_stress_windows and gate/evaluate.COVID_START/
COVID_END verbatim -- does NOT recompute the grid, redraw any latency/reject
samples, or touch gate/latency_model.py, gate/policies.py, gate/evaluate.py.

Outputs:
    results/partC_policy_map.png       net_impact_bps vs tau_ms, Policy H vs S,
                                    one subplot per instrument (uninformed
                                    rejection only; median regime by linestyle)
    results/partC_policy_summary.csv   per (instrument, median, tau): which policy
                                    wins under uninformed rejection, and by
                                    how much
    results/partC_stress_windows.csv   COVID + each instrument's own top-5%-vol
                                    clustered windows, as concrete dates
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys, pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import pathlib

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

from simulator.data_loader import load_m1, TICK_SIZE
from gate.evaluate import instrument_stress_windows, COVID_START, COVID_END
from gate.policies import POLICY_H, POLICY_S, UNINFORMED

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
GRID_CSV = RESULTS / "partC_gate_policy_grid.csv"


# ---------------------------------------------------------------------------
# 1. Policy map: net_impact_bps vs tau_ms, H vs S, per instrument
# ---------------------------------------------------------------------------
def plot_policy_map(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    uninformed = df[df["rejection_mode"] == UNINFORMED]
    instruments = list(TICK_SIZE.keys())
    medians = sorted(uninformed["gate_median_ms"].unique())
    tau_ticks = sorted(uninformed["tau_ms"].unique())
    linestyles = {medians[0]: "-", medians[-1]: "--"}
    colors = {POLICY_H: "tab:red", POLICY_S: "tab:blue"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, sym in zip(axes.flat, instruments):
        sub = uninformed[uninformed["instrument"] == sym]
        for median_ms in medians:
            for policy in (POLICY_H, POLICY_S):
                row = sub[(sub["gate_median_ms"] == median_ms) & (sub["policy"] == policy)]
                row = row.sort_values("tau_ms")
                ax.plot(row["tau_ms"], row["net_impact_bps"],
                        color=colors[policy], linestyle=linestyles[median_ms],
                        marker="o", markersize=3,
                        label=f"{policy}, median={median_ms:.0f}ms")
        # y=0 reference line: consistently visible in every panel so a
        # policy's net win/loss vs. the ungated baseline is legible at a glance.
        ax.axhline(0.0, color="grey", linewidth=0.8, zorder=0)

        # Log-scale x-axis: the tau grid is not linearly spaced and most of
        # the interesting variation sits compressed at the low end. Explicit
        # ticks at exactly the evaluated tau values, not matplotlib's
        # automatic log ticks/minor ticks.
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(tau_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in tau_ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelbottom=True)  # force visible x-tick labels on every
                                          # subplot regardless of grid row (no
                                          # sharex, so nothing to override here,
                                          # but explicit per the requirement)

        ax.set_title(sym)
        ax.set_xlabel("tau (ms)")
        ax.set_ylabel("net impact (bps, summed)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Part C: net performance impact vs. timeout budget "
                "(uninformed rejection)", y=0.99)
    # Panels are NOT on a shared y-axis (instrument-level edge magnitude
    # differs by orders of magnitude) -- say so explicitly so absolute
    # heights are not misread as comparable across panels.
    fig.text(0.5, 0.935,
             "Note: y-axis scale differs per panel due to large differences "
             "in instrument-level edge magnitude.",
             ha="center", fontsize=7.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Condensed policy-comparison table (H vs S winner per tau, uninformed)
# ---------------------------------------------------------------------------
def build_policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    uninformed = df[df["rejection_mode"] == UNINFORMED]
    rows = []
    for (sym, median_ms), grp in uninformed.groupby(["instrument", "gate_median_ms"]):
        baseline = float(grp["baseline_pnl_bps"].iloc[0])
        for tau_ms, tau_grp in grp.groupby("tau_ms"):
            h_impact = float(tau_grp.loc[tau_grp["policy"] == POLICY_H, "net_impact_bps"].iloc[0])
            s_impact = float(tau_grp.loc[tau_grp["policy"] == POLICY_S, "net_impact_bps"].iloc[0])
            winner = POLICY_H if h_impact >= s_impact else POLICY_S
            margin_bps = abs(h_impact - s_impact)
            rows.append({
                "instrument": sym, "gate_median_ms": median_ms, "tau_ms": tau_ms,
                "baseline_pnl_bps": baseline,
                "net_impact_H_bps": h_impact, "net_impact_S_bps": s_impact,
                "winning_policy": winner, "margin_bps": margin_bps,
            })
    out = pd.DataFrame(rows).sort_values(["instrument", "gate_median_ms", "tau_ms"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Stress-window dates (COVID + instrument-own), reusing gate/evaluate.py
# ---------------------------------------------------------------------------
def build_stress_windows_table() -> pd.DataFrame:
    rows = []
    for sym in TICK_SIZE:
        rows.append({
            "instrument": sym, "window_type": "covid",
            "start_date": COVID_START.date().isoformat(),
            "end_date": COVID_END.date().isoformat(),
        })
        m1 = load_m1(sym)
        for start, end in instrument_stress_windows(m1):
            rows.append({
                "instrument": sym, "window_type": "own_stress",
                "start_date": pd.Timestamp(start).date().isoformat(),
                "end_date": pd.Timestamp(end).date().isoformat(),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(GRID_CSV)

    plot_policy_map(df, RESULTS / "partC_policy_map.png")
    print("wrote results/partC_policy_map.png")

    summary = build_policy_summary(df)
    summary.to_csv(RESULTS / "partC_policy_summary.csv", index=False)
    print(f"wrote results/partC_policy_summary.csv ({len(summary)} rows)")

    windows = build_stress_windows_table()
    windows.to_csv(RESULTS / "partC_stress_windows.csv", index=False)
    print(f"wrote results/partC_stress_windows.csv ({len(windows)} rows)")
