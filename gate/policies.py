"""
Part C timeout policies and rejection resolution.

Fixed, non-reversible per-trade evaluation order (see gate/evaluate.py module
docstring for the full modelling rationale):

    1. timeout_i = g_i > tau
    2. if timeout_i:  apply Policy H or Policy S. The rejection flag is never
       consulted here -- there is no gate response to reject, only a missed
       deadline.
    3. if not timeout_i (g_i <= tau): apply the rejection flag. A rejected
       timely response yields no trade; an approved timely response trades at
       full size, priced at signal_time + g_i.

This order guarantees a timeout trade is never additionally blocked by a
reject flag that was only ever meant for the timely-response branch.
"""

from __future__ import annotations

import numpy as np

POLICY_H = "H"   # hard block: no fill on timeout
POLICY_S = "S"   # fail-safe scale-down: half size, priced at signal_time + tau
POLICIES = (POLICY_H, POLICY_S)
POLICY_S_SCALE = 0.5

UNINFORMED = "uninformed"
INFORMED = "informed"
REJECTION_MODES = (UNINFORMED, INFORMED)
INFORMED_REJECT_RATE = 0.02


def is_timeout(g_ms: np.ndarray, tau_ms: float) -> np.ndarray:
    """timeout_i = g_i > tau (section 4/9 of the spec)."""
    return g_ms > tau_ms


def informed_reject_flags(
    baseline_pnl_bps: np.ndarray, symbols: np.ndarray, rate: float = INFORMED_REJECT_RATE,
) -> np.ndarray:
    """Deterministic (no RNG): per instrument, flag the lowest `rate` share of
    baseline PnL trades. Requires baseline_pnl_bps to already be computed --
    the informed flags are a function of the baseline ranking, not an
    independent draw (section 3b/7: baseline must be computed first)."""
    flags = np.zeros(len(baseline_pnl_bps), dtype=bool)
    for sym in np.unique(symbols):
        mask = symbols == sym
        n_sym = int(mask.sum())
        n_reject = int(np.floor(n_sym * rate))
        if n_reject == 0:
            continue
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(baseline_pnl_bps[idx], kind="stable")]  # ascending
        flags[order[:n_reject]] = True
    return flags


def resolve_trade_outcome(
    g_ms: np.ndarray, tau_ms: float, policy: str, reject_flags: np.ndarray,
) -> dict:
    """Per-trade outcome under one (tau, policy, rejection-mode) configuration.

    Returns a dict of length-n arrays:
        timeout           bool          g_i > tau
        blocked           bool          no trade at all (H-timeout or a
                                         timely reject)
        scaled_timeout     bool          Policy-S timeout (half size, not a
                                         full block -- see gate/evaluate.py
                                         section 6 for why this is excluded
                                         from the foregone/avoided split)
        size               float         0.0, 0.5 (Policy-S timeout), or 1.0
        entry_latency_ms   float         latency to feed the pricing pipeline
                                         for non-blocked trades: g_i for an
                                         approved timely response, tau for a
                                         Policy-S timeout. 0.0 (unused, since
                                         size=0) for blocked trades.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")

    timeout = is_timeout(g_ms, tau_ms)
    n = len(g_ms)
    size = np.zeros(n)
    entry_latency_ms = np.zeros(n)
    blocked = np.zeros(n, dtype=bool)
    scaled_timeout = np.zeros(n, dtype=bool)

    # --- timely branch (g_i <= tau): rejection applies, priced at g_i ---
    timely = ~timeout
    approved_timely = timely & ~reject_flags
    rejected_timely = timely & reject_flags
    size[approved_timely] = 1.0
    entry_latency_ms[approved_timely] = g_ms[approved_timely]
    blocked[rejected_timely] = True

    # --- timeout branch (g_i > tau): policy decides, reject flag skipped ---
    if policy == POLICY_H:
        blocked[timeout] = True
    else:  # POLICY_S
        size[timeout] = POLICY_S_SCALE
        entry_latency_ms[timeout] = tau_ms
        scaled_timeout[timeout] = True

    return {
        "timeout": timeout,
        "blocked": blocked,
        "scaled_timeout": scaled_timeout,
        "size": size,
        "entry_latency_ms": entry_latency_ms,
    }
