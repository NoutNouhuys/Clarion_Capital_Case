"""
Part C gate: gate-decision latency and the uninformed-rejection draw.

z_i (one standard normal per master ENTRY order, drawn ONCE with a fixed seed)
underlies the gate's decision latency at every median regime:

    g_i(m) = m * exp(sigma * z_i),   sigma = 1.0

This is mathematically equivalent to an independent lognormal draw with median
m and sigma=1.0, but because the SAME z_i is reused for m = GATE_MEDIAN_MS_LOW
(30ms) and m = GATE_MEDIAN_MS_HIGH (150ms), the two median regimes sit on the
same underlying quantile draw. Differences between the regimes are therefore
attributable to the regime itself, not sampling noise -- the same Common
Random Numbers logic Part B already applies across its delay grid and its two
replication modes (see replication/delay_models.py).

r_i (uninformed reject flag, Bernoulli(UNINFORMED_REJECT_RATE)) is drawn once
per trade with its own fixed seed and reused across the full 40-configuration
grid (2 median x 5 tau x 2 policy x 2 rejection-modes). It is consumed ONLY by
the uninformed rejection variant (gate/policies.py); the informed variant is a
deterministic per-instrument ranking of baseline PnL (see
gate/policies.informed_reject_flags), not a random draw, and therefore has no
r_i-style seed of its own.
"""

from __future__ import annotations

import numpy as np

# Fixed seeds, documented per section 7 of the spec. Distinct from Part A/B's
# FrictionConfig.seed (42) and ReplicationConfig.seed (2024) so gate draws
# never alias onto those streams.
GATE_Z_SEED = 7301          # z_i: underlying standard normal for gate latency
GATE_REJECT_SEED = 7302     # r_i: uninformed reject Bernoulli flags

GATE_LATENCY_SIGMA = 1.0
UNINFORMED_REJECT_RATE = 0.02

# Gate decision-latency median regimes (milliseconds).
GATE_MEDIAN_MS_LOW = 30.0
GATE_MEDIAN_MS_HIGH = 150.0
GATE_MEDIAN_GRID_MS = (GATE_MEDIAN_MS_LOW, GATE_MEDIAN_MS_HIGH)

# Hard timeout budget grid (milliseconds), per the spec.
DEFAULT_TAU_GRID_MS = (10.0, 25.0, 50.0, 100.0, 250.0)


def draw_gate_z(n: int, seed: int = GATE_Z_SEED) -> np.ndarray:
    """One standard normal per trade, shared across both median regimes."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def gate_latency_ms(z: np.ndarray, median_ms: float, sigma: float = GATE_LATENCY_SIGMA) -> np.ndarray:
    """g_i(m) = m * exp(sigma * z_i); equivalent to an independent lognormal
    draw with median m and the given sigma, but reusing the SAME z_i across
    median regimes (CRN, see module docstring)."""
    return median_ms * np.exp(sigma * z)


def draw_uninformed_reject_flags(
    n: int, rate: float = UNINFORMED_REJECT_RATE, seed: int = GATE_REJECT_SEED,
) -> np.ndarray:
    """r_i: independent Bernoulli(rate) reject flags, one per trade, reused
    across the full grid. Only consumed by the uninformed rejection variant."""
    rng = np.random.default_rng(seed)
    return rng.random(n) < rate
