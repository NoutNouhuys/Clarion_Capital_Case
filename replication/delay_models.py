"""
Follower replication-delay model (Part B).

Each follower account receives every order after an *additional* random delay
`d` on top of the master. The default model (usable without justification) is
lognormal with sigma = 0.75 and a median swept over the grid
{0.3, 3, 30, 120, 600} seconds:

    d = exp( log(median_s) + sigma * z ),   z ~ N(0, 1)

Why lognormal: replication + execution latency is strictly positive and
right-skewed (queuing / congestion tails), the same reasoning as the master
latency in Part A.

Common Random Numbers (CRN). The standard-normal draws `z` are generated from a
fixed seed and depend only on (n_followers, n_trades, seed) -- NOT on the median.
Scaling to a delay is a deterministic shift of `mu = log(median_s)`. Therefore:

  * across the delay grid, every grid point reuses the *same* z, so only the
    median moves and grid comparisons carry no sampling noise;
  * across signal-side vs fill-side replication, both modes reuse the same z (and
    the same master latencies), so the only thing that differs is the structural
    +master-latency offset of fill-side.

Entry and exit orders are two separate orders, so they draw independent delays.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SIGMA = 0.75
DELAY_GRID_S = (0.3, 3.0, 30.0, 120.0, 600.0)


def standard_normals(n_followers: int, n_trades: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Median-independent N(0,1) draws for entry/exit, shape (n_followers, n_trades).

    Seeded and reused across the whole delay grid and both replication modes so
    that every comparison is on common random numbers.
    """
    rng = np.random.default_rng(seed)
    z_entry = rng.standard_normal((n_followers, n_trades))
    z_exit = rng.standard_normal((n_followers, n_trades))
    return z_entry, z_exit


def follower_delays(
    median_s: float,
    z_entry: np.ndarray,
    z_exit: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the shared normals to per-(follower, order) delays in seconds for one
    median. Because only `mu = log(median_s)` changes, calling this for each grid
    point keeps the draws common across the grid."""
    mu = np.log(median_s)
    return np.exp(mu + sigma * z_entry), np.exp(mu + sigma * z_exit)
