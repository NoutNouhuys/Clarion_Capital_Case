"""
Tests for Part B replication: the delay model (CRN) and signal/fill mechanics.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from replication import delay_models as dm
from replication.followers import (
    _follower_latency, run_master, run_followers, ReplicationConfig,
    _price_follower, _order_diag, SIGNAL_SIDE, FILL_SIDE,
)
from simulator.frictions import FrictionConfig
from simulator.data_loader import load_m1, resample_h1
from simulator import reference_strategy as strat


# --- delay model -----------------------------------------------------------
def test_standard_normals_reproducible():
    a = dm.standard_normals(5, 20, seed=7)
    b = dm.standard_normals(5, 20, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_zero_normal_gives_exact_median():
    z = np.zeros((3, 4))
    d_e, d_x = dm.follower_delays(30.0, z, z, sigma=0.75)
    assert np.allclose(d_e, 30.0) and np.allclose(d_x, 30.0)


def test_crn_reuses_same_normals_across_grid():
    # Same z at two medians -> delays differ only by the multiplicative median
    # ratio (log-shift). This is what makes the grid a common-random-number sweep.
    z_e, z_x = dm.standard_normals(4, 10, seed=1)
    a_e, _ = dm.follower_delays(3.0, z_e, z_x)
    b_e, _ = dm.follower_delays(30.0, z_e, z_x)
    assert np.allclose(b_e / a_e, 10.0)  # 30 / 3


# --- signal vs fill latency structure --------------------------------------
def test_fill_side_is_signal_plus_master_latency():
    d_e = np.array([2.0, 5.0]); d_x = np.array([1.0, 4.0])
    lat_e = np.array([0.5, 0.5]); lat_x = np.array([0.5, 0.5])
    se, sx = _follower_latency(SIGNAL_SIDE, d_e, d_x, lat_e, lat_x)
    fe, fx = _follower_latency(FILL_SIDE, d_e, d_x, lat_e, lat_x)
    assert np.allclose(se, d_e) and np.allclose(sx, d_x)
    assert np.allclose(fe, d_e + lat_e) and np.allclose(fx, d_x + lat_x)
    # fill-side can never be faster than signal-side (structural lower-bound lag)
    assert (fe >= se).all() and (fx >= sx).all()


# --- integration on real data (small, for speed) ---------------------------
@pytest.fixture(scope="module")
def spx_small():
    m1 = load_m1("SPXUSD")
    h1 = resample_h1(m1)
    trades = strat.run(h1, "SPXUSD")[:40]
    return m1, h1, trades


def test_replication_deterministic(spx_small):
    m1, h1, trades = spx_small
    fcfg = FrictionConfig(); rcfg = ReplicationConfig(n_followers=3)
    df, le, lx = run_master(trades, h1, m1, fcfg)
    z_e, z_x = dm.standard_normals(rcfg.n_followers, len(trades), rcfg.seed)
    a = run_followers(trades, h1, m1, fcfg, rcfg, 30.0, SIGNAL_SIDE, le, lx, z_e, z_x)
    b = run_followers(trades, h1, m1, fcfg, rcfg, 30.0, SIGNAL_SIDE, le, lx, z_e, z_x)
    assert np.array_equal(a, b)


def test_tiny_delay_is_near_master(spx_small):
    # A sub-second follower delay lands in the master's bar -> almost no decay.
    m1, h1, trades = spx_small
    fcfg = FrictionConfig(); rcfg = ReplicationConfig(n_followers=3)
    df, le, lx = run_master(trades, h1, m1, fcfg)
    from replication.followers import net_bps_per_trade
    master_bps = net_bps_per_trade(df).mean()
    z_e, z_x = dm.standard_normals(rcfg.n_followers, len(trades), rcfg.seed)
    sig = run_followers(trades, h1, m1, fcfg, rcfg, 0.3, SIGNAL_SIDE, le, lx, z_e, z_x)
    assert abs(sig.mean() - master_bps) < 0.05  # bps


def test_fill_side_lag_never_negative(spx_small):
    # Timing identity: a fill-side follower starts only after the master fill,
    # so its per-order lag vs the master is >= 0 by construction (both entries
    # and exits). A signal-side follower can be faster (lag < 0) at small delay.
    m1, h1, trades = spx_small
    fcfg = FrictionConfig(); rcfg = ReplicationConfig(n_followers=3)
    master_df, le, lx = run_master(trades, h1, m1, fcfg)
    assert "entry_fill_time" in master_df and "exit_fill_time" in master_df
    z_e, z_x = dm.standard_normals(rcfg.n_followers, len(trades), rcfg.seed)
    d_e, d_x = dm.follower_delays(0.3, z_e, z_x, rcfg.sigma)

    fe, fx = _follower_latency(FILL_SIDE, d_e[0], d_x[0], le, lx)
    lag, faster, _ = _order_diag(master_df, _price_follower(trades, h1, m1, fcfg, fe, fx), m1)
    assert (lag >= -1e-9).all() and not faster.any()

    se, sx = _follower_latency(SIGNAL_SIDE, d_e[0], d_x[0], le, lx)
    slag, sfast, _ = _order_diag(master_df, _price_follower(trades, h1, m1, fcfg, se, sx), m1)
    assert sfast.any()  # some orders beat the master when d < L_master


def test_decay_distribution_shape_and_order(spx_small):
    # Distribution helper is deterministic and internally consistent: one decay
    # per follower, worst >= mean >= p10, dispersion grows with delay.
    from replication.distribution import follower_decay_distribution, distribution_stats
    m1, h1, trades = spx_small
    fcfg = FrictionConfig(); rcfg = ReplicationConfig(n_followers=8)
    mbps, dec = follower_decay_distribution(trades, h1, m1, fcfg, rcfg)
    for median_s, arr in dec.items():
        assert arr.shape == (rcfg.n_followers,)
    stats = distribution_stats("SPXUSD", mbps, dec)
    assert (stats["worst_decay_bps"] >= stats["mean_decay_bps"] - 1e-9).all()
    assert (stats["mean_decay_bps"] >= stats["p10_decay_bps"] - 1e-9).all()
    # dispersion is monotone-ish: 600s std exceeds 0.3s std
    s = stats.set_index("median_delay_s")["std_decay_bps"]
    assert s.loc[600.0] > s.loc[0.3]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
