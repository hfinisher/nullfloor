"""Tests for the resampling tests.

These assert statistical *behaviour* -- that a test fires on signal and stays
quiet on noise -- which means they depend on the seed. Seeds are fixed and
sample sizes are large enough that the margins are not delicate.
"""

import numpy as np
import pytest

from nullfloor import permute


@pytest.fixture
def noise():
    """750 days of nothing: zero mean, realistic daily volatility."""
    return np.random.default_rng(101).normal(0.0, 0.01, 750)


@pytest.fixture
def strong_edge():
    """750 days with an edge large enough that any honest test must find it."""
    return np.random.default_rng(102).normal(0.0025, 0.01, 750)


class TestSignFlip:
    def test_quiet_on_noise(self, noise):
        assert permute.sign_flip_test(noise).p_value > 0.10

    def test_fires_on_real_edge(self, strong_edge):
        r = permute.sign_flip_test(strong_edge)
        assert r.p_value < 0.01
        assert r.significant

    def test_p_value_is_never_exactly_zero(self, strong_edge):
        """With n sims the strongest honest claim is 1/(n+1), not 0."""
        r = permute.sign_flip_test(strong_edge, n_sims=500)
        assert r.p_value == pytest.approx(1.0 / 501.0)
        assert r.p_value > 0

    def test_null_distribution_centres_on_zero(self, strong_edge):
        r = permute.sign_flip_test(strong_edge, n_sims=4000)
        assert r.null_mean == pytest.approx(0.0, abs=5e-4)

    def test_reproducible_under_seed(self, noise):
        a = permute.sign_flip_test(noise, seed=7)
        b = permute.sign_flip_test(noise, seed=7)
        assert a.p_value == b.p_value

    def test_keeps_distribution_only_on_request(self, noise):
        assert permute.sign_flip_test(noise, n_sims=100).null_distribution is None
        kept = permute.sign_flip_test(noise, n_sims=100, keep_distribution=True)
        assert kept.null_distribution is not None
        assert kept.null_distribution.size == 100


class TestDrawdownPermutation:
    def test_ordinary_ordering_is_unremarkable(self, noise):
        """Randomly ordered returns should not look suspicious."""
        pct = permute.drawdown_test(noise, n_sims=600).percentile
        assert 5.0 < pct < 95.0

    def test_detects_suspiciously_convenient_sequencing(self):
        """Interleaved losses give an implausibly mild drawdown path.

        This is the synthetic signature of fitted exits or look-ahead: the very
        same returns, arranged so losses never get to compound. Note that
        *sorting* the series would do the opposite -- all gains then all losses
        is the worst possible path, not the best -- so the rigged ordering has
        to alternate sign to keep the equity curve from ever falling far.
        """
        rng = np.random.default_rng(103)
        r = rng.normal(0.0005, 0.015, 400)
        gains = np.sort(r[r > 0])[::-1]
        losses = np.sort(r[r <= 0])
        rigged = np.empty_like(r)
        rigged[0::2][: gains.size] = gains[: rigged[0::2].size]
        rigged[1::2][: losses.size] = losses[: rigged[1::2].size]
        leftover = r.size - min(gains.size, rigged[0::2].size) - min(
            losses.size, rigged[1::2].size
        )
        if leftover:  # uneven split: park the remainder at the end
            rigged = np.concatenate(
                [np.ravel(np.column_stack([gains[: losses.size], losses[: gains.size]])),
                 gains[losses.size:], losses[gains.size:]]
            )
        natural = permute.drawdown_test(r, n_sims=400).percentile
        rigged_pct = permute.drawdown_test(rigged, n_sims=400).percentile
        assert rigged_pct < natural
        assert rigged_pct < 10.0, "an interleaved path should look clearly suspicious"

    def test_records_the_null_it_tested(self, noise):
        assert "ordering" in permute.drawdown_test(noise, n_sims=100).null


class TestBootstrapSharpe:
    def test_interval_brackets_a_known_sharpe(self):
        """A large sample's CI should contain its own point estimate."""
        r = np.random.default_rng(104).normal(0.001, 0.01, 2000)
        b = permute.bootstrap_sharpe(r, n_sims=800)
        assert b["ci_low"] < b["sharpe"] < b["ci_high"]

    def test_noise_interval_straddles_zero(self, noise):
        b = permute.bootstrap_sharpe(noise, n_sims=800)
        assert b["ci_low"] < 0 < b["ci_high"]

    def test_real_edge_interval_clears_zero(self, strong_edge):
        b = permute.bootstrap_sharpe(strong_edge, n_sims=800)
        assert b["ci_low"] > 0
        assert b["p_sharpe_le_zero"] < 0.01

    def test_wider_confidence_gives_wider_interval(self, noise):
        narrow = permute.bootstrap_sharpe(noise, n_sims=800, confidence=0.80)
        wide = permute.bootstrap_sharpe(noise, n_sims=800, confidence=0.99)
        assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])

    def test_stationary_bootstrap_runs_and_is_labelled(self, noise):
        b = permute.bootstrap_sharpe(noise, n_sims=400, block=10)
        assert "stationary" in b["method"]
        assert b["ci_low"] < b["ci_high"]

    def test_stationary_widens_interval_for_autocorrelated_returns(self):
        """Ignoring dependence understates uncertainty -- the classic error."""
        rng = np.random.default_rng(105)
        eps = rng.normal(0.0008, 0.01, 1500)
        ar = np.zeros_like(eps)
        for t in range(1, eps.size):
            ar[t] = 0.6 * ar[t - 1] + eps[t]
        iid = permute.bootstrap_sharpe(ar, n_sims=600)
        blk = permute.bootstrap_sharpe(ar, n_sims=600, block=20)
        assert (blk["ci_high"] - blk["ci_low"]) > (iid["ci_high"] - iid["ci_low"])

    def test_rejects_unknown_alternative(self, noise):
        with pytest.raises(ValueError, match="alternative"):
            permute._summarise("x", "y", 0.0, np.zeros(10), "sideways", False)
