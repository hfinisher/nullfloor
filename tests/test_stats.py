"""Tests for the closed-form statistics.

Where a published formula has a value that can be worked out by hand, the
expected number is hardcoded rather than recomputed from the implementation --
a test that re-derives the result it is checking proves only that the code
agrees with itself.
"""

import math

import numpy as np
import pytest

from nullfloor import stats


class TestSharpe:
    def test_known_value(self):
        # mean 0.02, population-corrected sd of [0.01,0.02,0.03] is 0.01
        assert stats.sharpe([0.01, 0.02, 0.03]) == pytest.approx(2.0)

    def test_constant_series_is_undefined_not_infinite(self):
        assert math.isnan(stats.sharpe([0.01, 0.01, 0.01]))

    def test_risk_free_shifts_numerator(self):
        r = [0.01, 0.02, 0.03]
        assert stats.sharpe(r, rf=0.02) == pytest.approx(0.0)

    def test_invariant_under_permutation(self):
        """The reason order-permutation cannot test Sharpe significance.

        Reordering a series changes neither its mean nor its standard
        deviation, so the Sharpe ratio is identical. Any tool reporting a
        p-value for Sharpe from shuffling the order is reporting noise.
        """
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, 500)
        for _ in range(5):
            assert stats.sharpe(rng.permutation(r)) == pytest.approx(stats.sharpe(r))

    def test_annualisation(self):
        assert stats.annualise_sharpe(0.1, 252) == pytest.approx(0.1 * math.sqrt(252))


class TestInputHandling:
    def test_drops_non_finite(self):
        assert stats.as_returns([0.01, np.nan, 0.02, np.inf]).size == 2

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError, match="at least 2"):
            stats.as_returns([0.01])

    def test_accepts_pandas_like(self):
        class FakeSeries:
            values = np.array([0.01, 0.02, 0.03])

        assert stats.sharpe(FakeSeries()) == pytest.approx(2.0)


class TestMoments:
    def test_kurtosis_is_pearson_not_excess(self):
        """The PSR/DSR formulas assume normal kurtosis = 3, not 0."""
        rng = np.random.default_rng(1)
        m = stats.moments(rng.normal(0, 1, 20000))
        assert m["kurtosis"] == pytest.approx(3.0, abs=0.15)
        assert m["skew"] == pytest.approx(0.0, abs=0.10)


class TestProbabilisticSharpe:
    def test_known_value_under_normality(self):
        # sr=0.1, n=101, skew=0, kurt=3 -> z = 0.1*10/sqrt(1.005) = 0.99751
        got = stats.probabilistic_sharpe(sr=0.1, n=101, skew=0.0, kurtosis=3.0)
        assert got == pytest.approx(0.8407, abs=1e-3)

    def test_zero_sharpe_is_a_coin_flip(self):
        got = stats.probabilistic_sharpe(sr=0.0, n=500, skew=0.0, kurtosis=3.0)
        assert got == pytest.approx(0.5, abs=1e-9)

    def test_rises_with_sample_size(self):
        vals = [
            stats.probabilistic_sharpe(sr=0.08, n=n, skew=0.0, kurtosis=3.0)
            for n in (50, 200, 1000, 5000)
        ]
        assert vals == sorted(vals)

    def test_negative_skew_and_fat_tails_penalise(self):
        base = stats.probabilistic_sharpe(sr=0.1, n=500, skew=0.0, kurtosis=3.0)
        skewed = stats.probabilistic_sharpe(sr=0.1, n=500, skew=-1.5, kurtosis=3.0)
        tailed = stats.probabilistic_sharpe(sr=0.1, n=500, skew=0.0, kurtosis=12.0)
        assert skewed < base
        assert tailed < base

    def test_requires_complete_arguments(self):
        with pytest.raises(ValueError, match="supply"):
            stats.probabilistic_sharpe(sr=0.1)


class TestMinTrackRecordLength:
    def test_infinite_when_no_edge(self):
        assert stats.min_track_record_length(sr=0.0, skew=0.0, kurtosis=3.0) == math.inf
        assert stats.min_track_record_length(sr=-0.1, skew=0.0, kurtosis=3.0) == math.inf

    def test_shrinks_as_edge_grows(self):
        weak = stats.min_track_record_length(sr=0.02, skew=0.0, kurtosis=3.0)
        strong = stats.min_track_record_length(sr=0.20, skew=0.0, kurtosis=3.0)
        assert strong < weak

    def test_agrees_with_psr_at_the_boundary(self):
        """At n = MinTRL, PSR should sit exactly at the target confidence."""
        sr, skew, kurt = 0.06, -0.3, 4.5
        n = stats.min_track_record_length(sr=sr, skew=skew, kurtosis=kurt, confidence=0.95)
        psr = stats.probabilistic_sharpe(sr=sr, n=n, skew=skew, kurtosis=kurt)
        assert psr == pytest.approx(0.95, abs=1e-6)


class TestExpectedMaxSharpe:
    def test_single_trial_has_nothing_to_deflate(self):
        assert stats.expected_max_sharpe(1, 0.01) == 0.0

    def test_zero_variance_means_no_search_dispersion(self):
        assert stats.expected_max_sharpe(1000, 0.0) == 0.0

    def test_grows_with_trials(self):
        vals = [stats.expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000, 10000)]
        assert vals == sorted(vals)
        assert all(v > 0 for v in vals)

    def test_scales_with_sqrt_variance(self):
        a = stats.expected_max_sharpe(500, 0.01)
        b = stats.expected_max_sharpe(500, 0.04)  # 4x variance -> 2x scale
        assert b == pytest.approx(2.0 * a)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            stats.expected_max_sharpe(0, 0.01)
        with pytest.raises(ValueError):
            stats.expected_max_sharpe(10, -1.0)


class TestDeflatedSharpe:
    def test_deflation_never_flatters(self):
        kw = dict(sr=0.12, n=1000, skew=0.0, kurtosis=3.0)
        psr = stats.probabilistic_sharpe(**kw)
        dsr = stats.deflated_sharpe(n_trials=500, sr_variance=0.01, **kw)
        assert dsr < psr

    def test_more_trials_deflate_harder(self):
        kw = dict(sr=0.12, n=1000, skew=0.0, kurtosis=3.0)
        vals = [
            stats.deflated_sharpe(n_trials=n, sr_variance=0.01, **kw)
            for n in (10, 100, 1000, 10000)
        ]
        assert vals == sorted(vals, reverse=True)

    def test_infers_search_scale_from_trial_sharpes(self):
        rng = np.random.default_rng(3)
        trials = rng.normal(0, 0.05, 300)
        explicit = stats.deflated_sharpe(
            sr=0.2, n=1000, skew=0.0, kurtosis=3.0,
            n_trials=300, sr_variance=float(trials.var(ddof=1)),
        )
        inferred = stats.deflated_sharpe(
            sr=0.2, n=1000, skew=0.0, kurtosis=3.0, trial_sharpes=trials,
        )
        assert inferred == pytest.approx(explicit)

    def test_needs_a_search_scale(self):
        with pytest.raises(ValueError, match="sr_variance"):
            stats.deflated_sharpe(sr=0.1, n=100, skew=0.0, kurtosis=3.0, n_trials=10)


class TestDrawdown:
    def test_known_value(self):
        # equity 1.10, 0.55, 0.66 against peak 1.10 -> worst decline 50%
        assert stats.max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(0.5)

    def test_monotonic_rise_never_draws_down(self):
        assert stats.max_drawdown([0.01] * 50) == pytest.approx(0.0)

    def test_total_loss_is_capped_at_one(self):
        assert stats.max_drawdown([0.1, -1.0, 0.5]) == pytest.approx(1.0)

    def test_longest_underwater_run(self):
        # peak at index 0; indices 1,2,3 below it; index 4 makes a new high
        assert stats.longest_drawdown([0.5, -0.1, -0.1, 0.05, 1.0]) == 3

    def test_simple_vs_compound_differ(self):
        r = [0.5, -0.4, 0.3, -0.2]
        assert stats.max_drawdown(r, compound=True) != stats.max_drawdown(r, compound=False)


class TestLoAutocorrelation:
    def test_independent_returns_recover_sqrt_q(self):
        rng = np.random.default_rng(5)
        r = rng.normal(0, 0.01, 30000)
        factor = stats.lo_annualisation_factor(r, 12)
        assert factor == pytest.approx(math.sqrt(12), rel=0.06)

    def test_positive_autocorrelation_reduces_the_factor(self):
        """Trend-like returns make naive sqrt(q) annualisation too generous."""
        rng = np.random.default_rng(6)
        eps = rng.normal(0, 0.01, 30000)
        ar = np.zeros_like(eps)
        for t in range(1, eps.size):
            ar[t] = 0.5 * ar[t - 1] + eps[t]
        assert stats.lo_annualisation_factor(ar, 12) < math.sqrt(12) * 0.85

    def test_degenerate_frequency(self):
        assert stats.lo_annualisation_factor([0.01, 0.02, 0.03], 1) == 1.0

    def test_autocorrelations_shape_and_range(self):
        rng = np.random.default_rng(7)
        rho = stats.autocorrelations(rng.normal(0, 1, 1000), max_lag=10)
        assert rho.shape == (10,)
        assert np.all(np.abs(rho) <= 1.0)
