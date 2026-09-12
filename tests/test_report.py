"""Tests for the verdict layer, including regressions for two shipped bugs."""

import math

import numpy as np
import pytest

from nullfloor import NOISE, ROBUST, UNPROVEN, report


@pytest.fixture
def noise():
    return np.random.default_rng(201).normal(0.0, 0.01, 900)


@pytest.fixture
def strong_edge():
    return np.random.default_rng(202).normal(0.0022, 0.01, 900)


class TestVerdicts:
    def test_noise_is_called_noise(self, noise):
        assert report(noise, n_sims=500).verdict == NOISE

    def test_strong_edge_survives(self, strong_edge):
        assert report(strong_edge, n_sims=500).verdict == ROBUST

    def test_short_sample_cannot_reach_the_top_grade(self):
        rng = np.random.default_rng(203)
        rep = report(rng.normal(0.004, 0.008, 60), n_sims=500)
        assert rep.verdict != ROBUST
        assert any("fragile" in f for f in rep.flags)


class TestDeflationRegression:
    """Regression: --trials once printed 'deflated' while deflating nothing.

    The cause was sr_variance defaulting to 0.0, which makes the expected
    maximum Sharpe 0.0 and the deflation an identity. The symptom was DSR
    exactly equal to PSR, with reassuring wording.
    """

    def test_deflation_actually_moves_the_number(self, strong_edge):
        rep = report(strong_edge, n_trials=600, n_sims=500)
        assert rep.dsr is not None
        assert rep.dsr < rep.psr, "deflation must penalise a large search"

    def test_more_trials_means_a_harsher_verdict(self, strong_edge):
        few = report(strong_edge, n_trials=10, n_sims=500)
        many = report(strong_edge, n_trials=20000, n_sims=500)
        assert many.dsr < few.dsr

    def test_enough_trials_can_overturn_a_strong_result(self, strong_edge):
        """The product's whole claim: a big enough search explains a good Sharpe."""
        assert report(strong_edge, n_trials=1, n_sims=500).verdict == ROBUST
        assert report(strong_edge, n_trials=10**7, n_sims=500).verdict == UNPROVEN

    def test_estimated_dispersion_is_disclosed(self, strong_edge):
        rep = report(strong_edge, n_trials=100, n_sims=500)
        assert any("estimated" in f for f in rep.flags)

    def test_measured_dispersion_is_not_flagged_as_estimated(self, strong_edge):
        trials = np.random.default_rng(204).normal(0, 0.03, 100)
        rep = report(strong_edge, trial_sharpes=trials, n_sims=500)
        assert not any("estimated" in f for f in rep.flags)
        assert rep.n_trials == 100

    def test_undeclared_search_is_flagged(self, strong_edge):
        rep = report(strong_edge, n_sims=500)
        assert rep.dsr is None
        assert any("n_trials=1" in f for f in rep.flags)


class TestAutocorrelationRegression:
    """Regression: the Lo correction once reported a Sharpe above the naive one.

    With q-1 = 251 lags estimated from 900 daily observations the sum is pure
    sampling noise, which inflated rather than corrected the figure. A
    correction that flatters the strategy is worse than none.
    """

    def test_suppressed_when_sample_is_too_short(self, noise):
        rep = report(noise, frequency="daily", n_sims=400)
        assert rep.sharpe_annual_lo_corrected is None
        assert any("suppressed" in f for f in rep.flags)

    def test_never_exceeds_the_naive_figure(self):
        rng = np.random.default_rng(205)
        for n in (500, 1500, 4000, 12000):
            rep = report(rng.normal(0.0005, 0.01, n), frequency="monthly", n_sims=300)
            if rep.sharpe_annual_lo_corrected is not None:
                assert abs(rep.sharpe_annual_lo_corrected) <= abs(rep.sharpe_annual) + 1e-9

    def test_applied_when_there_is_enough_data(self):
        """Monthly q=12 with a long history is the regime Lo's formula suits."""
        rng = np.random.default_rng(206)
        rep = report(rng.normal(0.004, 0.03, 3000), frequency="monthly", n_sims=300)
        assert rep.sharpe_annual_lo_corrected is not None


class TestDiagnosticFlags:
    def test_negative_skew_flagged(self):
        rng = np.random.default_rng(207)
        r = -np.abs(rng.standard_t(3, 600)) * 0.01 + 0.012
        rep = report(r, n_sims=400)
        assert any("skew" in f for f in rep.flags)

    def test_fat_tails_flagged(self):
        r = np.random.default_rng(208).standard_t(2.5, 800) * 0.01
        assert any("tails" in f for f in report(r, n_sims=400).flags)

    def test_per_trade_annualisation_is_disclaimed(self, noise):
        rep = report(noise, frequency="trade", n_sims=300)
        assert any("nominal" in f for f in rep.flags)


class TestReportMechanics:
    def test_unknown_frequency_rejected(self, noise):
        with pytest.raises(ValueError, match="unknown frequency"):
            report(noise, frequency="fortnightly", n_sims=200)

    def test_periods_per_year_override(self, noise):
        assert report(noise, periods_per_year=99.0, n_sims=200).periods_per_year == 99.0

    def test_renders_without_error(self, strong_edge):
        text = str(report(strong_edge, n_trials=50, n_sims=300))
        assert "nullfloor verdict" in text
        assert "Deflated Sharpe" in text

    def test_p_value_floor_is_not_printed_as_zero(self, strong_edge):
        """1/(n+1) is the strongest claim available; 0.000 would overstate it."""
        text = str(report(strong_edge, n_sims=500))
        assert "0.000 " not in text
        assert "<0.002" in text

    def test_min_track_record_infinite_for_losers(self):
        rng = np.random.default_rng(209)
        rep = report(rng.normal(-0.001, 0.01, 600), n_sims=300)
        assert math.isinf(rep.min_track_record)
        assert rep.verdict == NOISE
