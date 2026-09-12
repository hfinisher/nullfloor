"""Core risk statistics, with the small-sample corrections most backtests skip.

Everything here operates on a 1-D series of *per-period* returns (simple, not
log, unless stated). Functions take and return per-period Sharpe ratios; the
annualisation factor is applied only at the reporting boundary, because every
correction in the literature is defined in per-period units and mixing the two
is the single most common error in retail backtest statistics.

References
----------
Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
    Journal of Risk 15(2).  -- PSR, MinTRL
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", Journal of
    Portfolio Management 40(5).  -- DSR, expected maximum Sharpe
Lo (2002), "The Statistics of Sharpe Ratios", Financial Analysts Journal 58(4).
    -- autocorrelation-corrected annualisation
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats as _sps

EULER_GAMMA = 0.5772156649015329
"""Euler-Mascheroni constant, used in the expected-maximum-Sharpe estimator."""


def as_returns(x) -> np.ndarray:
    """Coerce input to a clean 1-D float array of returns.

    Accepts anything array-like (list, numpy array, pandas Series). Drops
    non-finite values, since a single NaN silently poisons every moment
    estimate downstream.
    """
    a = np.asarray(getattr(x, "values", x), dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < 2:
        raise ValueError(
            f"need at least 2 finite returns to compute any statistic, got {a.size}"
        )
    return a


def sharpe(returns, rf: float = 0.0, ddof: int = 1) -> float:
    """Per-period Sharpe ratio: mean excess return over its standard deviation.

    `rf` is the per-period risk-free rate, in the same units as `returns`.
    Returns nan for a constant series, where the ratio is undefined rather
    than infinite.
    """
    r = as_returns(returns) - rf
    sd = r.std(ddof=ddof)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd)


def annualise_sharpe(sr_period: float, periods_per_year: float) -> float:
    """Scale a per-period Sharpe by sqrt(periods per year).

    This is the naive factor, valid only for serially independent returns.
    See `lo_annualisation_factor` for the corrected version.
    """
    return sr_period * math.sqrt(periods_per_year)


def moments(returns) -> dict:
    """Sample moments in the conventions the PSR/DSR formulas expect.

    `skew` is the standard sample skewness. `kurtosis` is *Pearson* kurtosis
    (3.0 for a normal distribution), not the excess kurtosis that most
    libraries return by default -- the published formulas assume Pearson.
    """
    r = as_returns(returns)
    return {
        "n": int(r.size),
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        "skew": float(_sps.skew(r, bias=False)),
        "kurtosis": float(_sps.kurtosis(r, fisher=False, bias=False)),
    }


def _psr_denominator(sr: float, skew: float, kurt: float) -> float:
    """Standard error multiplier for the Sharpe estimator under non-normality.

    This is the sqrt term from Bailey & Lopez de Prado: negative skew and fat
    tails both inflate it, which is how the PSR penalises strategies whose
    returns are 'pick up pennies, occasionally get flattened'.
    """
    var = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    # Clamp at a small positive number: extreme sample moments in tiny samples
    # can drive this negative, where the asymptotic expansion has broken down.
    return math.sqrt(max(var, 1e-12))


def probabilistic_sharpe(
    returns=None,
    benchmark_sr: float = 0.0,
    *,
    sr: Optional[float] = None,
    n: Optional[int] = None,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
) -> float:
    """P(true Sharpe > `benchmark_sr`), correcting for sample size and shape.

    Pass either `returns`, or the precomputed quadruple (`sr`, `n`, `skew`,
    `kurtosis`). All Sharpe values are per-period.

    A PSR of 0.95 is the conventional bar: it means that after accounting for
    how short and how non-normal the sample is, there is a 95% chance the true
    Sharpe exceeds the benchmark.
    """
    if returns is not None:
        m = moments(returns)
        sr = sharpe(returns) if sr is None else sr
        n, skew, kurtosis = m["n"], m["skew"], m["kurtosis"]
    if sr is None or n is None or skew is None or kurtosis is None:
        raise ValueError("supply `returns`, or all of sr, n, skew and kurtosis")
    if not math.isfinite(sr):
        return float("nan")
    z = (sr - benchmark_sr) * math.sqrt(n - 1) / _psr_denominator(sr, skew, kurtosis)
    return float(_sps.norm.cdf(z))


def min_track_record_length(
    returns=None,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
    *,
    sr: Optional[float] = None,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
) -> float:
    """Observations needed before a Sharpe this high would be believable.

    Returns the sample length at which PSR would reach `confidence`. If the
    observed sample is already longer than this, the track record is long
    enough; if not, the honest reading is 'promising but unproven'.

    Returns +inf when the observed Sharpe does not exceed the benchmark, since
    no amount of additional data makes a non-edge significant.
    """
    if returns is not None:
        m = moments(returns)
        sr = sharpe(returns) if sr is None else sr
        skew, kurtosis = m["skew"], m["kurtosis"]
    if sr is None or skew is None or kurtosis is None:
        raise ValueError("supply `returns`, or all of sr, skew and kurtosis")
    if not math.isfinite(sr) or sr <= benchmark_sr:
        return float("inf")
    z = _sps.norm.ppf(confidence)
    denom = _psr_denominator(sr, skew, kurtosis)
    return float(1.0 + (denom**2) * (z / (sr - benchmark_sr)) ** 2)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected highest Sharpe among `n_trials` genuinely skill-free strategies.

    Search hard enough and something will look good by luck alone. This is the
    height of that luck: the expected maximum of `n_trials` draws from a
    zero-mean normal with variance `sr_variance`, using the standard
    extreme-value approximation.

    `sr_variance` is the variance of the per-period Sharpe ratios *across the
    trials you ran* -- it measures how widely your search scattered.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if sr_variance < 0:
        raise ValueError(f"sr_variance must be non-negative, got {sr_variance}")
    if n_trials == 1 or sr_variance == 0:
        return 0.0
    a = _sps.norm.ppf(1.0 - 1.0 / n_trials)
    b = _sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(
    returns=None,
    n_trials: int = 1,
    sr_variance: Optional[float] = None,
    *,
    sr: Optional[float] = None,
    n: Optional[int] = None,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
    trial_sharpes=None,
) -> float:
    """PSR measured against the best Sharpe that luck alone would have produced.

    This is the number that matters when you have searched a parameter space.
    Supply the search's scale in one of two ways:

    - `trial_sharpes`: the per-period Sharpes of every variant you tested.
      Preferred -- `n_trials` and `sr_variance` are then both inferred.
    - `n_trials` plus `sr_variance` explicitly.

    A DSR below 0.95 means the result is within reach of a lucky search, no
    matter how good the headline Sharpe looks.
    """
    if trial_sharpes is not None:
        t = np.asarray(getattr(trial_sharpes, "values", trial_sharpes), dtype=float)
        t = t[np.isfinite(t)]
        if t.size < 2:
            raise ValueError("trial_sharpes needs at least 2 finite values")
        n_trials = int(t.size)
        sr_variance = float(t.var(ddof=1))
    if sr_variance is None:
        raise ValueError("supply `sr_variance` or `trial_sharpes`")
    if returns is not None:
        m = moments(returns)
        sr = sharpe(returns) if sr is None else sr
        n, skew, kurtosis = m["n"], m["skew"], m["kurtosis"]
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe(
        benchmark_sr=benchmark, sr=sr, n=n, skew=skew, kurtosis=kurtosis
    )


def autocorrelations(returns, max_lag: int) -> np.ndarray:
    """Sample autocorrelations for lags 1..max_lag."""
    r = as_returns(returns)
    r = r - r.mean()
    denom = float(np.dot(r, r))
    if denom == 0:
        return np.zeros(max_lag)
    out = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        out[k - 1] = float(np.dot(r[:-k], r[k:]) / denom) if k < r.size else 0.0
    return out


def newey_west_bandwidth(n: int) -> int:
    """Standard automatic lag truncation: floor(4 * (n/100) ** (2/9)).

    Lag truncation is not an optional refinement here. Lo's formula sums over
    q-1 lags, but estimating 251 autocorrelations from 900 observations yields
    noise, not information -- the accumulated sampling error can swamp the
    statistic and even flip its sign. Truncating at a bandwidth that grows
    slowly with the sample is what makes the estimator usable.
    """
    return max(1, int(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def lo_annualisation_factor(
    returns, periods_per_year: int, max_lag: Optional[int] = None
) -> float:
    """Autocorrelation-corrected replacement for sqrt(periods_per_year).

    Lo (2002): when returns are serially correlated, scaling by sqrt(q)
    misstates the annualised Sharpe. Positively autocorrelated returns -- the
    signature of a trend follower, or of stale marks -- make sqrt(q) too
    generous.

    `max_lag` defaults to a Newey-West bandwidth capped at q-1 and at a
    quarter of the sample, because the textbook q-1 lags are unestimable at
    daily frequency unless the history is enormous. Returns nan when the
    sample is too short for the correction to mean anything, so callers can
    fall back to the naive factor rather than quote a fabricated one.

    Compare against sqrt(periods_per_year) to see the size of the distortion.
    """
    q = int(periods_per_year)
    if q <= 1:
        return 1.0
    r = as_returns(returns)
    n = r.size
    if n < 4 * q:
        # Too little data for the long-run variance to be identified at all.
        return float("nan")
    lag = newey_west_bandwidth(n) if max_lag is None else int(max_lag)
    lag = max(1, min(lag, q - 1, n // 4))
    rho = autocorrelations(r, max_lag=lag)
    weights = np.arange(q - 1, q - 1 - lag, -1, dtype=float)  # (q-k), k = 1..lag
    denom = q + 2.0 * float(np.dot(weights, rho))
    if denom <= 0:
        return float("nan")
    return float(q / math.sqrt(denom))


def max_drawdown(returns, compound: bool = True) -> float:
    """Maximum peak-to-trough decline of the equity curve, as a positive number.

    With `compound=True` the curve is a cumulative product (the realistic case
    for a reinvested book); otherwise returns are summed.
    """
    r = as_returns(returns)
    equity = np.cumprod(1.0 + r) if compound else 1.0 + np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    # Guard against a curve that has gone non-positive: drawdown is total.
    if np.any(peak <= 0):
        return 1.0
    return float(np.max(1.0 - equity / peak))


def longest_drawdown(returns, compound: bool = True) -> int:
    """Longest run of periods spent below a previous equity peak."""
    r = as_returns(returns)
    equity = np.cumprod(1.0 + r) if compound else 1.0 + np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    underwater = equity < peak
    longest = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)
