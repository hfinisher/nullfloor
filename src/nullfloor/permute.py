"""Resampling tests: what would this track record look like with the skill removed?

Every function here answers one question -- how extreme is the observed result
inside a distribution generated under an explicit null hypothesis? The null is
named in each docstring, because a p-value without its null is decoration.

A warning that shapes this whole module
---------------------------------------
Shuffling the *order* of a return series does not change its mean or its
standard deviation, so it cannot change the Sharpe ratio. Order-permutation is
therefore useless as a test of Sharpe significance, and any tool that claims
otherwise is reporting a p-value of exactly nothing.

What order-permutation *does* change is everything path-dependent: drawdown,
losing streaks, time under water. That is what `drawdown_test` uses it for.
To test whether mean return is distinguishable from zero, use `sign_flip_test`.
To put an interval on the Sharpe itself, use the bootstraps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .stats import as_returns, max_drawdown, sharpe


@dataclass
class TestResult:
    """Outcome of one resampling test.

    Attributes
    ----------
    name: human-readable test name.
    null: the hypothesis being tested against, in words.
    observed: the statistic measured on the real series.
    p_value: probability of a result at least this extreme under the null.
    null_mean, null_std: centre and spread of the simulated null distribution.
    percentile: where `observed` falls in the null distribution, 0-100.
    n_sims: number of simulations run.
    alternative: 'greater', 'less', or 'two-sided'.
    """

    name: str
    null: str
    observed: float
    p_value: float
    null_mean: float
    null_std: float
    percentile: float
    n_sims: int
    alternative: str
    null_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def significant(self) -> bool:
        """True at the conventional 5% level. A convenience, not a verdict."""
        return self.p_value < 0.05


def _summarise(
    name: str,
    null: str,
    observed: float,
    sims: np.ndarray,
    alternative: str,
    keep: bool,
) -> TestResult:
    """Package an observed statistic against its simulated null distribution.

    The p-value uses the (1 + count) / (1 + n) convention, which keeps it
    strictly positive: with 1000 simulations the strongest claim available is
    p = 0.001, and reporting p = 0 would overstate the evidence.
    """
    sims = sims[np.isfinite(sims)]
    n = sims.size
    if n == 0:
        raise RuntimeError(f"{name}: every simulation produced a non-finite statistic")
    if alternative == "greater":
        count = int(np.sum(sims >= observed))
    elif alternative == "less":
        count = int(np.sum(sims <= observed))
    elif alternative == "two-sided":
        centre = float(np.mean(sims))
        count = int(np.sum(np.abs(sims - centre) >= abs(observed - centre)))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return TestResult(
        name=name,
        null=null,
        observed=float(observed),
        p_value=(1.0 + count) / (1.0 + n),
        null_mean=float(np.mean(sims)),
        null_std=float(np.std(sims, ddof=1)) if n > 1 else 0.0,
        percentile=float(np.mean(sims < observed) * 100.0),
        n_sims=n,
        alternative=alternative,
        null_distribution=sims if keep else None,
    )


def sign_flip_test(
    returns,
    n_sims: int = 2000,
    seed: Optional[int] = 0,
    keep_distribution: bool = False,
) -> TestResult:
    """Is the average return distinguishable from zero?

    Null: returns are symmetrically distributed about zero -- the strategy has
    no directional edge, and each period's sign is a coin flip.

    Implementation: randomly negate each return and recompute the mean. This
    preserves the magnitude distribution exactly, so it is robust to fat tails
    in a way that a t-test is not.
    """
    r = as_returns(returns)
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_sims, r.size))
    sims = (signs * r).mean(axis=1)
    return _summarise(
        "sign-flip test of mean return",
        "returns are symmetric about zero (no directional edge)",
        float(r.mean()),
        sims,
        "greater",
        keep_distribution,
    )


def drawdown_test(
    returns,
    n_sims: int = 2000,
    seed: Optional[int] = 0,
    compound: bool = True,
    keep_distribution: bool = False,
) -> TestResult:
    """Is the drawdown suspiciously mild for this distribution of returns?

    Null: the returns are in a random order -- their sequencing carries no
    information, so timing contributed nothing.

    A very low percentile here is a red flag rather than a triumph. It says the
    losses arrived in an unusually convenient order, which is the fingerprint
    of exit rules fitted to the sample, or of look-ahead leaking into the
    backtest. Genuinely good risk management does show up here, but so does
    curve-fitting, and this test cannot tell them apart -- it can only tell you
    that the sequencing is doing suspicious amounts of work.
    """
    r = as_returns(returns)
    rng = np.random.default_rng(seed)
    sims = np.empty(n_sims)
    for i in range(n_sims):
        sims[i] = max_drawdown(rng.permutation(r), compound=compound)
    return _summarise(
        "max-drawdown permutation test",
        "return ordering is uninformative (timing added nothing)",
        max_drawdown(r, compound=compound),
        sims,
        "less",
        keep_distribution,
    )


def _iid_indices(rng, n: int, size: int) -> np.ndarray:
    return rng.integers(0, n, size=size)


def _stationary_indices(rng, n: int, size: int, mean_block: float) -> np.ndarray:
    """Politis & Romano stationary bootstrap index path.

    Walks forward through the series, restarting at a uniformly random position
    with probability 1/mean_block at each step. Geometric block lengths make the
    resampled series stationary, which matters because fixed blocks do not.
    """
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(size, dtype=np.int64)
    cur = int(rng.integers(0, n))
    restarts = rng.random(size) < p
    for t in range(size):
        if t > 0:
            cur = int(rng.integers(0, n)) if restarts[t] else (cur + 1) % n
        idx[t] = cur
    return idx


def bootstrap_sharpe(
    returns,
    n_sims: int = 2000,
    seed: Optional[int] = 0,
    confidence: float = 0.95,
    block: Optional[float] = None,
    statistic: Optional[Callable[[np.ndarray], float]] = None,
) -> dict:
    """Confidence interval for the per-period Sharpe ratio by resampling.

    With `block=None` this is an IID bootstrap, valid when returns are serially
    independent. Pass `block` as a mean block length to use the stationary
    bootstrap instead, which preserves short-range dependence -- the right
    choice for trend-following or any strategy whose returns autocorrelate.

    Returns a dict with the point estimate, the percentile interval, the
    bootstrap standard error, and the share of resamples with Sharpe <= 0,
    which reads as an informal one-sided p-value.
    """
    r = as_returns(returns)
    n = r.size
    rng = np.random.default_rng(seed)
    stat = statistic if statistic is not None else (lambda a: sharpe(a))
    sims = np.empty(n_sims)
    for i in range(n_sims):
        idx = (
            _iid_indices(rng, n, n)
            if block is None
            else _stationary_indices(rng, n, n, block)
        )
        sims[i] = stat(r[idx])
    sims = sims[np.isfinite(sims)]
    if sims.size == 0:
        raise RuntimeError("every bootstrap resample gave a non-finite statistic")
    alpha = 1.0 - confidence
    lo, hi = np.quantile(sims, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "sharpe": sharpe(r),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": confidence,
        "std_error": float(np.std(sims, ddof=1)),
        "p_sharpe_le_zero": float(np.mean(sims <= 0.0)),
        "method": "iid" if block is None else f"stationary(mean_block={block:g})",
        "n_sims": int(sims.size),
    }
