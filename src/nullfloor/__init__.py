"""nullfloor -- is your backtest a real edge, or is it noise?

The null floor is the level of performance the null hypothesis already
explains. A strategy only has something to show you above it.

    >>> import nullfloor as nf
    >>> print(nf.report(daily_returns, n_trials=500))
"""

from .report import NOISE, PLAUSIBLE, ROBUST, UNPROVEN, Report, report
from .permute import bootstrap_sharpe, drawdown_test, sign_flip_test
from .stats import (
    deflated_sharpe,
    expected_max_sharpe,
    lo_annualisation_factor,
    longest_drawdown,
    max_drawdown,
    min_track_record_length,
    moments,
    probabilistic_sharpe,
    sharpe,
)

__version__ = "0.1.0"
__all__ = [
    "report", "Report", "NOISE", "UNPROVEN", "PLAUSIBLE", "ROBUST",
    "sharpe", "moments", "probabilistic_sharpe", "deflated_sharpe",
    "expected_max_sharpe", "min_track_record_length", "lo_annualisation_factor",
    "max_drawdown", "longest_drawdown",
    "sign_flip_test", "drawdown_test", "bootstrap_sharpe",
]
