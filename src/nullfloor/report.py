"""The verdict layer: run every test and say, in plain words, what they mean.

The statistics in `stats` and `permute` are individually unremarkable -- they are
in the literature and anyone can implement them. What this module does is refuse
to let a good headline number stand unchallenged: it runs the full battery,
collects the caveats, and grades the track record on the weakest link rather
than the strongest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from . import permute, stats

PERIODS = {"daily": 252, "weekly": 52, "monthly": 12, "hourly": 1638, "trade": 252}
"""Conventional periods-per-year. 'trade' is a nominal stand-in: per-trade
returns have no calendar frequency, so any annualisation of them is a fiction
and the report will say so."""

# Grades, weakest to strongest. The verdict is the weakest supported rung.
NOISE = "NOISE"
UNPROVEN = "UNPROVEN"
PLAUSIBLE = "PLAUSIBLE"
ROBUST = "ROBUST"


@dataclass
class Report:
    """Full diagnostic of one return series."""

    n: int
    periods_per_year: float
    sharpe_period: float
    sharpe_annual: float
    sharpe_annual_lo_corrected: Optional[float]
    mean: float
    std: float
    skew: float
    kurtosis: float
    max_drawdown: float
    longest_drawdown: int
    psr: float
    dsr: Optional[float]
    n_trials: int
    min_track_record: float
    bootstrap: dict
    sign_flip: permute.TestResult
    drawdown: permute.TestResult
    verdict: str
    flags: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only for the top grade. Deliberately strict."""
        return self.verdict == ROBUST

    def __str__(self) -> str:  # pragma: no cover - presentation
        w = 64
        primary = self.dsr if self.dsr is not None else self.psr
        label = "Deflated Sharpe" if self.dsr is not None else "Probabilistic Sharpe"
        trl = (
            "never at this Sharpe"
            if math.isinf(self.min_track_record)
            else f"{self.min_track_record:,.0f} periods"
        )
        lo = (
            "n/a"
            if self.sharpe_annual_lo_corrected is None
            or not math.isfinite(self.sharpe_annual_lo_corrected)
            else f"{self.sharpe_annual_lo_corrected:+.2f}"
        )
        floor = 1.0 / (1.0 + self.sign_flip.n_sims)
        p_txt = (
            f"<{floor:.4f}".rstrip("0")
            if self.sign_flip.p_value <= floor
            else f"{self.sign_flip.p_value:.3f}"
        )
        lines = [
            "=" * w,
            f"  nullfloor verdict:  {self.verdict}",
            "=" * w,
            f"  observations            {self.n:,}",
            f"  Sharpe (annualised)     {self.sharpe_annual:+.2f}"
            f"   [per period {self.sharpe_period:+.4f}]",
            f"  Sharpe, autocorr-adj    {lo}",
            f"  Sharpe 95% CI           [{self.bootstrap['ci_low']*math.sqrt(self.periods_per_year):+.2f},"
            f" {self.bootstrap['ci_high']*math.sqrt(self.periods_per_year):+.2f}]"
            f"  ({self.bootstrap['method']})",
            f"  max drawdown            {self.max_drawdown*100:.1f}%"
            f"   (longest {self.longest_drawdown:,} periods underwater)",
            f"  skew / kurtosis         {self.skew:+.2f} / {self.kurtosis:.2f}",
            "-" * w,
            f"  {label:<22}  {primary:.3f}"
            + (f"   (deflated for {self.n_trials:,} trials)" if self.dsr is not None else ""),
            f"  mean-return p-value     {p_txt}"
            f"   (sign-flip, {self.sign_flip.n_sims:,} sims)",
            f"  drawdown percentile     {self.drawdown.percentile:.1f}"
            f"   (vs {self.drawdown.n_sims:,} shuffles)",
            f"  track record needed     {trl}",
            "-" * w,
        ]
        if self.flags:
            lines.append("  flags")
            lines.extend(f"    - {f}" for f in self.flags)
        else:
            lines.append("  flags                   none")
        lines.append("=" * w)
        return "\n".join(lines)


def _grade(psr_or_dsr: float, sign_p: float, ci_low: float, n: int) -> str:
    """Grade on the weakest link, not the headline.

    The bar rises through four rungs, and every rung must hold: a positive
    lower confidence bound, then a significant mean, then the 0.95 PSR/DSR
    convention, then a sample long enough for any of it to mean much.
    """
    if not math.isfinite(psr_or_dsr) or ci_low <= 0 or sign_p > 0.10:
        return NOISE
    if psr_or_dsr < 0.95 or sign_p > 0.05:
        return UNPROVEN
    if n < 100:
        return PLAUSIBLE
    return ROBUST


def report(
    returns,
    frequency: str = "daily",
    periods_per_year: Optional[float] = None,
    n_trials: int = 1,
    trial_sharpes=None,
    block: Optional[float] = None,
    n_sims: int = 2000,
    seed: Optional[int] = 0,
    confidence: float = 0.95,
) -> Report:
    """Run the full battery on a return series and grade it.

    Parameters
    ----------
    returns: per-period simple returns (list, array or pandas Series).
    frequency: one of 'daily', 'weekly', 'monthly', 'hourly', 'trade'.
    periods_per_year: overrides `frequency` when your bars are irregular.
    n_trials: how many strategy variants you tested before picking this one.
        Be honest here -- every parameter you swept counts, and leaving it at 1
        when you swept 500 is the single easiest way to fool yourself.
    trial_sharpes: the per-period Sharpes of all variants tested. Preferred
        over `n_trials`, since it measures how widely your search scattered
        instead of assuming it.
    block: mean block length for the stationary bootstrap. Defaults to IID;
        set it if your returns autocorrelate.
    """
    r = stats.as_returns(returns)
    ppy = periods_per_year if periods_per_year is not None else PERIODS.get(frequency)
    if ppy is None:
        raise ValueError(
            f"unknown frequency {frequency!r}; pass periods_per_year explicitly"
        )
    m = stats.moments(r)
    sr = stats.sharpe(r)
    flags: List[str] = []

    # Autocorrelation check needs more observations than the lags it measures.
    lo_adj = None
    naive = math.sqrt(ppy)
    factor = stats.lo_annualisation_factor(r, int(ppy))
    if not (math.isfinite(factor) and factor > 0):
        # Not enough history to identify the long-run variance. Say so: a blank
        # row with no explanation reads as 'no autocorrelation', which is a
        # different and much more reassuring claim than 'not measurable'.
        flags.append(
            "autocorrelation correction suppressed: needs at least "
            f"{4 * int(ppy):,} observations at {int(ppy)} periods/year, has {r.size:,}"
        )
    elif factor > naive:
        # A correction larger than the naive factor means the autocorrelation
        # estimates are dominated by sampling noise, not that the Sharpe is
        # genuinely understated. Suppress it rather than flatter the strategy.
        flags.append(
            "autocorrelation correction suppressed: too few observations "
            f"({r.size:,}) relative to {int(ppy)} periods/year to estimate it reliably"
        )
    else:
        lo_adj = sr * factor
        if factor < naive * 0.85:
            flags.append(
                f"serial correlation inflates the naive annualised Sharpe by "
                f"~{naive/factor:.2f}x; the autocorr-adjusted figure is the honest one"
            )

    dsr = None
    if trial_sharpes is not None or n_trials > 1:
        sr_var = None
        if trial_sharpes is None:
            # Without the actual trial Sharpes we must estimate how widely the
            # search scattered. Under the null of no skill the Sharpe estimator
            # has sampling variance ~ (1 + sr^2 / 2) / n, which is the standard
            # stand-in. It is an approximation, and the flag below says so --
            # but it is vastly better than assuming zero dispersion, which
            # would make the deflation a no-op while still printing the word.
            sr_var = (1.0 + (sr**2) / 2.0) / m["n"]
            flags.append(
                f"search dispersion estimated as ~{sr_var:.2e} from sample size, "
                "not measured; pass trial_sharpes for an accurate deflation"
            )
        dsr = stats.deflated_sharpe(
            sr=sr,
            n=m["n"],
            skew=m["skew"],
            kurtosis=m["kurtosis"],
            n_trials=n_trials,
            sr_variance=sr_var,
            trial_sharpes=trial_sharpes,
        )
    else:
        flags.append(
            "n_trials=1 assumes this is the only strategy you ever tested; if you "
            "swept parameters, pass n_trials or trial_sharpes or the verdict is too generous"
        )

    psr = stats.probabilistic_sharpe(sr=sr, n=m["n"], skew=m["skew"], kurtosis=m["kurtosis"])
    boot = permute.bootstrap_sharpe(
        r, n_sims=n_sims, seed=seed, confidence=confidence, block=block
    )
    sf = permute.sign_flip_test(r, n_sims=n_sims, seed=seed)
    dd = permute.drawdown_test(r, n_sims=min(n_sims, 1000), seed=seed)
    trl = stats.min_track_record_length(
        sr=sr, skew=m["skew"], kurtosis=m["kurtosis"], confidence=confidence
    )

    if m["n"] < 100:
        flags.append(f"only {m['n']} observations; every estimate here is fragile")
    if math.isfinite(trl) and trl > m["n"]:
        flags.append(
            f"needs ~{trl:,.0f} observations to be {confidence:.0%} confident, has {m['n']}"
        )
    if m["skew"] < -0.5:
        flags.append(
            f"negative skew ({m['skew']:+.2f}): small steady gains, rare large losses"
        )
    if m["kurtosis"] > 6:
        flags.append(f"fat tails (kurtosis {m['kurtosis']:.1f}): tail risk understated by normal assumptions")
    if dd.percentile < 5:
        flags.append(
            f"drawdown sits at the {dd.percentile:.1f}th percentile of shuffled orderings -- "
            "the sequencing is doing suspicious work; check for look-ahead or fitted exits"
        )
    if frequency == "trade":
        flags.append("per-trade returns have no calendar frequency; annualised figures here are nominal")

    return Report(
        n=m["n"],
        periods_per_year=ppy,
        sharpe_period=sr,
        sharpe_annual=stats.annualise_sharpe(sr, ppy),
        sharpe_annual_lo_corrected=lo_adj,
        mean=m["mean"],
        std=m["std"],
        skew=m["skew"],
        kurtosis=m["kurtosis"],
        max_drawdown=stats.max_drawdown(r),
        longest_drawdown=stats.longest_drawdown(r),
        psr=psr,
        dsr=dsr,
        n_trials=int(n_trials if trial_sharpes is None else len(trial_sharpes)),
        min_track_record=trl,
        bootstrap=boot,
        sign_flip=sf,
        drawdown=dd,
        verdict=_grade(dsr if dsr is not None else psr, sf.p_value, boot["ci_low"], m["n"]),
        flags=flags,
    )
