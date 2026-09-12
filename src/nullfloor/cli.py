"""Command line interface.

Reads a returns or equity series from CSV (or stdin) and prints the verdict.
Uses only the standard library for parsing, so the CLI has no dependency
beyond numpy and scipy.

The `--fail-under` flag makes this usable as a CI gate: a strategy whose
verdict does not reach the required grade exits non-zero, so a fitted backtest
cannot quietly make it into a repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import List, Optional, Sequence

from . import __version__
from .report import NOISE, PERIODS, PLAUSIBLE, ROBUST, UNPROVEN, report

GRADES = [NOISE, UNPROVEN, PLAUSIBLE, ROBUST]


def _read_column(path: str, column: Optional[str], has_header: bool) -> List[float]:
    """Pull one numeric column out of a CSV, by name or zero-based index.

    Accepts '-' for stdin. Non-numeric cells are skipped rather than fatal,
    which keeps trailing blank lines and stray footer rows from killing a run.
    """
    handle = sys.stdin if path == "-" else open(path, newline="")
    try:
        rows = list(csv.reader(handle))
    finally:
        if handle is not sys.stdin:
            handle.close()
    if not rows:
        raise SystemExit(f"nullfloor: {path} is empty")

    header: Optional[Sequence[str]] = None
    if has_header:
        header, rows = rows[0], rows[1:]

    idx = 0
    if column is None:
        # Single-column files are unambiguous; otherwise demand an explicit choice.
        if header is not None and len(header) > 1:
            raise SystemExit(
                f"nullfloor: {path} has {len(header)} columns "
                f"({', '.join(header)}); pick one with --column"
            )
    elif column.isdigit():
        idx = int(column)
    elif header is None:
        raise SystemExit(
            f"nullfloor: --column {column!r} is a name, but the file was read "
            "without a header; drop --no-header or use a numeric index"
        )
    else:
        try:
            idx = list(header).index(column)
        except ValueError:
            raise SystemExit(
                f"nullfloor: no column {column!r} in {path}; "
                f"available: {', '.join(header)}"
            )

    out: List[float] = []
    for row in rows:
        if idx >= len(row):
            continue
        try:
            value = float(row[idx].strip())
        except (ValueError, AttributeError):
            continue
        if math.isfinite(value):
            out.append(value)
    if len(out) < 2:
        raise SystemExit(
            f"nullfloor: found only {len(out)} numeric values in column {idx} of {path}"
        )
    return out


def _equity_to_returns(equity: Sequence[float]) -> List[float]:
    """Convert an equity curve to period-over-period simple returns."""
    out = []
    for prev, cur in zip(equity, equity[1:]):
        if prev == 0:
            raise SystemExit("nullfloor: equity curve passes through zero")
        out.append(cur / prev - 1.0)
    return out


def _as_dict(rep) -> dict:
    """Flatten a Report for JSON, converting non-finite floats to null."""

    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    return {
        "verdict": rep.verdict,
        "observations": rep.n,
        "sharpe_period": clean(rep.sharpe_period),
        "sharpe_annual": clean(rep.sharpe_annual),
        "sharpe_annual_autocorr_adjusted": clean(rep.sharpe_annual_lo_corrected),
        "sharpe_ci_low_period": clean(rep.bootstrap["ci_low"]),
        "sharpe_ci_high_period": clean(rep.bootstrap["ci_high"]),
        "bootstrap_method": rep.bootstrap["method"],
        "probabilistic_sharpe": clean(rep.psr),
        "deflated_sharpe": clean(rep.dsr),
        "n_trials": rep.n_trials,
        "mean_return_p_value": clean(rep.sign_flip.p_value),
        "drawdown_percentile": clean(rep.drawdown.percentile),
        "max_drawdown": clean(rep.max_drawdown),
        "longest_drawdown_periods": rep.longest_drawdown,
        "skew": clean(rep.skew),
        "kurtosis": clean(rep.kurtosis),
        "min_track_record_length": clean(rep.min_track_record),
        "flags": rep.flags,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nullfloor",
        description="Test whether a backtest's edge survives contact with statistics.",
        epilog=(
            "examples:\n"
            "  nullfloor returns.csv --column pnl_pct --trials 500\n"
            "  nullfloor equity.csv --column nav --equity --frequency daily\n"
            "  cat r.csv | nullfloor - --no-header --json\n"
            "  nullfloor r.csv --trials 200 --fail-under PLAUSIBLE   # CI gate\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", help="CSV file of returns, or '-' for stdin")
    p.add_argument("--column", "-c", help="column name, or zero-based index")
    p.add_argument(
        "--no-header", action="store_true", help="treat the first row as data"
    )
    p.add_argument(
        "--equity",
        action="store_true",
        help="input is an equity curve; convert to returns first",
    )
    p.add_argument(
        "--frequency",
        "-f",
        default="daily",
        choices=sorted(PERIODS),
        help="bar frequency (default: daily)",
    )
    p.add_argument(
        "--periods-per-year", type=float, help="override --frequency for irregular bars"
    )
    p.add_argument(
        "--trials",
        "-t",
        type=int,
        default=1,
        metavar="N",
        help="strategy variants tested before picking this one; counts every "
        "parameter you swept. Leaving it at 1 after a sweep inflates the verdict.",
    )
    p.add_argument(
        "--trial-sharpes",
        metavar="CSV",
        help="file of per-period Sharpes for every variant tested; more accurate "
        "than --trials because it measures how widely the search scattered",
    )
    p.add_argument(
        "--block",
        type=float,
        metavar="L",
        help="mean block length for the stationary bootstrap; use when returns "
        "autocorrelate (default: IID bootstrap)",
    )
    p.add_argument("--sims", type=int, default=2000, help="simulations per test")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument(
        "--fail-under",
        choices=GRADES,
        metavar="GRADE",
        help=f"exit 1 if the verdict is weaker than this ({', '.join(GRADES)})",
    )
    p.add_argument("--version", action="version", version=f"nullfloor {__version__}")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    series = _read_column(args.path, args.column, has_header=not args.no_header)
    if args.equity:
        series = _equity_to_returns(series)

    trial_sharpes = None
    if args.trial_sharpes:
        trial_sharpes = _read_column(args.trial_sharpes, None, has_header=False)

    rep = report(
        series,
        frequency=args.frequency,
        periods_per_year=args.periods_per_year,
        n_trials=args.trials,
        trial_sharpes=trial_sharpes,
        block=args.block,
        n_sims=args.sims,
        seed=args.seed,
    )
    print(json.dumps(_as_dict(rep), indent=2) if args.json else str(rep))

    if args.fail_under and GRADES.index(rep.verdict) < GRADES.index(args.fail_under):
        print(
            f"nullfloor: verdict {rep.verdict} is weaker than required {args.fail_under}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
