# nullfloor

**Is your backtest a real edge, or is it noise?**

[![PyPI](https://img.shields.io/pypi/v/nullfloor.svg)](https://pypi.org/project/nullfloor/)
[![Python](https://img.shields.io/pypi/pyversions/nullfloor.svg)](https://pypi.org/project/nullfloor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Most backtests are overfit. Not because the people running them are dishonest, but
because searching a parameter space and keeping the best result is *guaranteed* to
produce something that looks profitable — even when there is nothing there at all.

The **null floor** is the level of performance that luck alone already explains.
`nullfloor` measures where that floor sits for your search, and tells you whether
your result clears it.

```bash
pip install nullfloor
```

## The one number that matters

Here is a real strategy with a genuine edge, tested at increasing levels of honesty
about how many variants were tried before this one was picked. **The returns are
identical in every row.** Only the declared search size changes.

| variants tested | Probabilistic Sharpe | Deflated Sharpe | verdict |
|---|---|---|---|
| 1 | 1.000 | — | **ROBUST** |
| 10 | 1.000 | 0.994 | **ROBUST** |
| 100 | 1.000 | 0.938 | **UNPROVEN** |
| 1,000 | 1.000 | 0.793 | **UNPROVEN** |
| 10,000 | 1.000 | 0.584 | **UNPROVEN** |

That is the entire argument. A result with a Probabilistic Sharpe of
1.000
— overwhelming by any conventional reading — stops being evidence once you admit how
hard you searched for it. Every parameter you swept counts. Every threshold you nudged
counts. If you do not deflate for them, your backtest is measuring your persistence,
not the market.

## Usage

### Command line

```bash
nullfloor returns.csv --column ret --trials 2000
```

```
================================================================
  nullfloor verdict:  UNPROVEN
================================================================
  observations            1,400
  Sharpe (annualised)     +1.73   [per period +0.1092]
  Sharpe, autocorr-adj    n/a
  Sharpe 95% CI           [+0.86, +2.59]  (iid)
  max drawdown            13.5%   (longest 153 periods underwater)
  skew / kurtosis         -0.05 / 3.03
----------------------------------------------------------------
  Deflated Sharpe         0.734   (deflated for 2,000 trials)
  mean-return p-value     <0.0005   (sign-flip, 2,000 sims)
  drawdown percentile     13.0   (vs 1,000 shuffles)
  track record needed     231 periods
----------------------------------------------------------------
  flags
    - autocorrelation correction suppressed: too few observations (1,400) relative to 252 periods/year to estimate it reliably
    - search dispersion estimated as ~7.19e-04 from sample size, not measured; pass trial_sharpes for an accurate deflation
================================================================
```

Reads returns or an equity curve from CSV or stdin:

```bash
nullfloor equity.csv -c nav --equity            # equity curve instead of returns
cat returns.csv | nullfloor - --no-header       # from a pipe
nullfloor r.csv -c ret --json                   # machine-readable
nullfloor r.csv -c ret -t 500 --fail-under PLAUSIBLE   # CI gate, exits 1 on failure
```

That last one is worth a habit: put it in CI and a curve-fitted strategy cannot
quietly make it into your repository.

### Python

```python
import nullfloor as nf

rep = nf.report(daily_returns, n_trials=2000)
print(rep)

rep.verdict      # 'NOISE' | 'UNPROVEN' | 'PLAUSIBLE' | 'ROBUST'
rep.dsr          # deflated Sharpe, the number to quote
rep.flags        # everything wrong with your sample, in plain words
```

If you kept the Sharpe of every variant you tested — and you should — pass them
instead of a count. It measures how widely your search actually scattered rather
than assuming it:

```python
rep = nf.report(returns, trial_sharpes=[s for s in all_variant_sharpes])
```

## What it tests

| test | question it answers | null hypothesis |
|---|---|---|
| **Deflated Sharpe** | Does this beat the best result luck would have produced across my search? | the best of N skill-free strategies |
| **Probabilistic Sharpe** | Is the Sharpe distinguishable from the benchmark, given sample size, skew and fat tails? | true Sharpe equals benchmark |
| **Sign-flip test** | Is mean return distinguishable from zero? | returns are symmetric about zero |
| **Drawdown permutation** | Is the drawdown suspiciously mild for these returns? | return ordering carries no information |
| **Stationary bootstrap** | How wide is the real confidence interval on the Sharpe? | — (interval estimate) |
| **Min track record length** | How much history would I need for this to be believable? | — (design calculation) |
| **Lo autocorrelation** | Is serial correlation inflating my annualised Sharpe? | returns are serially independent |

### A warning this library takes seriously

**Shuffling the order of a return series cannot test Sharpe significance.** Reordering
changes neither the mean nor the standard deviation, so the Sharpe ratio is *identical*
— there is a test asserting exactly this. Any tool that reports a p-value for Sharpe by
shuffling returns is reporting nothing at all.

Order-permutation is still useful, but only for path-dependent statistics: drawdown,
losing streaks, time underwater. That is what `drawdown_test` uses it for. To test mean
return, use `sign_flip_test`. To bound the Sharpe, use the bootstraps.

## Honest limitations

- **It cannot detect look-ahead bias or bad data.** Feed it returns computed with
  tomorrow's prices and it will cheerfully call them robust. Statistics cannot see a
  flawed simulation; `drawdown_test` only hints when sequencing is doing suspicious work.
- **`ROBUST` is not permission to trade.** It means this sample is not explained by the
  nulls tested here. Regime change, transaction costs, slippage, capacity and execution
  are all outside its scope.
- **The deflation is only as honest as your trial count.** Guess low and you get a
  flattering answer. This is the one input the library cannot verify for you.
- **Estimated search dispersion is an approximation.** Without `trial_sharpes` the
  dispersion is inferred from sample size, and the report flags when it has done so.
- **The autocorrelation correction needs a long history** — at least 4 years of daily
  data. Below that it is suppressed rather than guessed, and the report says so.

## References

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio Management 40(5)
- Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier*, Journal of Risk 15(2)
- Lo (2002), *The Statistics of Sharpe Ratios*, Financial Analysts Journal 58(4)
- Politis & Romano (1994), *The Stationary Bootstrap*, JASA 89(428)

## Disclaimer

For research and educational use. Not investment advice, and not a recommendation to
trade any strategy or instrument. `nullfloor` evaluates the statistical strength of a
return series; it makes no claim about future performance.

MIT licensed.
