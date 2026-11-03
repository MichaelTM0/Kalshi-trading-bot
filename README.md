# Kalshi Crypto Prediction Market Bot — Quantitative Research Project

An automated trading system built to test whether a classic options-pricing
model could find a real edge in Kalshi's short-duration crypto prediction
markets. Live-deployed on AWS EC2, backed by a real backtesting and
diagnostics pipeline, and tested rigorously over several thousand live
trades before reaching a conclusion.

**TL;DR: after weeks of live testing and four separate, evidence-based
diagnostic investigations, the strategy did not find a statistically
significant edge on this specific market and timeframe. That negative
result — and the process that produced it — is the actual deliverable of
this project.**

---

## What this is

Kalshi runs 15-minute "will the price be up or down" markets on BTC, ETH,
SOL, and XRP. This bot:

1. Pulls live spot price and short-term volatility for each asset
2. Feeds them into a Black-Scholes-derived probability model (the same
   `N(d2)` math used to price options) to estimate the true probability
   the asset finishes above the market's reference price
3. Compares that estimate to what Kalshi's current bid/ask implies the
   market believes
4. If the two disagree by more than a threshold, flags a trade
5. Sizes and (in `DRY_RUN` mode) simulates the trade, with a daily
   circuit breaker after 5 consecutive losses

Nothing here ever touched real money — every trade was paper-traded
against live market data, then checked against Kalshi's actual settlement
results afterward.

## Architecture

```
data/            Kalshi API client (RSA-PSS authenticated), CoinGecko
                  price/volatility fetchers with request caching
models/           Black-Scholes-derived probability model
strategy/         Signal generation — compares model vs. market probability
execution/        Position sizing, circuit breaker, DRY_RUN order handling
backtest/         Result analysis, calibration testing, diagnostic tooling
main.py           The live polling loop tying it all together
```

Deployed on a persistent AWS EC2 instance, with credentials excluded from
version control and a schema-safe CSV trade log that auto-detects and
archives itself if its own data format ever changes mid-run, rather than
silently corrupting.

## The investigation

The headline number stayed stubbornly in the mid-30s% win rate across
every version tested — well below the ~50% a random strategy would get on
these near-even-money markets, and consistent across a sample of several
thousand live trades. Rather than accept that number at face value, each
of the following was tested individually with real data:

| Question | Method | Finding |
|---|---|---|
| **Is the probability model itself sound?** | Backtested the model against real historical price data (independent of Kalshi's pricing) across ~2,300 simulated 15-minute windows | Well-calibrated. Brier score of 0.166 vs. 0.25 for random guessing; predicted probabilities closely matched actual outcome frequencies across every bucket |
| **Is our price data source wrong?** | Compared our live CoinGecko price against Kalshi's actual settlement value at the exact same moment, across 69 same-day resolved markets | No — average difference of 0.024%, 91.3% directional agreement. Ruled out |
| **Are early-window signals worse than late-window ones?** | Bucketed win rate by minutes remaining at decision time | No meaningful improvement near expiry; the theory wasn't supported by the data |
| **Is single-cycle noise the problem?** | Required a signal to hold the same direction across 2 consecutive scans before trading, instead of acting on the first reading | Made results *worse* (30.7% vs. ~38% baseline) — likely because genuine mispricings get corrected by the market quickly, so *persisting* disagreement is more often a sign the market is right and the model is wrong, not the reverse |

Also fixed along the way: a genuine bug in the *original* version of this
bot (crashed on any market with a null field, meaning it silently never
traded for months), an authentication signing bug, and a CSV schema
mismatch that was corrupting logged data after a mid-project field change.

## Honest conclusion

A Black-Scholes-style probability model, even a well-calibrated one, does
not appear to have an exploitable edge against Kalshi's short-duration
crypto markets at an 8-percentage-point disagreement threshold. The most
likely explanation: these are fast, liquid, likely well-priced markets,
and the model's disagreements with the market price more often reflect
noise in a per-cycle volatility estimate than genuine mispricing.

This isn't a coding failure to fix — it's a real answer, arrived at
through the same process a professional quant would use to test a
hypothesis: build it, deploy it, measure it, and trust the data over the
narrative.

## Stack

Python · AWS EC2 · Kalshi Trade API (RSA-PSS authenticated) · CoinGecko API
· pandas/numpy/scipy · Git/GitHub deployment workflow

---

*Built as a live, iterative research project — not a demo. Every number
above came from real trades against real market data.*
