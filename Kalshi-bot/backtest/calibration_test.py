"""
calibration_test.py

Tests whether our probability model is actually CALIBRATED — meaning,
when the model says "70% chance," does the outcome actually happen
about 70% of the time historically? This is independent of Kalshi's
specific pricing; it's purely a test of whether our math holds up
against real price history.

METHOD: using one day of fine-grained (5-minute) historical price
data per asset, we simulate many overlapping 15-minute windows:
- "spot" = the price at the start of each window
- "strike" = same as spot (matching how Kalshi's 15-min markets work
  — the reference price IS the opening price)
- "outcome" = did the price 15 minutes later end up >= that reference?

For each window we compute our model's probability, then compare
predicted probabilities to what actually happened, bucketed into
deciles (0-10%, 10-20%, etc.) — a well-calibrated model's average
predicted probability in each bucket should roughly match the actual
win rate in that bucket.

IMPORTANT CAVEAT: this uses one day's overall realized volatility as
a single volatility input for every window in that day, for
simplicity. A more rigorous version would use a rolling volatility
estimate calculated only from data available before each window
(to avoid "lookahead bias" — using information from the future to
judge a prediction about the past). Treat this as a solid first
signal, not a final, fully rigorous backtest.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from data.volatility import fetch_recent_fine_grained_prices, calculate_annualized_volatility
from models.probability_model import probability_above_strike

ASSETS = ["BTC", "ETH", "SOL", "XRP"]

# How many 5-minute bars ahead constitutes "15 minutes" — depends on
# CoinGecko's actual returned spacing, which we measure rather than assume.
def bars_for_minutes(timestamps_ms: list[float], minutes: int) -> int:
    avg_interval_sec = (timestamps_ms[-1] - timestamps_ms[0]) / 1000 / (len(timestamps_ms) - 1)
    return max(1, round((minutes * 60) / avg_interval_sec))


def run_calibration_for_asset(asset: str) -> list[tuple[float, int]]:
    """
    Returns a list of (predicted_probability, actual_outcome) pairs
    for one asset, where actual_outcome is 1 if price ended at/above
    the window's fixed reference strike, 0 otherwise.

    IMPORTANT: this mirrors what the LIVE bot actually does — for
    each 15-minute window, the strike is FIXED at the window's
    opening price, but we evaluate the model's probability at every
    point DURING that window as price drifts away from that strike
    and time-to-expiry shrinks. An earlier version of this test
    mistakenly re-set strike=current price at every evaluation point,
    which trivially always gives ~50% (of course "will price end
    above itself right now" is a coin flip at the instant you ask).
    This version tests the real scenario: strike fixed, spot moving.
    """
    timestamps_ms, prices = fetch_recent_fine_grained_prices(asset, days=1)

    if len(prices) < 20:
        print(f"[WARN] Not enough data for {asset}, skipping")
        return []

    avg_interval_sec = (timestamps_ms[-1] - timestamps_ms[0]) / 1000 / (len(timestamps_ms) - 1)
    periods_per_year = (365 * 24 * 60 * 60) / avg_interval_sec
    day_volatility = calculate_annualized_volatility(prices, periods_per_year=periods_per_year)

    horizon_bars = bars_for_minutes(timestamps_ms, 15)

    results = []
    # For each window, the strike is FIXED at the opening price.
    for i in range(len(prices) - horizon_bars):
        strike = prices[i]
        final_price = prices[i + horizon_bars]
        outcome = 1 if final_price >= strike else 0

        # Evaluate at every point DURING this window (not just at the
        # start), as the live bot actually does — spot moves, strike
        # stays fixed, time-to-expiry shrinks toward zero.
        for j in range(i + 1, i + horizon_bars):
            spot = prices[j]
            bars_remaining = (i + horizon_bars) - j
            time_to_expiry_years = (bars_remaining * avg_interval_sec) / (365 * 24 * 60 * 60)

            if time_to_expiry_years <= 0:
                continue

            try:
                predicted_prob = probability_above_strike(
                    spot_price=spot,
                    strike_price=strike,
                    time_to_expiry_years=time_to_expiry_years,
                    volatility=day_volatility,
                )
            except ValueError:
                continue

            results.append((predicted_prob, outcome))

    return results


def analyze_calibration(all_results: list[tuple[float, int]]) -> None:
    """Bucket predictions into deciles and compare predicted vs. actual rates."""
    probs = np.array([r[0] for r in all_results])
    outcomes = np.array([r[1] for r in all_results])

    print(f"\nTotal windows analyzed: {len(all_results)}")

    brier_score = np.mean((probs - outcomes) ** 2)
    print(f"Brier score: {brier_score:.4f}  (0 = perfect, 0.25 = random guessing, 1 = always wrong)")

    print(f"\n{'Predicted range':<18}{'Avg predicted':<16}{'Actual rate':<14}{'Count'}")
    print("-" * 60)
    bins = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probs >= low) & (probs < high)
        if mask.sum() == 0:
            continue
        avg_pred = probs[mask].mean()
        actual_rate = outcomes[mask].mean()
        print(f"{low:.0%}-{high:.0%}{'':<10}{avg_pred:<16.1%}{actual_rate:<14.1%}{mask.sum()}")


def main():
    all_results = []
    for asset in ASSETS:
        print(f"Fetching and analyzing {asset}...")
        results = run_calibration_for_asset(asset)
        all_results.extend(results)
        print(f"  {len(results)} windows collected")

    analyze_calibration(all_results)


if __name__ == "__main__":
    main()
