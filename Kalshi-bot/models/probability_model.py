"""
probability_model.py

This module answers one question: "What is the true probability that
an asset (e.g. BTC) will be above a given strike price by a given
expiry time?"

We use the same core math behind the Black-Scholes options pricing
model. Kalshi's crypto markets are structured like binary options
("Will BTC be above $X at time T?"), so instead of pricing an actual
option premium, we just need the *probability* piece of that model,
often called N(d2) in options textbooks.

No trading logic lives here. This file's only job is: given market
conditions, compute a probability. Keeping it isolated like this
means we can unit test it independently and swap/improve the model
later without touching the rest of the bot.
"""

import math
from scipy.stats import norm


def probability_above_strike(
    spot_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Estimate the probability that the asset's price will be ABOVE
    strike_price at expiry.

    Parameters
    ----------
    spot_price : float
        Current price of the underlying asset (e.g. BTC spot price in USD).
    strike_price : float
        The threshold price the Kalshi contract is asking about.
    time_to_expiry_years : float
        Time remaining until the contract settles, expressed in YEARS.
        (e.g. 4 hours = 4 / 8760)
    volatility : float
        Annualized volatility of the asset, as a decimal (e.g. 0.60 for 60%).
        This should be estimated from recent historical price data —
        see data/volatility.py (built separately).
    risk_free_rate : float, optional
        Annualized risk-free interest rate, as a decimal. For short-dated
        crypto contracts (hours), this has a negligible effect, so it
        defaults to 0.0. Included for completeness / correctness.

    Returns
    -------
    float
        Probability (0.0 to 1.0) that spot_price ends up above
        strike_price at expiry.

    Raises
    ------
    ValueError
        If inputs are non-physical (e.g. negative price, zero time,
        zero/negative volatility) — better to fail loudly here than
        silently return a nonsense probability.
    """
    if spot_price <= 0 or strike_price <= 0:
        raise ValueError("spot_price and strike_price must be positive")
    if time_to_expiry_years <= 0:
        raise ValueError("time_to_expiry_years must be positive (contract already expired?)")
    if volatility <= 0:
        raise ValueError("volatility must be positive")

    # Step 1: log of the ratio between spot and strike.
    # This measures the % distance between where we are and the
    # threshold we're asking about, in a way that plays nicely with
    # volatility math (which is naturally multiplicative, not additive).
    log_moneyness = math.log(spot_price / strike_price)

    # Step 2: the drift term. This accounts for the fact that, under
    # risk-neutral pricing assumptions, the "expected" path of the
    # asset isn't flat — it drifts slightly based on volatility and
    # the risk-free rate. For short crypto windows this term is small
    # but we include it for correctness.
    drift = (risk_free_rate - 0.5 * volatility ** 2) * time_to_expiry_years

    # Step 3: scale volatility down to the size of our actual time window.
    # Volatility is quoted annualized, so we shrink it using sqrt(time) —
    # the standard scaling rule (variance scales linearly with time,
    # so standard deviation scales with the square root of time).
    vol_scaled = volatility * math.sqrt(time_to_expiry_years)

    # Step 4: compute d2 — how many "standard deviations" away the
    # strike is from spot, given time and volatility. This is the
    # same d2 from the Black-Scholes formula.
    d2 = (log_moneyness + drift) / vol_scaled

    # Step 5: convert d2 into an actual probability using the
    # cumulative distribution function (CDF) of the standard normal
    # distribution — i.e. "what fraction of the bell curve lies below
    # this many standard deviations?"
    probability = norm.cdf(d2)

    return probability


if __name__ == "__main__":
    # Quick sanity check using the exact example we walked through:
    # BTC at $65,000, strike $66,000, 4 hours out, 60% annualized vol.
    # Expected result: roughly 11-12% probability of finishing above strike.
    example_prob = probability_above_strike(
        spot_price=65000,
        strike_price=66000,
        time_to_expiry_years=4 / 8760,
        volatility=0.60,
    )
    print(f"Probability BTC > $66,000 in 4 hours: {example_prob:.2%}")
