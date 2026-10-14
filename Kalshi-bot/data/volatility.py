"""
volatility.py

This module answers: "How volatile has this crypto asset actually
been recently?" — which is the missing input our probability model
needs (we can't just guess 60% forever; real volatility shifts
day to day, and differs between BTC, ETH, SOL, and XRP).

Approach:
1. Pull recent historical hourly prices for an asset from CoinGecko's
   free public API.
2. Convert that price series into "log returns" (the standard way
   quant finance measures period-to-period price changes).
3. Compute the standard deviation of those returns — that's our
   volatility measurement.
4. Annualize it, so it plugs directly into probability_model.py.

Kept separate from probability_model.py so each piece can be tested,
debugged, and improved independently.

CACHING: CoinGecko's free tier rate-limits aggressively (429 errors
if we call it too often, which we hit scanning 4 assets back to back).
Spot prices and volatility don't change meaningfully second-to-second,
so we cache both briefly in memory.
"""

import time
import requests
import numpy as np

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
}

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

_spot_price_cache = {}
_volatility_cache = {}

SPOT_PRICE_CACHE_SECONDS = 15
VOLATILITY_CACHE_SECONDS = 300  # 5 minutes — volatility barely moves cycle to cycle


def get_current_spot_prices(assets: list[str]) -> dict[str, float]:
    """
    Fetch current live prices for multiple assets in ONE API call,
    instead of one call per asset. Faster, and avoids CoinGecko's
    rate limiting when scanning all 4 assets each cycle.
    """
    now = time.time()

    if all(a in _spot_price_cache and (now - _spot_price_cache[a][1]) < SPOT_PRICE_CACHE_SECONDS for a in assets):
        return {a: _spot_price_cache[a][0] for a in assets}

    unknown = [a for a in assets if a not in COINGECKO_IDS]
    if unknown:
        raise ValueError(f"Unknown asset(s) {unknown}. Supported: {list(COINGECKO_IDS.keys())}")

    coingecko_ids = ",".join(COINGECKO_IDS[a] for a in assets)
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coingecko_ids, "vs_currencies": "usd"}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    prices = {}
    for asset in assets:
        cg_id = COINGECKO_IDS[asset]
        price = data[cg_id]["usd"]
        prices[asset] = price
        _spot_price_cache[asset] = (price, now)

    return prices


def get_current_spot_price(asset: str) -> float:
    """Convenience wrapper for a single asset. Prefer get_current_spot_prices()
    when fetching multiple assets in the same cycle."""
    return get_current_spot_prices([asset])[asset]


def fetch_recent_fine_grained_prices(asset: str, days: int = 1) -> tuple[list[float], list[float]]:
    """
    Fetch CoinGecko's finest available granularity — asking for a
    short window (days=1) returns ~5-minute interval data instead of
    hourly, which matters a lot for a 15-minute strategy: hourly data
    is simply too coarse to measure how the price actually behaves
    within a 15-minute window.

    Parameters
    ----------
    asset : str
        One of "BTC", "ETH", "SOL", "XRP".
    days : int
        How many days back to pull. CoinGecko switches to hourly data
        above roughly 1-2 days, so keep this small to preserve
        fine-grained spacing.

    Returns
    -------
    tuple[list[float], list[float]]
        (timestamps_ms, prices) — we return timestamps too because we
        need to measure the ACTUAL spacing between data points rather
        than assume it, since CoinGecko's exact interval isn't
        officially guaranteed and can vary slightly.
    """
    if asset not in COINGECKO_IDS:
        raise ValueError(f"Unknown asset '{asset}'. Supported: {list(COINGECKO_IDS.keys())}")

    coingecko_id = COINGECKO_IDS[asset]
    url = COINGECKO_URL.format(id=coingecko_id)
    params = {"vs_currency": "usd", "days": days}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    timestamps = [point[0] for point in data["prices"]]
    prices = [point[1] for point in data["prices"]]

    return timestamps, prices


def get_short_term_volatility(asset: str) -> float:
    """
    Calculate annualized volatility from RECENT, FINE-GRAINED price
    data, instead of scaling a 7-day hourly estimate down
    mathematically via sqrt(time).

    WHY THIS MATTERS: volatility measured over hours/days doesn't
    reliably predict how much an asset moves in a 15-minute window —
    short horizons often behave calmer (relatively speaking) than a
    naive square-root-of-time scaling from longer data assumes. Live
    trading data showed our old approach was overestimating short-term
    volatility, making the probability model overconfident about big
    swings being plausible even when the market had already correctly
    converged toward a near-certain outcome.

    This function measures volatility directly from data at close to
    the actual timescale we're trading, rather than extrapolating from
    a different regime.

    Parameters
    ----------
    asset : str
        One of "BTC", "ETH", "SOL", "XRP".

    Returns
    -------
    float
        Annualized volatility as a decimal, based on genuinely
        short-horizon recent price behavior.
    """
    now = time.time()
    cache_key = ("short_term", asset)

    if cache_key in _volatility_cache:
        cached_value, cached_time = _volatility_cache[cache_key]
        if (now - cached_time) < VOLATILITY_CACHE_SECONDS:
            return cached_value

    timestamps_ms, prices = fetch_recent_fine_grained_prices(asset, days=1)

    if len(timestamps_ms) < 10:
        # Not enough fine-grained data to trust — fall back to the
        # older hourly-based estimate rather than compute volatility
        # from too few data points (which would be noisy/unreliable).
        return get_current_volatility(asset)

    # Measure the ACTUAL average spacing between data points, in
    # seconds, instead of assuming a fixed interval — CoinGecko's
    # exact granularity isn't officially guaranteed.
    intervals_seconds = [(timestamps_ms[i] - timestamps_ms[i - 1]) / 1000 for i in range(1, len(timestamps_ms))]
    avg_interval_seconds = sum(intervals_seconds) / len(intervals_seconds)

    seconds_per_year = 365 * 24 * 60 * 60
    periods_per_year = seconds_per_year / avg_interval_seconds

    volatility = calculate_annualized_volatility(prices, periods_per_year=periods_per_year)
    _volatility_cache[cache_key] = (volatility, now)

    return volatility


def fetch_hourly_prices(asset: str, days: int = 7) -> list[float]:
    """Fetch recent hourly closing prices for a given asset."""
    if asset not in COINGECKO_IDS:
        raise ValueError(f"Unknown asset '{asset}'. Supported: {list(COINGECKO_IDS.keys())}")

    coingecko_id = COINGECKO_IDS[asset]
    url = COINGECKO_URL.format(id=coingecko_id)
    params = {"vs_currency": "usd", "days": days}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    prices = [point[1] for point in data["prices"]]
    return prices


def calculate_annualized_volatility(prices: list[float], periods_per_year: int = 8760) -> float:
    """Calculate annualized volatility from a series of prices using log returns."""
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to calculate volatility")

    prices_arr = np.array(prices)
    log_returns = np.diff(np.log(prices_arr))
    period_volatility = np.std(log_returns)
    annualized_volatility = period_volatility * np.sqrt(periods_per_year)

    return float(annualized_volatility)


def get_current_volatility(asset: str, lookback_days: int = 7) -> float:
    """Fetch recent price history for an asset and return its current
    annualized volatility. Cached since this is a heavier call and the
    result barely changes cycle to cycle."""
    now = time.time()
    cache_key = (asset, lookback_days)

    if cache_key in _volatility_cache:
        cached_value, cached_time = _volatility_cache[cache_key]
        if (now - cached_time) < VOLATILITY_CACHE_SECONDS:
            return cached_value

    prices = fetch_hourly_prices(asset, days=lookback_days)
    volatility = calculate_annualized_volatility(prices, periods_per_year=8760)
    _volatility_cache[cache_key] = (volatility, now)

    return volatility


if __name__ == "__main__":
    prices = get_current_spot_prices(["BTC", "ETH", "SOL", "XRP"])
    for symbol, price in prices.items():
        print(f"{symbol}: ${price:,.2f}")

    for symbol in ["BTC", "ETH", "SOL", "XRP"]:
        try:
            vol = get_current_volatility(symbol)
            print(f"{symbol}: {vol:.1%} annualized volatility")
        except Exception as e:
            print(f"{symbol}: failed to fetch — {e}")
