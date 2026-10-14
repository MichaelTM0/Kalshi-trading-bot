"""
signal_generator.py

This is where everything we've built comes together. For each open
Kalshi crypto market, we:

1. Figure out what asset, strike price, and expiry time the market
   is asking about.
2. Get that asset's current spot price and recent volatility.
3. Feed all of that into our probability model to get OUR estimate
   of the true probability.
4. Compare that to what Kalshi's current price implies the crowd
   thinks the probability is.
5. If the gap between the two is bigger than our threshold AND has
   persisted across multiple consecutive scans, generate a trade signal.

This module does NOT place any orders — it only decides what a good
trade would look like. Actually executing trades (with position
sizing, risk limits, the circuit breaker) is the execution layer's job.
Keeping these separate means we can unit test "is this a good signal?"
completely independently from "should we actually risk money on it?"

PERSISTENCE FILTER: after weeks of live testing showed a stable ~38%
win rate despite the model itself being well-calibrated in isolation,
the leading theory is that single-cycle gaps are often just noise in
our per-cycle volatility estimate, not real mispricing — we end up
trading whenever OUR estimate is off, not necessarily when the market
is. This filter requires a signal to hold the same direction across
multiple consecutive scans before we act on it, to test whether
filtering out one-off noisy readings improves results.
"""

from datetime import datetime, timezone
from models.probability_model import probability_above_strike
from data.volatility import get_short_term_volatility, get_current_spot_price

# Only trade signals where our model and the market disagree by at
# least this many percentage points. Below this, the gap is more
# likely just noise/estimation error than a real edge. This is a
# tunable parameter we'll refine with backtesting.
SIGNAL_THRESHOLD_PCT = 8.0

# NEW: how many consecutive scans (same ticker, same direction) a gap
# must hold before we treat it as tradeable, instead of acting on the
# very first reading. 1 = old behavior (no filter). Testing 2 first —
# cheap to change and re-test.
REQUIRED_PERSISTENCE_CYCLES = 2

# Tracks, per ticker, which direction the gap has been pointing and for
# how many consecutive scans in a row. Module-level so it persists across
# scan cycles within one running process (main.py's long-lived loop).
# Doesn't need to survive a restart — if the bot restarts, we just start
# counting fresh, which is fine (worst case: a few extra cycles before
# the first trade after a restart).
_signal_persistence = {}

# Kalshi's actual series tickers for short-duration (15-minute)
# crypto up/down markets, confirmed by inspecting live API data.
# Each market in these series asks "will the price in 15 minutes be
# >= the reference price at market open?" — which maps directly onto
# our probability model (strike = today's reference price instead of
# an arbitrary distant target, same underlying math).
ASSET_SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}

TRADED_ASSETS = list(ASSET_SERIES.keys())


def get_strike(market: dict) -> float:
    """
    Get the reference/strike price for a market. Confirmed via live
    API inspection: Kalshi's 15-minute crypto markets expose this as
    'floor_strike' — for these markets it represents the reference
    price at market open, and the market resolves YES if the price
    15 minutes later is at or above it.

    Parameters
    ----------
    market : dict
        A raw market object.

    Returns
    -------
    float or None
        The strike/reference price, or None if missing.
    """
    strike = market.get("floor_strike")
    return float(strike) if strike is not None else None


def get_time_to_expiry_years(market: dict) -> float:
    """
    Calculate time remaining until this market settles, in years
    (matching what probability_model.py expects).

    Parameters
    ----------
    market : dict
        A raw market object.

    Returns
    -------
    float or None
        Time to expiry in years, or None if the market has no valid
        close_time or it's already expired.
    """
    close_time_str = market.get("close_time")
    if not close_time_str:
        return None

    try:
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    now = datetime.now(timezone.utc)
    seconds_remaining = (close_time - now).total_seconds()

    if seconds_remaining <= 0:
        return None

    return seconds_remaining / (365 * 24 * 60 * 60)


def get_market_implied_probability(market: dict) -> float:
    """
    Convert Kalshi's current bid/ask prices into an implied probability.
    Confirmed via live API inspection: Kalshi's 'yes_bid_dollars' and
    'yes_ask_dollars' fields are already decimal probabilities (e.g.
    "0.8400" means 84%), NOT cents — no division needed.

    Parameters
    ----------
    market : dict
        A raw market object.

    Returns
    -------
    float or None
        Implied probability (0.0-1.0), or None if bid/ask data is missing.
    """
    yes_bid_str = market.get("yes_bid_dollars")
    yes_ask_str = market.get("yes_ask_dollars")

    if yes_bid_str is None or yes_ask_str is None:
        return None

    try:
        yes_bid = float(yes_bid_str)
        yes_ask = float(yes_ask_str)
    except (ValueError, TypeError):
        return None

    return (yes_bid + yes_ask) / 2


def _update_persistence(ticker: str, side: str) -> int:
    """
    Update the consecutive-cycle counter for this ticker's current
    gap direction. Returns the updated count.

    If the direction flipped (was "yes", now "no", or vice versa),
    the streak resets to 1 — a flip means the gap wasn't actually
    stable, so it shouldn't count toward persistence.
    """
    existing = _signal_persistence.get(ticker)

    if existing and existing["side"] == side:
        existing["count"] += 1
    else:
        existing = {"side": side, "count": 1}

    _signal_persistence[ticker] = existing
    return existing["count"]


def _clear_persistence(ticker: str) -> None:
    """Remove a ticker's persistence tracking once it's no longer relevant
    (gap dropped back below threshold, or the ticker's market resolved)."""
    _signal_persistence.pop(ticker, None)


def generate_signal(market: dict, asset: str) -> dict:
    """
    Evaluate a single market and decide if it's a tradeable signal.

    Parameters
    ----------
    market : dict
        A raw market object from KalshiClient.get_open_markets().
    asset : str
        Which asset this market belongs to (known from which series
        we fetched it from — e.g. "BTC" for anything pulled via the
        KXBTC15M series).

    Returns
    -------
    dict
        Always returns a dict describing the outcome, even when there's
        no trade — this makes logging/debugging easier than returning
        None in some cases and a dict in others. Shape:
        {
            "ticker": str,
            "asset": str,
            "tradeable": bool,
            "reason": str,       # why not tradeable, if applicable
            "side": str | None,  # "yes" or "no" if tradeable
            "model_prob": float | None,
            "market_prob": float | None,
            "gap_pct": float | None,
            "spot_price": float | None,
            "strike": float | None,
            "persistence_count": int | None,  # how many consecutive
                                               # cycles this direction held
        }
    """
    ticker = market.get("ticker", "UNKNOWN")
    result = {
        "ticker": ticker,
        "asset": asset,
        "tradeable": False,
        "reason": None,
        "side": None,
        "model_prob": None,
        "market_prob": None,
        "gap_pct": None,
        "spot_price": None,
        "strike": None,
        "persistence_count": None,
    }

    strike = get_strike(market)
    if strike is None:
        result["reason"] = "Missing floor_strike"
        _clear_persistence(ticker)
        return result

    time_to_expiry = get_time_to_expiry_years(market)
    if time_to_expiry is None:
        result["reason"] = "Missing or invalid expiry time"
        _clear_persistence(ticker)
        return result

    market_prob = get_market_implied_probability(market)
    if market_prob is None:
        result["reason"] = "Missing bid/ask data"
        return result

    try:
        spot_price = get_current_spot_price(asset)
        volatility = get_short_term_volatility(asset)
    except Exception as e:
        result["reason"] = f"Failed to fetch market data: {e}"
        return result

    try:
        model_prob = probability_above_strike(
            spot_price=spot_price,
            strike_price=strike,
            time_to_expiry_years=time_to_expiry,
            volatility=volatility,
        )
    except ValueError as e:
        result["reason"] = f"Probability model rejected inputs: {e}"
        return result

    gap_pct = (model_prob - market_prob) * 100

    result["model_prob"] = model_prob
    result["market_prob"] = market_prob
    result["gap_pct"] = gap_pct
    result["spot_price"] = spot_price
    result["strike"] = strike

    if abs(gap_pct) < SIGNAL_THRESHOLD_PCT:
        result["reason"] = f"Gap ({gap_pct:+.1f}pp) below threshold ({SIGNAL_THRESHOLD_PCT}pp)"
        _clear_persistence(ticker)  # gap dropped out — reset the streak
        return result

    # Gap exceeds threshold — figure out direction and update persistence.
    side = "yes" if gap_pct > 0 else "no"
    persistence_count = _update_persistence(ticker, side)
    result["persistence_count"] = persistence_count

    if persistence_count < REQUIRED_PERSISTENCE_CYCLES:
        result["reason"] = (f"Gap ({gap_pct:+.1f}pp) exceeds threshold but only "
                             f"{persistence_count}/{REQUIRED_PERSISTENCE_CYCLES} consecutive cycles so far")
        return result

    # Persisted long enough — this is a real, tradeable signal.
    result["tradeable"] = True
    result["side"] = side
    result["reason"] = (f"Model/market gap of {gap_pct:+.1f}pp exceeded threshold and held for "
                         f"{persistence_count} consecutive cycles")

    return result


def scan_markets_for_signals(client) -> list[dict]:
    """
    Scan open 15-minute markets across all our traded assets and
    return every evaluated result (tradeable or not) — useful for
    logging/debugging what the bot is seeing each cycle, not just
    what it decides to trade.

    Parameters
    ----------
    client : KalshiClient
        An authenticated client instance.

    Returns
    -------
    list[dict]
        One result dict (see generate_signal) per market evaluated.
    """
    results = []

    for asset, series_ticker in ASSET_SERIES.items():
        try:
            markets = client.get_open_markets(series_ticker=series_ticker)
        except Exception as e:
            print(f"[WARN] Failed to fetch markets for {asset} ({series_ticker}): {e}", flush=True)
            continue

        for market in markets:
            try:
                result = generate_signal(market, asset)
                results.append(result)
            except Exception as e:
                # Exactly the lesson from the old bot: one bad market
                # should never crash the whole scan. Log and continue.
                print(f"[WARN] Unexpected error evaluating market {market.get('ticker', 'UNKNOWN')}: {e}", flush=True)
                continue

    return results
