"""
price_source_check.py

The CORRECT version of the CoinGecko-vs-Kalshi comparison. Our earlier
attempt in analyze_results.py compared our DECISION-TIME price against
Kalshi's SETTLEMENT-TIME price — two different moments — which meant
it was really just re-measuring normal price movement over time, not
a genuine data-source discrepancy.

This version fixes that: for each resolved market, we fetch
CoinGecko's fine-grained price at the closest point to the market's
ACTUAL close_time, and compare that directly to Kalshi's real
settlement value (expiration_value) at that same moment. This is
the fair, same-moment comparison we actually need.

LIMITATION: CoinGecko's free tier only provides fine-grained (~5min)
history for roughly the last 24 hours. Markets older than that get
skipped here rather than compared using misleadingly coarse data.
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from data.kalshi_client import KalshiClient
from data.volatility import fetch_recent_fine_grained_prices

load_dotenv()

TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.csv")

OLD_SCHEMA_FIELDS = ["timestamp", "ticker", "asset", "side", "model_prob", "market_prob",
                      "gap_pct", "action_taken", "size_dollars", "dry_run"]
NEW_SCHEMA_FIELDS = ["timestamp", "ticker", "asset", "side", "model_prob", "market_prob",
                      "gap_pct", "spot_price", "strike", "action_taken", "size_dollars", "dry_run"]


def load_tickers() -> list[dict]:
    """Load unique (ticker, asset) pairs from trades.csv, same schema-safe
    parsing as analyze_results.py."""
    import csv
    seen = set()
    trades = []
    with open(TRADE_LOG_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for raw_row in reader:
            if len(raw_row) == len(OLD_SCHEMA_FIELDS):
                row = dict(zip(OLD_SCHEMA_FIELDS, raw_row))
            elif len(raw_row) == len(NEW_SCHEMA_FIELDS):
                row = dict(zip(NEW_SCHEMA_FIELDS, raw_row))
            else:
                continue

            if row["action_taken"] and row["action_taken"].startswith(("DRY RUN", "LIVE ORDER")):
                if row["ticker"] not in seen:
                    seen.add(row["ticker"])
                    trades.append(row)
    return trades


def find_nearest_price(timestamps_ms: list[float], prices: list[float], target_ms: float) -> tuple[float, float]:
    """Returns (price, actual_time_diff_seconds) for the closest available data point."""
    idx = min(range(len(timestamps_ms)), key=lambda i: abs(timestamps_ms[i] - target_ms))
    diff_seconds = abs(timestamps_ms[idx] - target_ms) / 1000
    return prices[idx], diff_seconds


def main():
    client = KalshiClient(
        api_key_id=os.getenv("API_KEY_ID"),
        private_key_path=os.getenv("PRIVATE_KEY_PATH"),
    )

    trades = load_tickers()
    print(f"Checking {len(trades)} unique tickers for same-moment price agreement...\n")

    # Cache CoinGecko history per asset so we only fetch it once each.
    price_history_cache = {}

    checked = 0
    agree = 0
    skipped_too_old = 0
    skipped_unresolved = 0
    errors = 0
    price_diffs_pct = []

    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    COINGECKO_WINDOW_MS = 23 * 60 * 60 * 1000  # be conservative, CoinGecko gives ~24h of fine data
    MAX_ACCEPTABLE_GAP_SECONDS = 600  # if the nearest CoinGecko point is >10min away, don't trust the comparison

    for trade in trades:
        ticker = trade["ticker"]
        asset = trade["asset"]
        strike = trade.get("strike")

        try:
            market = client.get_market(ticker)
        except Exception:
            errors += 1
            continue

        if market.get("status") != "finalized" or not market.get("expiration_value"):
            skipped_unresolved += 1
            continue

        close_time_str = market.get("close_time")
        try:
            close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
            close_ms = close_time.timestamp() * 1000
        except (ValueError, TypeError):
            errors += 1
            continue

        if (now_ms - close_ms) > COINGECKO_WINDOW_MS:
            skipped_too_old += 1
            continue

        if asset not in price_history_cache:
            try:
                ts, prices = fetch_recent_fine_grained_prices(asset, days=1)
                price_history_cache[asset] = (ts, prices)
            except Exception as e:
                print(f"[ERROR] Could not fetch price history for {asset}: {e}")
                errors += 1
                continue

        ts_list, price_list = price_history_cache[asset]
        coingecko_price, gap_seconds = find_nearest_price(ts_list, price_list, close_ms)

        if gap_seconds > MAX_ACCEPTABLE_GAP_SECONDS:
            skipped_too_old += 1
            continue

        kalshi_price = float(market["expiration_value"])
        strike_val = float(strike) if strike else None

        checked += 1
        pct_diff = abs(coingecko_price - kalshi_price) / kalshi_price * 100
        price_diffs_pct.append(pct_diff)

        if strike_val:
            coingecko_side = coingecko_price >= strike_val
            kalshi_side = kalshi_price >= strike_val
            if coingecko_side == kalshi_side:
                agree += 1

        print(f"{ticker:<32} CoinGecko=${coingecko_price:,.2f}  Kalshi=${kalshi_price:,.2f}  "
              f"diff={pct_diff:.3f}%  gap_to_nearest_data={gap_seconds:.0f}s")

    print("\n" + "=" * 60)
    print(f"Checked: {checked}  |  Skipped (too old for CoinGecko fine data): {skipped_too_old}  |  "
          f"Skipped (unresolved): {skipped_unresolved}  |  Errors: {errors}")
    if checked > 0:
        print(f"Same-moment directional agreement: {agree/checked*100:.1f}% ({agree}/{checked})")
        avg_diff = sum(price_diffs_pct) / len(price_diffs_pct)
        max_diff = max(price_diffs_pct)
        print(f"Average price difference: {avg_diff:.4f}%  |  Max price difference: {max_diff:.4f}%")


if __name__ == "__main__":
    main()
