"""
analyze_results.py

Reads logs/trades.csv (every trade decision the bot has logged, real
or DRY_RUN), checks which of those markets have actually resolved by
now, and reports whether each trade would have won or lost — plus
overall win rate and estimated P&L.

This is what turns "here's what the bot decided" into "here's whether
the strategy actually works." Run this any time you want an updated
read on performance.

P&L ESTIMATE CAVEAT: we don't log the exact fill price for DRY_RUN
trades (there's no real fill). We approximate using the market's
implied probability at the time of the signal as the entry price —
a reasonable estimate, but real fills would include a small amount
of slippage this doesn't capture.
"""

import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from data.kalshi_client import KalshiClient

load_dotenv()

TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.csv")


# Every historical column layout the trade log has used, oldest first.
# The file's header can't be trusted (see load_dry_run_trades below),
# so we identify each row's layout purely by its column COUNT.
OLD_SCHEMA_FIELDS = ["timestamp", "ticker", "asset", "side", "model_prob", "market_prob",
                      "gap_pct", "action_taken", "size_dollars", "dry_run"]
SPOT_STRIKE_SCHEMA_FIELDS = ["timestamp", "ticker", "asset", "side", "model_prob", "market_prob",
                      "gap_pct", "spot_price", "strike", "action_taken", "size_dollars", "dry_run"]
PERSISTENCE_SCHEMA_FIELDS = ["timestamp", "ticker", "asset", "side", "model_prob", "market_prob",
                      "gap_pct", "spot_price", "strike", "persistence_count", "action_taken",
                      "size_dollars", "dry_run"]

ALL_SCHEMAS = [OLD_SCHEMA_FIELDS, SPOT_STRIKE_SCHEMA_FIELDS, PERSISTENCE_SCHEMA_FIELDS]


def load_dry_run_trades() -> list[dict]:
    """
    Load only the rows that represent an actual simulated/real trade
    (skip SKIPPED rows — those never entered a position).

    IMPORTANT: this file's header only reflects whichever schema was
    active the last time order_manager.py detected a schema change and
    started a fresh file — but rows within a file are internally
    consistent with each OTHER at least. We identify each row's schema
    by its column COUNT against every layout we've ever used, so this
    script keeps working correctly even after future schema changes,
    as long as each new layout gets added to ALL_SCHEMAS above.
    """
    trades = []
    with open(TRADE_LOG_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the (unreliable) header line
        for raw_row in reader:
            matching_schema = None
            for schema in ALL_SCHEMAS:
                if len(raw_row) == len(schema):
                    matching_schema = schema
                    break

            if matching_schema is None:
                continue  # malformed or unrecognized row, skip rather than crash

            row = dict(zip(matching_schema, raw_row))
            row.setdefault("spot_price", None)
            row.setdefault("strike", None)
            row.setdefault("persistence_count", None)

            if row["action_taken"] and row["action_taken"].startswith(("DRY RUN", "LIVE ORDER")):
                trades.append(row)
    return trades


def estimate_pnl(side: str, entry_prob: float, size_dollars: float, won: bool) -> float:
    """
    Estimate profit/loss for a single trade, given the side we took,
    the market-implied probability at entry (used as our estimated
    entry price), the amount risked, and whether it won.
    """
    entry_price = entry_prob if side == "yes" else (1 - entry_prob)
    entry_price = max(entry_price, 0.01)  # avoid divide-by-zero on extreme prices

    if won:
        return size_dollars * (1 / entry_price - 1)
    else:
        return -size_dollars


def main():
    client = KalshiClient(
        api_key_id=os.getenv("API_KEY_ID"),
        private_key_path=os.getenv("PRIVATE_KEY_PATH"),
    )

    trades = load_dry_run_trades()
    print(f"Found {len(trades)} logged trades to check.\n")

    wins, losses, unresolved, errors = 0, 0, 0, 0
    total_pnl = 0.0
    results_rows = []

    for trade in trades:
        ticker = trade["ticker"]
        side = trade["side"]
        size_dollars = float(trade["size_dollars"])
        entry_prob = float(trade["market_prob"])

        try:
            market = client.get_market(ticker)
        except Exception as e:
            print(f"[ERROR] Could not fetch {ticker}: {e}")
            errors += 1
            continue

        status = market.get("status", "")
        result = market.get("result", "")

        if status != "finalized" or result not in ("yes", "no"):
            unresolved += 1
            continue

        won = (result == side)
        pnl = estimate_pnl(side, entry_prob, size_dollars, won)
        total_pnl += pnl

        if won:
            wins += 1
        else:
            losses += 1

        # Pull our logged spot price (recorded at decision time) and
        # Kalshi's real settlement price (their BRTI-based value),
        # so we can check whether our price source agrees with what
        # Kalshi actually used to resolve the market.
        our_spot_price = trade.get("spot_price")
        kalshi_settlement_price = market.get("expiration_value")
        strike = trade.get("strike")

        price_source_agrees = None
        if our_spot_price and kalshi_settlement_price and strike:
            try:
                our_spot_price = float(our_spot_price)
                kalshi_settlement_price = float(kalshi_settlement_price)
                strike = float(strike)
                # Did our price and Kalshi's real price agree on which
                # side of the strike the asset was on?
                our_direction = our_spot_price >= strike
                kalshi_direction = kalshi_settlement_price >= strike
                price_source_agrees = (our_direction == kalshi_direction)
            except (ValueError, TypeError):
                pass

        # How much time was left on the clock when we made this decision?
        minutes_remaining = None
        try:
            decision_time = datetime.fromisoformat(trade["timestamp"])
            close_time = datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
            minutes_remaining = round((close_time - decision_time).total_seconds() / 60, 1)
        except (ValueError, TypeError, KeyError):
            pass

        persistence_count = trade.get("persistence_count")
        try:
            persistence_count = int(persistence_count) if persistence_count not in (None, "", "None") else None
        except (ValueError, TypeError):
            persistence_count = None

        results_rows.append({
            "ticker": ticker,
            "asset": trade["asset"],
            "side": side,
            "gap_pct": trade["gap_pct"],
            "result": result,
            "won": won,
            "estimated_pnl": round(pnl, 2),
            "our_spot_price": our_spot_price,
            "kalshi_settlement_price": kalshi_settlement_price,
            "strike": strike,
            "price_source_agrees": price_source_agrees,
            "minutes_remaining": minutes_remaining,
            "persistence_count": persistence_count,
        })

        outcome_str = "WIN " if won else "LOSS"
        print(f"[{outcome_str}] {ticker:<30} side={side:<3} gap={float(trade['gap_pct']):+.1f}pp  est_pnl=${pnl:+.2f}")

    print("\n" + "=" * 50)
    print(f"Resolved trades: {wins + losses}  |  Unresolved (still open): {unresolved}  |  Errors: {errors}")
    if wins + losses > 0:
        win_rate = wins / (wins + losses) * 100
        print(f"Win rate: {win_rate:.1f}%  ({wins}W / {losses}L)")
        print(f"Estimated total P&L: ${total_pnl:+.2f}")
    else:
        print("No resolved trades yet to calculate a win rate.")

    agree_checks = [r["price_source_agrees"] for r in results_rows if r["price_source_agrees"] is not None]
    if agree_checks:
        agree_rate = sum(agree_checks) / len(agree_checks) * 100
        print(f"\nCoinGecko-vs-Kalshi price agreement: {agree_rate:.1f}% ({sum(agree_checks)}/{len(agree_checks)} checked)")
    else:
        print("\nNo trades with logged spot_price yet to check price-source agreement (older trades predate that fix).")

    # Save detailed results for further review.
    results_file = os.path.join(os.path.dirname(__file__), "..", "logs", "resolved_results.csv")
    if results_rows:
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"\nDetailed results saved to {results_file}")


if __name__ == "__main__":
    main()
