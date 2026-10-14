"""
order_manager.py

This is the only module that actually places orders. Everything else
(signal_generator.py, risk_manager.py) just decides WHAT to do —
this module is where a decision becomes a real action.

DRY_RUN mode: when True, this module does everything a real trade
would do (logging, position sizing, circuit breaker checks) EXCEPT
actually call Kalshi's create_order endpoint. This lets us watch the
bot make real decisions against live data with zero financial risk,
before ever going live.

POSITION TRACKING: each Kalshi 15-minute market has a unique ticker
tied to its specific time window, so once we've entered a given
ticker, we should not enter it again — even across multiple scan
cycles before it expires. We track "already entered" tickers in a
small local JSON file (works identically for DRY_RUN and live, since
DRY_RUN never creates real Kalshi positions to check against). Old
entries are cleaned up automatically after they're old enough that
the underlying 15-minute market must have expired.
"""

import csv
import json
import time
import os
from datetime import datetime, timezone

from execution.risk_manager import (
    is_trading_halted,
    calculate_position_size,
    record_trade_result,
)

TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.csv")
POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "open_positions.json")

# 15-minute markets are stale well before this — generous buffer so we
# never accidentally forget a position while it's still genuinely open.
POSITION_MEMORY_SECONDS = 30 * 60  # 30 minutes


def _load_open_positions() -> dict:
    """Load our local record of tickers we've already entered, as {ticker: timestamp}."""
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_open_positions(positions: dict) -> None:
    os.makedirs(os.path.dirname(POSITIONS_FILE), exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f)


def _cleanup_stale_positions(positions: dict) -> dict:
    """Drop any tracked ticker old enough that its 15-minute market must have expired."""
    now = time.time()
    return {ticker: ts for ticker, ts in positions.items() if (now - ts) < POSITION_MEMORY_SECONDS}


def already_has_position(ticker: str) -> bool:
    """Check whether we've already entered this exact market ticker recently."""
    positions = _cleanup_stale_positions(_load_open_positions())
    return ticker in positions


def _mark_position_entered(ticker: str) -> None:
    """Record that we've just entered a position on this ticker."""
    positions = _cleanup_stale_positions(_load_open_positions())
    positions[ticker] = time.time()
    _save_open_positions(positions)


def _log_trade(row: dict) -> None:
    """
    Append a trade record to a CSV log file, creating it with a header
    the first time.

    SCHEMA SAFETY: if the code's fields ever change (like when we
    added spot_price/strike), the existing file's header would
    silently become wrong for every new row — the header describes
    the OLD layout while new rows have a different one, causing
    misaligned/mislabeled data with no error raised. To prevent that
    happening again, we check whether the file's current header
    actually matches what we're about to write; if not, we archive
    the old file (renamed with a timestamp, nothing is deleted) and
    start a fresh file with a correct, matching header.
    """
    os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
    file_exists = os.path.exists(TRADE_LOG_FILE)
    expected_header = list(row.keys())

    if file_exists:
        with open(TRADE_LOG_FILE, "r") as f:
            existing_header = f.readline().strip().split(",")
        if existing_header != expected_header:
            archive_path = TRADE_LOG_FILE + f".old_schema_{int(time.time())}"
            os.rename(TRADE_LOG_FILE, archive_path)
            print(f"[INFO] Trade log schema changed — archived old file to {archive_path}", flush=True)
            file_exists = False

    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=expected_header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def execute_signal(client, signal: dict, dry_run: bool = True) -> dict:
    """
    Act on a tradeable signal from signal_generator.py: check the
    circuit breaker, check for an existing position on this same
    market, size the position, and place (or simulate) the order.

    Parameters
    ----------
    client : KalshiClient
        Authenticated client, used to check balance and place orders.
    signal : dict
        A signal dict from signal_generator.generate_signal(), where
        signal["tradeable"] is True.
    dry_run : bool
        If True, simulate the trade (log it, but never call
        client.create_order). If False, place a REAL order with REAL
        money. Defaults to True as a safety default.

    Returns
    -------
    dict
        Outcome record: what happened and why, always returned even
        when no trade was placed.
    """
    outcome = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": signal["ticker"],
        "asset": signal["asset"],
        "side": signal["side"],
        "model_prob": signal["model_prob"],
        "market_prob": signal["market_prob"],
        "gap_pct": signal["gap_pct"],
        "spot_price": signal.get("spot_price"),
        "strike": signal.get("strike"),
        "persistence_count": signal.get("persistence_count"),
        "action_taken": None,
        "size_dollars": None,
        "dry_run": dry_run,
    }

    if is_trading_halted():
        outcome["action_taken"] = "SKIPPED — circuit breaker active (daily loss limit hit)"
        _log_trade(outcome)
        return outcome

    if already_has_position(signal["ticker"]):
        outcome["action_taken"] = "SKIPPED — already have an open position on this market"
        _log_trade(outcome)
        return outcome

    try:
        balance = client.get_balance()
    except Exception as e:
        outcome["action_taken"] = f"SKIPPED — failed to fetch balance: {e}"
        _log_trade(outcome)
        return outcome

    size_dollars = calculate_position_size(balance)
    outcome["size_dollars"] = size_dollars

    if size_dollars <= 0:
        outcome["action_taken"] = "SKIPPED — calculated position size was zero (insufficient balance?)"
        _log_trade(outcome)
        return outcome

    if dry_run:
        outcome["action_taken"] = f"DRY RUN — would buy '{signal['side']}' on {signal['ticker']} for ${size_dollars:.2f}"
        print(f"[DRY RUN] {outcome['action_taken']}", flush=True)
        _mark_position_entered(signal["ticker"])
        _log_trade(outcome)
        return outcome

    # Real order placement — only reached when dry_run=False.
    try:
        order_response = client.create_order(
            ticker=signal["ticker"],
            side=signal["side"],
            count=int(size_dollars),  # Kalshi contracts are priced near $1 each at $1 notional
            action="buy",
            order_type="market",
        )
        outcome["action_taken"] = f"LIVE ORDER PLACED — {order_response}"
        print(f"[LIVE] {outcome['action_taken']}", flush=True)
        _mark_position_entered(signal["ticker"])
    except Exception as e:
        outcome["action_taken"] = f"FAILED — order placement error: {e}"
        print(f"[ERROR] {outcome['action_taken']}", flush=True)

    _log_trade(outcome)
    return outcome
