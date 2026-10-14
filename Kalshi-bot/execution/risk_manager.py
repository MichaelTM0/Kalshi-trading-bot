"""
risk_manager.py

This module decides HOW MUCH to risk on a trade, and WHETHER we're
even allowed to trade right now at all. It does not evaluate whether
a signal is good (that's signal_generator.py's job) — it only answers
"given a good signal, is it safe to act on it, and how big should the
position be?"

Two responsibilities:
1. Position sizing — never risk more than a small, fixed fraction of
   the account balance on any single trade.
2. Circuit breaker — track consecutive losses. After MAX_CONSECUTIVE_LOSSES
   in a row, stop trading for the rest of the day and only resume the
   next calendar day. This protects against a bad regime (e.g. our
   volatility estimate being off) from compounding into a wipeout.
"""

import json
import os
from datetime import datetime, timezone

# Never risk more than this fraction of account balance on one trade.
MAX_RISK_PCT_PER_TRADE = 0.02  # 2%

# Absolute dollar cap per trade too, regardless of balance — a safety
# net against a balance/percentage calculation bug risking too much.
MAX_RISK_DOLLARS_PER_TRADE = 25.0

# Circuit breaker: stop trading for the day after this many losses in a row.
MAX_CONSECUTIVE_LOSSES = 5

# Where we persist circuit breaker state between bot restarts. If the
# bot crashes and restarts, it should NOT forget it already hit the
# loss limit today — that would defeat the whole point of the breaker.
STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "risk_state.json")


def _load_state() -> dict:
    """
    Load circuit breaker state from disk. If the file doesn't exist yet
    (first run) or is corrupted, start fresh rather than crashing —
    this is exactly the "don't let one bad thing take down everything"
    principle from earlier in the project.
    """
    default_state = {"consecutive_losses": 0, "last_trade_date": None, "halted": False}

    if not os.path.exists(STATE_FILE):
        return default_state

    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default_state


def _save_state(state: dict) -> None:
    """Persist circuit breaker state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def _today_str() -> str:
    """Today's date in UTC, as a simple string. Used to detect day changes."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def is_trading_halted() -> bool:
    """
    Check whether the circuit breaker is currently active. Automatically
    resets at the start of a new calendar day (UTC) — this is what
    makes it a DAILY circuit breaker rather than a permanent one.

    Returns
    -------
    bool
        True if trading should be paused right now.
    """
    state = _load_state()
    today = _today_str()

    # New day since the halt was triggered -> reset automatically.
    if state["last_trade_date"] != today:
        state["consecutive_losses"] = 0
        state["halted"] = False
        state["last_trade_date"] = today
        _save_state(state)
        return False

    return state["halted"]


def record_trade_result(won: bool) -> None:
    """
    Update the circuit breaker state after a trade resolves. Call this
    whenever a position closes and we know if it won or lost.

    Parameters
    ----------
    won : bool
        True if the trade was profitable, False if it lost.
    """
    state = _load_state()
    today = _today_str()

    if state["last_trade_date"] != today:
        state["consecutive_losses"] = 0
        state["halted"] = False

    if won:
        state["consecutive_losses"] = 0
    else:
        state["consecutive_losses"] += 1
        if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
            state["halted"] = True
            print(f"[CIRCUIT BREAKER] {MAX_CONSECUTIVE_LOSSES} consecutive losses reached — halting trading for the rest of today.", flush=True)

    state["last_trade_date"] = today
    _save_state(state)


def calculate_position_size(account_balance: float) -> float:
    """
    Determine how many dollars to risk on a trade, given current
    account balance. Takes the SMALLER of a percentage-based limit
    and a hard dollar cap — two independent safety nets rather than
    relying on just one.

    Parameters
    ----------
    account_balance : float
        Current account balance in dollars.

    Returns
    -------
    float
        Dollar amount to risk on this trade.
    """
    percent_based = account_balance * MAX_RISK_PCT_PER_TRADE
    return round(min(percent_based, MAX_RISK_DOLLARS_PER_TRADE), 2)
