"""
main.py

The bot's entry point. Runs the full cycle on a loop:

1. Scan all 4 assets' 15-minute markets for signals.
2. For every tradeable signal found, run it through the execution
   layer (circuit breaker check, position sizing, place/simulate order).
3. Sleep, then repeat.

DRY_RUN is set at the top — leave it True until we've watched the bot
run correctly for a while and are confident in its behavior. This is
the single switch that separates "watching the bot think" from
"the bot spending real money."
"""

import os
import time
from dotenv import load_dotenv

from data.kalshi_client import KalshiClient
from strategy.signal_generator import scan_markets_for_signals
from execution.order_manager import execute_signal

load_dotenv()

# THE MOST IMPORTANT LINE IN THIS FILE. Keep this True until we've
# confirmed the bot behaves correctly over many cycles.
DRY_RUN = True

# How often to scan for new signals, in seconds. These are 15-minute
# markets, so scanning every 60 seconds gives us plenty of chances to
# catch a good entry without hammering the APIs.
POLL_INTERVAL_SECONDS = 60


def build_client() -> KalshiClient:
    api_key_id = os.getenv("API_KEY_ID")
    private_key_path = os.getenv("PRIVATE_KEY_PATH")

    if not api_key_id or not private_key_path:
        raise RuntimeError("API_KEY_ID and PRIVATE_KEY_PATH must be set in .env")

    return KalshiClient(api_key_id=api_key_id, private_key_path=private_key_path)


def run_one_cycle(client: KalshiClient) -> None:
    """
    Run a single scan-and-act cycle. Wrapped in try/except at the call
    site so that an unexpected error in one cycle doesn't crash the
    whole bot — it logs the error and the loop continues next cycle.
    """
    results = scan_markets_for_signals(client)
    tradeable = [r for r in results if r["tradeable"]]

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Scanned {len(results)} markets, {len(tradeable)} tradeable signal(s)", flush=True)

    for signal in tradeable:
        execute_signal(client, signal, dry_run=DRY_RUN)


def main() -> None:
    print(f"Starting Kalshi bot. DRY_RUN = {DRY_RUN}", flush=True)
    client = build_client()

    while True:
        try:
            run_one_cycle(client)
        except Exception as e:
            # Exactly the lesson from the old bot: never let one bad
            # cycle crash the entire process. Log it and keep going.
            print(f"[ERROR] Cycle failed: {e}", flush=True)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
