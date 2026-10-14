"""
kalshi_client.py

A minimal, direct client for Kalshi's REST API. We're deliberately NOT
using the kalshi_python_sync SDK here — that's the library whose
strict Pydantic auto-validation crashed the old bot's entire market
scan whenever a single market had a null field.

This client just makes plain HTTP requests and returns raw JSON
(Python dicts/lists). We pull out exactly the fields we need, with
safe defaults, so one weird market from Kalshi's API can never take
down anything else.

Handles two kinds of requests:
- Public endpoints (like fetching open markets) — no auth needed.
- Private/authenticated endpoints (balance, positions, orders) — these
  require a signed request using your Kalshi private key.
"""

import time
import base64
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    def __init__(self, api_key_id: str, private_key_path: str, host: str = "https://api.elections.kalshi.com/trade-api/v2"):
        """
        Parameters
        ----------
        api_key_id : str
            Your Kalshi API Key ID (from .env's API_KEY_ID).
        private_key_path : str
            Path to your private key .pem file (from .env's PRIVATE_KEY_PATH).
        host : str
            Kalshi's API base URL.
        """
        self.api_key_id = api_key_id
        self.host = host

        with open(private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign_request(self, method: str, path: str) -> dict:
        """
        Build the authentication headers Kalshi requires for private
        endpoints. Kalshi uses RSA-PSS signatures: we sign a string
        made of (timestamp + HTTP method + path) with our private key,
        and send that signature along with our Key ID and timestamp.
        Kalshi verifies it on their end using the public key tied to
        our Key ID.

        Returns
        -------
        dict
            Headers to attach to the request.
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")

        signature = self.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        signature_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    def get_open_markets(self, series_ticker: str = None, limit: int = 200) -> list[dict]:
        """
        Fetch currently open markets as raw dicts (no SDK validation).
        This is a public endpoint — no signing needed.

        Parameters
        ----------
        series_ticker : str, optional
            Filter to a specific series (e.g. crypto price-range series).
            If None, fetches across all open markets.
        limit : int
            Max markets to return per page.

        Returns
        -------
        list[dict]
            Raw market objects exactly as Kalshi's API returns them.
            Fields may be missing or null — callers must handle that,
            not assume every field is present (this is the lesson from
            the old bot's crash).
        """
        path = "/markets"
        params = {"status": "open", "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker

        response = requests.get(f"{self.host}{path}", params=params, timeout=10)
        response.raise_for_status()

        return response.json().get("markets", [])

    def get_market(self, ticker: str) -> dict:
        """
        Fetch a single market by its exact ticker, regardless of whether
        it's still open or already resolved. Used for checking whether
        a trade we logged has settled, and what it resolved to.
        Public endpoint — no signing needed.

        Parameters
        ----------
        ticker : str
            The exact market ticker (e.g. "KXBTC15M-26SEP011200-00").

        Returns
        -------
        dict
            The raw market object, including 'status' ("active",
            "finalized", etc.) and 'result' ("yes", "no", or "" if
            not yet resolved).
        """
        path = f"/markets/{ticker}"
        response = requests.get(f"{self.host}{path}", timeout=10)
        response.raise_for_status()
        return response.json().get("market", {})

    def get_balance(self) -> float:
        """
        Fetch account balance in dollars. Authenticated endpoint.
        """
        path = "/trade-api/v2/portfolio/balance"
        headers = self._sign_request("GET", path)

        response = requests.get(f"{self.host}/portfolio/balance", headers=headers, timeout=10)
        response.raise_for_status()

        balance_cents = response.json().get("balance", 0)
        return balance_cents / 100

    def get_positions(self) -> dict:
        """
        Fetch current open positions. Authenticated endpoint.
        Returns raw dict with 'market_positions' and 'event_positions' lists.
        """
        path = "/trade-api/v2/portfolio/positions"
        headers = self._sign_request("GET", path)

        response = requests.get(f"{self.host}/portfolio/positions", headers=headers, timeout=10)
        response.raise_for_status()

        return response.json()

    def create_order(self, ticker: str, side: str, count: int, action: str = "buy", order_type: str = "market") -> dict:
        """
        Place an order. Authenticated endpoint.

        Parameters
        ----------
        ticker : str
            Market ticker to trade.
        side : str
            'yes' or 'no'.
        count : int
            Number of contracts.
        action : str
            'buy' or 'sell'.
        order_type : str
            'market' or 'limit'.

        Returns
        -------
        dict
            Kalshi's order confirmation response.
        """
        path = "/trade-api/v2/portfolio/orders"
        headers = self._sign_request("POST", path)
        body = {
            "ticker": ticker,
            "side": side,
            "count": count,
            "action": action,
            "type": order_type,
        }

        response = requests.post(f"{self.host}/portfolio/orders", headers=headers, json=body, timeout=10)
        response.raise_for_status()

        return response.json()
