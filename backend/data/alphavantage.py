import os
import threading
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
MIN_SECONDS_BETWEEN_REQUESTS = 1.05

_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_TS = 0.0


def load_alphavantage_api_key() -> str | None:
    load_dotenv(ROOT_DIR / ".env")
    return os.getenv("ALPHAVANTAGE_API_KEY")


def alphavantage_get_json(params: dict, timeout: int = 30) -> dict:
    global _LAST_REQUEST_TS

    with _REQUEST_LOCK:
        now = time.monotonic()
        wait_seconds = MIN_SECONDS_BETWEEN_REQUESTS - (now - _LAST_REQUEST_TS)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _LAST_REQUEST_TS = time.monotonic()

    response = requests.get(ALPHAVANTAGE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()
