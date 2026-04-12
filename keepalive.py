import logging
import os
import threading
import time

import requests


LOGGER = logging.getLogger(__name__)
_START_LOCK = threading.Lock()
_STARTED = False


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ping_loop(target_url: str, interval_seconds: int, timeout_seconds: int, service_name: str) -> None:
    session = requests.Session()
    while True:
        try:
            response = session.get(target_url, timeout=timeout_seconds)
            response.raise_for_status()
        except Exception as exc:
            LOGGER.warning("%s keepalive ping failed for %s: %s", service_name, target_url, exc)
        time.sleep(interval_seconds)


def start_keepalive_thread(service_name: str) -> bool:
    global _STARTED

    enabled = _env_flag("KEEPALIVE_ENABLED", default=False)
    target_url = os.getenv("KEEPALIVE_URL", "").strip()
    interval_seconds = int(os.getenv("KEEPALIVE_INTERVAL_SECONDS", "600"))
    timeout_seconds = int(os.getenv("KEEPALIVE_TIMEOUT_SECONDS", "30"))

    if not enabled or not target_url:
        return False

    with _START_LOCK:
        if _STARTED:
            return False

        thread = threading.Thread(
            target=_ping_loop,
            args=(target_url, interval_seconds, timeout_seconds, service_name),
            daemon=True,
        )
        thread.start()
        _STARTED = True
        LOGGER.info(
            "%s keepalive enabled: pinging %s every %ss",
            service_name,
            target_url,
            interval_seconds,
        )
        return True
