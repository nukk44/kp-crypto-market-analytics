import os
import time
import requests
from datetime import datetime, timezone
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Можно переопределить через переменную окружения BINANCE_HOSTS
_HOSTS = [
    h.strip() for h in os.getenv(
        "BINANCE_HOSTS",
        "https://api4.binance.com,https://api.binance.com,https://api1.binance.com,https://api2.binance.com,https://api3.binance.com"
    ).split(",")
    if h.strip()
]

def _ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def _get_json(path: str, params: dict, timeout: int = 20):
    """Ходим по списку хостов, пока один не ответит 200."""
    last_err = None
    for base in _HOSTS:
        url = f"{base}{path}"
        try:
            r = requests.get(url, params=params, timeout=timeout)
            # 451/403/429/5xx — пробуем следующий хост
            if r.status_code in (451, 403, 429, 500, 502, 503):
                last_err = requests.HTTPError(f"{r.status_code} from {url}")
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            continue
        finally:
            time.sleep(0.1)  # чуть-чуть притормозим
    raise last_err or RuntimeError("All Binance hosts failed")

def get_klines(*,
               symbol: str,
               interval: str,
               start_time: datetime,
               end_time: datetime,
               limit: int = 1000,
               sleep_sec: float = 0.1,
               ) -> List[List]:
    """
    Возвращает "сырые" свечи Binance (или синтетические в OFFLINE).
    """
    # OFFLINE режим для тестов/CI: возвращаем маленький синтетический набор
    if os.getenv("OFFLINE") == "1":
        start_ms = _ms(start_time)
        step = 60_000  # 1m
        out = []
        for i in range(min(limit, 10)):
            t = start_ms + i * step
            out.append([t, "1.0", "1.0", "1.0", "1.0", "0", t + step - 1, "0", 0, "0", "0", "0"])
        time.sleep(sleep_sec)
        return out

    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": _ms(start_time),
        "endTime": _ms(end_time),
        "limit": limit,
    }
    data = _get_json("/api/v3/klines", params)
    time.sleep(sleep_sec)
    return data
