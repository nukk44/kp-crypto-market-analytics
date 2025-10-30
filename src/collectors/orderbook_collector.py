import os
import csv
import time
import requests
from datetime import datetime, timezone
from typing import Iterable


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_order_book(symbol: str, limit: int = 100):
    url = "https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def run(pairs: Iterable[str], out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)

    for sym in pairs:
        data = get_order_book(sym)
        update_id = data["lastUpdateId"]
        now_utc = _utc_now()

        path = os.path.join(out_dir, f"orderbook_{sym}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "price", "qty", "side", "update_id", "update_time"])

            for price, qty in data["bids"]:
                w.writerow([sym, price, qty, "bid", update_id, now_utc])
            for price, qty in data["asks"]:
                w.writerow([sym, price, qty, "ask", update_id, now_utc])

        print(f"[collector] saved {path} ({len(data['bids']) + len(data['asks'])} rows)")
        time.sleep(0.2)
