import os
import csv
import time
import requests
from datetime import datetime, timezone
from typing import Iterable


def ms_to_iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def get_recent_trades(symbol: str, limit: int = 1000):
    url = f"https://api.binance.com/api/v3/trades"
    params = {"symbol": symbol, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def run(pairs: Iterable[str], out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)

    for sym in pairs:
        trades = get_recent_trades(sym)
        path = os.path.join(out_dir, f"trades_{sym}.csv")

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "symbol", "trade_id", "price", "qty", "quote_qty",
                "trade_time", "is_buyer_maker", "is_best_match"
            ])

            for t in trades:
                w.writerow([
                    sym,
                    t["id"],
                    t["price"],
                    t["qty"],
                    t["quoteQty"],
                    ms_to_iso_utc(int(t["time"])),
                    t["isBuyerMaker"],
                    t["isBestMatch"],
                ])

        print(f"[collector] saved {path} ({len(trades)} rows)")
        time.sleep(0.2)
