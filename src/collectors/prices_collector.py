import os, csv
from datetime import datetime, timedelta, timezone
from typing import Iterable
from ..binance.api import get_klines

def _utc_now():
    return datetime.now(timezone.utc)

def run(pairs: Iterable[str], tf: str = "1m", days: int = 3, out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)
    end   = _utc_now()
    start = end - timedelta(days=days)

    for sym in pairs:
        rows = get_klines(
            symbol=sym,
            interval=tf,
            start_time=start,
            end_time=end,
            limit=1000,
            sleep_sec=0.2,
        )
        path = os.path.join(out_dir, f"klines_{sym}_{tf}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote"])
            for r in rows:
                w.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]])
        print(f"[collector] saved {path} ({len(rows)} rows)")
