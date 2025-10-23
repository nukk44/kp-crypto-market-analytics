import os, csv
from datetime import datetime, timedelta, timezone
from typing import Iterable
from ..binance.api import get_klines

def _utc_now():
    return datetime.now(timezone.utc)

def ms_to_iso_utc(ms: int) -> str:
    """Переводит миллисекунды Binance в ISO-строку UTC."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

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
            w.writerow(["symbol", "tf", "open_time", "open", "high", "low", "close", "volume", "num_trades"])

            for r in rows:
                w.writerow([
                    sym,  # торговая пара (BTCUSDT, ETHUSDT ...)
                    tf,  # таймфрейм (1m, 5m ...)
                    ms_to_iso_utc(int(r[0])),  # время открытия в UTC ISO
                    r[1],  # open
                    r[2],  # high
                    r[3],  # low
                    r[4],  # close
                    r[5],  # volume
                    int(r[8]),  # num_trades (индекс 8 в API Binance)
                ])

        print(f"[collector] saved {path} ({len(rows)} rows)")
