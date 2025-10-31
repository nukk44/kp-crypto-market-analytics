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
        print(f"[collector] collecting {sym} ({tf}, {days}d) ...")

        all_rows = []
        current_start = start

        while current_start < end:
            # вызываем API на 1000 свечей максимум
            rows = get_klines(
                symbol=sym,
                interval=tf,
                start_time=current_start,
                end_time=end,
                limit=1000,
                sleep_sec=0.2,
            )

            if not rows:
                break

            all_rows.extend(rows)

            # вычисляем новое начало — последняя свеча + 1 миллисекунда
            last_open_ms = int(rows[-1][0])
            last_open_time = datetime.fromtimestamp(last_open_ms / 1000, tz=timezone.utc)
            current_start = last_open_time + timedelta(milliseconds=1)

        # сохраняем в CSV
        path = os.path.join(out_dir, f"klines_{sym}_{tf}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "tf", "open_time", "open", "high", "low", "close", "volume", "num_trades"])

            for r in all_rows:
                w.writerow([
                    sym,
                    tf,
                    ms_to_iso_utc(int(r[0])),
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    int(r[8]),
                ])

        print(f"[collector] saved {path} ({len(all_rows)} rows)")
