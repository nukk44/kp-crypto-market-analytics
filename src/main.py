import argparse, time, os
from typing import List
from dotenv import load_dotenv

from .binance.api import get_klines  # уже есть
from .binance.api import _get_json   # добавим импорт для тикера
from .collectors.prices_collector import run as collect_run

load_dotenv()

def price_ticks(n: int = 5, delay: float = 1.0) -> List[float]:
    """Собрать n цен BTCUSDT. В OFFLINE=1 возвращает константы."""
    if os.getenv("OFFLINE") == "1":
        return [100.0 for _ in range(n)]
    prices = []
    for _ in range(n):
        j = _get_json("/api/v3/ticker/price", {"symbol": "BTCUSDT"})
        prices.append(round(float(j["price"]), 2))
        time.sleep(delay)
    return prices


def main():
    p = argparse.ArgumentParser("kp-crypto")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ticks = sub.add_parser("price-ticks", help="Собрать n цен BTCUSDT /ticker/price")
    p_ticks.add_argument("--n", type=int, default=5)
    p_ticks.add_argument("--delay", type=float, default=1.0)

    p_collect = sub.add_parser("collect-klines", help="Собрать свечи (klines) по списку пар")
    p_collect.add_argument("--pairs", default=os.getenv("PAIRS", "BTCUSDT,ETHUSDT"))
    p_collect.add_argument("--tf", default=os.getenv("TF", "1m"))
    p_collect.add_argument("--days", type=int, default=int(os.getenv("DAYS", "3")))
    p_collect.add_argument("--out", default=os.getenv("OUT_DIR", "data"))

    args = p.parse_args()

    if args.cmd == "price-ticks":
        vals = price_ticks(n=args.n, delay=args.delay)
        print({"count": len(vals), "min": min(vals), "max": max(vals), "values": vals})
    elif args.cmd == "collect-klines":
        pairs = [s.strip() for s in args.pairs.split(",") if s.strip()]
        collect_run(pairs=pairs, tf=args.tf, days=args.days, out_dir=args.out)

if __name__ == "__main__":
    main()
