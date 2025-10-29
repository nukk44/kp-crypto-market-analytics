# src/main.py
import argparse
import os
import time
from typing import List

from dotenv import load_dotenv

# Абсолютные импорты — удобно для запуска файла через IDE
from src.binance.api import _get_json  # вспомогательная функция для тикера
from src.collectors.prices_collector import run as collect_run
from src.collectors.trades_collector import run as collect_trades_run


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


def parse_pairs(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("kp-crypto", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=False)

    p_ticks = sub.add_parser("price-ticks", help="Собрать n цен BTCUSDT (/ticker/price)")
    p_ticks.add_argument("--n", type=int, default=5, help="сколько цен собрать")
    p_ticks.add_argument("--delay", type=float, default=1.0, help="задержка между запросами, сек")

    p_collect = sub.add_parser("collect-klines", help="Собрать свечи (klines) по списку пар")
    p_collect.add_argument("--pairs", default=os.getenv("PAIRS", "BTCUSDT,ETHUSDT"),
                           help="список пар через запятую")
    p_collect.add_argument("--tf", default=os.getenv("TF", "1m"),
                           help="таймфрейм: 1m,5m,15m,1h,4h,1d")
    p_collect.add_argument("--days", type=int, default=int(os.getenv("DAYS", "1")),
                           help="за сколько дней назад до сейчас")
    p_collect.add_argument("--out", default=os.getenv("OUT_DIR", "data"),
                           help="папка для сохранения данных")

    p_trades = sub.add_parser("collect-trades", help="Собрать сделки (trades) по списку пар")
    p_trades.add_argument("--pairs", default=os.getenv("PAIRS", "BTCUSDT,ETHUSDT"),
                          help="список пар через запятую")
    p_trades.add_argument("--out", default=os.getenv("OUT_DIR", "data"),
                          help="папка для сохранения данных")
    return p


def interactive_menu():
    """Интерактивный режим по умолчанию (если параметров нет)."""
    # читаем дефолты из .env
    def_pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT")
    def_tf    = os.getenv("TF", "1m")
    def_days  = os.getenv("DAYS", "1")
    def_out   = os.getenv("OUT_DIR", "data")

    print("\n=== KP Crypto — интерактивный режим ===")
    print("1) price-ticks  — собрать N цен BTCUSDT (ticker/price)")
    print("2) collect-klines — скачать свечи Binance для пар")
    print("q) выйти")

    choice = input("\nВыберите пункт [1/2/q]: ").strip().lower()
    if choice == "1":
        try:
            n = int(input("Сколько цен собрать? [5]: ") or "5")
            delay = float(input("Задержка между запросами, сек? [1.0]: ") or "1.0")
        except ValueError:
            print("Некорректный ввод. Запускаю по умолчанию n=5, delay=1.0.")
            n, delay = 5, 1.0
        vals = price_ticks(n=n, delay=delay)
        print({"count": len(vals), "min": min(vals), "max": max(vals), "values": vals})
        return

    if choice == "2":
        pairs = input(f"Пары через запятую [{def_pairs}]: ").strip() or def_pairs
        tf    = input(f"Таймфрейм [{def_tf}]: ").strip() or def_tf
        days_str = input(f"Сколько дней назад? [{def_days}]: ").strip() or def_days
        out   = input(f"Куда сохранить? [{def_out}]: ").strip() or def_out
        try:
            days = int(days_str)
        except ValueError:
            print("Некорректный ввод дней. Использую 1.")
            days = 1
        collect_run(pairs=parse_pairs(pairs), tf=tf, days=days, out_dir=out)
        return

    if choice in ("q", "quit", "exit"):
        print("Выход.")
        return

    print("Неизвестная команда. Ничего не выполнено.")



def main():
    parser = build_parser()
    args = parser.parse_args()

    # Если пользователь запустил без подкоманд -> интерактивный режим
    if not args.cmd:
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
        return

    # Обычный режим через аргументы
    if args.cmd == "price-ticks":
        vals = price_ticks(n=args.n, delay=args.delay)
        print({"count": len(vals), "min": min(vals), "max": max(vals), "values": vals})
    elif args.cmd == "collect-klines":
        collect_run(pairs=parse_pairs(args.pairs), tf=args.tf, days=args.days, out_dir=args.out)
    elif args.cmd == "collect-trades":
        collect_trades_run(pairs=parse_pairs(args.pairs), out_dir=args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
