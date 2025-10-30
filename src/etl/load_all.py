import os
from pathlib import Path
import psycopg2

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://crypto:crypto@localhost:5432/crypto"
)

DATA_DIR = Path("data")


def load_csv_to_table(csv_path: Path, table_name: str, columns: str, conflict: str, update_set: str):
    print(f"Загрузка файла: {csv_path} → {table_name}")

    conn = psycopg2.connect(POSTGRES_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(f"CREATE TEMP TABLE tmp_{table_name} (LIKE {table_name} INCLUDING ALL) ON COMMIT DROP;")

    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)
        cur.copy_expert(
            f"COPY tmp_{table_name} ({columns}) FROM STDIN WITH (FORMAT csv);",
            f
        )

    cur.execute(f"""
        INSERT INTO {table_name}
        SELECT * FROM tmp_{table_name}
        ON CONFLICT ({conflict}) DO UPDATE
        SET {update_set};
    """)

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Загрузка {table_name} завершена.\n")


def main():
    for file in DATA_DIR.glob("*.csv"):
        name = file.name.lower()

        # --- свечи ---
        if "kline" in name:
            load_csv_to_table(
                file,
                "candles",
                "symbol, tf, open_time, open, high, low, close, volume, num_trades",
                "symbol, tf, open_time",
                "open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume, num_trades = EXCLUDED.num_trades"
            )

        # --- сделки ---
        elif "trade" in name:
            load_csv_to_table(
                file,
                "trades",
                "symbol, trade_id, price, qty, quote_qty, trade_time, is_buyer_maker, is_best_match",
                "symbol, trade_id",  # << было 'trade_id'
                "price = EXCLUDED.price, qty = EXCLUDED.qty, quote_qty = EXCLUDED.quote_qty, "
                "trade_time = EXCLUDED.trade_time, is_buyer_maker = EXCLUDED.is_buyer_maker, "
                "is_best_match = EXCLUDED.is_best_match"
            )

        # --- ордербуки ---
        elif "orderbook" in name:
            load_csv_to_table(
                file,
                "order_books",
                "symbol, price, qty, side, update_id, update_time",
                "symbol, side, price, update_id",  # << было в другом порядке
                "qty = EXCLUDED.qty, update_time = EXCLUDED.update_time"
            )

        else:
            print(f"⚠️ Пропуск файла {file.name} — неизвестный тип данных.\n")


if __name__ == "__main__":
    main()
