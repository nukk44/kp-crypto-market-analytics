import os
from pathlib import Path
import psycopg2

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://crypto:crypto@localhost:5432/crypto"
)


def load_csv_to_postgres(csv_path: Path):
    """Загружает CSV-файл со сделками в таблицу trades через COPY + UPSERT."""
    print(f"Загрузка файла: {csv_path}")

    conn = psycopg2.connect(POSTGRES_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    # Временная таблица для быстрой загрузки
    cur.execute("""
        CREATE TEMP TABLE tmp_trades (LIKE trades INCLUDING ALL) ON COMMIT DROP;
    """)

    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)  # пропустить заголовок
        cur.copy_expert("""
            COPY tmp_trades (symbol, trade_id, price, qty, quote_qty, trade_time, is_buyer_maker, is_best_match)
            FROM STDIN WITH (FORMAT csv);
        """, f)

    # UPSERT в основную таблицу
    cur.execute("""
        INSERT INTO trades
        SELECT * FROM tmp_trades
        ON CONFLICT (symbol, trade_id) DO UPDATE
        SET
            price = EXCLUDED.price,
            qty = EXCLUDED.qty,
            quote_qty = EXCLUDED.quote_qty,
            trade_time = EXCLUDED.trade_time,
            is_buyer_maker = EXCLUDED.is_buyer_maker,
            is_best_match = EXCLUDED.is_best_match;
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Загрузка завершена успешно.\n")


if __name__ == "__main__":
    data_dir = Path("data")
    for csv_file in data_dir.glob("trades_*.csv"):
        load_csv_to_postgres(csv_file)
