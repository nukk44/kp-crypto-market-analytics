import os
from pathlib import Path
import psycopg2

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://crypto:crypto@localhost:5432/crypto"
)


def load_csv_to_postgres(csv_path: Path):
    print(f"Загрузка файла: {csv_path}")

    conn = psycopg2.connect(POSTGRES_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    # Временная таблица для быстрой загрузки
    cur.execute("""
        CREATE TEMP TABLE tmp_order_books (LIKE order_books INCLUDING ALL) ON COMMIT DROP;
    """)

    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)  # пропустить заголовок
        cur.copy_expert("""
            COPY tmp_order_books (symbol, price, qty, side, update_id, update_time)
            FROM STDIN WITH (FORMAT csv);
        """, f)

    # UPSERT в основную таблицу
    cur.execute("""
        INSERT INTO order_books
        SELECT * FROM tmp_order_books
        ON CONFLICT (symbol, side, price, update_id) DO UPDATE
        SET
            qty = EXCLUDED.qty,
            update_time = EXCLUDED.update_time;
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Загрузка завершена успешно.\n")


if __name__ == "__main__":
    data_dir = Path("data")
    for csv_file in data_dir.glob("orderbook_*.csv"):
        load_csv_to_postgres(csv_file)
