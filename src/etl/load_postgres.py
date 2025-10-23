import os
from pathlib import Path
import psycopg2

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://crypto:crypto@localhost:5432/crypto"
)

def load_csv_to_postgres(csv_path: Path):
    """Загружает CSV-файл со свечами в таблицу candles через COPY + UPSERT."""
    print(f"Загрузка файла: {csv_path}")

    conn = psycopg2.connect(POSTGRES_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    # Временная таблица для быстрой загрузки
    cur.execute("""
        CREATE TEMP TABLE tmp_candles (LIKE candles INCLUDING ALL) ON COMMIT DROP;
    """)

    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)  # пропустить заголовок
        cur.copy_expert("""
            COPY tmp_candles (symbol, tf, open_time, open, high, low, close, volume, num_trades)
            FROM STDIN WITH (FORMAT csv);
        """, f)

    # Обновляем или вставляем данные (UPSERT)
    cur.execute("""
        INSERT INTO candles
        SELECT * FROM tmp_candles
        ON CONFLICT (symbol, tf, open_time) DO UPDATE
        SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            num_trades = EXCLUDED.num_trades;
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Загрузка завершена успешно.\n")


if __name__ == "__main__":
    data_dir = Path("data")
    for csv_file in data_dir.glob("*.csv"):
        load_csv_to_postgres(csv_file)
