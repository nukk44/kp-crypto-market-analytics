import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def load_to_postgres(csv_path: str, table_name: str = "candle_1m"):
    """Пример функции загрузки CSV в PostgreSQL"""
    conn = psycopg2.connect(os.getenv("POSTGRES_DSN"))
    cur = conn.cursor()

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        cur.execute(
            f"""
            INSERT INTO {table_name} (open_time, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (row.open_time, row.open, row.high, row.low, row.close, row.volume),
        )
    conn.commit()
    conn.close()
