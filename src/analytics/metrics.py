import os
import psycopg2
import pandas as pd

DB_SETTINGS = {
    "dbname": os.getenv("POSTGRES_DB", "crypto"),
    "user": os.getenv("POSTGRES_USER", "crypto"),
    "password": os.getenv("POSTGRES_PASSWORD", "crypto"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}


def fetch_metrics():
    query = """
        SELECT
          symbol,
          ROUND(AVG((open + high + low + close) / 4), 2)  AS avg_price,
          ROUND(STDDEV(close), 2)                        AS volatility,
          ROUND(SUM(volume), 2)                          AS total_volume,
          SUM(num_trades)                                AS total_trades
        FROM candles
        GROUP BY symbol
        ORDER BY symbol;
    """

    with psycopg2.connect(**DB_SETTINGS) as conn:
        df = pd.read_sql_query(query, conn)
    return df


def main():
    df = fetch_metrics()

    print("\nКлючевые рыночные метрики:")
    print(df.to_string(index=False))

    out_path = "data/metrics_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nСохранено в {out_path}")


if __name__ == "__main__":
    main()
