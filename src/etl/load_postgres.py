import os
import io
import pandas as pd
import psycopg2
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://crypto:crypto@localhost:5432/crypto"
)

def load_csv_to_postgres(csv_path: Path) -> None:
    print(f"Загрузка файла: {csv_path}")

    dsn = os.getenv("POSTGRES_DSN") or "postgresql://crypto:crypto@localhost:5432/crypto"


    # извлекаем symbol и tf из имени файла: klines_BTCUSDT_1m.csv
    name = csv_path.stem  # klines_BTCUSDT_1m
    _, symbol, tf = name.split("_")

    df = pd.read_csv(csv_path)

    # приводим open_time из ms в datetime
    s = df["open_time"]

    # 1) если пришло число/строка-число (unix ms)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().all():
        df["open_time"] = pd.to_datetime(s_num.astype("int64"), unit="ms", utc=True)
    else:
        # 2) если пришла строка даты
        df["open_time"] = pd.to_datetime(s, utc=True, errors="raise")

    # --- map trades column safely ---
    trades_col = None
    for c in ("trades", "num_trades", "trade_count", "count"):
        if c in df.columns:
            trades_col = c
            break

    if trades_col is None:
        print("[WARN] Column with number of trades not found; setting num_trades=0")
        num_trades = 0
    else:
        num_trades = pd.to_numeric(df[trades_col], errors="coerce").fillna(0).astype("int64")

    out = pd.DataFrame({
        "symbol": symbol,
        "tf": tf,
        "open_time": df["open_time"],
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        "num_trades": num_trades,
    })

    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE tmp_candles (LIKE candles INCLUDING ALL);")

            with io.StringIO() as buf:
                out.to_csv(buf, index=False, header=False)
                buf.seek(0)
                cur.copy_expert(
                    """
                    COPY tmp_candles(
                        symbol, tf, open_time,
                        open, high, low, close,
                        volume, num_trades
                    )
                    FROM STDIN WITH (FORMAT CSV)
                    """,
                    buf,
                )

            cur.execute(
                """
                INSERT INTO candles(
                    symbol, tf, open_time,
                    open, high, low, close,
                    volume, num_trades
                )
                SELECT
                    symbol, tf, open_time,
                    open, high, low, close,
                    volume, num_trades
                FROM tmp_candles
                ON CONFLICT (symbol, tf, open_time) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low  = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    num_trades = EXCLUDED.num_trades;
                """
            )
    finally:
        conn.close()



if __name__ == "__main__":
    data_dir = Path("data")
    for csv_file in data_dir.glob("klines_*.csv"):
        load_csv_to_postgres(csv_file)
