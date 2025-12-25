"""
Binance course analysis (offline-friendly) — adapted for project layout.
...
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()
"""
Binance course analysis (offline-friendly) — adapted for project layout.

Expected project layout:
  <repo_root>/
    data/               # input CSVs (klines_...csv, optional trades_...csv, orderbook_...csv)
    figs/               # output PNG figures
    src/analysis/binance_analysis.py

What it does:
  - Loads all klines_*_1m.csv from data/
  - Filters to 2025-02-01..2025-04-30 (UTC)
  - Computes market metrics, time-of-day patterns, anomalies
  - Optionally analyzes large trades impact if trades CSV exists
  - Optionally analyzes orderbook snapshot if orderbook CSV exists
  - Exports figures (PNG) into figs/ and summary tables (CSV/JSON) into data/

Notes:
  - The script will NOT crash if trades/orderbook files are missing; it writes an "error" field in JSON.
  - Supports multi-symbol klines; can compute cross-asset correlation if >=2 symbols exist.
"""


# ----------------------------
# Paths (project-root relative)
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../src/analysis/binance_analysis.py -> repo root
DATA_DIR = ROOT / "data"
FIGS_DIR = ROOT / "figs"

DATA_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

# ----------------------------
# Analysis window (UTC)
# ----------------------------
START = None
END_EXCL = None



# ----------------------------
# Helpers
# ----------------------------
def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median absolute deviation (MAD)."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return x - med
    return 0.6745 * (x - med) / mad


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _infer_symbol_from_filename(p: Path) -> str:
    # klines_BTCUSDT_1m.csv -> BTCUSDT
    name = p.stem  # klines_BTCUSDT_1m
    parts = name.split("_")
    if len(parts) >= 3 and parts[0].lower() == "klines":
        return parts[1]
    return "UNKNOWN"


def _infer_tf_from_filename(p: Path) -> str:
    # klines_BTCUSDT_1m.csv -> 1m
    name = p.stem
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[-1]
    return "UNKNOWN"


def load_klines_files() -> pd.DataFrame:
    """Load klines CSVs from <repo_root>/data and filter to analysis window."""
    files = sorted(DATA_DIR.glob("klines_*_1m.csv"))
    if not files:
        raise FileNotFoundError(f"No klines_*_1m.csv files found in: {DATA_DIR}")

    parts: list[pd.DataFrame] = []

    for p in files:
        df = pd.read_csv(p)

        if "open_time" not in df.columns:
            raise ValueError(f"Missing 'open_time' column in {p.name}")

        # Robust parsing: supports ms timestamps or ISO strings
        s = df["open_time"]

        # If numeric-like: treat as milliseconds since epoch
        if pd.api.types.is_numeric_dtype(s):
            df["open_time"] = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
        else:
            # Try parse strings; if they look like big integers in strings -> ms
            s2 = pd.to_numeric(s, errors="coerce")
            if s2.notna().mean() > 0.9 and s2.dropna().astype("int64").median() > 10**10:
                df["open_time"] = pd.to_datetime(s2, unit="ms", utc=True, errors="coerce")
            else:
                df["open_time"] = pd.to_datetime(s, utc=True, errors="coerce")

        df = df.dropna(subset=["open_time"]).copy()

        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "symbol" not in df.columns:
            df["symbol"] = _infer_symbol_from_filename(p)
        if "tf" not in df.columns:
            df["tf"] = _infer_tf_from_filename(p)

        # Do not filter here; date window is applied in main()
        #df = df[(df["open_time"] >= START) & (df["open_time"] < END_EXCL)].copy()

        df = df.copy()

        # Debug: show what survived
        if df.empty:
            print(f"[WARN] {p.name}: 0 rows after date filter ({START.date()}..{(END_EXCL - pd.Timedelta(days=1)).date()})")
        else:
            print(f"[OK]   {p.name}: {len(df)} rows, range {df['open_time'].min()} .. {df['open_time'].max()}")

        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    return out

def load_klines_postgres() -> pd.DataFrame:
    """
    Load candles from PostgreSQL table `candles`.
    Keeps the same schema as CSV loader so downstream analytics stays unchanged.
    """
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN is not set (.env). Cannot load from PostgreSQL.")

    tf = os.getenv("TF", "1m")
    pairs_env = os.getenv("PAIRS", "").strip()
    pairs = [p.strip().upper() for p in pairs_env.split(",") if p.strip()] if pairs_env else None

    with psycopg2.connect(dsn) as conn:
        # If pairs not provided, take all available symbols for given tf
        if not pairs:
            q_sym = "SELECT DISTINCT symbol FROM candles WHERE tf = %s ORDER BY symbol;"
            pairs = [r[0] for r in pd.read_sql_query(q_sym, conn, params=(tf,))["symbol"].tolist()]

        # Pull candles
        q = """
            SELECT
                symbol, tf, open_time,
                open, high, low, close,
                volume, num_trades
            FROM candles
            WHERE tf = %s
              AND symbol = ANY(%s)
            ORDER BY symbol, open_time;
        """
        df = pd.read_sql_query(q, conn, params=(tf, pairs))

    if df.empty:
        return df

    # Ensure open_time is UTC-aware for consistency with CSV loader
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["open_time"]).copy()

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "num_trades" in df.columns:
        df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce")

    print(f"[OK]   postgres: {len(df)} rows, tf={tf}, symbols={pairs}, range {df['open_time'].min()} .. {df['open_time'].max()}")
    return df


def load_klines() -> pd.DataFrame:
    """
    Router: CSV mode (default) or DB mode.
    Controlled by env DATA_SOURCE=csv|db
    """
    src = os.getenv("DATA_SOURCE", "csv").strip().lower()
    if src == "db":
        return load_klines_postgres()
    return load_klines_files()


def add_features(kl: pd.DataFrame) -> pd.DataFrame:
    kl = kl.sort_values(["symbol", "open_time"]).reset_index(drop=True)

    # Validate required numeric columns
    required_cols = ["high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in kl.columns]
    if missing:
        raise ValueError(f"Missing required columns in klines data: {missing}")

    kl["typical_price"] = (kl["high"] + kl["low"] + kl["close"]) / 3.0
    kl["log_close"] = np.log(kl["close"])
    kl["log_ret"] = kl.groupby("symbol")["log_close"].diff()
    kl["abs_ret"] = kl["log_ret"].abs()
    kl["log_volume"] = np.log1p(kl["volume"])

    kl["hour"] = kl["open_time"].dt.hour
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    kl["weekday"] = pd.Categorical(
        kl["open_time"].dt.day_name(),
        categories=weekday_order,
        ordered=True,
    )

    # Rolling hourly volatility (per symbol): std of 1m log returns in 60-min window, annualization-like scaling
    kl["vol_60m"] = (
        kl.groupby("symbol")["log_ret"]
        .rolling(60, min_periods=30)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(60)
    )

    # Anomaly scores (robust)
    kl["z_absret"] = robust_z(kl["abs_ret"].fillna(0).to_numpy())
    kl["z_logvol"] = robust_z(kl["log_volume"].fillna(0).to_numpy())
    kl["anomaly_score"] = np.maximum(np.abs(kl["z_absret"]), np.abs(kl["z_logvol"]))
    return kl


# ----------------------------
# Plots
# ----------------------------
def plot_price_and_vol(df: pd.DataFrame, symbol: str, period_label: str) -> Path:

    d = df[df["symbol"] == symbol].copy()
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(d["open_time"], d["close"])
    ax1.set_title(f"{symbol} 1m: цена закрытия и часовая волатильность ({period_label})")


    ax1.set_xlabel("Время (UTC)")
    ax1.set_ylabel("Цена закрытия, USDT")

    ax2 = ax1.twinx()
    ax2.plot(d["open_time"], d["vol_60m"])
    ax2.set_ylabel("Волатильность (std лог-доходностей, 60м)")

    fig.tight_layout()
    out = FIGS_DIR / "price_and_vol.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_returns_hist(df: pd.DataFrame, symbol: str, period_label: str) -> Path:

    d = df[df["symbol"] == symbol]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(d["log_ret"].dropna(), bins=200)
    ax.set_title(f"Распределение 1-минутных лог-доходностей ({symbol}, {period_label})")

    ax.set_xlabel("log-return")
    ax.set_ylabel("Частота")
    fig.tight_layout()
    out = FIGS_DIR / "returns_hist.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def heatmap_plot(mat: pd.DataFrame, title: str, outname: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(mat.values, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Час (UTC)")
    ax.set_ylabel("День недели")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([str(x)[:3] for x in mat.index.astype(str)])
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xticklabels([str(h) for h in range(0, 24, 2)])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    out = FIGS_DIR / outname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def monthly_bar(monthly: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(monthly["month"], monthly["volume"])
    ax.set_title("Объём по месяцам (сумма минутных объёмов, BTC)")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Volume (BTC)")

    # <<< ВОТ ЭТО ГЛАВНОЕ ИСПРАВЛЕНИЕ >>>
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["month"], rotation=45, ha="right")

    fig.tight_layout()
    out = FIGS_DIR / "monthly_volume_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out



# ----------------------------
# Optional analyses: trades / orderbook
# ----------------------------
def _pick_first_existing(patterns: list[str]) -> Path | None:
    for pat in patterns:
        cand = sorted(DATA_DIR.glob(pat))
        if cand:
            return cand[0]
    return None


def analyze_large_trades(kl: pd.DataFrame) -> dict:
    """
    Large trades analysis for BTCUSDT day 2025-03-15.
    Accepts any of these patterns in data/:
      - trades_BTCUSDT_2025-03-15.csv
      - trades_BTCUSDT.csv
      - trades_BTCUSDT*.csv
      - trades_*.csv (fallback)
    """
    trades_path = _pick_first_existing(
        [
            "trades_BTCUSDT_2025-03-15.csv",
            "trades_BTCUSDT.csv",
            "trades_BTCUSDT*.csv",
            "trades_*.csv",
        ]
    )
    if trades_path is None:
        return {"error": f"missing trades CSV in {DATA_DIR} (expected trades_*.csv)"}

    tr = pd.read_csv(trades_path)

    # Flexible time columns: timestamp(ms) / trade_time(ISO) / ts
    if "timestamp" in tr.columns:
        tr["ts"] = pd.to_datetime(tr["timestamp"], unit="ms", utc=True, errors="coerce")
    elif "trade_time" in tr.columns:
        tr["ts"] = pd.to_datetime(tr["trade_time"], utc=True, errors="coerce")
    elif "ts" in tr.columns:
        tr["ts"] = pd.to_datetime(tr["ts"], utc=True, errors="coerce")
    else:
        return {"error": f"missing time column in {trades_path.name} (expected timestamp/trade_time/ts)"}

    for col in ("price", "qty"):
        if col not in tr.columns:
            return {"error": f"missing '{col}' column in {trades_path.name}"}
        tr[col] = pd.to_numeric(tr[col], errors="coerce")

    tr = tr.dropna(subset=["ts", "price", "qty"]).copy()
    tr["notional"] = tr["price"] * tr["qty"]

    day0 = pd.Timestamp("2025-03-15", tz="UTC")
    day1 = pd.Timestamp("2025-03-16", tz="UTC")
    tr = tr[(tr["ts"] >= day0) & (tr["ts"] < day1)].copy()

    # Candle close series for that day
    d = kl[
        (kl["symbol"] == "BTCUSDT")
        & (kl["open_time"] >= day0)
        & (kl["open_time"] < day1)
    ].copy()

    if d.empty:
        return {"error": "No BTCUSDT candles for 2025-03-15 in klines data."}

    close = d.set_index("open_time")["close"].dropna().copy()

    # Align trades to minute buckets
    tr["minute"] = tr["ts"].dt.floor("min")
    close_idx = close.index
    pos = {t: i for i, t in enumerate(close_idx)}

    threshold = float(tr["notional"].quantile(0.995))
    large = tr[tr["notional"] >= threshold].copy()
    if large.empty:
        return {"error": "No large trades found by 99.5% notional threshold."}

    def fwd_ret(minute: pd.Timestamp, k: int) -> float:
        i = pos.get(minute, np.nan)
        if np.isnan(i):
            return np.nan
        i = int(i)
        if i + k >= len(close.index):
            return np.nan
        p0 = close.iloc[i]
        pk = close.iloc[i + k]
        if p0 <= 0 or pk <= 0:
            return np.nan
        return float(np.log(pk / p0))

    rows = []
    for _, r in large.sort_values("notional", ascending=False).head(50).iterrows():
        m = r["minute"]
        rows.append(
            {
                "ts": r["ts"].isoformat(),
                "minute": m.isoformat() if pd.notna(m) else None,
                "price": float(r["price"]),
                "qty": float(r["qty"]),
                "notional": float(r["notional"]),
                "ret_5m": fwd_ret(m, 5),
                "ret_15m": fwd_ret(m, 15),
                "ret_60m": fwd_ret(m, 60),
            }
        )

    out = {
        "source_file": trades_path.name,
        "day": "2025-03-15",
        "rows_total": int(len(tr)),
        "threshold_notional_q995": float(threshold),
        "rows_large": int(len(large)),
        "top_rows": rows,
    }
    return out


def analyze_orderbook() -> dict:
    """
    Orderbook snapshot analysis.
    Accepts any of these patterns in data/:
      - orderbook_BTCUSDT_snapshot.csv
      - orderbook_BTCUSDT.csv
      - orderbook_BTCUSDT*.csv
      - orderbook_*.csv (fallback)
    Expected columns: timestamp (ms) or update_time (ISO), side (bid/ask), price, qty, level (optional).
    """
    ob_path = _pick_first_existing(
        [
            "orderbook_BTCUSDT_snapshot.csv",
            "orderbook_BTCUSDT.csv",
            "orderbook_BTCUSDT*.csv",
            "orderbook_*.csv",
        ]
    )
    if ob_path is None:
        return {"error": f"missing orderbook CSV in {DATA_DIR} (expected orderbook_*.csv)"}

    ob = pd.read_csv(ob_path)

    if "timestamp" in ob.columns:
        ob["ts"] = pd.to_datetime(ob["timestamp"], unit="ms", utc=True, errors="coerce")
    elif "update_time" in ob.columns:
        ob["ts"] = pd.to_datetime(ob["update_time"], utc=True, errors="coerce")
    else:
        return {"error": f"missing time column in {ob_path.name} (expected timestamp/update_time)"}

    for col in ("side", "price", "qty"):
        if col not in ob.columns:
            return {"error": f"missing '{col}' column in {ob_path.name}"}

    ob["price"] = pd.to_numeric(ob["price"], errors="coerce")
    ob["qty"] = pd.to_numeric(ob["qty"], errors="coerce")
    ob = ob.dropna(subset=["ts", "side", "price", "qty"]).copy()

    # Normalize side
    ob["side"] = ob["side"].astype(str).str.lower().str.strip()
    ob = ob[ob["side"].isin(["bid", "ask"])].copy()
    if ob.empty:
        return {"error": "Orderbook has no bid/ask rows after filtering."}

    # If "level" missing, synthesize by sorting best to worst
    if "level" not in ob.columns:
        bids = ob[ob["side"] == "bid"].sort_values("price", ascending=False).copy()
        asks = ob[ob["side"] == "ask"].sort_values("price", ascending=True).copy()
        bids["level"] = range(1, len(bids) + 1)
        asks["level"] = range(1, len(asks) + 1)
        ob = pd.concat([bids, asks], ignore_index=True)
    else:
        ob["level"] = pd.to_numeric(ob["level"], errors="coerce")

    # Best bid/ask
    best_bid = ob[ob["side"] == "bid"]["price"].max()
    best_ask = ob[ob["side"] == "ask"]["price"].min()
    spread = float(best_ask - best_bid) if pd.notna(best_bid) and pd.notna(best_ask) else None
    mid = float((best_ask + best_bid) / 2) if spread is not None else None

    # Depth (top N)
    N = 20
    bids_top = ob[ob["side"] == "bid"].sort_values("price", ascending=False).head(N).copy()
    asks_top = ob[ob["side"] == "ask"].sort_values("price", ascending=True).head(N).copy()

    depth_bid_qty = float(bids_top["qty"].sum())
    depth_ask_qty = float(asks_top["qty"].sum())
    depth_bid_notional = float((bids_top["qty"] * bids_top["price"]).sum())
    depth_ask_notional = float((asks_top["qty"] * asks_top["price"]).sum())

    out = {
        "source_file": ob_path.name,
        "rows_total": int(len(ob)),
        "ts_min": ob["ts"].min().isoformat() if pd.notna(ob["ts"].min()) else None,
        "ts_max": ob["ts"].max().isoformat() if pd.notna(ob["ts"].max()) else None,
        "best_bid": float(best_bid) if pd.notna(best_bid) else None,
        "best_ask": float(best_ask) if pd.notna(best_ask) else None,
        "spread": spread,
        "mid": mid,
        "depth_top20_bid_qty": depth_bid_qty,
        "depth_top20_ask_qty": depth_ask_qty,
        "depth_top20_bid_notional": depth_bid_notional,
        "depth_top20_ask_notional": depth_ask_notional,
    }
    return out



def correlation_heatmap(kl: pd.DataFrame) -> dict:
    """Compute correlation of log returns across symbols (if >=2 symbols exist)."""
    symbols = sorted(kl["symbol"].unique().tolist())
    if len(symbols) < 2:
        return {"error": "need >=2 symbols in klines files to compute correlation heatmap"}

    pivot = (
        kl.pivot_table(index="open_time", columns="symbol", values="log_ret", aggfunc="mean")
        .dropna(how="all")
        .fillna(0.0)
    )
    corr = pivot.corr()

    # Save CSV
    corr.to_csv(DATA_DIR / "correlation_matrix.csv", index=True)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, interpolation="nearest")
    ax.set_title("Корреляция лог-доходностей (1m) между активами")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    return {
        "symbols": symbols,
        "csv": "data/correlation_matrix.csv",
        "png": "figs/correlation_heatmap.png",
    }


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    kl = load_klines()
    if kl.empty:
        raise RuntimeError("No klines loaded at all")

    if START is None or END_EXCL is None:
        start_auto = kl["open_time"].min().floor("D")
        end_auto = kl["open_time"].max().floor("D") + pd.Timedelta(days=1)
        print(f"[INFO] Auto date window: {start_auto} .. {end_auto}")
        START_AUTO = start_auto
        END_EXCL_AUTO = end_auto
    else:
        START_AUTO = START
        END_EXCL_AUTO = END_EXCL

    kl = kl[(kl["open_time"] >= START_AUTO) & (kl["open_time"] < END_EXCL_AUTO)].copy()

    PERIOD_LABEL = f"{START_AUTO.date()} – {(END_EXCL_AUTO - pd.Timedelta(minutes=1)).date()}"

    kl = add_features(kl)

    symbols = sorted(kl["symbol"].unique().tolist())
    if not symbols:
        raise RuntimeError(
            f"No symbols found in klines after filtering. "
            f"Check input files in {DATA_DIR} and open_time parsing."
        )

    focus = "BTCUSDT" if "BTCUSDT" in symbols else symbols[0]

    # Coverage summary for focus symbol
    d0 = kl[kl["symbol"] == focus].copy()
    if d0.empty:
        raise RuntimeError(f"No rows for focus symbol={focus}. Check input klines files in {DATA_DIR}")

    start_ts = d0["open_time"].min()
    end_ts = d0["open_time"].max()
    end_excl = end_ts + pd.Timedelta(minutes=1)
    expected = int((end_excl - start_ts) / pd.Timedelta(minutes=1))

    tf_val = str(d0["tf"].iloc[0]) if "tf" in d0.columns and len(d0["tf"].dropna()) else "1m"
    coverage = {
        "symbol": focus,
        "tf": tf_val,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "rows": int(len(d0)),
        "expected_rows": expected,
        "missing_minutes": int(max(0, expected - len(d0))),
        "data_source": os.getenv("DATA_SOURCE", "csv").strip().lower(),
    }

    save_json(coverage, DATA_DIR / "summary_coverage.json")

    # Overall metrics (focus)
    # Guard correlation computations if series have insufficient length
    log_ret = d0["log_ret"].dropna()
    abs_ret = d0["abs_ret"].dropna()
    log_vol = d0.loc[d0["log_ret"].notna(), "log_volume"]

    corr_lr_lv = float(np.corrcoef(log_ret, log_vol)[0, 1]) if len(log_ret) > 10 else np.nan
    corr_ar_lv = (
        float(np.corrcoef(abs_ret, d0.loc[d0["abs_ret"].notna(), "log_volume"])[0, 1])
        if len(abs_ret) > 10
        else np.nan
    )

    overall = {
        "symbol": focus,
        "avg_price": float(d0["typical_price"].mean()),
        "median_price": float(d0["typical_price"].median()),
        "total_volume": float(d0["volume"].sum()),
        "avg_minute_volume": float(d0["volume"].mean()),
        "total_num_trades": int(d0["num_trades"].sum()) if "num_trades" in d0.columns else None,
        "avg_minute_num_trades": float(d0["num_trades"].mean()) if "num_trades" in d0.columns else None,
        "mean_abs_ret": float(d0["abs_ret"].mean()),
        "p95_abs_ret": float(d0["abs_ret"].quantile(0.95)),
        "p99_abs_ret": float(d0["abs_ret"].quantile(0.99)),
        "corr_logret_logvol": None if np.isnan(corr_lr_lv) else corr_lr_lv,
        "corr_absret_logvol": None if np.isnan(corr_ar_lv) else corr_ar_lv,
    }
    save_json(overall, DATA_DIR / "summary_overall.json")

    # Daily summaries
    d0["date"] = d0["open_time"].dt.date
    agg_dict = {
        "avg_price": ("typical_price", "mean"),
        "volume": ("volume", "sum"),
        "abs_ret_mean": ("abs_ret", "mean"),
        "abs_ret_max": ("abs_ret", "max"),
        "anom_max": ("anomaly_score", "max"),
    }
    if "num_trades" in d0.columns:
        agg_dict["num_trades"] = ("num_trades", "sum")

    daily = d0.groupby("date").agg(**agg_dict).reset_index()
    daily.to_csv(DATA_DIR / "daily_summary.csv", index=False)

    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = daily.groupby("month").agg(volume=("volume", "sum")).reset_index()
    monthly.to_csv(DATA_DIR / "monthly_summary.csv", index=False)
    monthly_bar(monthly)

    # Hourly patterns / DOW patterns
    hourly_agg = {
        "vol_mean": ("abs_ret", "mean"),
        "volume_mean": ("volume", "mean"),
    }
    if "num_trades" in d0.columns:
        hourly_agg["trades_mean"] = ("num_trades", "mean")

    hourly = d0.groupby("hour").agg(**hourly_agg).reset_index()
    hourly.to_csv(DATA_DIR / "hourly_pattern.csv", index=False)

    dow = d0.groupby("weekday", observed=True).agg(**hourly_agg).reset_index()
    dow.to_csv(DATA_DIR / "dow_pattern.csv", index=False)

    vol_heat = d0.pivot_table(index="weekday", columns="hour", values="abs_ret", aggfunc="mean", observed=True)
    volu_heat = d0.pivot_table(index="weekday", columns="hour", values="volume", aggfunc="mean", observed=True)
    heatmap_plot(volu_heat, f"Средний объём по часам и дням недели ({focus}, {PERIOD_LABEL})", "heatmap_volume.png")
    heatmap_plot(vol_heat, f"Средняя |лог-доходность| по часам и дням недели ({focus}, {PERIOD_LABEL})","heatmap_volatility.png")

    # Plots
    plot_price_and_vol(kl, focus, PERIOD_LABEL)

    plot_returns_hist(kl, focus, PERIOD_LABEL)


    # Anomalies plot
    top_idx = d0.sort_values("anomaly_score", ascending=False).head(200).index
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(d0["open_time"], d0["abs_ret"])
    ax.scatter(d0.loc[top_idx, "open_time"], d0.loc[top_idx, "abs_ret"], s=10)
    ax.set_title("Аномальные минуты: |лог-доходность| (точки = топ-200 по anomaly score)")
    ax.set_xlabel("Время (UTC)")
    ax.set_ylabel("|log-return|")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "anomalies_absret.png", dpi=200)
    plt.close(fig)

    # Vol vs volume scatter
    samp = d0.sample(min(5000, len(d0)), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(samp["log_volume"], samp["abs_ret"], s=5, alpha=0.5)
    ax.set_title("Связь активности и волатильности: |log-ret| vs log(1+volume)")
    ax.set_xlabel("log(1+volume)")
    ax.set_ylabel("|log-return|")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "vol_vs_volume_scatter.png", dpi=200)
    plt.close(fig)

    # Optional: correlation heatmap across symbols
    save_json(correlation_heatmap(kl), DATA_DIR / "summary_correlation.json")

    # Optional: large trades & orderbook
    save_json(analyze_large_trades(kl), DATA_DIR / "summary_large_trades.json")
    save_json(analyze_orderbook(), DATA_DIR / "summary_orderbook.json")

    print("Done. Outputs:")
    print(f"  Figures:  {FIGS_DIR}")
    print(f"  Summaries: {DATA_DIR}/*.csv and {DATA_DIR}/summary_*.json")


if __name__ == "__main__":
    main()
