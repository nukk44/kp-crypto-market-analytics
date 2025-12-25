"""
Streamlit dashboard for Binance market analytics (pyarrow-free).

Important:
- Streamlit's st.dataframe/st.table require pyarrow in many versions.
- On Windows pyarrow often fails with: "No module named pyarrow.lib".
- This dashboard NEVER uses st.dataframe/st.table. Tables are shown as text.

Reads:
  - figures from ./figs
  - tables/json from ./data
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ----------------------------
# Repo-aware paths
# ----------------------------
APP_DIR = Path(__file__).resolve().parent

REPO_ROOT = None
for p in [APP_DIR, *APP_DIR.parents]:
    if (p / "src").is_dir() and (p / "docker-compose.yml").exists():
        REPO_ROOT = p
        break
    if (p / "src").is_dir() and (p / ".git").exists():
        REPO_ROOT = p
        break

if REPO_ROOT is None:
    REPO_ROOT = APP_DIR.parents[1]

DATA_DIR = (REPO_ROOT / "data").resolve()
FIG_DIR = (REPO_ROOT / "figs").resolve()

DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Binance Analysis Dashboard", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def show_img(filename: str, caption: str | None = None) -> None:
    p = FIG_DIR / filename
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Нет файла: {filename}. Запусти анализ для генерации артефактов.")


def show_df_text(df: pd.DataFrame | None, title: str, max_rows: int = 50) -> None:
    """
    Safe table rendering without pyarrow:
    - shows head(max_rows) as text table
    """
    st.subheader(title)
    if df is None:
        st.info("Файл отсутствует.")
        return
    if df.empty:
        st.info("Таблица пустая.")
        return

    if len(df) > max_rows:
        st.warning(f"Показаны первые {max_rows} строк из {len(df)} (режим без pyarrow).")
        df_show = df.head(max_rows)
    else:
        df_show = df

    # Pretty fixed-width table
    st.code(df_show.to_string(index=False), language="text")


# ----------------------------
# Header
# ----------------------------
st.title("Binance Market Data Analysis (Course Dashboard)")
st.caption(f"DATA_DIR: {DATA_DIR}")
st.caption(f"FIG_DIR: {FIG_DIR}")

# PostgreSQL status
dsn = os.getenv("POSTGRES_DSN")
if dsn:
    try:
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), MIN(open_time), MAX(open_time) FROM candles;")
            cnt, dmin, dmax = cur.fetchone()
        st.success(f"PostgreSQL: candles={cnt}, period={dmin} — {dmax}")
    except Exception as e:
        st.warning(f"PostgreSQL недоступен: {e}")
else:
    st.info("POSTGRES_DSN не задан — дашборд работает по локальным артефактам.")


# Read summaries
coverage = read_json(DATA_DIR / "summary_coverage.json") or {}
overall = read_json(DATA_DIR / "summary_overall.json") or {}
corr_summary = read_json(DATA_DIR / "summary_correlation.json") or {}

artifact_source = None
if isinstance(coverage, dict):
    artifact_source = coverage.get("data_source")
if artifact_source:
    st.caption(f"Артефакты анализа были сгенерированы из: {artifact_source}")

# period
start = None
end = None
symbols = None
if isinstance(coverage, dict):
    start = coverage.get("start") or coverage.get("start_utc") or coverage.get("date_min")
    end = coverage.get("end") or coverage.get("end_utc") or coverage.get("date_max")
    symbols = coverage.get("symbols") or coverage.get("assets") or coverage.get("pairs")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Период")
    if start and end:
        st.write(f"{start} — {end}")
    else:
        st.write("Период определяется автоматически по данным.")
with c2:
    st.subheader("Активы")
    if symbols:
        st.write(", ".join(map(str, symbols)))
    else:
        st.write("BTCUSDT/ETHUSDT (по данным проекта)")
with c3:
    st.subheader("Режим")
    st.write("Offline (без API, по локальным CSV/PNG)")

st.divider()

# Sidebar navigation
st.sidebar.header("Навигация")
page = st.sidebar.radio(
    "Раздел",
    ["Обзор", "Цена и волатильность", "Паттерны по времени", "Аномалии", "Корреляции", "Таблицы"],
    index=0,
)
st.sidebar.header("Артефакты")
st.sidebar.write("Сначала запусти анализ:")
st.sidebar.code("DATA_SOURCE=db python src/analytics/binance_analysis.py")


# ----------------------------
# Pages
# ----------------------------
if page == "Обзор":
    st.subheader("Ключевые графики")
    left, right = st.columns(2)
    with left:
        show_img("price_and_vol.png", "Цена и часовая волатильность")
        show_img("returns_hist.png", "Распределение лог-доходностей")
    with right:
        show_img("heatmap_volume.png", "Теплокарта объёма: час × день недели")
        show_img("heatmap_volatility.png", "Теплокарта волатильности: час × день недели")

    st.subheader("Дополнительно")
    show_img("monthly_volume_bar.png", "Объём по месяцам (агрегация)")
    show_img("vol_vs_volume_scatter.png", "|log-return| vs log(1+volume)")

    if overall:
        st.subheader("Сводка (JSON)")
        st.json(overall)

elif page == "Цена и волатильность":
    st.subheader("Цена и волатильность")
    show_img("price_and_vol.png")
    st.subheader("Распределение лог-доходностей")
    show_img("returns_hist.png")

elif page == "Паттерны по времени":
    st.subheader("Паттерны: час суток / день недели")
    show_img("heatmap_volume.png")
    show_img("heatmap_volatility.png")

elif page == "Аномалии":
    st.subheader("Аномальные минуты")
    show_img("anomalies_absret.png", "Топ-аномалии по |log-return|")
    df_anom = read_csv(DATA_DIR / "anomalies_top10.csv")
    show_df_text(df_anom, "Топ-10 аномалий (CSV)", max_rows=20)

elif page == "Корреляции":
    st.subheader("Корреляции доходностей между активами")
    show_img("correlation_heatmap.png")
    df_corr = read_csv(DATA_DIR / "correlation_matrix.csv")
    show_df_text(df_corr, "Матрица корреляций (CSV)", max_rows=20)
    if corr_summary:
        st.subheader("Сводка (JSON)")
        st.json(corr_summary)

elif page == "Таблицы":
    st.subheader("Сводные таблицы")
    show_df_text(read_csv(DATA_DIR / "daily_summary.csv"), "daily_summary.csv", max_rows=30)
    show_df_text(read_csv(DATA_DIR / "hourly_pattern.csv"), "hourly_pattern.csv", max_rows=30)
    show_df_text(read_csv(DATA_DIR / "dow_pattern.csv"), "dow_pattern.csv", max_rows=30)
    show_df_text(read_csv(DATA_DIR / "monthly_summary.csv"), "monthly_summary.csv", max_rows=30)
