import os
import streamlit as st

st.set_page_config(page_title="Crypto KP", layout="wide")
st.title("Crypto Analytics (KP)")

st.info("Пока заглушка: OFFLINE режим = %s" % os.getenv("OFFLINE", "0"))

st.write("Команды запуска:")
st.code("python -m src.main price-ticks --n 3 --delay 0.2\n"
        "python -m src.main collect-klines --pairs BTCUSDT,ETHUSDT --tf 1m --days 1 --out data")
