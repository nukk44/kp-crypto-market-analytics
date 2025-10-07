# kp-crypto-market-analytics
Course project: Crypto market ETL + analytics (Binance klines/trades), ClickHouse, dashboards.
## Быстрый старт
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip -r requirements.txt

# офлайн-демо (без обращения в интернет)
OFFLINE=1 python -m src.main price-ticks --n 3 --delay 0.2
OFFLINE=1 python -m src.main collect-klines --pairs BTCUSDT,ETHUSDT --tf 1m --days 1 --out data
