Маленький образцы данных для тестов

docker/docker-compose.yml

db/schema.sql (или миграции Alembic, если есть)

pyproject.toml (зависимости)

Входная точка / CLI: src/main.py

Ingest с Binance: src/binance/api.py и src/collectors/prices_collector.py

ETL: src/etl/normalize.py и src/etl/load_postgres.py

Первая метрика: src/analytics/volatility.py (или другой модуль аналитики)

Утилиты: src/utils/logging.py

Пример данных (маленький): 5–10 строк того, что сохраняет collector (jsonl/csv).

Тест (если есть): tests/test_prices.py.