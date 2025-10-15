CREATE TABLE IF NOT EXISTS candles (
  symbol     TEXT      NOT NULL,   -- пара, напр. BTCUSDT
  tf         TEXT      NOT NULL,   -- таймфрейм: 1m, 5m, 1h...
  open_time  TIMESTAMP NOT NULL,   -- начало свечи (UTC)

  open       NUMERIC   NOT NULL,
  high       NUMERIC   NOT NULL,
  low        NUMERIC   NOT NULL,
  close      NUMERIC   NOT NULL,

  volume     NUMERIC,              -- объём в базовой валюте
  num_trades INTEGER,              -- число сделок за интервал

  PRIMARY KEY (symbol, tf, open_time)
);
