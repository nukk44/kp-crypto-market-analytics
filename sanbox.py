from datetime import datetime, timedelta, timezone
from src.binance.api import get_klines

end = datetime.now(timezone.utc)
start = end - timedelta(hours=1)

data = get_klines("BTCUSDT", "1m", start, end)
print(f"Получено {len(data)} свечей")
print("Пример первой свечи:", data[0])
