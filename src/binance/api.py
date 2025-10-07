import os
import time
import requests
from datetime import datetime, timezone
from typing import List
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла (если он есть)
# Например: BINANCE_BASE_URL, DATA_DIR и т.д.
load_dotenv()

# Получаем базовый URL Binance API из окружения или используем дефолт
BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")


# Переводит datetime в миллисекунды (Binance использует время в ms).
def _ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def get_klines(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 1000,
    sleep_sec: float = 0.1,
) -> List[List]:
    """
    Запрашивает исторические свечи (klines) для заданного символа на Binance.

    symbol: торговая пара, например "BTCUSDT"
    interval: таймфрейм (например, "1m", "5m", "1h")
    start_time, end_time: временной диапазон для выгрузки
    limit: максимальное количество свечей за 1 запрос (макс. 1000 у Binance)
    sleep_sec: небольшая пауза между запросами, чтобы не словить rate limit

    Возвращает "сырые" свечи — список списков вида:
    [
        [
            1499040000000,      // Open time
            "0.01634790",       // Open
            "0.80000000",       // High
            "0.01575800",       // Low
            "0.01577100",       // Close
            "148976.11427815",  // Volume
            1499644799999,      // Close time
            "2434.19055334",    // Quote asset volume
            308,               // Number of trades
            "1756.87402397",    // Taker buy base volume
            "28.46694368",      // Taker buy quote volume
            "0"                // Ignore
        ],
        ...
    ]
    """
    # Формируем URL для REST запроса
    url = f"{BASE_URL}/api/v3/klines"

    # Подготавливаем параметры запроса
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": _ms(start_time),
        "endTime": _ms(end_time),
        "limit": limit,
    }

    # Отправляем GET-запрос
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()  # если ошибка → будет исключение

    # Разбираем JSON-ответ от Binance
    data = response.json()

    # Небольшая задержка, чтобы не попасть под rate limit API
    time.sleep(sleep_sec)

    return data
