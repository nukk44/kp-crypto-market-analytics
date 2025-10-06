import requests
import time


def get_bitcoin_price():
    """Возвращает текущую цену BTC/USDT с Binance."""
    response = requests.get(
        'https://api.binance.com/api/v3/ticker/price',
        params={'symbol': 'BTCUSDT'}
    )
    response.raise_for_status()
    data = response.json()
    return float(data['price'])


def collect_prices(n=5, delay=1):
    """Собирает n цен с заданным интервалом delay (в секундах)."""
    prices = []
    for _ in range(n):
        try:
            price = round(get_bitcoin_price(), 2)
            prices.append(price)
        except requests.RequestException as e:
            print("Ошибка:", e)
        time.sleep(delay)
    return prices


if __name__ == "__main__":
    prices = collect_prices(5, 1)
    print("Собранные цены:", prices)
    print("Количество:", len(prices))
    print("Максимум:", max(prices))
    print("Минимум:", min(prices))
