import ccxt
import os
from dotenv import load_dotenv
import requests

# === Загружаем ключи из .env ===
load_dotenv()
api_key = os.getenv("BYBIT_API_KEY")
secret_key = os.getenv("BYBIT_SECRET_KEY")
proxy = os.getenv("PROXY")

if not api_key or not secret_key:
    raise ValueError("❌ Ключи API не найдены! Проверь .env")

# === Настраиваем requests.Session с прокси ===
session = requests.Session()
session.proxies = {
    "http": proxy,
    "https": proxy,
}
session.verify = True   # проверка SSL (обязательно!)

# === Подключаемся к Bybit, привязываем session ===
exchange = ccxt.bybit({
    "apiKey": api_key,
    "secret": secret_key,
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})
exchange.session = session  # 🚀 теперь все запросы ccxt идут через наш прокси

# === Проверка соединения ===
try:
    print("⏳ Проверка баланса...")
    balance = exchange.fetch_balance()
    print("✅ Успешное подключение!")
    print("Баланс:", balance["total"])

    ticker = exchange.fetch_ticker("BTC/USDT")
    print("📊 Цена BTC:", ticker["last"])
except Exception as e:
    print("❌ Ошибка подключения:", str(e))


