import ccxt
import os
from dotenv import load_dotenv

# === Загружаем ключи из .env ===
load_dotenv()
api_key = os.getenv("BYBIT_API_KEY")
secret_key = os.getenv("BYBIT_SECRET_KEY")
proxy = os.getenv("PROXY")

if not api_key or not secret_key:
    raise ValueError("❌ Ключи API не найдены! Проверь .env")

# === Подключаемся к Bybit с прокси ===
exchange = ccxt.bybit({
    "apiKey": api_key,
    "secret": secret_key,
    "enableRateLimit": True,
    "params": {"defaultType": "spot"},
})

# Если задан прокси, добавляем его
if proxy:
    exchange.proxies = {
        "http": proxy,
        "https": proxy,
    }

# === Проверка соединения (баланс) ===
try:
    balance = exchange.fetch_balance()
    print("✅ Успешное подключение!")
    print("Баланс:", balance["total"])
except Exception as e:
    print("❌ Ошибка подключения:", str(e))
