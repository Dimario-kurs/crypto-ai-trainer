import ccxt
import os
from dotenv import load_dotenv

# === Загружаем ключи и прокси из .env ===
load_dotenv()
api_key = os.getenv("BYBIT_API_KEY")
secret_key = os.getenv("BYBIT_SECRET_KEY")
proxy = os.getenv("PROXY")

if not api_key or not secret_key:
    raise ValueError("❌ API ключи не найдены! Проверь .env")

# === Подключаемся к Bybit через рабочий прокси ===
exchange = ccxt.bybit({
    "apiKey": api_key,
    "secret": secret_key,
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
    "proxies": {
        "http": proxy,
        "https": proxy,
    },
    "requests_trust_env": False,   # игнор системных прокси, использовать только наш
})

# === Проверка соединения (баланс) ===
try:
    balance = exchange.fetch_balance()
    print("✅ Успешное подключение!")
    print("Баланс:", balance["total"])
except Exception as e:
    print("❌ Ошибка подключения:", e)

