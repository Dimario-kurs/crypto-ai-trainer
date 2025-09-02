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

# === Подключаемся к Bybit ===
exchange = ccxt.bybit({
    "apiKey": api_key,
    "secret": secret_key,
    "enableRateLimit": True,
    "options": {"defaultType": "spot"}
})

# Если есть прокси → добавляем
if proxy:
    exchange.session = requests.Session()
    exchange.session.proxies = {
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

