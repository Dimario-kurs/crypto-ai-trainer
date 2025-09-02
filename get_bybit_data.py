import ccxt
import os
from dotenv import load_dotenv

# === Загружаем ключи из .env ===
load_dotenv()
api_key = os.getenv("BYBIT_API_KEY")
secret_key = os.getenv("BYBIT_SECRET_KEY")

if not api_key or not secret_key:
    raise ValueError("❌ Ключи API не найдены! Проверь файл .env")

# === Подключаемся к Bybit ===
exchange = ccxt.bybit({
    "apiKey": api_key,
    "secret": secret_key,
    "enableRateLimit": True,
    "options": {"defaultType": "spot"}  # можно 'linear' для фьючей
})

# === Проверка соединения (баланс) ===
try:
    balance = exchange.fetch_balance()
    print("✅ Подключение успешно!")
    print("Баланс:", balance["total"])
except Exception as e:
    print("❌ Ошибка подключения:", str(e))
