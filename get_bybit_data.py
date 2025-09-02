import ccxt

# === Подключаемся к Binance ===
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"}
})

# === Получаем OHLCV по BTC/USDT ===
symbol = "BTC/USDT"
timeframe = "15m"   # можно "1h", "4h", "1d"
limit = 100

try:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    print(f"✅ Получили {len(ohlcv)} свечей для {symbol} ({timeframe})")
    print("Первая свеча:", ohlcv[0])
    print("Последняя свеча:", ohlcv[-1])
except Exception as e:
    print("❌ Ошибка:", str(e))
