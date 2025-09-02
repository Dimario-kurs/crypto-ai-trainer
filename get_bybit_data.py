import ccxt
import pandas as pd
import numpy as np

# === 1. Подключение к Bybit ===
exchange = ccxt.bybit({"enableRateLimit": True})

# === 2. Получаем последние 500 свечей (15m) ===
symbol = "BTC/USDT"
timeframe = "15m"
limit = 500

ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# === 3. В DataFrame ===
df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
df["time"] = pd.to_datetime(df["time"], unit="ms")

# === 4. Добавляем фичи ===
# SMA
df["sma20"] = df["close"].rolling(20).mean()
df["sma50"] = df["close"].rolling(50).mean()

# ATR
df["hl"] = df["high"] - df["low"]
df["hc"] = (df["high"] - df["close"].shift()).abs()
df["lc"] = (df["low"] - df["close"].shift()).abs()
df["atr"] = df[["hl","hc","lc"]].max(axis=1).rolling(14).mean()

# RSI
delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / (loss + 1e-9)
df["rsi"] = 100 - (100 / (1 + rs))

# Spread (разница close - SMA20)
df["spread"] = df["close"] - df["sma20"]

# Чистим
df = df.fillna(0)

# === 5. Сохраняем ===
df.to_csv("bybit_data.csv", index=False)
print("✅ Данные сохранены в bybit_data.csv")
print(df.tail())
