import ccxt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# === Модель ===
class Net(nn.Module):
    def __init__(self, input_size, hidden=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# === Загружаем скейлер и модель ===
scaler = joblib.load("scaler.pkl")

input_size = 4  # open, high, low, close
model = Net(input_size)
state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

# === Подключение к Binance ===
exchange = ccxt.binance()

symbol = "BTC/USDT"
timeframe = "15m"
limit = 200  # количество последних свечей

ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# Превращаем в DataFrame
df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["time"] = pd.to_datetime(df["timestamp"], unit="ms")

# === Симуляция торговли ===
balance = 1000.0  # стартовый депозит USDT
trade_size = 10.0
positions = []
history = []

for _, row in df.iterrows():
    x = np.array([[row["open"], row["high"], row["low"], row["close"]]], dtype=np.float32)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled)

    with torch.no_grad():
        pred = model(x_tensor)
        action = pred.argmax(1).item()  # 0=держим, 1=покупка, 2=продажа

    price = row["close"]

    if action == 1 and balance >= trade_size:  # покупка
        positions.append(price)
        balance -= trade_size
        history.append((row["time"], "BUY", price, balance))
    elif action == 2 and positions:  # продажа
        entry = positions.pop(0)
        pnl = (price - entry) / entry * trade_size
        balance += trade_size + pnl
        history.append((row["time"], "SELL", price, balance))

# === Итоги ===
print(f"Стартовый баланс: 1000 USDT")
print(f"Финальный баланс: {balance:.2f} USDT")
print(f"Сделок всего: {len(history)}")

print("\nПоследние 5 сделок:")
for h in history[-5:]:
    print(h)

# Сохраняем историю в CSV
results = pd.DataFrame(history, columns=["time", "action", "price", "balance"])
results.to_csv("simulation_results.csv", index=False)
print("✅ Результаты сохранены в simulation_results.csv")


