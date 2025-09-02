import ccxt
import torch
import pandas as pd
import numpy as np
from torch import nn
import joblib

# === Параметры ===
symbol = "BTC/USDT"
timeframe = "15m"
limit = 200
initial_balance = 1000
trade_size = 10

# === Подключение к Binance ===
exchange = ccxt.binance({"enableRateLimit": True})

# === Загружаем модель и scaler ===
model = torch.load("model.pth")
model.eval()
scaler = joblib.load("scaler.pkl")

# === Определяем архитектуру сети (как при обучении) ===
class Net(nn.Module):
    def __init__(self, input_size, hidden=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

# === Функция предсказания ===
def predict_signal(row):
    X = scaler.transform([row])
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X).argmax(1).item()
    return pred  # -1 = short, 0 = hold, 1 = long

# === Загружаем свечи ===
ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
df["ts"] = pd.to_datetime(df["ts"], unit="ms")

# === Баланс ===
balance = initial_balance
trades = []

for i in range(1, len(df)):
    row = df.iloc[i][["open","high","low","close","volume"]].values
    signal = predict_signal(row)

    if signal == 1:  # long
        profit = trade_size * ((df.iloc[i]["close"] - df.iloc[i]["open"]) / df.iloc[i]["open"])
    elif signal == -1:  # short
        profit = trade_size * ((df.iloc[i]["open"] - df.iloc[i]["close"]) / df.iloc[i]["open"])
    else:
        profit = 0

    balance += profit
    trades.append({
        "time": df.iloc[i]["ts"],
        "open": df.iloc[i]["open"],
        "close": df.iloc[i]["close"],
        "signal": signal,
        "profit": profit,
        "balance": balance
    })

# === Результаты ===
print(f"Начальный баланс: {initial_balance} USDT")
print(f"Финальный баланс: {balance:.2f} USDT")
print(f"Сделок: {len(trades)}")

print("\nПоследние 5 сделок:")
for t in trades[-5:]:
    print(t)

# === Сохраняем в CSV ===
results_df = pd.DataFrame(trades)
results_df.to_csv("simulation_results.csv", index=False)
print("✅ История сохранена в simulation_results.csv")
