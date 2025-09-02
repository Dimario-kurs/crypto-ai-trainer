import ccxt
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import datetime
import json

# === 1. Модель ===
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

# Загружаем конфиг
with open("config.json", "r") as f:
    config = json.load(f)

# Восстанавливаем модель
model = Net(config["input_size"])
model.load_state_dict(torch.load("model_best.pth"))
model.eval()

# Загружаем скейлер
scaler = joblib.load("scaler.pkl")

# === 2. Binance данные ===
exchange = ccxt.binance()
symbol = "BTC/USDT"
timeframe = "15m"
limit = 200

ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
df["time"] = pd.to_datetime(df["time"], unit="ms")

# Для простоты — используем только базовые фичи
X_live = df[["open","high","low","close","volume"]].values
X_live = scaler.transform(X_live.astype(float))
X_live = torch.tensor(X_live, dtype=torch.float32)

# === 3. Симуляция торговли ===
balance = 1000.0   # стартовый баланс USDT
trade_size = 10.0  # размер сделки (USDT)
position = 0       # текущая позиция (BTC)
fee_rate = 0.001   # комиссия Binance 0.1%
results = []

for i in range(len(X_live)):
    row = X_live[i].unsqueeze(0)
    with torch.no_grad():
        pred = model(row)
        action = pred.argmax(1).item()  # 0=hold, 1=buy, 2=sell

    price = df.iloc[i]["close"]

    if action == 1 and balance >= trade_size:  # BUY
        btc_bought = (trade_size / price) * (1 - fee_rate)
        position += btc_bought
        balance -= trade_size
        results.append((df.iloc[i]["time"], "BUY", price, balance, position))

    elif action == 2 and position > 0:  # SELL
        usdt_received = position * price * (1 - fee_rate)
        balance += usdt_received
        results.append((df.iloc[i]["time"], "SELL", price, balance, 0))
        position = 0

# Закрываем позицию в конце
if position > 0:
    balance += position * df.iloc[-1]["close"] * (1 - fee_rate)
    results.append((df.iloc[-1]["time"], "SELL_END", df.iloc[-1]["close"], balance, 0))

# === 4. Результаты ===
res_df = pd.DataFrame(results, columns=["time","action","price","balance","position"])
res_df.to_csv("simulation_results.csv", index=False)

# График equity
plt.plot(res_df["time"], res_df["balance"], label="Balance")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("equity_curve.png")

print("✅ Simulation finished")
print("Final balance:", balance)



